# attributions.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import numba
import torch
import pandas
import logomaker

from tqdm import tqdm
from tqdm import trange

from captum.attr import DeepLiftShap as CaptumDeepLiftShap
from numba import NumbaDeprecationWarning

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)


class DeepLiftShap():
	"""A vectorized version of the DeepLIFT/SHAP algorithm from Captum.

	This approach is based on the Captum approach of assigning hooks to
	layers that modify the gradients to implement the rescale rule. This
	implementation is vectorized in a manner that can accept unique references
	for each example to be explained as well as multiple references for each
	example.

	The implementation is minimal and currently only supports the operations
	used in bpnet-lite. This is not meant to be a general-purpose implementation
	of the algorithm and may not work with custom architectures.
	

	Parameters
	----------
	model: bpnetlite.BPNet or bpnetlite.ChromBPNet
		A BPNet or ChromBPNet module as implemented in this repo.

	attribution_func: function or None, optional
		This function is used to aggregate the gradients after calculation.
		Useful when trying to handle the implications of one-hot encodings. If
		None, return the gradients as calculated. Default is None.

	eps: float, optional
		An epsilon with which to threshold gradients to ensure that there
		isn't an explosion. Default is 1e-10.

	warning_threshold: float, optional
		A threshold on the convergence delta that will always raise a warning
		if the delta is larger than it. Normal deltas are in the range of
		1e-6 to 1e-8. Note that convergence deltas are calculated on the
		gradients prior to the attribution_func being applied to them. Default 
		is 0.001. 

	verbose: bool, optional
		Whether to print the convergence delta for each example that is
		explained, regardless of whether it surpasses the warning threshold.
		Note that convergence deltas are calculated on the gradients prior to 
		the attribution_func being applied to them. Default is False.
	"""

	def __init__(self, model, attribution_func=None, eps=1e-6, 
		warning_threshold=0.001, verbose=False):
		for module in model.named_modules():
			if isinstance(module[1], torch.nn.modules.pooling._MaxPoolNd):
				raise ValueError("Cannot use this implementation of " + 
					"DeepLiftShap with max pooling layers. Please use the " +
					"implementation in Captum.")

		self.model = model
		self.attribution_func = attribution_func
		self.eps = eps
		self.warning_threshold = warning_threshold
		self.verbose = verbose

		self.forward_handles = []
		self.backward_handles = []

	def attribute(self, inputs, baselines, args=None):
		assert inputs.shape[1:] == baselines.shape[2:]
		n_inputs, n_baselines = baselines.shape[:2]

		inputs = inputs.repeat_interleave(n_baselines, dim=0).requires_grad_()
		baselines = baselines.reshape(-1, *baselines.shape[2:]).requires_grad_()

		if args is not None:
			args = (arg.repeat_interleave(n_baselines, dim=0) for arg in args)
		else:
			args = None

		###

		try:
			self.model.apply(self._register_hooks)
			inputs_ = torch.cat([inputs, baselines])

			# Calculate the gradients using the rescale rule
			with torch.autograd.set_grad_enabled(True):
				if args is not None:
					args = (torch.cat([arg, arg]) for arg in 
						args)
					outputs = self.model(inputs_, *args)
				else:
					outputs = self.model(inputs_)

				outputs_ = torch.chunk(outputs, 2)[0].sum()
				gradients = torch.autograd.grad(outputs_, inputs)[0]

			output_diff = torch.sub(*torch.chunk(outputs[:,0], 2))
			input_diff = torch.sum((inputs - baselines) * gradients, dim=(1, 2)) 
			convergence_deltas = output_diff - input_diff
			
			if any(convergence_deltas > self.warning_threshold):
				raise Warning("Convergence deltas too high: ", 
					convergence_deltas)

			if self.verbose:
				print(convergence_deltas)

			# Process the gradients to get attributions
			if self.attribution_func is None:
				attributions = gradients
			else:
				attributions = self.attribution_func((gradients,), (inputs,), 
					(baselines,))[0]

		finally:
			for forward_handle in self.forward_handles:
				forward_handle.remove()
			for backward_handle in self.backward_handles:
				backward_handle.remove()

		###

		attr_shape = (n_inputs, n_baselines) + attributions.shape[1:]
		attributions = torch.mean(attributions.view(attr_shape), dim=1, 
			keepdim=False)
		return attributions

	def _forward_pre_hook(self, module, inputs):
		module.input = inputs[0].clone().detach()

	def _forward_hook(self, module, inputs, outputs):
		module.output = outputs.clone().detach()

	def _backward_hook(self, module, grad_input, grad_output):
		delta_in_ = torch.sub(*module.input.chunk(2))
		delta_out_ = torch.sub(*module.output.chunk(2))

		delta_in = torch.cat([delta_in_, delta_in_])
		delta_out = torch.cat([delta_out_, delta_out_])

		delta = delta_out / delta_in

		grad_input = (torch.where(
			abs(delta_in) < self.eps, grad_input[0], grad_output[0] * delta),
		)
		return grad_input

	def _can_register_hook(self, module):
		return not (len(module._backward_hooks) > 0 or not isinstance(module, 
			(torch.nn.ReLU, _ProfileLogitScaling)))

	def _register_hooks(self, module, attribute_to_layer_input=True):
		if not self._can_register_hook(module) or (
			not attribute_to_layer_input and module is self.layer
		):
			return

		# adds forward hook to leaf nodes that are non-linear
		forward_handle = module.register_forward_hook(self._forward_hook)
		pre_forward_handle = module.register_forward_pre_hook(self._forward_pre_hook)
		backward_handle = module.register_full_backward_hook(self._backward_hook)

		self.forward_handles.append(forward_handle)
		self.forward_handles.append(pre_forward_handle)
		self.backward_handles.append(backward_handle)


class _ProfileLogitScaling(torch.nn.Module):
	"""This ugly class is necessary because of Captum.

	Captum internally registers classes as linear or non-linear. Because the
	profile wrapper performs some non-linear operations, those operations must
	be registered as such. However, the inputs to the wrapper are not the
	logits that are being modified in a non-linear manner but rather the
	original sequence that is subsequently run through the model. Hence, this
	object will contain all of the operations performed on the logits and
	can be registered.


	Parameters
	----------
	logits: torch.Tensor, shape=(-1, -1)
		The logits as they come out of a Chrom/BPNet model.
	"""

	def __init__(self):
		super(_ProfileLogitScaling, self).__init__()

	def forward(self, logits):
		y = torch.exp(logits - torch.logsumexp(logits, dim=-1, keepdims=True))
		return logits * y.detach()


class ProfileWrapper(torch.nn.Module):
	"""A wrapper class that returns transformed profiles.

	This class takes in a trained model and returns the weighted softmaxed
	outputs of the first dimension. Specifically, it takes the predicted
	"logits" and takes the dot product between them and the softmaxed versions
	of those logits. This is for convenience when using captum to calculate
	attribution scores.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super(ProfileWrapper, self).__init__()
		self.model = model
		self.scaling = _ProfileLogitScaling()

	def forward(self, X, X_ctl=None, **kwargs):
		logits = self.model(X, X_ctl, **kwargs)[0]
		logits = logits.reshape(X.shape[0], -1)
		logits = logits - torch.mean(logits, dim=-1, keepdims=True)

		y = self.scaling(logits)
		return y.sum(axis=-1, keepdims=True)


class CountWrapper(torch.nn.Module):
	"""A wrapper class that only returns the predicted counts.

	This class takes in a trained model and returns only the second output.
	For BPNet models, this means that it is only returning the count
	predictions. This is for convenience when using captum to calculate
	attribution scores.

	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super(CountWrapper, self).__init__()
		self.model = model

	def forward(self, X, X_ctl=None, **kwargs):
		return self.model(X, X_ctl, **kwargs)[1]


def hypothetical_attributions(multipliers, inputs, baselines):
	"""A function for aggregating contributions into hypothetical attributions.

	When handling categorical data, like one-hot encodings, the attributions
	returned by a method like DeepLIFT/SHAP may need to be modified slightly.
	Specifically, one needs to account for each nucleotide change actually
	being the addition of one category AND the subtraction of another category.
	Basically, once you've calculated the multipliers, you need to subtract
	out the contribution of the nucleotide actually present and then add in
	the contribution of the nucleotide you are becomming.

	These values are then averaged over all references.


	Parameters
	----------
	multipliers: torch.tensor, shape=(n_baselines, 4, length)
		The multipliers determined by DeepLIFT

	inputs: torch.tensor, shape=(n_baselines, 4, length)
		The one-hot encoded sequence being explained, copied several times.

	baselines: torch.tensor, shape=(n_baselines, 4, length)
		The one-hot encoded baseline sequences.


	Returns
	-------
	projected_contribs: torch.tensor, shape=(1, 4, length)
		The attribution values for each nucleotide in the input.
	"""

	projected_contribs = torch.zeros_like(baselines[0], dtype=baselines[0].dtype)
	
	for i in range(inputs[0].shape[1]):
		hypothetical_input = torch.zeros_like(inputs[0], dtype=baselines[0].dtype)
		hypothetical_input[:, i] = 1.0
		hypothetical_diffs = hypothetical_input - baselines[0]
		hypothetical_contribs = hypothetical_diffs * multipliers[0]
		
		projected_contribs[:, i] = torch.sum(hypothetical_contribs, dim=1)

	return (projected_contribs,)


params = 'void(int64, int64[:], int64[:], int32[:, :], int32[:,], '
params += 'int32[:, :], float32[:, :, :], int32)'
@numba.jit(params, nopython=False)
def _fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, 
	counters, shuffled_sequences, random_state):
	"""An internal function for fast shuffling using numba."""

	numpy.random.seed(random_state)

	for i in range(n_shuffles):
		for char in chars:
			n = next_idxs_counts[char]

			next_idxs_ = numpy.arange(n)
			next_idxs_[:-1] = numpy.random.permutation(n-1)  # Keep last index same
			next_idxs[char, :n] = next_idxs[char, :n][next_idxs_]

		idx = 0
		shuffled_sequences[i, idxs[idx], 0] = 1
		for j in range(1, len(idxs)):
			char = idxs[idx]
			count = counters[i, char]
			idx = next_idxs[char, count]

			counters[i, char] += 1
			shuffled_sequences[i, idxs[idx], j] = 1


def dinucleotide_shuffle(sequence, n_shuffles=10, random_state=None):
	"""Given a one-hot encoded sequence, dinucleotide shuffle it.

	This function takes in a one-hot encoded sequence (not a string) and
	returns a set of one-hot encoded sequences that are dinucleotide
	shuffled. The approach constructs a transition matrix between
	nucleotides, keeps the first and last nucleotide constant, and then
	randomly at uniform selects transitions until all nucleotides have
	been observed. This is a Eulerian path. Because each nucleotide has
	the same number of transitions into it as out of it (except for the
	first and last nucleotides) the greedy algorithm does not need to
	check at each step to make sure there is still a path.

	This function has been adapted to work on PyTorch tensors instead of
	numpy arrays. Code has been adapted from
	https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

	Parameters
	----------
	sequence: torch.tensor, shape=(k, -1)
		The one-hot encoded sequence. k is usually 4 for nucleotide sequences
		but can be anything in practice.

	n_shuffles: int, optional
		The number of dinucleotide shuffles to return. Default is 10.

	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism. If None, the
		process is not deterministic. Default is None. 

	Returns
	-------
	shuffled_sequences: torch.tensor, shape=(n_shuffles, k, -1)
		The shuffled sequences.
	"""

	if random_state is None:
		random_state = numpy.random.randint(0, 9999999)

	chars, idxs = torch.unique(sequence.argmax(axis=0), return_inverse=True)
	chars, idxs = chars.numpy(), idxs.numpy()

	next_idxs = numpy.zeros((len(chars), sequence.shape[1]), dtype=numpy.int32)
	next_idxs_counts = numpy.zeros(max(chars)+1, dtype=numpy.int32)

	for char in chars:
		next_idxs_ = numpy.where(idxs[:-1] == char)[0]
		n = len(next_idxs_)

		next_idxs[char][:n] = next_idxs_ + 1
		next_idxs_counts[char] = n

	shuffled_sequences = numpy.zeros((n_shuffles, *sequence.shape), 
		dtype=numpy.float32)
	counters = numpy.zeros((n_shuffles, len(chars)), dtype=numpy.int32)

	_fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, 
		counters, shuffled_sequences, random_state)
	
	shuffled_sequences = torch.from_numpy(shuffled_sequences)
	return shuffled_sequences


def create_references(X, algorithm='dinucleotide', n_shuffles=20, 
	random_state=None):
	"""Generate references for a batch of sequences.

	This function will take in a batch of sequences and return a tensor of
	references given the specified strategy. The returned tensor will have an
	additional dimension corresponding to the number of shuffles.


	Parameters
	----------
	X: torch.tensor, shape=(-1, 4, -1)
		A one-hot encoded sequence input to the model.

	algorithm: "dinucleotide", "freq", "zeros", torch.Tensor
		The algorithm to use for generating references. Can be "dinucleotide",
		"freq", or "zero". If "dinucleotide", generate dinucleotide shuffled 
		sequences for each input sequence. If "freq", set each value to
		0.25. If "zeros", set each value to 0. If a tensor is passed in, assume
		that it is the references being passed by the user and simply return it.
		Default is 'dinucleotide'.

	n_shuffles: int, optional
		The number of dinucleotide shuffles to return. Only needed when
		algorithm="dinucleoide". Default is 20.

	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism. If None, the
		process is not deterministic. Default is None. 


	Returns
	-------
	_references: torch.tensor, shape=(-1, 4, -1)
		A one-hot encoded sequence of shuffles.
	"""

	if algorithm == 'dinucleotide':
		references = torch.stack([dinucleotide_shuffle(x.cpu(), 
			n_shuffles=n_shuffles, random_state=random_state).to(
			X.dtype) for x in X]).to(X.device)
	elif algorithm == 'zero':
		references = torch.zeros(e-s, n_shuffles, *X.shape[1:], 
			dtype=X.dtype, device=X.device)
	elif algorithm == 'freq':
		references = torch.zeros(e-s, n_shuffles, *X.shape[1:], 
			dtype=X.dtype, device=X.device) + 0.25	
	else:
		raise ValueError("Algorithm must be one of " +
			"'dinucleotide', 'zero', or 'freq'.")

	return references


@torch.no_grad()
def ism(model, X_0, args=None, batch_size=128, verbose=False):
	"""In-silico mutagenesis saliency scores. 

	This function will perform in-silico mutagenesis in a naive manner, i.e.,
	where each input sequence has a single mutation in it and the entirety
	of the sequence is run through the given model. It returns the ISM score,
	which is a vector of the L2 difference between the reference sequence 
	and the perturbed sequences with one value for each output of the model.

	Parameters
	----------
	model: torch.nn.Module
		The model to use.

	X_0: torch.tensor, shape=(batch_size, 4, seq_len)
		The one-hot encoded sequence to calculate saliency for.

	args: tuple or None, optional
		Additional arguments to pass into the forward function. If None,
		pass nothing additional in. Default is None.

	batch_size: int, optional
		The size of the batches.

	verbose: bool, optional
		Whether to display a progress bar as positions are being mutated. One
		display bar will be printed for each sequence being analyzed. Default
		is False.

	Returns
	-------
	X_ism: torch.tensor, shape=(batch_size, 4, seq_len)
		The saliency score for each perturbation.
	"""
	
	n_seqs, n_choices, seq_len = X_0.shape
	X_idxs = X_0.argmax(axis=1)

	n = seq_len*(n_choices-1)
	X = torch.tile(X_0, (n, 1, 1))
	X = X.reshape(n, n_seqs, n_choices, seq_len).permute(1, 0, 2, 3)

	for i in range(n_seqs):
		for k in range(1, n_choices):
			idx = numpy.arange(seq_len)*(n_choices-1) + (k-1)

			X[i, idx, X_idxs[i], numpy.arange(seq_len)] = 0
			X[i, idx, (X_idxs[i]+k) % n_choices, numpy.arange(seq_len)] = 1


	model = model.eval()

	if args is None:
		reference = model(X_0).unsqueeze(1)
	else:
		reference = model(X_0, *args).unsqueeze(1)

	starts = numpy.arange(0, X.shape[1], batch_size)
	isms = []
	for i in range(n_seqs):
		ism = []
		for start in tqdm(starts, disable=not verbose):
			X_ = X[i, start:start+batch_size].cuda()

			if args is None:
				y = model(X_)
			else:
				args_ = tuple(a[i:i+1] for a in args)
				y = model(X_, *args_)

			ism.append(y - reference[i])

		ism = torch.cat(ism)
		if len(ism.shape) > 1:
			ism = ism.sum(dim=list(range(1, len(ism.shape))))
		isms.append(ism)

	isms = torch.stack(isms)
	isms = isms.reshape(n_seqs, seq_len, n_choices-1)

	j_idxs = torch.arange(n_seqs*seq_len)
	X_ism = torch.zeros(n_seqs*seq_len, n_choices, device='cuda')
	for i in range(1, n_choices):
		i_idxs = (X_idxs.flatten() + i) % n_choices
		X_ism[j_idxs, i_idxs] = isms[:, :, i-1].flatten()

	X_ism = X_ism.reshape(n_seqs, seq_len, n_choices).permute(0, 2, 1)
	X_ism = (X_ism - X_ism.mean(dim=1, keepdims=True))
	return X_ism
	

def calculate_attributions(model, X, args=None, model_output="profile", 
	attribution_func=hypothetical_attributions, hypothetical=False,
	algorithm="deepliftshap", references='dinucleotide', n_shuffles=20, 
	batch_size=32, return_references=False, warning_threshold=0.001, 
	print_convergence_deltas=False, verbose=False, random_state=None):
	"""Calculate attributions using DeepLift/Shap and a given model. 

	This function will calculate DeepLift/Shap attributions on a set of
	sequences. It assumes that the model returns "logits" in the first output,
	not softmax probabilities, and count predictions in the second output.
	It will create GC-matched negatives to use as a reference and proceed
	using the given batch size.


	Parameters
	----------
	model: torch.nn.Module
		The model to use, either BPNet or one of it's variants.

	X: torch.tensor, shape=(-1, 4, -1)
		A one-hot encoded sequence input to the model.

	args: tuple or None, optional
		Additional arguments to pass into the forward function. If None,
		pass nothing additional in. Default is None.

	model_output: None, "profile" or "count", optional
		If None, then no wrapper is applied to the model. If "profile", wrap 
		the model using ProfileWrapper and calculate attributions with respect 
		to the profile. If "count", wrap the model using CountWrapper and 
		calculate attributions with respect to the count. Default is "profile".

	hypothetical: bool, optional
		Whether to return attributions for all possible characters at each
		position or only for the character that is actually at the sequence.
		Practically, whether to return the returned attributions from captum
		with the one-hot encoded sequence. Default is False.

	algorithm: "deepliftshap" or "ism", optional
		The algorithm to use to calculate attributions. Must be one of
		"deepliftshap", which uses the DeepLiftShap object, or "ism", which
		uses the naive_ism method. Default is "deepliftshap".

	references: "dinucleotide", "freq", "zeros", optional
		The reference to use when algorithm is "deepliftshap". If "dinucleotide"
		generate dinucleotide shuffled sequences. If "freq", set each value to
		0.25. If "zeros", set each value to 0.

	n_shuffles: int, optional
		The number of dinucleotide shuffles to return. Only needed when
		algorithm is "deepliftshap". Default is 10.

	batch_size: int, optional
		The number of attributions to calculate at the same time. This is
		limited by GPU memory. Default is 8.

	return_references: bool, optional
		Whether to return the references that were generated during this
		process.

	warning_threshold: float, optional
		A threshold on the convergence delta that will always raise a warning
		if the delta is larger than it. Normal deltas are in the range of
		1e-6 to 1e-8. Note that convergence deltas are calculated on the
		gradients prior to the attribution_func being applied to them. Default 
		is 0.001. 

	print_convergence_deltas: bool, optional
		Whether to print the convergence deltas for each example when using
		DeepLiftShap. Default is False.

	verbose: bool, optional
		Whether to display a progress bar.

	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism. If None, the
		process is not deterministic. Default is None. 


	Returns
	-------
	attributions: torch.tensor
		The attributions calculated for each input sequence, with the same
		shape as the input sequences.

	references: torch.tensor, optional
		The references used for each input sequence, with the shape
		(n_input_sequences, n_shuffles, 4, length). Only returned if
		`return_references = True`. 
	"""

	if model_output is None:
		wrapper = model
	elif model_output == "profile":
		wrapper = ProfileWrapper(model)
	elif model_output == "count":
		wrapper = CountWrapper(model)
	else:
		raise ValueError("model_output must be None, 'profile' or 'count'.")

	attributions = []
	references_ = []
	dev = next(model.parameters()).device

	for i in trange(0, len(X), batch_size, disable=not verbose):
		s, e = i, min(i+batch_size, len(X))
		_X = X[s:e].to(dev)
		args_ = None if args is None else tuple([a[s:e].to(dev) for a in args])

		if algorithm == 'deepliftshap':
			# Calculate references
			if isinstance(references, torch.Tensor):
				_references = references[s:e]
			else:
				_references = create_references(_X, algorithm=references, 
					n_shuffles=n_shuffles)

			# Run DeepLiftShap
			dl = DeepLiftShap(wrapper, attribution_func=attribution_func, 
				warning_threshold=warning_threshold, 
				verbose=print_convergence_deltas)			
			attr = dl.attribute(_X, _references, args=args_)
		
			if return_references:
				references_.append(_references.cpu().detach())
	
		elif algorithm == 'ism':
			attr = ism(wrapper, _X, args=args, verbose=False)
		
		else:
			raise ValueError("Must pass in one of 'deepliftshap' or 'ism'.")

		attr = attr if hypothetical else attr * _X
		attributions.append(attr.cpu().detach())

	attributions = torch.cat(attributions)
	
	if return_references:
		return attributions, torch.cat(references_)
	return attributions


def plot_attributions(X_attr, ax):
	"""Plot the attributions using logomaker.

	Takes in a matrix of attributions and plots the attribution-weighted
	sequence using logomaker. This is a convenience function.


	Parameters
	----------
	X_attr: torch.tensor, shape=(4, -1)
		A tensor of the attributions. Can be either the hypothetical
		attributions, where the entire matrix has values, or the projected
		attributions, where only the actual bases have their attributions
		stored, i.e., 3 values per column are zero.
	"""

	df = pandas.DataFrame(X_attr.T, columns=['A', 'C', 'G', 'T'])
	df.index.name = 'pos'
	
	logo = logomaker.Logo(df, ax=ax)
	logo.style_spines(visible=False)
	return logo
