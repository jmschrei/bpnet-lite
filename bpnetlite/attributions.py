# attributions.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import numba
import torch
import pandas
import logomaker

from tqdm import trange
from captum.attr import DeepLiftShap


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

	def forward(self, X, X_ctl=None, **kwargs):
		logits = self.model(X, X_ctl, **kwargs)[0]
		logits = logits.reshape(X.shape[0], -1)
		logits = logits - torch.mean(logits, dim=-1, keepdims=True)
		l = torch.clone(logits).detach()

		y = torch.exp(l - torch.logsumexp(l, dim=-1, keepdims=True))
		return (logits * y).sum(axis=-1, keepdims=True)


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


@numba.jit('void(int64, int64[:], int64[:], int32[:, :], int32[:,], int32[:, :], float32[:, :, :])')
def _fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, counters, shuffled_sequences):
	"""An internal function for fast shuffling using numba."""

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

	if not isinstance(random_state, numpy.random.RandomState):
		random_state = numpy.random.RandomState(random_state)

	chars, idxs = torch.unique(sequence.argmax(axis=0), return_inverse=True)
	chars, idxs = chars.numpy(), idxs.numpy()

	next_idxs = numpy.zeros((len(chars), sequence.shape[1]), dtype=numpy.int32)
	next_idxs_counts = numpy.zeros(max(chars)+1, dtype=numpy.int32)

	for char in chars:
		next_idxs_ = numpy.where(idxs[:-1] == char)[0]
		n = len(next_idxs_)

		next_idxs[char][:n] = next_idxs_ + 1
		next_idxs_counts[char] = n

	shuffled_sequences = numpy.zeros((n_shuffles, *sequence.shape), dtype=numpy.float32)
	counters = numpy.zeros((n_shuffles, len(chars)), dtype=numpy.int32)

	_fast_shuffle(n_shuffles, chars, idxs, next_idxs, next_idxs_counts, 
		counters, shuffled_sequences)
	
	shuffled_sequences = torch.from_numpy(shuffled_sequences)
	return shuffled_sequences


def calculate_attributions(model, X, args=None, model_output="profile", 
	hypothetical=False, n_shuffles=20, return_references=False, verbose=False, 
	random_state=None):
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

	model_output: str, "profile" or "count", optional
		If "profile", wrap the model using ProfileWrapper and calculate
		attributions with respect to the profile. If "count", wrap the model
		using CountWrapper and calculate attributions with respect to the
		count. Default is "profile".

	hypothetical: bool, optional
		Whether to return attributions for all possible characters at each
		position or only for the character that is actually at the sequence.
		Practically, whether to return the returned attributions from captum
		with the one-hot encoded sequence. Default is False.

	n_shuffles: int, optional
		The number of dinucleotide shuffles to return. Default is 10.

	batch_size: int, optional
		The number of attributions to calculate at the same time. This is
		limited by GPU memory. Default is 8.

	return_references: bool, optional
		Whether to return the references that were generated during this
		process.

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

	if model_output == "profile":
		wrapper = ProfileWrapper(model)
	elif model_output == "count":
		wrapper = CountWrapper(model)
	else:
		raise ValueError("model_output must be one of 'profile' or 'count'.")

	ig = DeepLiftShap(wrapper)
	
	attributions = []
	references = []
	with torch.no_grad():
		for i in trange(len(X), disable=not verbose):
			X_ = X[i:i+1]
			reference = dinucleotide_shuffle(X_[0], n_shuffles=n_shuffles, 
				random_state=random_state).cuda()

			X_ = X_.cuda()

			if args is None:
				args_ = None
			else:
				args_ = tuple([arg[i:i+1].cuda() for arg in args])
						
			attr = ig.attribute(X_, reference, target=0, 
				additional_forward_args=args_, 
				custom_attribution_func=hypothetical_attributions)

			if not hypothetical:
				attr = (attr * X_)
			
			if return_references:
				references.append(reference.unsqueeze(0))

			attributions.append(attr.cpu())
	
	attributions = torch.cat(attributions)

	if return_references:
		return attributions, torch.cat(references)
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
