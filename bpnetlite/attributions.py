# attributions.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import numba
import torch

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
		
		y = torch.nn.functional.log_softmax(logits, dim=-1)
		y = torch.exp(y)
		return (logits * y).sum(axis=-1).unsqueeze(-1)

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
	hypothetical=False, n_shuffles=20, verbose=False, random_state=None):
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

	verbose: bool, optional
		Whether to display a progress bar.

	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism. If None, the
		process is not deterministic. Default is None. 
	"""

	if model_output == "profile":
		wrapper = ProfileWrapper(model)
	elif model_output == "count":
		wrapper = CountWrapper(model)
	else:
		raise ValueError("model_output must be one of 'profile' or 'count'.")

	ig = DeepLiftShap(wrapper)
	
	attributions = []
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
						
			attr = ig.attribute(X_, reference, target=0, additional_forward_args=args_)

			if not hypothetical:
				attr = (attr * X_)
			
			attributions.append(attr.cpu())
	
	attributions = torch.cat(attributions)    
	return attributions