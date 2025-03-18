# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

import numpy
import torch

from tqdm import tqdm
from tangermeme.io import extract_loci


class DataGenerator(torch.utils.data.Dataset):
	"""A data generator for BPNet inputs.

	This generator takes in an extracted set of sequences, output signals,
	and control signals, and will return a single element with random
	jitter and reverse-complement augmentation applied. Jitter is implemented
	efficiently by taking in data that is wider than the in/out windows by
	two times the maximum jitter and windows are extracted from that.
	Essentially, if an input window is 1000 and the maximum jitter is 128, one
	would pass in data with a length of 1256 and a length 1000 window would be
	extracted starting between position 0 and 256. This  generator must be 
	wrapped by a PyTorch generator object.

	Parameters
	----------
	sequences: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
		A one-hot encoded tensor of `n` example sequences, each of input 
		length `in_window`. See description above for connection with jitter.

	signals: torch.tensor, shape=(n, t, out_window+2*max_jitter)
		The signals to predict, usually counts, for `n` examples with
		`t` output tasks (usually 2 if stranded, 1 otherwise), each of 
		output length `out_window`. See description above for connection 
		with jitter.

	controls: torch.tensor, shape=(n, t, out_window+2*max_jitter) or None, optional
		The control signal to take as input, usually counts, for `n`
		examples with `t` strands and output length `out_window`. If
		None, does not return controls.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 0.

	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default is False.

	random_state: int or None, optional
		Whether to use a deterministic seed or not.
	"""

	def __init__(self, sequences, signals, controls=None, in_window=2114, 
		out_window=1000, max_jitter=0, reverse_complement=False, 
		random_state=None):
		self.in_window = in_window
		self.out_window = out_window
		self.max_jitter = max_jitter
		
		self.reverse_complement = reverse_complement
		self.random_state = numpy.random.RandomState(random_state)

		self.signals = signals
		self.controls = controls
		self.sequences = sequences

	def __len__(self):
		return len(self.sequences)

	def __getitem__(self, idx):
		i = self.random_state.choice(len(self.sequences))
		j = 0 if self.max_jitter == 0 else self.random_state.randint(
			self.max_jitter*2) 

		X = self.sequences[i][:, j:j+self.in_window]
		y = self.signals[i][:, j:j+self.out_window]

		if self.controls is not None:
			X_ctl = self.controls[i][:, j:j+self.in_window]

		if self.reverse_complement and self.random_state.choice(2) == 1:
			X = torch.flip(X, [0, 1])
			y = torch.flip(y, [0, 1])

			if self.controls is not None:
				X_ctl = torch.flip(X_ctl, [0, 1])

		if self.controls is not None:
			return X, X_ctl, y

		return X, y	


def PeakGenerator(loci, sequences, signals, controls=None, chroms=None, 
	in_window=2114, out_window=1000, max_jitter=128, reverse_complement=True, 
	min_counts=None, max_counts=None, random_state=None, pin_memory=True, 
	num_workers=0, batch_size=32, verbose=False):
	"""This is a constructor function that handles all IO.

	This function will extract signal from all signal and control files,
	pass that into a DataGenerator, and wrap that using a PyTorch data
	loader. This is the only function that needs to be used.

	Parameters
	----------
	loci: str or pandas.DataFrame or list/tuple of such
		Either the path to a bed file or a pandas DataFrame object containing
		three columns: the chromosome, the start, and the end, of each locus
		to train on. Alternatively, a list or tuple of strings/DataFrames where
		the intention is to train on the interleaved concatenation, i.e., when
		you want ot train on peaks and negatives.

	sequences: str or dictionary
		Either the path to a fasta file to read from or a dictionary where the
		keys are the unique set of chromosoms and the values are one-hot
		encoded sequences as numpy arrays or memory maps.

	signals: list of strs or list of dictionaries
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps.

	controls: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps. If None, no control tensor is returned. Default is None. 

	chroms: list or None, optional
		A set of chromosomes to extact loci from. Loci in other chromosomes
		in the locus file are ignored. If None, all loci are used. Default is
		None.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 128.

	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default is True.

	min_counts: float or None, optional
		The minimum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no minimum. Default 
		is None.

	max_counts: float or None, optional
		The maximum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no maximum. Default 
		is None.  

	random_state: int or None, optional
		Whether to use a deterministic seed or not.

	pin_memory: bool, optional
		Whether to pin page memory to make data loading onto a GPU easier.
		Default is True.

	num_workers: int, optional
		The number of processes fetching data at a time to feed into a model.
		If 0, data is fetched from the main process. Default is 0.

	batch_size: int, optional
		The number of data elements per batch. Default is 32.
	
	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.

	Returns
	-------
	X: torch.utils.data.DataLoader
		A PyTorch DataLoader wrapped DataGenerator object.
	"""

	X = extract_loci(loci=loci, sequences=sequences, signals=signals, 
		in_signals=controls, chroms=chroms, in_window=in_window, 
		out_window=out_window, max_jitter=max_jitter, min_counts=min_counts,
		max_counts=max_counts, ignore=list('QWERYUIOPSDFHJKLZXVBNM'), 
		verbose=verbose)

	if controls is not None:
		sequences, signals_, controls_ = X
	else:
		sequences, signals_ = X
		controls_ = None

	#sequences = sequences.float()

	X_gen = DataGenerator(sequences, signals_, controls=controls_, 
		in_window=in_window, out_window=out_window, max_jitter=max_jitter,
		reverse_complement=reverse_complement, random_state=random_state)

	X_gen = torch.utils.data.DataLoader(X_gen, pin_memory=pin_memory,
		num_workers=num_workers, batch_size=batch_size) 

	return X_gen
