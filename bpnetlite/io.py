# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>
# Code adapted from Alex Tseng, Avanti Shrikumar, and Ziga Avsec

import numpy
import torch
import pandas

import pyfaidx
import pyBigWig

from tqdm import tqdm
from numba import njit


def read_meme(filename):
	"""Read a MEME file and return a dictionary of PWMs.

	This method takes in the filename of a MEME-formatted file to read in
	and returns a dictionary of the PWMs where the keys are the metadata
	line and the values are the PWMs.


	Parameters
	----------
	filename: str
		The filename of the MEME-formatted file to read in


	Returns
	-------
	motifs: dict
		A dictionary of the motifs in the MEME file.
	"""

	motifs = {}

	with open(filename, "r") as infile:
		motif, width, i = None, None, 0

		for line in infile:
			if motif is None:
				if line[:5] == 'MOTIF':
					motif = line.split()[1]
				else:
					continue

			elif width is None:
				if line[:6] == 'letter':
					width = int(line.split()[5])
					pwm = numpy.zeros((width, 4))

			elif i < width:
				pwm[i] = list(map(float, line.split()))
				i += 1

			else:
				motifs[motif] = pwm
				motif, width, i = None, None, 0

	return motifs


def one_hot_encode(sequence, alphabet=['A', 'C', 'G', 'T'], dtype='int8', 
	desc=None, verbose=False, **kwargs):
	"""Converts a string or list of characters into a one-hot encoding.

	This function will take in either a string or a list and convert it into a
	one-hot encoding. If the input is a string, each character is assumed to be
	a different symbol, e.g. 'ACGT' is assumed to be a sequence of four 
	characters. If the input is a list, the elements can be any size.

	Although this function will be used here primarily to convert nucleotide
	sequences into one-hot encoding with an alphabet of size 4, in principle
	this function can be used for any types of sequences.

	Parameters
	----------
	sequence : str or list
		The sequence to convert to a one-hot encoding.

	alphabet : set or tuple or list
		A pre-defined alphabet where the ordering of the symbols is the same
		as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
		the returned tensor will have a 1 at index 0 if the character was 'A'.
		Characters outside the alphabet are ignored and none of the indexes are
		set to 1. Default is ['A', 'C', 'G', 'T'].

	dtype : str or numpy.dtype, optional
		The data type of the returned encoding. Default is int8.

	desc : str or None, optional
		The title to display in the progress bar.

	verbose : bool or str, optional
		Whether to display a progress bar. If a string is passed in, use as the
		name of the progressbar. Default is False.

	kwargs : arguments
		Arguments to be passed into tqdm. Default is None.

	Returns
	-------
	ohe : numpy.ndarray
		A binary matrix of shape (alphabet_size, sequence_length) where
		alphabet_size is the number of unique elements in the sequence and
		sequence_length is the length of the input sequence.
	"""

	d = verbose is False
	alphabet_lookup = {char: i for i, char in enumerate(alphabet)}

	ohe = numpy.zeros((len(sequence), len(alphabet)), dtype=dtype)
	for i, char in tqdm(enumerate(sequence), disable=d, desc=desc, **kwargs):
		idx = alphabet_lookup.get(char, -1)
		if idx != -1:
			ohe[i, idx] = 1
	
	return ohe


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
		j = 0 if self.max_jitter == 0 else self.random_state.randint(self.max_jitter*2) 

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


def extract_loci(loci, sequences, signals=None, controls=None, chroms=None, 
	in_window=2114, out_window=1000, max_jitter=0, min_counts=None,
	max_counts=None, n_loci=None, verbose=False):
	"""Extract sequences and signals at coordinates from a locus file.

	This function will take in genome-wide sequences, signals, and optionally
	controls, and extract the values of each at the coordinates specified in
	the locus file/s and return them as tensors.

	Signals and controls are both lists with the length of the list, n_s
	and n_c respectively, being the middle dimension of the returned
	tensors. Specifically, the returned tensors of size 
	(len(loci), n_s/n_c, (out_window/in_wndow)+max_jitter*2).

	The values for sequences, signals, and controls, can either be filepaths
	or dictionaries of numpy arrays or a mix of the two. When a filepath is 
	passed in it is loaded using pyfaidx or pyBigWig respectively.   

	Parameters
	----------
	loci: str or pandas.DataFrame or list/tuple of such
		Either the path to a bed file or a pandas DataFrame object containing
		three columns: the chromosome, the start, and the end, of each locus
		to train on. Alternatively, a list or tuple of strings/DataFrames where
		the intention is to train on the interleaved concatenation, i.e., when
		you want to train on peaks and negatives.

	sequences: str or dictionary
		Either the path to a fasta file to read from or a dictionary where the
		keys are the unique set of chromosoms and the values are one-hot
		encoded sequences as numpy arrays or memory maps.

	signals: list of strs or list of dictionaries or None, optional
		A list of filepaths to bigwig files, where each filepath will be read
		using pyBigWig, or a list of dictionaries where the keys are the same
		set of unique chromosomes and the values are numpy arrays or memory
		maps. If None, no signal tensor is returned. Default is None.

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
		midpoints that are passed in. Default is 0.

	min_counts: float or None, optional
		The minimum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no minimum. Default 
		is None.

	max_counts: float or None, optional
		The maximum number of counts, summed across the length of each example
		and across all tasks, needed to be kept. If None, no maximum. Default 
		is None.  

	n_loci: int or None, optional
		A cap on the number of loci to return. Note that this is not the
		number of loci that are considered. The difference is that some
		loci may be filtered out for various reasons, and those are not
		counted towards the total. If None, no cap. Default is None.

	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.

	Returns
	-------
	seqs: torch.tensor, shape=(n, 4, in_window+2*max_jitter)
		The extracted sequences in the same order as the loci in the locus
		file after optional filtering by chromosome.

	signals: torch.tensor, shape=(n, len(signals), out_window+2*max_jitter)
		The extracted signals where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of signal files.
		If no signal files are given, this is not returned.

	controls: torch.tensor, shape=(n, len(controls), out_window+2*max_jitter)
		The extracted controls where the first dimension is in the same order
		as loci in the locus file after optional filtering by chromosome and
		the second dimension is in the same order as the list of control files.
		If no control files are given, this is not returned.
	"""

	seqs, signals_, controls_ = [], [], []
	in_width, out_width = in_window // 2, out_window // 2

	# Load the sequences
	if isinstance(sequences, str):
		sequences = pyfaidx.Fasta(sequences)

	names = ['chrom', 'start', 'end']
	if not isinstance(loci, (tuple, list)):
		loci = [loci]

	loci_dfs = []
	for i, df in enumerate(loci):
		if isinstance(df, str):
			df = pandas.read_csv(df, sep='\t', usecols=[0, 1, 2], 
				header=None, index_col=False, names=names)
		elif isinstance(df, pandas.DataFrame):
			df = df.iloc[:, [0, 1, 2]].copy()

		df['idx'] = numpy.arange(len(df)) * len(loci) + i
		loci_dfs.append(df)

	loci = pandas.concat(loci_dfs).set_index("idx").sort_index().reset_index(drop=True)

	if chroms is not None:
		loci = loci[numpy.isin(loci['chrom'], chroms)]

	# Load the signal and optional control tracks if filenames are given
	_signals = []
	if signals is not None:
		for i, signal in enumerate(signals):
			if isinstance(signal, str):
				signal = pyBigWig.open(signal)
			_signals.append(signal)

		signals = _signals
				
	_controls = []
	if controls is not None:
		for i, control in enumerate(controls):
			if isinstance(control, str):
				control = pyBigWig.open(control, "r")
			_controls.append(control)
			
		controls = _controls

	desc = "Loading Loci"
	d = not verbose

	max_width = max(in_width, out_width)
	loci_count = 0
	for chrom, start, end in tqdm(loci.values, disable=d, desc=desc):
		mid = start + (end - start) // 2

		if start - max_width - max_jitter < 0:
			continue

		if end + max_width + max_jitter >= len(sequences[chrom]):
			continue

		if n_loci is not None and loci_count == n_loci:
			break 

		start = mid - out_width - max_jitter
		end = mid + out_width + max_jitter

		# Extract the signal from each of the signal files
		if signals is not None:
			signals_.append([])
			for signal in signals:
				if isinstance(signal, dict):
					signal_ = signal[chrom][start:end]
				else:
					try:
						signal_ = signal.values(chrom, start, end, numpy=True)
					except:
						print(f"Warning: {chrom} {start} {end} not " +
							"valid bigwig indexes. Using zeros instead.")
						signal_ = numpy.zeros(end-start)

					signal_ = numpy.nan_to_num(signal_)

				signals_[-1].append(signal_)

		# For the sequences and controls extract a window the size of the input
		start = mid - in_width - max_jitter
		end = mid + in_width + max_jitter

		# Extract the controls from each of the control files
		if controls is not None:
			controls_.append([])
			for control in controls:
				if isinstance(control, dict):
					control_ = control[chrom][start:end]
				else:
					control_ = control.values(chrom, start, end, numpy=True)
					control_ = numpy.nan_to_num(control_)

				controls_[-1].append(control_)

		# Extract the sequence
		if isinstance(sequences, dict):
			seq = sequences[chrom][start:end].T
		else:
			seq = one_hot_encode(sequences[chrom][start:end].seq.upper(),
				alphabet=['A', 'C', 'G', 'T']).T

		seqs.append(seq)
		loci_count += 1

	seqs = torch.tensor(numpy.array(seqs), dtype=torch.float32)

	if signals is not None:
		signals_ = torch.tensor(numpy.array(signals_), dtype=torch.float32)

		idxs = torch.ones(signals_.shape[0], dtype=torch.bool)
		if max_counts is not None:
			idxs = (idxs) & (signals_.sum(dim=(1, 2)) < max_counts)
		if min_counts is not None:
			idxs = (idxs) & (signals_.sum(dim=(1, 2)) > min_counts)

		if controls is not None:
			controls_ = torch.tensor(numpy.array(controls_), dtype=torch.float32)
			return seqs[idxs], signals_[idxs], controls_[idxs]

		return seqs[idxs], signals_[idxs]
	else:
		if controls is not None:
			controls_ = torch.tensor(numpy.array(controls_), dtype=torch.float32)
			return seqs, controls_

		return seqs			


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
		controls=controls, chroms=chroms, in_window=in_window, 
		out_window=out_window, max_jitter=max_jitter, min_counts=min_counts,
		max_counts=max_counts, verbose=verbose)

	if controls is not None:
		sequences, signals_, controls_ = X
	else:
		sequences, signals_ = X
		controls_ = None

	X_gen = DataGenerator(sequences, signals_, controls=controls_, 
		in_window=in_window, out_window=out_window, max_jitter=max_jitter,
		reverse_complement=reverse_complement, random_state=random_state)

	X_gen = torch.utils.data.DataLoader(X_gen, pin_memory=pin_memory,
		num_workers=num_workers, batch_size=batch_size) 

	return X_gen
