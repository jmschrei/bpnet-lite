# io.py
# Author: Jacob Schreiber
# Code adapted from Avanti Shrikumar and Ziga Avsec

import numpy
import torch
import pandas
import pyBigWig

from tqdm import tqdm

def one_hot_encode(sequence, ignore='N', alphabet=None, dtype='int8', 
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

	ignore : str, optional
		A character to indicate setting nothing to 1 for that row, keeping the
		encoding entirely 0's for that row. In the context of genomics, this is
		the N character. Default is 'N'.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet. If None is passed in, the alphabet will be
		determined from the sequence, but this may be time consuming for
		large sequences. Default is None.

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

	if isinstance(sequence, str):
		sequence = list(sequence)

	alphabet = alphabet or numpy.unique(sequence)
	alphabet = [char for char in alphabet if char != ignore]
	alphabet_lookup = {char: i for i, char in enumerate(alphabet)}

	ohe = numpy.zeros((len(sequence), len(alphabet)), dtype=dtype)
	for i, char in tqdm(enumerate(sequence), disable=d, desc=desc, **kwargs):
		if char != ignore:
			idx = alphabet_lookup[char]
			ohe[i, idx] = 1

	return ohe

def read_fasta(filename, include_chroms=None, exclude_chroms=None, 
	ignore='N', alphabet=['A', 'C', 'G', 'T', 'N'], verbose=True):
	"""Read in a FASTA file and output a dictionary of sequences.

	This function will take in the path to a FASTA-formatted file and output
	a string containing the sequence for each chromosome. Optionally,
	the user can specify a set of chromosomes to include or exclude from
	the returned dictionary.

	Parameters
	----------
	filename : str
		The path to the FASTA-formatted file to open.

	include_chroms : set or tuple or list, optional
		The exact names of chromosomes in the FASTA file to include, excluding
		all others. If None, include all chromosomes (except those specified by
		exclude_chroms). Default is None.

	exclude_chroms : set or tuple or list, optional
		The exact names of chromosomes in the FASTA file to exclude, including
		all others. If None, include all chromosomes (or the set specified by
		include_chroms). Default is None.

	ignore : str, optional
		A character to indicate setting nothing to 1 for that row, keeping the
		encoding entirely 0's for that row. In the context of genomics, this is
		the N character. Default is 'N'.

	alphabet : set or tuple or list, optional
		A pre-defined alphabet. If None is passed in, the alphabet will be
		determined from the sequence, but this may be time consuming for
		large sequences. Must include the ignore character. Default is
		['A', 'C', 'G', 'T', 'N'].

	verbose : bool or str, optional
		Whether to display a progress bar. If a string is passed in, use as the
		name of the progressbar. Default is False.

	Returns
	-------
	chroms : dict
		A dictionary of one-hot encoded sequences where the keys are the names
		of the chromosomes (exact strings from the header lines in the FASTA file)
		and the values are the strings.
	"""

	sequences = {}
	name, sequence = None, None
	skip_chrom = False

	with open(filename, "r") as infile:
		for line in tqdm(infile, disable=not verbose, desc="Reading FASTA"):
			if line.startswith(">"):
				if name is not None and skip_chrom is False:
					sequences[name] = ''.join(sequence)

				sequence = []
				name = line[1:].strip("\n")
				if include_chroms is not None and name not in include_chroms:
					skip_chrom = True
				elif exclude_chroms is not None and name in exclude_chroms:
					skip_chrom = True
				else:
					skip_chrom = False

			else:
				if skip_chrom == False:
					sequence.append(line.rstrip("\n").upper())

	return sequences


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

	controls: torch.tensor, shape=(n, t, out_window+2*max_jitter)
		The control signal to take as input, usually counts, for `n`
		examples with `t` strands and output length `out_window`. 

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 128.

	reverse_complement: bool, optional
		Whether to reverse complement-augment half of the data. Default is True.

	random_state: int or None, optional
		Whether to use a deterministic seed or not.
	"""

	def __init__(self, sequences, signals, controls, in_window, out_window, 
		max_jitter=128, reverse_complement=True, random_state=None):
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
		j = self.random_state.randint(self.max_jitter*2)

		X = self.sequences[i][:, j:j+self.in_window]
		X_ctl = self.controls[i][:, j:j+self.out_window]
		y = self.signals[i][:, j:j+self.out_window]

		if self.reverse_complement and numpy.random.choice(2) == 1:
			X = X[::-1][:, ::-1]
			y = y[::-1][:, ::-1]
			X_ctl = X_ctl[::-1][:, ::-1]

		X = torch.tensor(X.copy(), dtype=torch.float32)
		X_ctl = torch.tensor(X_ctl.copy(), dtype=torch.float32)
		y = torch.tensor(y.copy())
		return X, X_ctl, y


def extract_peaks(sequences, plus_bw_path, minus_bw_path, plus_ctl_bw_path, 
	minus_ctl_bw_path, peak_path, chroms, in_window=2114, out_window=1000, 
	max_jitter=128, verbose=False):
	"""Extract data directly from fasta and bigWig files.

	This function will take in the file path to a fasta file and stranded
	signal and control files as well as other parameters. It will then
	extract the data to the specified window lengths with jitter added to
	each side for efficient jitter extraction. If you don't want jitter,
	set that to 0.

	Parameters
	----------
	sequence_path: str or dictionary
		Either the path to a fasta file to read from or a dictionary where the
		keys are the unique set of chromosoms and the values are one-hot
		encoded sequences as numpy arrays or memory maps.

	plus_bw_path: str
		Path to the bigWig containing the signal values on the positive strand.

	minus_bw_path: str
		Path to the bigWig containing the signal values on the negative strand.

	plus_ctl_bw_path: str
		Path to the bigWig containing the control values on the positive strand.

	minus_ctl_bw_path: str
		Path to the bigWig containing the control values on the negative strand.

	peak_path: str
		Path to a peak bed file. The file can have more than three columns as
		long as the first three columns are (chrom, start, end).

	chroms: list
		A set of chromosomes to extact peaks from. Peaks in other chromosomes
		are ignored.

	in_window: int, optional
		The input window size. Default is 2114.

	out_window: int, optional
		The output window size. Default is 1000.

	max_jitter: int, optional
		The maximum amount of jitter to add, in either direction, to the
		midpoints that are passed in. Default is 128.

	verbose: bool, optional
		Whether to display a progress bar while loading. Default is False.

	Returns
	-------
	seqs: numpy.ndarray, shape=(n, 4, in_window+2*max_jitter)
		The extracted sequences in the same order as the chrom and mid arrays.

	signals: numpy.ndarray, shape=(n, 2, out_window+2*max_jitter)
		The extracted stranded signals in the same order as the chrom and mid
		arrays.

	controls: numpy.ndarray, shape=(n, 2, out_window+2*max_jitter)
		The extracted stranded signals in the same order as the chrom and mid
		arrays.
	"""

	seqs, signals, controls = [], [], []
	in_width, out_width = in_window // 2, out_window // 2

	if isinstance(sequences, str):
		sequences = read_fasta(sequences, include_chroms=chroms, 
			verbose=verbose)

	names = ['chrom', 'start', 'end']
	peaks = pandas.read_csv(peak_path, sep="\t", usecols=(0, 1, 2), 
		header=None, index_col=False, names=names)
	peaks = peaks[numpy.isin(peaks['chrom'], chroms)]

	plus_bw = pyBigWig.open(plus_bw_path, "r")
	minus_bw = pyBigWig.open(minus_bw_path, "r")

	plus_ctl_bw = pyBigWig.open(plus_ctl_bw_path, "r")
	minus_ctl_bw = pyBigWig.open(minus_ctl_bw_path, "r")

	desc = "Loading Peaks"
	d = not verbose
	for _, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d, desc=desc):
		mid = start + (end - start) // 2
		start = mid - out_width - max_jitter
		end = mid + out_width + max_jitter

		sequence = sequences[chrom]

		# Load plus strand signal
		plus_sig = plus_bw.values(chrom, start, end, numpy=True)
		plus_sig = numpy.nan_to_num(plus_sig)

		# Load minus strand signal
		minus_sig = minus_bw.values(chrom, start, end, numpy=True)
		minus_sig = numpy.nan_to_num(minus_sig)

		# Load plus strand control
		plus_ctl = plus_ctl_bw.values(chrom, start, end, numpy=True)
		plus_ctl = numpy.nan_to_num(plus_ctl)

		# Load minus strand control
		minus_ctl = minus_ctl_bw.values(chrom, start, end, numpy=True)
		minus_ctl = numpy.nan_to_num(minus_ctl)

		# Append signal to growing signal list
		sig = numpy.array([plus_sig, minus_sig])
		signals.append(sig)

		# Append control to growing control list
		ctl = numpy.array([plus_ctl, minus_ctl])
		controls.append(ctl)

		# Append sequence to growing sequence list
		s = mid - in_width - max_jitter
		e = mid + in_width + max_jitter

		if isinstance(sequence, str):
			seq = one_hot_encode(sequence[s:e], alphabet=['A', 'C', 'G', 'T', 
				'N']).T
		else:
			seq = sequence[s:e].T
		
		seqs.append(seq)

	signals = numpy.array(signals)
	controls = numpy.array(controls)
	seqs = numpy.array(seqs)
	return seqs, signals, controls