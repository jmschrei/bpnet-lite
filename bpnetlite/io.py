# io.py
# Author: Jacob Schreiber
# Code adapted from Avanti Shrikumar and Ziga Avsec

import numpy
import pyBigWig

from tqdm import tqdm

def one_hot_encode(sequence, ignore='N', alphabet=None, dtype='int8', 
	verbose=False, **kwargs):
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

	name = None if verbose in (True, False) else verbose
	d = verbose is False

	if isinstance(sequence, str):
		sequence = list(sequence)

	alphabet = alphabet or numpy.unique(sequence)
	alphabet = [char for char in alphabet if char != ignore]
	alphabet_lookup = {char: i for i, char in enumerate(alphabet)}

	ohe = numpy.zeros((len(sequence), len(alphabet)), dtype=dtype)
	for i, char in tqdm(enumerate(sequence), disable=d, desc=name, **kwargs):
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
		A dictionary of strings where the keys are the names of the
		chromosomes (exact strings from the header lines in the FASTA file)
		and the values are the strings encoded there.
	"""

	sequences = {}
	name, sequence = None, None
	skip_chrom = False

	with open(filename, "r") as infile:
		for line in tqdm(infile, disable=not verbose):
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


def extract_subset(fasta, control_positive_bw, control_negative_bw,
 	output_positive_bw, output_negative_bw, peaks, chroms=None, 
 	window_width=1000, verbose=True):
	"""Take in the filenames of the input data and returns an extracted set.

	This function will take in the filename of the genome FASTA file, the
	two control bigwig files (positive and negative strand), the two
	output bigwig files (positive and negative strand), the coordinates
	of peaks, and the chromosomes for which to extract data for. It will
	return extracted matrices for the one-hot encoded sequence and the
	four signals at each peak in the specified chromosomes, centered in
	the middle of the peak.

	Parameters
	----------
	fasta : str
		The filename of the FASTA file to use. Cannot be gzipped.

	control_positive_bw : str
		The filename of the bigwig containing the control signal on the
		positive strand.

	control_negative_bw : str
		The filename of the bigwig containing the control signal on the
		negative strand.

	output_positive_bw : str
		The filename of the bigwig containing the target signal on the
		positive strand.

	output_negative_bw : str
		The filename of the bigwig containing the target signal on the
		negative strand.

	peaks : str
		The filename of the bedgraph file containing a list of peaks, or
		more generally, the locations to extract data from. Each row in
		the returned matrix will correspond to a row in this file if the
		row comes from the specified chromosomes. The examples will be
		extracted from the center of these peaks and extend half a
		window width in either direction.

	chroms : list or tuple or None, optional
		The chromosomes to extract examples from. If None, then uses
		chr1-22 + chrX. Default is None.

	window_width : int, optional
		The width of the window for extracting examples. Default is 1000.

	verbose : bool, optional
		Whether to display a progress bar as examples are being extracted.

	Returns
	-------
	X_sequence : numpy.ndarray, shape=(n, 1000, 4)
		A one-hot encoded set of examples, derived from each peak falling
		on the specified chromosomes.

	X_control_positives : numpy.ndarray, shape=(n, 1000)
		A base-pair resolution readout of signal from the positive
		strand of the control track, derived from each peak falling on the
		specified chromosomes.

	X_control_negatives : numpy.ndarray, shape=(n, 1000)
		A base-pair resolution readout of signal from the negative
		strand of the control track, derived from each peak falling on the
		specified chromosomes.

	y_output_positives : numpy.ndarray, shape=(n, 1000)
		A base-pair resolution readout of signal from the positive
		strand of the target signal track, derived from each peak falling on 
		the specified chromosomes.

	y_output_negatives : numpy.ndarray, shape=(n, 1000)
		A base-pair resolution readout of signal from the negative
		strand of the target signal track, derived from each peak falling on 
		the specified chromosomes.
	"""

	if chroms is None:
		chroms = ['chr{}'.format(i) for i in range(1, 23)] + ['chrX']

	peaks = pandas.read_csv(peaks, sep='\t', usecols=[0, 1, 2], 
		names=['chrom', 'start', 'end'])
	
	sequences = read_fasta(fasta, include_chroms=chroms, verbose=True)
	control_positive_bw = pyBigWig.open(control_positive_bw, 'r')
	control_negative_bw = pyBigWig.open(control_negative_bw, 'r')
	output_positive_bw = pyBigWig.open(output_positive_bw, 'r')
	output_negative_bw = pyBigWig.open(output_negative_bw, 'r')

	X_sequences = []
	X_control_positives = []
	X_control_negatives = []
	y_output_positives = []
	y_output_negatives = []

	n = peaks.shape[0]
	for i, peak in tqdm(peaks.iterrows(), disable=not verbose, total=n):
		chrom = peak['chrom']
		if chrom not in chroms:
			continue

		mid = (peak['end'] - peak['start']) // 2 + peak['start']
		start, end = mid - window_width // 2, mid + window_width // 2

		sequence = one_hot_encode(sequences[chrom][start:end])
		control_positive = control_positive_bw.values(chrom, start, end, numpy=True)
		control_positive = numpy.nan_to_num(control_positive)

		control_negative = control_negative_bw.values(chrom, start, end, numpy=True)
		control_negative = numpy.nan_to_num(control_negative)

		output_positive = output_positive_bw.values(chrom, start, end, numpy=True)
		output_positive = numpy.nan_to_num(output_positive)

		output_negative = output_negative_bw.values(chrom, start, end, numpy=True)
		output_negative = numpy.nan_to_num(output_negative)

		X_sequences.append(sequence)
		X_control_positives.append(control_positive)
		X_control_negatives.append(control_negative)
		y_output_positives.append(output_positive)
		y_output_negatives.append(output_negative)

	X_sequences = numpy.array(X_sequences)
	X_control_positives = numpy.array(X_control_positives)
	X_control_negatives = numpy.array(X_control_negatives)
	y_output_positives = numpy.array(y_output_positives)
	y_output_negatives = numpy.array(y_output_negatives)
	return (X_sequences, X_control_positives, X_control_negatives, 
		y_output_positives, y_output_negatives)
