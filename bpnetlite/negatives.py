# negatives.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
Provides functions for the calculation of GC-content genome-wide and the
sampling of GC-matched negatives.
"""

import numpy
import pandas
import pyfaidx
import pyBigWig

from tqdm import tqdm
from scipy.stats import rankdata
from scipy.stats import ks_2samp


def calculate_gc(sequence, width):
    """Get the GC percentage across an entire string.

    This function takes in a string `sequence' and calculates the GC percentage
    across each window of size `width'. The returned array is aligned such that
    the number represents the GC content with that position being the midpoint,
    i.e., `gc[i]'' is the GC percentage of `sequence[i-width//2:i+width//2]'

    Parameters
    ----------
    sequence: str
        A string made up of the alphabet 'A', 'C', 'G', 'T'.

    width: int
        The width of he window to calculate the GC content for.

    Returns
    -------
    gc: numpy.ndarray, shape=(len(sequence))
        An array of GC percentages that range between 0 and 1.
    """

    chars = ('C', 'G')
    n = len(sequence)
    k = width // 2

    is_gc = numpy.isin(list(sequence), chars)
    gc_sum = numpy.cumsum(is_gc)

    gc = numpy.zeros(n)
    gc[:k] = numpy.nan
    gc[-k:] = numpy.nan
    gc[k:-k] = (gc_sum[width:] - gc_sum[:-width]) / width
    return gc


def calculate_gc_genomewide(fasta, bigwig, width, include_chroms=None,
    verbose=False):
    """Calculate GC percentages across an entire fasta file.

    This function takes in the string names of a fasta file to calculate
    GC percentages for and a bigwig file to write these values out to.
    The width parameter is the width of the window that GC percentages
    are calculated for.

    This function does not explicitly return anything but writes the
    results out to the bigwig file.

    Parameters
    ----------
    fasta: str
        The filename of a properly formatted fasta file.

    bigwig: str
        The filename of a bigwig file to create with the results.

    width: int
        The width of the window to calculate GC percentages for.

    include_chroms: list or None
        A list of the chromosomes to process. The bigwig will only
        contain entries for these chromosomes. Each entry must be
        in the fasta file. If None, will use all chromosomes in the
        fasta file. Default is None. 

    verbose: bool, optional
        Whether to print status during calculation. Default is None.
    """

    fa = pyfaidx.Fasta(fasta, as_raw=True)

    chroms = include_chroms or list(fa.keys())
    chrom_sizes = {}
    gcs = {}

    for chrom in tqdm(chroms, desc="Calculating", disable=not verbose):
        sequence = fa[chrom][:].upper()
        chrom_sizes[chrom] = len(sequence)

        gc = calculate_gc(sequence, width)
        gcs[chrom] = gc

    bw = pyBigWig.open(bigwig, "w")
    bw.addHeader(list(chrom_sizes.items()), maxZooms=0)
    for chrom in tqdm(chroms, desc="Writing", disable=not verbose):
        bw.addEntries(chrom, 0, values=gcs[chrom], span=1, step=1)
    
    bw.close()
    fa.close()


def extract_values_and_masks(peaks, bw, width, verbose=False):
	"""Extract the average signal value for each entry in a bed file.

	This function takes in a bed file as a pandas DataFrame, a bigwig file
	as a pyBigWig object, and a fixed width. Importantly, this bigwig file
	must be precomputed rolling averages along that width instead of the
	underlying signal. Hence, the value that is reported is the value in the
	middle of each entry in the bed file, which will be the precomputed
	rolling average.

	Binary masks are also returned for each chromosome marking where these
	loci are so that they cannot be selected in the next stage.

	Parameters
	----------
	peaks: pandas.DataFrame
		A dataframe containing the chromosome, start, and end coordinates.

	bw: pyBigWig
		A pyBigWig opened file.

	width: int
		The width to mark in the binary masks for each selected example.

	verbose: bool, optional
		Whether to have a progress bar as examples are selected. Default is
		False.


	Returns
	-------
	values: numpy.ndarray
		A 1-D array of signal values in the same order as the bed file.

	masks: dict
		A dictionary of binary masks where the keys are the chromosome names
		and the values are numpy arrays that span the chromosome length.
	"""

	values = []
	chroms = bw.chroms()

	masks = {}
	for chrom, size in chroms.items():
		masks[chrom] = numpy.zeros(size, dtype=bool)

	d = not verbose
	n = len(peaks)
	for _, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d, total=n):
		if chrom not in chroms:
			continue

		if start < width:
			start, end = 0, width * 2
		elif end > chroms[chrom] - width:
			start, end = chroms[chrom] - width * 2, chroms[chrom]
		else:
			mid = (end - start) // 2 + start
			start, end = mid - width, mid + width

		mid = (end - start) // 2 + start
		masks[chrom][start:end] = True

		signal = bw.values(chrom, mid, mid+1, numpy=True)[0]
		values.append(signal)

	return numpy.array(values), masks


def extract_matching_loci(bed, bigwig, width, bin_width, verbose=False):
	"""Extract non-overlapping, matching, regions genome-wide.

	Given a genome-wide rolling average signal bigwig and a count for the
	number of loci to select at each signal bin, extract candidate regions
	from each chromosome. Importantly, this function will attempt to extract
	the total desired number of loci from each chromosome to ensure that
	at least those numbers are extracted genome-wide. For example, if 10 loci
	are requested at a signal bin of 0.2 and 15 are requested at 0.4, this
	number will be extracted from each chromosome. 

	Parameters
	----------
	bw: pyBigWig file
		A signal file with precomputed rolling averages at each position.

	masks: dict
		A dictionary of binary masks where the keys are the chromosome names
		and the values are numpy arrays that span the chromosome length.

	value_bin: list or numpy.ndarray
		The binned values to sample

	value_bin_counts: list or numpy.ndarray
		The number of each bin, in the same order as value_bin, to select.

	bin_width: float
		The width for which to bin signal when finding matches.

	verbose: bool, optional
		Whether to have a progress bar as examples are selected. Default is
		False.

	Returns
	-------
	reservoirs: dict
		A dictionary of candidate matching regions where the keys are GC bins
		and the entries are a subset of total indices, non-overlapping across
		bins.
	"""

	names = 'chrom', 'start', 'end'
	peaks = pandas.read_csv(bed, delimiter="\t", usecols=(0, 1, 2), 
		names=names)
	bw = pyBigWig.open(bigwig, "r")

	# Extract GC content from the given peaks and bin it
	orig_values, masks = extract_values_and_masks(peaks, bw, width, 
		verbose=verbose)
	values = ((orig_values + bin_width / 2) // bin_width).astype(int)
	value_bin, value_bin_counts = numpy.unique(values, return_counts=True)

	reservoirs = {value: [] for value in value_bin}

	chroms = bw.chroms().items()
	desc = "Choosing loci..."
	for chrom, length in tqdm(chroms, desc=desc, disable=not verbose):
		X = bw.values(chrom, 0, -1, numpy=True)
		X = numpy.nan_to_num(X)
		X = ((X + bin_width / 2) // bin_width).astype(int)
		X[masks[chrom]] = -1

		for value, count in zip(value_bin, value_bin_counts):
			idxs = numpy.where(X == value)[0]
			numpy.random.shuffle(idxs)
			n_selected = 0

			for idx in idxs:
				if X[idx] == -1:
					continue

				reservoirs[value].append((chrom, idx, length))
				X[idx-width:idx+width] = -1

				n_selected += 1
				if n_selected == count:
					break

	# In case there aren't enough for each bin, 
	n_to_select_from = [len(reservoirs[value]) for value in value_bin]
	n_to_select = [count for count in value_bin_counts]

	for i in range(len(value_bin)-1):
		k = max(0, n_to_select[i] - n_to_select_from[i])
		n_to_select[i+1] += k
		n_to_select[i] -= k

		k = max(0, n_to_select[-i-1] - n_to_select_from[-i-1])
		n_to_select[-i-2] += k
		n_to_select[-i-1] -= k

	chosen_idxs, chosen_values = {}, []
	for i, (value, count) in enumerate(zip(value_bin, n_to_select)):
		if count == 0:
			continue

		weights = numpy.array([idx[2] for idx in reservoirs[value]], 
			dtype='float64')
		weights = weights / weights.sum()

		r_idxs = numpy.random.choice(len(weights), size=count, replace=False, 
			p=weights)
		
		chosen_idxs[value] = [reservoirs[value][idx] for idx in r_idxs]
		chosen_values.extend([value] * count)

	values_idxs = rankdata(values, method="ordinal") - 1
	chosen_values = sorted(chosen_values)

	matched_loci = []
	for i, (value, value_idx) in enumerate(zip(values, values_idxs)):
		matched_value = chosen_values[value_idx]
		chrom, mid, _ = chosen_idxs[matched_value].pop(0)
		start, end = mid - width // 2, mid + width // 2
		matched_loci.append((chrom, start, end))

	matched_loci = pandas.DataFrame(matched_loci, columns=names)

	if verbose:
		matched_values, _ = extract_values_and_masks(matched_loci, bw, width, 
			verbose=verbose)

		stats = ks_2samp(orig_values, matched_values)
		print("GC paired t-test: {:3.3}, {:3.3}".format(stats.statistic, 
			stats.pvalue))

	return matched_loci
