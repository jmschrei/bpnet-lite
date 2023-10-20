# hit_calling.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import numpy
import torch

from .io import read_meme

class HitCaller(torch.nn.Module):
	"""A motif hit caller that operates on sequences and attributions.

	This is a method for calling motif hits by scanning PWMs across one-hot
	encoded sequences as convolutions. One can get these scores directly, or
	summaries based on hits -- positions where the score goes above a certain
	p-value threshold, as calculated using dynamic programming a la FIMO.

	For efficiency, all PWMs are put in the same convolution operation 
	regardless of size and so are simultaneously scanned across all of the
	sequences, and p-values are only used to calculate thresholds on the score 
	rather than calculated for each position. 
	
	Because this method is implemented in torch, you can easily use a GPU to
	accelerate scanning and use half precision for additional speed ups (if
	your GPU supports half precision).

	There are a few ways to use this method:

		(1) Use the `.predict` method to get raw scores of each motif on each
		example on both strands.
		(2) Use the `.hits` method to get a pandas DataFrame in bed format
		showing all locations where the score is higher than the score
		threshold at the provided p-value threshold.
		(3) Use the `.hits` method with `axis=1` to get a pandas DataFrame in
		bed format showing all motif hits at each example.
		(4) Use the `.hit_matrix` method to get a matrix of size (n_examples,
		n_motifs) showing the maximum score for each motif in each example.

	
	Parameters
	----------
	motifs: str or dict
		A set of motifs to scan with. If a string, this is interpreted as a 
		filepath to a MEME file. If a dictionary, the keys are interpreted as 
		the motif names and the values are interpreted as the PWMs.
	
	batch_size: int, optional
		The number of examples to process in parallel. Default is 256.

	bin_size: float, optional
		The bin size to use for the dynamic programming step when calculating 
		p-values. Default is 0.1.
	
	eps: float, optional
		A small value to add to a PWM to stabilize taking the log. Default is 
		1e-4.
	"""

	def __init__(self, motifs, batch_size=256, bin_size=0.1, eps=0.00005):
		super().__init__()
		
		self.batch_size = batch_size
		self.bin_size = bin_size
		
		if isinstance(motifs, str):
			motifs = read_meme(motifs)
			
		self.motif_names = numpy.array([name for name in motifs.keys()])
		self.motif_lengths = numpy.array([len(motif) for motif in motifs.values()])
		self.n_motifs = len(self.motif_names)
		
		motif_pwms = numpy.zeros((len(motifs), 4, max(self.motif_lengths)), dtype=numpy.float32)

		self._score_to_pval = []
		self._smallest = []
		for i, (name, motif) in enumerate(motifs.items()):
			motif_pwms[i, :, :len(motif)] = numpy.log(motif.T + eps) - numpy.log(0.25)

			smallest, mapping = self._pwm_to_mapping(motif_pwms[i])
			self._smallest.append(smallest)
			self._score_to_pval.append(mapping)

		self.motif_pwms = torch.nn.Parameter(torch.from_numpy(motif_pwms))
		self._smallest = numpy.array(self._smallest)

	def _pwm_to_mapping(self, log_pwm):
		"""An internal method for calculating score <-> p-value mappings.

		Use dynamic programming to calculate useful statistics necessary 
		for calculating p-values later on.
		"""
		
		log_bg = math.log(0.25)
		int_log_pwm = numpy.round(log_pwm / self.bin_size).astype(numpy.int32).T

		smallest = int(numpy.min(numpy.cumsum(numpy.min(int_log_pwm, axis=-1), axis=-1)))
		largest = int(numpy.max(numpy.cumsum(numpy.max(int_log_pwm, axis=-1), axis=-1)))
		
		logpdf = -numpy.inf * numpy.ones(largest - smallest + 1)
		for i in range(log_pwm.shape[0]):
			idx = int_log_pwm[0, i] - smallest
			logpdf[idx] = numpy.logaddexp(logpdf[idx], log_bg)

		old_logpdf = logpdf.copy()
		for i in range(1, log_pwm.shape[1]):
			logpdf = -numpy.inf * numpy.ones(largest - smallest + 1)

			for j, x in enumerate(old_logpdf):
				if x != -numpy.inf:
					for k in range(log_pwm.shape[0]):
						offset = int_log_pwm[i, k]
						
						logpdf[j + offset] = numpy.logaddexp(
							logpdf[j + offset], log_bg + x)

			old_logpdf = logpdf.copy()

		log1mcdf = logpdf.copy()
		for i in range(len(logpdf) - 2, -1, -1):
			log1mcdf[i] = numpy.logaddexp(log1mcdf[i], log1mcdf[i + 1])

		return smallest, log1mcdf

	def forward(self, X):
		"""Score a set of sequences.
		
		This method will run the PWMs against the sequences, reverse-complement 
		the sequences and run the PWMs against them again, and then return the 
		maximum per-position score after correcting for the flipping.
		
		
		Parameters
		----------
		X: torch.tensor, shape=(n, 4, length)
			A tensor containing one-hot encoded sequences.
		"""
		
		y_fwd = torch.nn.functional.conv1d(X, self.motif_pwms)
		y_bwd = torch.nn.functional.conv1d(X, torch.flip(self.motif_pwms, (1, 2)))
		return torch.stack([y_fwd, y_bwd]).permute(1, 2, 0, 3)
	
	@torch.no_grad()
	def predict(self, X):
		"""Score a potentially large number of sequences in batches.
		
		This method will apply the forward function to batches of sequences and
		handle moving the batches to the appropriate device and the results
		back to the CPU to not run out of memory.
		
		
		Parameters
		----------
		X: torch.tensor, shape=(n, 4, length)
			A tensor containing one-hot encoded sequences.
		"""

		scores = []
		
		for start in range(0, len(X), self.batch_size):
			X_ = X[start:start+self.batch_size].to(self.motif_pwms.device)
			
			scores_ = self(X_).cpu().float()
			scores.append(scores_)

		return torch.cat(scores)
	
	@torch.no_grad()
	def hits(self, X, X_attr=None, threshold=0.0001, dim=0):
		"""Find motif hits that pass the given threshold.
		
		This method will first scan the PWMs over all sequences, identify where
		those scores are above the per-motif score thresholds (by converting
		the provided p-value thresholds to scores), extract those coordinates
		and provide the hits in a convenient format.


		Parameters
		----------
		X: torch.tensor, shape=(n, 4, length)
			A tensor containing one-hot encoded sequences.

		X_attr: torch.tensor, shape=(n, 4, length), optional
			A tensor containing the per-position attributions. The values in
			this tensor will be summed across all four channels in the positions
			found by the hits, so make sure that the four channels are encoding
			something summable. You may want to multiply X_attr by X before
			passing it in. If None, do not sum attributions.

		threshold: float, optional
			The p-value threshold to use when calling hits. Default is 0.0001.

		dim: 0 or 1, optional
			The dimension to provide hits over. Similar to other APIs, one can
			view this as the dimension to remove from the returned results. If
			0, provide one DataFrame per motif that shows all hits for that
			motif across examples. If 1, provide one DataFrame per motif that
			shows all motif hits within each example.


		Returns
		-------
		dfs: list of pandas.DataFrame
			A list of DataFrames containing hit calls, either with one per
			example or one per motif.
		"""

		n = self.n_motifs if dim == 1 else len(X)
		hits = [[] for i in range(n)]
		letters = numpy.array(['A', 'C', 'G', 'T'])
		
		log_threshold = numpy.log(threshold)
		n_bins = numpy.array([len(scores) for scores in self._score_to_pval]) - 1
		
		scores = self.predict(X)        
		score_thresh = torch.empty(1, scores.shape[1], 1, 1)
		for i in range(scores.shape[1]):
			idx = numpy.where(model._score_to_pval[i] < log_threshold)[0][0]
			score_thresh[0, i] = (idx + self._smallest[i]) * self.bin_size                               
		
		hit_idxs = torch.where(scores > score_thresh)        
		for example_idx, motif_idx, strand_idx, pos_idx in zip(*hit_idxs):
			score = scores[example_idx, motif_idx, strand_idx, pos_idx].item()
			
			start = pos_idx.item()
			end = pos_idx.item() + self.motif_lengths[motif_idx]
			idxs = X[example_idx, :, start:end].argmax(axis=0).numpy(force=True)
			seq = ''.join(letters[idxs])
			strand = '+-'[strand_idx]

			if X_attr is not None:
				attr = X_attr[example_idx, :, start:end].sum(axis=1)
			else:
				attr = '.'
			
			entry_idx = example_idx.item() if dim == 0 else motif_idx.item()
			entry = entry_idx, start, end, strand, score, attr, seq
			
			idx = motif_idx if dim == 0 else example_idx
			hits[idx].append(entry)
		
		name = 'example_idx' if dim == 0 else 'motif'
		names = name, 'start', 'end', 'strand', 'score', 'attr', 'seq'        
		hits = [pandas.DataFrame(hits_, columns=names) for hits_ in hits]
		if dim == 1:
			for hits_ in hits:
				hits_['motif'] = self.motif_names[hits_['motif']]
		
		return hits
	
	@torch.no_grad()
	def hit_matrix(self, X):
		"""Return the maximum score per motif for each example.

		Parameters
		----------
		X: torch.tensor, shape=(n, 4, length)
			A tensor containing one-hot encoded sequences.
		"""

		return self.predict(X).max(dim=-1).values.max(dim=-1).values
		