# hit_calling.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import math
import numpy
import torch

from .io import read_meme


class HitCaller(torch.nn.Module):
	"""A hit caller.


	Parameters
	----------
	motifs: str or dict
	"""

	def __init__(self, motifs, bin_size=0.1, eps=1e-4):
		super().__init__()

		if isinstance(motifs, str):
			motifs = read_meme(motifs)

		n = len(motifs)
		lengths = numpy.array([len(motif) for motif in motifs.values()])
		max_len = max(lengths)
		pwms = torch.zeros((n, 4, max_len), dtype=torch.float32)

		self.bin_size = bin_size

		self.motif_names = []
		self._score_to_pval = []
		self._smallest = []

		for i, (name, motif) in enumerate(motifs.items()):
			pwm = torch.from_numpy(motif).T
			pwms[i, :, :len(motif)] = torch.log(pwm + eps) - math.log(0.25)

			smallest, mapping = self._pwm_to_mapping(pwms[i])

			self.motif_names.append(name)
			self._smallest.append(smallest)
			self._score_to_pval.append(mapping)

		self._smallest = torch.tensor(self._smallest)
		self.pwms = pwms
		self.lengths = lengths
		self.conv = torch.nn.Conv1d(4, n, kernel_size=max_len, bias=False)
		self.conv.weight = torch.nn.Parameter(pwms)


	def _pwm_to_mapping(self, log_pwm):
		bin_size = 0.2

		log_bg = math.log(0.25)
		int_log_pwm = torch.round(log_pwm / bin_size).type(torch.int32).T

		smallest = int(torch.min(torch.cumsum(torch.min(int_log_pwm, 
			dim=-1).values, dim=-1)).item())
		largest = int(torch.max(torch.cumsum(torch.max(int_log_pwm, 
			dim=-1).values, dim=-1)).item())

		logpdf = -torch.inf * torch.ones(largest - smallest + 1)
		for i in range(log_pwm.shape[0]):
			idx = int_log_pwm[0, i] - smallest
			logpdf[idx] = numpy.logaddexp(logpdf[idx], log_bg)

		old_logpdf = torch.clone(logpdf)
		for i in range(1, log_pwm.shape[1]):
			logpdf = -torch.inf * torch.ones(largest - smallest + 1)
			
			for j, x in enumerate(old_logpdf):
				if x != -torch.inf:
					for k in range(log_pwm.shape[0]):
						offset = int_log_pwm[i, k]
						logpdf[j + offset] = numpy.logaddexp(logpdf[j + offset], 
							log_bg + x)

			old_logpdf = torch.clone(logpdf)

		log1mcdf = torch.clone(logpdf)
		for i in range(len(logpdf) - 2, -1, -1):
			log1mcdf[i] = numpy.logaddexp(log1mcdf[i], log1mcdf[i + 1])

		return smallest, log1mcdf

	def forward(self, X):
		return self.conv(X)
		#X_rc = torch.flip(X, (1, 2))
		#return torch.maximum(self.conv(X), torch.flip(self.conv(X_rc), (-1,)))

	@torch.no_grad()
	def predict(self, X, batch_size=64):
		scores, log_pvals = [], []
		n_bins = torch.tensor([len(scores) for scores in self._score_to_pval])
		n_bins = n_bins.unsqueeze(0).unsqueeze(-1).numpy()
		_smallest = self._smallest.unsqueeze(0).unsqueeze(-1).numpy()

		for start in range(0, len(X)+batch_size, batch_size):
			X_ = X[start:start+batch_size].to(self.conv.weight.device)
			
			scores_ = self(X_).cpu()
			int_scores = torch.round(scores_ / 0.2).type(torch.int32).numpy()
			int_scores -= _smallest
			int_scores = numpy.maximum(int_scores, 0)
			int_scores = numpy.minimum(int_scores, n_bins).astype(numpy.int32)

			log_pvals_ = torch.stack([self._score_to_pval[i][int_scores[:, i]] 
				for i in range(int_scores.shape[1])]).permute(1, 0, 2)
			
			scores.append(scores_)
			log_pvals.append(log_pvals_)

		return torch.cat(scores), torch.cat(log_pvals)