# bpnet.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This module contains a reference implementation of BPNet that can be used
or adapted for your own circumstances. The implementation takes in a
stranded control track and makes predictions for stranded outputs.
"""

import time 
import numpy
import torch

from .losses import MNLLLoss
from .losses import log1pMSELoss

from .performance import pearson_corr
from .performance import calculate_performance_measures

torch.backends.cudnn.benchmark = True

class BPNet(torch.nn.Module):
	"""A basic BPNet model with stranded profile and total count prediction.

	This is a reference implementation for BPNet. The model takes in
	one-hot encoded sequence, runs it through: 

	(1) a single wide convolution operation 

	THEN 

	(2) a user-defined number of dilated residual convolutions

	THEN

	(3a) profile predictions done using a very wide convolution layer 
	that also takes in stranded control tracks 

	AND

	(3b) total count prediction done using an average pooling on the output
	from 2 followed by concatenation with the log1p of the sum of the
	stranded control tracks and then run through a dense layer.

	This implementation differs from the original BPNet implementation in
	two ways:

	(1) The model concatenates stranded control tracks for profile
	prediction as opposed to adding the two strands together and also then
	smoothing that track 

	(2) The control input for the count prediction task is the log1p of
	the strand-wise sum of the control tracks, as opposed to the raw
	counts themselves.

	(3) A single log softmax is applied across both strands such that
	the logsumexp of both strands together is 0. Put another way, the
	two strands are concatenated together, a log softmax is applied,
	and the MNLL loss is calculated on the concatenation. 

	(4) The count prediction task is predicting the total counts across
	both strands. The counts are then distributed across strands according
	to the single log softmax from 3.

	Parameters
	----------
	n_filters: int, optional
		The number of filters to use per convolution. Default is 64.

	n_layers: int, optional
		The number of dilated residual layers to include in the model.
		Default is 8.

	n_outputs: int, optional
		The number of profile outputs from the model. Generally either 1 or 2 
		depending on if the data is unstranded or stranded. Default is 2.

	alpha: float, optional
		The weight to put on the count loss.

	name: str or None, optional
		The name to save the model to during training.

	trimming: int or None, optional
		The amount to trim from both sides of the input window to get the
		output window. This value is removed from both sides, so the total
		number of positions removed is 2*trimming.
	"""

	def __init__(self, n_filters=64, n_layers=8, n_outputs=2, 
		n_control_tracks=2, alpha=1, profile_output_bias=True, 
		count_output_bias=True, name=None, trimming=None):
		super(BPNet, self).__init__()
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.n_control_tracks = n_control_tracks

		self.alpha = alpha
		self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
		self.trimming = trimming or 2 ** n_layers

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)

		self.rconvs = torch.nn.ModuleList([
			torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
				dilation=2**i) for i in range(1, self.n_layers+1)
		])

		self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, kernel_size=75, 
			padding=37, bias=profile_output_bias)
		
		n_count_control = 1 if n_control_tracks > 0 else 0
		self.linear = torch.nn.Linear(n_filters+n_count_control, 1, 
			bias=count_output_bias)

	def forward(self, X, X_ctl=None):
		"""A forward pass of the model.

		This method takes in a nucleotide sequence X, a corresponding
		per-position value from a control track, and a per-locus value
		from the control track and makes predictions for the profile 
		and for the counts. This per-locus value is usually the
		log(sum(X_ctl_profile)+1) when the control is an experimental
		read track but can also be the output from another model.

		Parameters
		----------
		X: torch.tensor, shape=(batch_size, 4, sequence_length)
			The one-hot encoded batch of sequences.

		X_ctl: torch.tensor, shape=(batch_size, n_strands, sequence_length)
			A value representing the signal of the control at each position in the
			sequence.

		Returns
		-------
		y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
			The output predictions for each strand.
		"""

		start, end = self.trimming, X.shape[2] - self.trimming

		X = torch.nn.ReLU()(self.iconv(X))
		for i in range(self.n_layers):
			X_conv = torch.nn.ReLU()(self.rconvs[i](X))
			X = torch.add(X, X_conv)

		if X_ctl is None:
			X_w_ctl = X
		else:
			X_w_ctl = torch.cat([X, X_ctl], dim=1)

		y_profile = self.fconv(X_w_ctl)[:, :, start:end]

		# counts prediction
		X = torch.mean(X[:, :, start-37:end+37], axis=2)

		if X_ctl is not None:
			X_ctl = torch.sum(X_ctl[:, :, start-37:end+37], axis=(1, 2))
			X_ctl = X_ctl.unsqueeze(-1)
			X = torch.cat([X, torch.log(X_ctl+1)], dim=-1)

		y_counts = self.linear(X).reshape(X.shape[0], 1)
		return y_profile, y_counts

	def predict(self, X, X_ctl=None, batch_size=64):
		with torch.no_grad():
			starts = numpy.arange(0, X.shape[0], batch_size)
			ends = starts + batch_size

			y_profiles, y_counts = [], []
			for start, end in zip(starts, ends):
				X_batch = X[start:end].cuda()
				X_ctl_batch = None if X_ctl is None else X_ctl[start:end].cuda()

				y_profiles_, y_counts_ = self(X_batch, X_ctl_batch)
				y_profiles_ = y_profiles_.cpu()
				y_counts_ = y_counts_.cpu()
				
				y_profiles.append(y_profiles_)
				y_counts.append(y_counts_)

			y_profiles = torch.cat(y_profiles)
			y_counts = torch.cat(y_counts)
			return y_profiles, y_counts

	def fit_generator(self, training_data, optimizer, X_valid=None, 
		X_ctl_valid=None, y_valid=None, max_epochs=100, batch_size=64, 
		validation_iter=100, verbose=True):

		if X_valid is not None:
			X_valid = X_valid.cuda()
			y_valid_counts = y_valid.sum(axis=2)

		if X_ctl_valid is not None:
			X_ctl_valid = X_ctl_valid.cuda()

		columns = "Epoch\tIteration\tTraining Time\tValidation Time\t"
		columns += "T MNLL\tT Count log1pMSE\t"
		columns += "V MNLL\tV Profile Pearson\tV Count Pearson\tV Count log1pMSE"
		columns += "\tSaved?"
		if verbose:
			print(columns)

		iteration = 0
		best_loss = float("inf")

		for epoch in range(max_epochs):
			tic = time.time()

			for data in training_data:
				if len(data) == 3:
					X, X_ctl, y = data
					X, X_ctl, y = X.cuda(), X_ctl.cuda(), y.cuda()
				else:
					X, y = data
					X, y = X.cuda(), y.cuda()
					X_ctl = None

				# Clear the optimizer and set the model to training mode
				optimizer.zero_grad()
				self.train()

				# Run forward pass
				y_profile, y_counts = self(X, X_ctl)
				y_profile = y_profile.reshape(y_profile.shape[0], -1)
				y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
				
				y = y.reshape(y.shape[0], -1)

				# Calculate the profile and count losses
				profile_loss = MNLLLoss(y_profile, y).mean()
				count_loss = log1pMSELoss(y_counts, y.sum(dim=-1).reshape(-1, 1)).mean()

				# Extract the profile loss for logging
				profile_loss_ = profile_loss.item()
				count_loss_ = count_loss.item()

				# Mix losses together and update the model
				loss = profile_loss + self.alpha * count_loss
				loss.backward()
				optimizer.step()

				# Report measures if desired
				if verbose and iteration % validation_iter == 0:
					train_time = time.time() - tic

					with torch.no_grad():
						self.eval()

						tic = time.time()
						y_profile, y_counts = self.predict(X_valid, X_ctl_valid)
						valid_time = time.time() - tic

						z = y_profile.shape
						y_profile = y_profile.reshape(y_profile.shape[0], -1)
						y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
						y_profile = y_profile.reshape(*z)

						measures = calculate_performance_measures(y_profile, 
							y_valid, y_counts, kernel_sigma=7, 
							kernel_width=81, measures=['profile_mnll', 
							'profile_pearson', 'count_pearson', 'count_mse'])

						line = "{}\t{}\t{:4.4}\t{:4.4}\t".format(epoch, iteration,
							train_time, valid_time)

						line += "{:4.4}\t{:4.4}\t".format(profile_loss_, 
							count_loss_)

						line += "{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}".format(
							measures['profile_mnll'].mean(), 
							numpy.nan_to_num(measures['profile_pearson']).mean(),
							numpy.nan_to_num(measures['count_pearson']).mean(), 
							measures['count_mse'].mean()
						)

						valid_loss = measures['profile_mnll'].mean() + self.alpha * measures['count_mse'].mean()
						line += "\t{}".format(valid_loss < best_loss)

						if valid_loss < best_loss:
							torch.save(self, "{}.torch".format(self.name))
							best_loss = valid_loss
					
						print(line)

				iteration += 1

		torch.save(self, "{}.final.torch".format(self.name))
