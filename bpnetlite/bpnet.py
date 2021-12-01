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
from .performance import multinomial_log_probs
from .performance import compute_performance_metrics

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

	n_profile_outputs: int, optional
		The number of profile outputs from the model. Generally either 1 or 2 
		depending on if the data is unstranded or stranded. Default is 2.

	n_count_outputs: int, optional
		The number of count outputs from the model. Generally just 1, but
		can be larger than 1 when multiple experiments are being modeled
		at the same time. Default is 1.

	alpha: float, optional
		The weight to put on the count loss.

	trimming: int or None, optional
		The amount to trim from both sides of the input window to get the
		output window. This value is removed from both sides, so the total
		number of positions removed is 2*trimming.
	"""

	def __init__(self, n_filters=64, n_layers=8, n_outputs=2, alpha=1, 
		trimming=None):
		super(BPNet, self).__init__()
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.alpha = alpha
		self.trimming = trimming or 2 ** n_layers

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)

		self.rconvs = torch.nn.ModuleList([
			torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
				dilation=2**i) for i in range(1, self.n_layers+1)
		])

		self.fconv = torch.nn.Conv1d(n_filters+2, n_outputs, kernel_size=75, 
			padding=37)
		
		self.relu = torch.nn.ReLU()
		self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

		self.linear = torch.nn.Linear(n_filters+1, 1)

	def forward(self, X, X_ctl):
		start, end = self.trimming, X.shape[2] - self.trimming

		X = self.relu(self.iconv(X))
		for i in range(self.n_layers):
			X_conv = self.relu(self.rconvs[i](X))
			X = torch.add(X, X_conv)

		X = X[:, :, start:end]

		X_ = torch.cat([X, X_ctl], dim=1)
		y_profile = self.fconv(X_)
		y_profile = y_profile.reshape(X.shape[0], -1)
		y_profile = self.logsoftmax(y_profile)
		y_profile = y_profile.reshape(X.shape[0], self.n_outputs, -1)

		# counts prediction
		X_ctl = torch.sum(X_ctl, axis=(1, 2)).reshape(-1, 1)

		X = torch.mean(X, axis=2)
		X = torch.cat([X, torch.log(X_ctl+1)], dim=-1)

		y_counts = self.linear(X).reshape(X.shape[0], 1)
		return y_profile, y_counts

	def predict(self, X, X_ctl, batch_size=64):
		with torch.no_grad():
			starts = numpy.arange(0, X.shape[0], batch_size)
			ends = starts + batch_size

			y_profiles, y_counts = [], []
			for start, end in zip(starts, ends):
				X_batch = X[start:end]
				X_ctl_batch = X_ctl[start:end]

				y_profiles_, y_counts_ = self(X_batch, X_ctl_batch)
				
				y_profiles.append(y_profiles_.cpu().detach().numpy())
				y_counts.append(y_counts_.cpu().detach().numpy())

			y_profiles = numpy.concatenate(y_profiles)
			y_counts = numpy.concatenate(y_counts)
			return y_profiles, y_counts

	def fit_generator(self, training_data, optimizer, X_valid=None, 
		X_ctl_valid=None, y_valid=None, max_epochs=100, batch_size=64, 
		validation_iter=100, verbose=True):

		if X_valid is not None:
			X_valid = torch.tensor(X_valid, dtype=torch.float32).cuda()
			X_ctl_valid = torch.tensor(X_ctl_valid, dtype=torch.float32).cuda()

			y_valid = y_valid.reshape(y_valid.shape[0], -1)
			y_valid = numpy.expand_dims(y_valid, (1, 3))
			y_valid_counts = y_valid.sum(axis=2)

		columns = "Epoch\tIteration\tTraining Time\tValidation Time\t"
		columns += "T MNLL\tT Count log1pMSE\t"
		columns += "V MNLL\tV Profile Pearson\tV Count Pearson\tV Count log1pMSE"
		columns += "\tSaved?"
		if verbose:
			print(columns)

		start = time.time()
		iteration = 0
		best_loss = float("inf")

		for epoch in range(max_epochs):
			tic = time.time()

			for X, X_ctl, y in training_data:
				X, X_ctl, y = X.cuda(), X_ctl.cuda(), y.cuda()

				# Clear the optimizer and set the model to training mode
				optimizer.zero_grad()
				self.train()

				# Run forward pass
				y_profile, y_counts = self(X, X_ctl)

				# Calculate the profile and count losses
				profile_loss = MNLLLoss(y_profile, y)
				count_loss = log1pMSELoss(y_counts, y.sum(dim=(1, 2)).reshape(-1, 1))

				# Extract the profile loss for logging
				profile_loss_ = profile_loss.item()
				count_loss_ = count_loss.item()

				# Mix losses together and update the model
				loss = profile_loss + self.alpha * count_loss
				loss.backward()
				optimizer.step()

				# Report measures if desired
				if verbose and iteration % validation_iter == 0:
					train_time = time.time() - start

					with torch.no_grad():
						self.eval()

						tic = time.time()
						y_profile, y_counts = self.predict(X_valid, X_ctl_valid)
						valid_time = time.time() - tic

						y_profile = y_profile.reshape(y_profile.shape[0], -1)
						y_profile = numpy.expand_dims(y_profile, (1, 3))
						y_counts = numpy.expand_dims(y_counts, 1)

						measures = compute_performance_metrics(y_valid, y_profile, 
							y_valid_counts, y_counts, 7, 81)

						line = "{}\t{}\t{:4.4}\t{:4.4}\t".format(epoch, iteration,
							train_time, valid_time)

						line += "{:4.4}\t{:4.4}\t".format(profile_loss_, 
							count_loss_)

						line += "{:4.4}\t{:4.4}\t{:4.4}\t{:4.4}".format(
							measures['nll'].mean(), 
							measures['profile_pearson'].mean(),
							measures['count_pearson'].mean(), 
							measures['count_mse'].mean()
						)

						valid_loss = measures['nll'].mean() + self.alpha * measures['count_mse'].mean()
						line += "\t{}".format(valid_loss < best_loss)

						print(line)

						start = time.time()

						if valid_loss < best_loss:
							best_loss = valid_loss

							self = self.cpu()
							torch.save(self, "bpnet.{}.{}.torch".format(self.n_filters, self.n_layers))
							self = self.cuda()
					
				iteration += 1
