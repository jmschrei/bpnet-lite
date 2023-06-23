# bpnet.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This module contains a reference implementation of BPNet that can be used
or adapted for your own circumstances. The implementation takes in a
stranded control track and makes predictions for stranded outputs.
"""

import h5py
import time 
import numpy
import torch

from .losses import MNLLLoss
from .losses import log1pMSELoss
from .performance import pearson_corr
from .performance import calculate_performance_measures
from .logging import Logger

from tqdm import tqdm

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

	Note that this model is also used as components in the ChromBPNet model,
	as both the bias model and the accessibility model. Both components are
	the same BPNet architecture but trained on different loci.


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

	n_control_tracks: int, optional
		The number of control tracks to feed into the model. When predicting
		TFs, this is usually 2. When predicting accessibility, this is usualy
		0. When 0, this input is removed from the model. Default is 2.

	alpha: float, optional
		The weight to put on the count loss.

	profile_output_bias: bool, optional
		Whether to include a bias term in the final profile convolution.
		Removing this term can help with attribution stability and will usually
		not affect performance. Default is True.

	count_output_bias: bool, optional
		Whether to include a bias term in the linear layer used to predict
		counts. Removing this term can help with attribution stability but
		may affect performance. Default is True.

	name: str or None, optional
		The name to save the model to during training.

	trimming: int or None, optional
		The amount to trim from both sides of the input window to get the
		output window. This value is removed from both sides, so the total
		number of positions removed is 2*trimming.

	verbose: bool, optional
		Whether to display statistics during training. Setting this to False
		will still save the file at the end, but does not print anything to
		screen during training. Default is True.
	"""

	def __init__(self, n_filters=64, n_layers=8, n_outputs=2, 
		n_control_tracks=2, alpha=1, profile_output_bias=True, 
		count_output_bias=True, name=None, trimming=None, verbose=True):
		super(BPNet, self).__init__()
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_outputs = n_outputs
		self.n_control_tracks = n_control_tracks

		self.alpha = alpha
		self.name = name or "bpnet.{}.{}".format(n_filters, n_layers)
		self.trimming = trimming or 2 ** n_layers

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.irelu = torch.nn.ReLU()

		self.rconvs = torch.nn.ModuleList([
			torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, padding=2**i, 
				dilation=2**i) for i in range(1, self.n_layers+1)
		])
		self.rrelus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(1, self.n_layers+1)
		])

		self.fconv = torch.nn.Conv1d(n_filters+n_control_tracks, n_outputs, 
			kernel_size=75, padding=37, bias=profile_output_bias)
		
		n_count_control = 1 if n_control_tracks > 0 else 0
		self.linear = torch.nn.Linear(n_filters+n_count_control, 1, 
			bias=count_output_bias)

		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Training Count MSE", 
			"Validation MNLL", "Validation Profile Pearson", 
			"Validation Count Pearson", "Validation Count MSE", "Saved?"], 
			verbose=verbose)


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
		X: torch.tensor, shape=(batch_size, 4, length)
			The one-hot encoded batch of sequences.

		X_ctl: torch.tensor or None, shape=(batch_size, n_strands, length)
			A value representing the signal of the control at each position in 
			the sequence. If no controls, pass in None. Default is None.

		Returns
		-------
		y_profile: torch.tensor, shape=(batch_size, n_strands, out_length)
			The output predictions for each strand trimmed to the output
			length.
		"""

		start, end = self.trimming, X.shape[2] - self.trimming

		X = self.irelu(self.iconv(X))
		for i in range(self.n_layers):
			X_conv = self.rrelus[i](self.rconvs[i](X))
			X = torch.add(X, X_conv)

		if X_ctl is None:
			X_w_ctl = X
		else:
			X_w_ctl = torch.cat([X, X_ctl], dim=1)

		y_profile = self.fconv(X_w_ctl)[:, :, start:end]

		# counts prediction
		X = torch.mean(X[:, :, start-37:end+37], dim=2)

		if X_ctl is not None:
			X_ctl = torch.sum(X_ctl[:, :, start-37:end+37], dim=(1, 2))
			X_ctl = X_ctl.unsqueeze(-1)
			X = torch.cat([X, torch.log(X_ctl+1)], dim=-1)

		y_counts = self.linear(X).reshape(X.shape[0], 1)
		return y_profile, y_counts


	def predict(self, X, X_ctl=None, batch_size=64, verbose=False):
		"""Make predictions for a large number of examples.

		This method will make predictions for a number of examples that exceed
		the batch size. It is similar to the forward method in terms of inputs 
		and outputs, but will run wrapped with `torch.no_grad()` to speed up
		computation and prevent information leakage into the model.


		Parameters
		----------
		X: torch.tensor, shape=(-1, 4, length)
			The one-hot encoded batch of sequences.

		X_ctl: torch.tensor or None, shape=(-1, n_strands, length)
			A value representing the signal of the control at each position in 
			the sequence. If no controls, pass in None. Default is None.

		batch_size: int, optional
			The number of examples to run at a time. Default is 64.

		verbose: bool
			Whether to print a progress bar during predictions.


		Returns
		-------
		y_profile: torch.tensor, shape=(-1, n_strands, out_length)
			The output predictions for each strand trimmed to the output
			length.
		"""


		with torch.no_grad():
			starts = numpy.arange(0, X.shape[0], batch_size)
			ends = starts + batch_size

			y_profiles, y_counts = [], []
			for start, end in tqdm(zip(starts, ends), disable=not verbose):
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

	def fit(self, training_data, optimizer, X_valid=None, X_ctl_valid=None, 
		y_valid=None, max_epochs=100, batch_size=64, validation_iter=100, 
		early_stopping=None, verbose=True):
		"""Fit the model to data and validate it periodically.

		This method controls the training of a BPNet model. It will fit the
		model to examples generated by the `training_data` DataLoader object
		and, if validation data is provided, will periodically validate the
		model against it and return those values. The periodicity can be
		controlled using the `validation_iter` parameter.

		Two versions of the model will be saved: the best model found during
		training according to the validation measures, and the final model
		at the end of training. Additionally, a log will be saved of the
		training and validation statistics, e.g. time and performance.


		Parameters
		----------
		training_data: torch.utils.data.DataLoader
			A generator that produces examples to train on. If n_control_tracks
			is greater than 0, must product two inputs, otherwise must produce
			only one input.

		optimizer: torch.optim.Optimizer
			An optimizer to control the training of the model.

		X_valid: torch.tensor or None, shape=(n, 4, 2114)
			A block of sequences to validate on periodically. If None, do not
			perform validation. Default is None.

		X_ctl_valid: torch.tensor or None, shape=(n, n_control_tracks, 2114)
			A block of control sequences to validate on periodically. If
			n_control_tracks is None, pass in None. Default is None.

		y_valid: torch.tensor or None, shape=(n, n_outputs, 1000)
			A block of signals to validate against. Must be provided if
			X_valid is also provided. Default is None.

		max_epochs: int
			The maximum number of epochs to train for, as measured by the
			number of times that `training_data` is exhausted. Default is 100.

		batch_size: int
			The number of examples to include in each batch. Default is 64.

		validation_iter: int
			The number of batches to train on before validating against the
			entire validation set. When the validation set is large, this
			enables the total validating time to be small compared to the
			training time by only validating periodically. Default is 100.

		early_stopping: int or None
			Whether to stop training early. If None, continue training until
			max_epochs is reached. If an integer, continue training until that
			number of `validation_iter` ticks has been hit without improvement
			in performance. Default is None.

		verbose: bool
			Whether to print out the training and evaluation statistics during
			training. Default is True.
		"""

		if X_valid is not None:
			X_valid = X_valid.cuda()
			y_valid_counts = y_valid.sum(dim=2)

		if X_ctl_valid is not None:
			X_ctl_valid = X_ctl_valid.cuda()


		iteration = 0
		early_stop_count = 0
		best_loss = float("inf")
		self.logger.start()

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

						z = y_profile.shape
						y_profile = y_profile.reshape(y_profile.shape[0], -1)
						y_profile = torch.nn.functional.log_softmax(y_profile, dim=-1)
						y_profile = y_profile.reshape(*z)

						measures = calculate_performance_measures(y_profile, 
							y_valid, y_counts, kernel_sigma=7, 
							kernel_width=81, measures=['profile_mnll', 
							'profile_pearson', 'count_pearson', 'count_mse'])

						profile_corr = measures['profile_pearson']
						count_corr = measures['count_pearson']
						
						valid_loss = measures['profile_mnll'].mean()
						valid_loss += self.alpha * measures['count_mse'].mean()
						valid_time = time.time() - tic

						self.logger.add([epoch, iteration, train_time, 
							valid_time, profile_loss_, count_loss_, 
							measures['profile_mnll'].mean().item(), 
							numpy.nan_to_num(profile_corr).mean(),
							numpy.nan_to_num(count_corr).mean(), 
							measures['count_mse'].mean().item(),
							(valid_loss < best_loss).item()])

						self.logger.save("{}.log".format(self.name))

						if valid_loss < best_loss:
							torch.save(self, "{}.torch".format(self.name))
							best_loss = valid_loss
							early_stop_count = 0
						else:
							early_stop_count += 1

				if early_stopping is not None and early_stop_count >= early_stopping:
					break

				iteration += 1

			if early_stopping is not None and early_stop_count >= early_stopping:
				break

		torch.save(self, "{}.final.torch".format(self.name))


	@classmethod
	def from_chrombpnet_lite(cls, filename):
		"""Loads a model from ChromBPNet-lite TensorFlow format.
	
		This method will load a ChromBPNet-lite model from TensorFlow format.
		Note that this is not the same as ChromBPNet format. Specifically,
		ChromBPNet-lite was a preceeding package that had a slightly different
		saving format, whereas ChromBPNet is the packaged version of that
		code that is applied at scale.

		This method does not load the entire ChromBPNet model. If that is
		the desired behavior, see the `ChromBPNet` object and its associated
		loading functions. Instead, this loads a single BPNet model -- either
		the bias model or the accessibility model, depending on what is encoded
		in the stored file.


		Parameters
		----------
		filename: str
			The name of the h5 file that stores the trained model parameters.


		Returns
		-------
		model: BPNet
			A BPNet model compatible with this repository in PyTorch.
		"""

		h5 = h5py.File(filename, "r")
		w = h5['model_weights']

		if 'model_1' in w.keys():
			w = w['model_1']
			bias = False
		else:
			bias = True

		k, b = 'kernel:0', 'bias:0'
		name = "conv1d_{}_1" if not bias else "conv1d_{0}/conv1d_{0}"

		layer_names = []
		for layer_name in w.keys():
			try:
				idx = int(layer_name.split("_")[1])
				layer_names.append(idx)
			except:
				pass

		n_filters = w[name.format(1)][k].shape[2]
		n_layers = max(layer_names) - 2

		model = BPNet(n_layers=n_layers, n_filters=n_filters, n_outputs=1,
			n_control_tracks=0, trimming=(2114-1000)//2)

		convert_w = lambda x: torch.nn.Parameter(torch.tensor(
			x[:]).permute(2, 1, 0))
		convert_b = lambda x: torch.nn.Parameter(torch.tensor(x[:]))

		model.iconv.weight = convert_w(w[name.format(1)][k])
		model.iconv.bias = convert_b(w[name.format(1)][b])
		model.iconv.padding = 12

		for i in range(2, n_layers+2):
			model.rconvs[i-2].weight = convert_w(w[name.format(i)][k])
			model.rconvs[i-2].bias = convert_b(w[name.format(i)][b])

		model.fconv.weight = convert_w(w[name.format(n_layers+2)][k])
		model.fconv.bias = convert_b(w[name.format(n_layers+2)][b])
		model.fconv.padding = 12

		name = "logcounts_1" if not bias else "logcounts/logcounts"
		model.linear.weight = torch.nn.Parameter(torch.tensor(w[name][k][:].T))
		model.linear.bias = convert_b(w[name][b])
		return model
