# chrombpnet.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch

from .losses import MNLLLoss
from .losses import log1pMSELoss

from .performance import calculate_performance_measures

from .logging import Logger


class ChromBPNet(torch.nn.Module):
	"""A ChromBPNet model.

	Parameters
	----------
	bias: torch.nn.Module 
		This model takes in sequence and outputs the shape one would expect in 
		ATAC-seq data due to Tn5 bias alone. This is usually a BPNet model
		from the bpnet-lite repo that has been trained on GC-matched non-peak
		regions.

	accessibility: torch.nn.Module
		This model takes in sequence and outputs the accessibility one would 
		expect due to the components of the sequence, but also takes in a cell 
		representation which modifies the parameters of the model, hence, 
		"dynamic." This model is usually a DynamicBPNet model, defined below.

	name: str
		The name to prepend when saving the file.
	"""

	def __init__(self, bias, accessibility, name):
		super(ChromBPNet, self).__init__()
		for parameter in bias.parameters():
			parameter.requires_grad = False

		self.bias = bias
		self.accessibility = accessibility
		self.name = name
		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Training Count MSE", 
			"Validation MNLL", "Validation Profile Correlation", 
			"Validation Count Pearson", "Validation Count MSE", "Saved?"], 
			verbose=True)

	def forward(self, X, X_ctl=None):
		"""A forward pass through the network.

		This function is usually accessed through calling the model, e.g.
		doing `model(x)`. The method defines how inputs are transformed into
		the outputs through interactions with each of the layers.


		Parameters
		----------
		X: torch.tensor, shape=(-1, 4, 2114)
			A one-hot encoded sequence tensor.

		X_ctl: ignore
			An ignored parameter for consistency with attribution functions.


		Returns
		-------
		y_profile: torch.tensor, shape=(-1, 1000)
			The predicted logit profile for each example. Note that this is not
			a normalized value.
		"""

		acc_profile, acc_counts = self.accessibility(X)
		bias_profile, bias_counts = self.bias(X)

		y_profile = acc_profile + bias_profile
		y_counts = torch.logsumexp(torch.stack([acc_counts, bias_counts]), 
			dim=0)
		
		return y_profile, y_counts

	@torch.no_grad()
	def predict(self, X, batch_size=16):
		"""A method for making batched predictions.

		This method will take in a large number of cell states and provide
		predictions in a batched manner without storing the gradients. Useful
		for inference step.


		Parameters
		----------
		X: torch.tensor, shape=(-1, 4, 2114)
			A one-hot encoded sequence tensor.

		batch_size: int, optional
			The number of elements to use in each batch. Default is 64.


		Returns
		-------
		y_hat: torch.tensor, shape=(-1, 1000)
			A tensor containing the predicted profiles.
		"""		

		y_profile, y_counts = [], []
		for start in range(0, len(X), batch_size):
			y_profile_, y_counts_ = self(X[start:start+batch_size])
			y_profile.append(y_profile_.cpu())
			y_counts.append(y_counts_.cpu())

		return torch.cat(y_profile), torch.cat(y_counts)

	def fit_generator(self, training_data, optimizer, X_valid=None, y_valid=None,
		max_epochs=100, batch_size=64, validation_iter=100, verbose=True):
		"""Fit an entire DragoNNFruit model to the data.
		"""

		X_valid = X_valid.cuda(non_blocking=True)
		y_bias_profile, y_bias_counts = self.bias.predict(X_valid)

		start, best_loss = time.time(), float("inf")
		self.logger.start()
		for epoch in range(max_epochs):
			for iteration, (X, y) in enumerate(training_data):
				self.accessibility.train()

				X = X.cuda()
				y = y.cuda()

				optimizer.zero_grad()

				acc_profile, acc_counts = self.accessibility(X)
				bias_profile, bias_counts = self.bias(X)

				y_profile = torch.nn.functional.log_softmax(acc_profile +
					bias_profile, dim=-1)

				y_counts = torch.logsumexp(torch.stack([acc_counts, 
					bias_counts]), dim=0)

				profile_loss = MNLLLoss(y_profile, y).mean()
				count_loss = log1pMSELoss(y_counts, y.sum(dim=-1).reshape(-1, 
					1)).mean()

				profile_loss_ = profile_loss.item()
				count_loss_ = count_loss.item()

				loss = profile_loss + self.accessibility.alpha * count_loss
				loss.backward()
				optimizer.step()

				if verbose and iteration % validation_iter == 0:
					train_time = time.time() - start
					tic = time.time()

					with torch.no_grad():
						self.accessibility.eval()

						y_profile, y_counts = self.accessibility.predict(X_valid)
						y_profile = torch.nn.functional.log_softmax(
							y_profile + y_bias_profile, dim=-1)

						y_counts = torch.logsumexp(torch.stack([y_counts,
							y_bias_counts]), dim=0)

						measures = calculate_performance_measures(y_profile, 
							y_valid, y_counts, kernel_sigma=7, 
							kernel_width=81, measures=['profile_mnll', 
							'profile_pearson', 'count_pearson', 'count_mse'])

						profile_corr = measures['profile_pearson']
						count_corr = measures['count_pearson']

						valid_loss = measures['profile_mnll'].mean()
						valid_loss += self.accessibility.alpha * measures['count_mse'].mean()
						valid_time = time.time() - tic

						self.logger.add([epoch, iteration, train_time, 
							valid_time, profile_loss_, count_loss_, 
							measures['profile_mnll'].mean().item(), 
							numpy.nan_to_num(profile_corr).mean(),
							numpy.nan_to_num(count_corr).mean(), 
							measures['count_mse'].mean().item(),
							(valid_loss < best_loss).item()])

						if valid_loss < best_loss:
							torch.save(self, "{}.torch".format(self.name))
							best_loss = valid_loss

					start = time.time()

			self.logger.save("{}.log".format(self.name))

		torch.save(self, "{}.final.torch".format(self.name, epoch))
