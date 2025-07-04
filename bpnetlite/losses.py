# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# Adapted from code by Alex Tseng

"""
This module contains the losses used by BPNet for training.
"""

import torch


def MNLLLoss(logps, true_counts):
	"""A loss function based on the multinomial negative log-likelihood.

	This loss function takes in a tensor of normalized log probabilities such
	that the logsumexp across the last dimension is equal to 0 (i.e., the
	sum of the exponentiated values is equal to 1), and a tensor of true
	integer counts, and returns the log probability of observing those counts
	given the predicted distributions.

	This function can accept tensors with any number of dimensions and calculates
	the loss across the last dimension. For example, if they are both A x B x L
	arrays, then the correlation of corresponding L-arrays will be computed and 
	returned in an A x B array.

	An important note about this loss is that, despite performing very well for
	these basepair resolution models, there is nothing spatial about this loss.
	Each position in the last dimension is viewed as an independent category, and
	having high counts one basepair away is just as bad as having it 100 basepairs
	away. There is nothing wrong with this, particularly if you care strongly
	about basepair resolution, but you should keep this in mind.

	Adapted from Alex Tseng.


	Parameters
	----------
	logps: torch.Tensor, shape=(n, ..., L)
		A tensor with `n` examples and `L` possible categories. 

	true_counts: torch.Tensor, shape=(n, ..., L)
		A tensor with `n` examples and `L` possible categories.


	Returns
	-------
	loss: torch.Tensor, shape=(n, ...)
		The multinomial log likelihood loss of the true counts given the
		predicted probabilities
	"""

	log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
	log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
	log_prod_exp = torch.sum(true_counts * logps, dim=-1)
	return -log_fact_sum + log_prod_fact - log_prod_exp


def log1pMSELoss(log_predicted_counts, true_counts):
	"""A MSE loss on the log(x+1) of the inputs.

	This loss will accept tensors of predicted counts and a vector of true
	counts and return the MSE on the log of the labels. The squared error
	is calculated for each position in the tensor and then averaged, regardless
	of the shape.

	Note: The predicted counts are in log space but the true counts are in the
	original count space.


	Parameters
	----------
	log_predicted_counts: torch.tensor, shape=(n, ...)
		A tensor of log predicted counts where the first axis is the number of
		examples. Important: these values are already in log space.

	true_counts: torch.tensor, shape=(n, ...)
		A tensor of the true counts where the first axis is the number of
		examples.


	Returns
	-------
	loss: torch.tensor, shape=(n, 1)
		The MSE loss on the log of the two inputs, averaged over all examples
		and all other dimensions.
	"""

	log_true = torch.log(true_counts+1)
	return torch.mean(torch.square(log_true - log_predicted_counts), dim=-1)
