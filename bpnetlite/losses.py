# losses.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""
This module contains the losses used by BPNet for training.
"""

import torch

def MNLLLoss(logps, true_counts):
	"""A loss function based on the multinomial negative log-likelihood.

	This loss function takes in a tensor of normalized log probabilities such
	that the sum of each row is equal to 1 (e.g. from a log softmax) and
	an equal sized tensor of true counts and returns the probability of
	observing the true counts given the predicted probabilities under a
	multinomial distribution. Can accept tensors with 2 or more dimensions
	and averages over all except for the last axis, which is the number
	of categories.

	Adapted from Alex Tseng.

	Parameters
	----------
	logps: torch.tensor, shape=(n, ..., L)
		A tensor with `n` examples and `L` possible categories. 

	true_counts: torch.tensor, shape=(n, ..., L)
		A tensor with `n` examples and `L` possible categories.

	Returns
	-------
	loss: float
		The multinomial log likelihood loss of the true counts given the
		predicted probabilities, averaged over all examples and all other
		dimensions.
	"""

	log_fact_sum = torch.lgamma(torch.sum(true_counts, dim=-1) + 1)
	log_prod_fact = torch.sum(torch.lgamma(true_counts + 1), dim=-1)
	log_prod_exp = torch.sum(true_counts * logps, dim=-1)
	return -torch.mean(log_fact_sum - log_prod_fact + log_prod_exp)

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
	loss: float
		The MSE loss on the log of the two inputs, averaged over all examples
		and all other dimensions.
	"""

	log_true = torch.log(true_counts+1)
	return torch.nn.MSELoss()(log_predicted_counts, log_true)

def pearson_corr(arr1, arr2):
    """A Pearson correlation function implemented in PyTorch.

    Computes the Pearson correlation in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the correlation of corresponding L-arrays will be
    computed and returned in an A x B array. This function is the same as
    the one in performance.py except it operates on PyTorch arrays.

    Parameters
    ----------
    arr1: torch.tensor
    	One of the tensor to correlate.

    arr2: torch.tensor
    	The other tensor to correlation.

    Returns
    -------
	correlation: torch.tensor
		The correlation for each element, calculated along the last axis.
    """

    mean1 = torch.mean(arr1, axis=-1).unsqueeze(-1)
    mean2 = torch.mean(arr2, axis=-1).unsqueeze(-1)
    dev1, dev2 = arr1 - mean1, arr2 - mean2

    sqdev1, sqdev2 = torch.square(dev1), torch.square(dev2)
    numer = torch.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = torch.sum(sqdev1, axis=-1), torch.sum(sqdev2, axis=-1)  # Variances
    denom = torch.sqrt(var1 * var2)
   
    # Divide numerator by denominator, but use 0 where the denominator is 0
    correlation = torch.zeros_like(numer)
    correlation[denom != 0] = numer[denom != 0] / denom[denom != 0]
    return correlation