# performance.py
# Authors: Alex Tseng <amtseng@stanford.edu>
#          Jacob Schreiber <jmschreiber91@gmail.com>

"""
This module contains performance measures that are used to evaluate the
model but are not explicitly used as losses for optimization.
"""

import numpy as np
import scipy.special
import scipy.ndimage

def multinomial_log_probs(
    category_log_probs, trials, query_counts, return_cross_entropy=True
):
    """
    Defines multinomial distributions and computes the probability of seeing
    the queried counts under these distributions. This defines D different
    distributions (that all have the same number of classes), and returns D
    probabilities corresponding to each distribution.

    Arguments:
        `category_log_probs`: a D x N array containing log probabilities (base
            e) of seeing each of the N classes/categories
        `trials`: a D-array containing the total number of trials for each
            distribution (can be different numbers)
        `query_counts`: a D x N array containing the observed count of each
            category in each distribution; the probability is computed for these
            observations
        `return_cross_entropy`: if True, also return a D-array of the cross
            entropy
    
    Returns a D-array containing the log probabilities (base e) of each observed
    query with its corresponding distribution. Note that D can be replaced with
    any shape (i.e. only the last dimension is reduced).
    """
    # Multinomial probability = n! / (x1!...xk!) * p1^x1 * ... pk^xk
    # Log prob = log(n!) - (log(x1!) ... + log(xk!)) + x1log(p1) ... + xklog(pk)
    log_n_fact = scipy.special.gammaln(trials + 1)
    log_counts_fact = scipy.special.gammaln(query_counts + 1)
    log_counts_fact_sum = np.sum(log_counts_fact, axis=-1)
    log_prob_pows = category_log_probs * query_counts  # Elementwise
    log_prob_pows_sum = np.sum(log_prob_pows, axis=-1)

    log_prob = log_n_fact - log_counts_fact_sum + log_prob_pows_sum
    if return_cross_entropy:
        cross_ent = (-log_prob_pows_sum) / trials
        return log_prob, cross_ent
    else:
        return log_prob


def profile_multinomial_nll(
    true_profs, log_pred_profs, true_counts, prof_smooth_kernel_sigma,
    prof_smooth_kernel_width, smooth_pred_profs=False,
    return_cross_entropy=True, batch_size=200
):
    """
    Computes the negative log likelihood of seeing the true profile, given the
    probabilities specified by the predicted profile. The NLL is computed
    separately for each sample, task, and strand, but the results are averaged
    across the strands.

    Arguments:
        `true_profs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, and O is the output profile
            length; contains the true profiles for each for each task and
            strand, as RAW counts
        `log_pred_profs`: a N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as LOG probabilities
        `true_counts`: a N x T x 2 array, containing the true total counts
            for each task and strand
        `smooth_pred_profs`: whether or not to smooth the predicted profiles
            before computing NLL
        `return_cross_entropy`: if True, also return an N x T array of the
            cross entropy
        `batch_size`: performs computation in a batch size of this many samples
    
    Returns an N x T array, containing the strand-pooled multinomial NLL for
    each sample and task.
    """
    num_samples = true_profs.shape[0]
    num_tasks = true_profs.shape[1]
    nlls = np.empty((num_samples, num_tasks))
    if return_cross_entropy:
        ces = np.empty((num_samples, num_tasks))

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        true_profs_batch = true_profs[start:end]
        log_pred_profs_batch = log_pred_profs[start:end]
        true_counts_batch = true_counts[start:end]

        # Swap axes on profiles to make them B x T x 2 x O
        true_profs_batch = np.swapaxes(true_profs_batch, 2, 3)
        log_pred_profs_batch = np.swapaxes(log_pred_profs_batch, 2, 3)

        # Smooth the predicted profile in probability space (by default this
        # doesn't happen)
        if prof_smooth_kernel_width == 0:
            sigma, truncate = 1, 0
        else:
            sigma = prof_smooth_kernel_sigma
            truncate = (prof_smooth_kernel_width - 1) / (2 * sigma)
        if smooth_pred_profs:
            # Transform to probability space
            pred_profs_batch = np.exp(log_pred_profs_batch)
            # Smooth
            pred_profs_batch_smooth = scipy.ndimage.gaussian_filter1d(
                pred_profs_batch, sigma, axis=-1, truncate=truncate
            )
            # Transform back to log-probability space
            log_pred_profs_batch = np.log(pred_profs_batch_smooth)

        result_batch = multinomial_log_probs(
            log_pred_profs_batch, true_counts_batch, true_profs_batch,
            return_cross_entropy
        )
        if return_cross_entropy:
            nll_batch, ce_batch = result_batch
            ce_batch_mean = np.mean(ce_batch, axis=2)
            ces[start:end] = ce_batch_mean
        else:
            nll_batch = result_batch
        nll_batch_mean = np.mean(-nll_batch, axis=2)  # Shape: B x T
        nlls[start:end] = nll_batch_mean

    if return_cross_entropy:
        return nlls, ces
    return nlls


def _kl_divergence(probs1, probs2):
    """
    Computes the KL divergence in the last dimension of `probs1` and `probs2`
    as KL(P1 || P2). `probs1` and `probs2` must be the same shape. For example,
    if they are both A x B x L arrays, then the KL divergence of corresponding
    L-arrays will be computed and returned in an A x B array. Does not
    renormalize the arrays. If probs2[i] is 0, that value contributes 0.
    """
    quot = np.divide(
        probs1, probs2, out=np.ones_like(probs1),
        where=((probs1 != 0) & (probs2 != 0))
        # No contribution if P1 = 0 or P2 = 0
    )
    return np.sum(probs1 * np.log(quot), axis=-1)


def jensen_shannon_distance(probs1, probs2):
    """
    Computes the Jesnsen-Shannon distance in the last dimension of `probs1` and
    `probs2`. `probs1` and `probs2` must be the same shape. For example, if they
    are both A x B x L arrays, then the KL divergence of corresponding L-arrays
    will be computed and returned in an A x B array. This will renormalize the
    arrays so that each subarray sums to 1. If the sum of a subarray is 0, then
    the resulting JSD will be NaN.
    """
    # Renormalize both distributions, and if the sum is NaN, put NaNs all around
    probs1_sum = np.sum(probs1, axis=-1, keepdims=True)
    probs1 = np.divide(
        probs1, probs1_sum, out=np.full_like(probs1, np.nan),
        where=(probs1_sum != 0)
    )
    probs2_sum = np.sum(probs2, axis=-1, keepdims=True)
    probs2 = np.divide(
        probs2, probs2_sum, out=np.full_like(probs2, np.nan),
        where=(probs2_sum != 0)
    )

    mid = 0.5 * (probs1 + probs2)
    return 0.5 * (_kl_divergence(probs1, mid) + _kl_divergence(probs2, mid))


def profile_jsd(
    true_prof_probs, pred_prof_probs, prof_smooth_kernel_sigma,
    prof_smooth_kernel_width, smooth_true_profs=True, smooth_pred_profs=False,
    batch_size=200
):
    """
    Computes the Jensen-Shannon divergence of the true and predicted profiles
    given their raw probabilities or counts. The inputs will be renormalized
    prior to JSD computation, so providing either raw probabilities or counts
    is sufficient. If any entries are negative, the arrays are assumed to be
    log probabilities and are exponentiated.

    Arguments:
        `true_prof_probs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, O is the output profile length;
            contains the true profiles for each task and strand, as RAW
            PROBABILITIES, LOG PROBABILITIES, or RAW COUNTS
        `pred_prof_probs`: N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as RAW PROBABILITIES, LOG
            PROBABILITIES, or RAW COUNTS
        `smooth_true_profs`: whether or not to smooth the true profiles before
            computing JSD
        `smooth_pred_profs`: whether or not to smooth the predicted profiles
            before computing JSD
        `batch_size`: performs computation in a batch size of this many samples
    
    Returns an N x T array, where the JSD is computed across the profiles and
    averaged between the strands, for each sample/task.
    """
    num_samples = true_prof_probs.shape[0]
    num_tasks = true_prof_probs.shape[1]
    jsds = np.empty((num_samples, num_tasks))

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        true_prof_probs_batch = true_prof_probs[start:end]
        pred_prof_probs_batch = pred_prof_probs[start:end]

        # Turn into probabilities from log probabilities, if needed
        if np.min(true_prof_probs_batch) < 0:
            true_prof_probs_batch = np.exp(true_prof_probs_batch)
        if np.min(pred_prof_probs_batch) < 0:
            pred_prof_probs_batch = np.exp(pred_prof_probs_batch)

        # Transpose to B x T x 2 x O, as JSD is computed along last dimension
        true_prof_swap = np.swapaxes(true_prof_probs_batch, 2, 3)
        pred_prof_swap = np.swapaxes(pred_prof_probs_batch, 2, 3)

        # Smooth the profiles (by default, only smooth true profile)
        if prof_smooth_kernel_width == 0:
            sigma, truncate = 1, 0
        else:
            sigma = prof_smooth_kernel_sigma
            truncate = (prof_smooth_kernel_width - 1) / (2 * sigma)
        if smooth_true_profs:
            true_prof_swap = scipy.ndimage.gaussian_filter1d(
                true_prof_swap, sigma, axis=-1, truncate=truncate
            )
        if smooth_pred_profs:
            pred_prof_swap = scipy.ndimage.gaussian_filter1d(
                pred_prof_swap, sigma, axis=-1, truncate=truncate
            )

        jsd_batch = jensen_shannon_distance(true_prof_swap, pred_prof_swap)
        jsd_batch_mean = np.mean(jsd_batch, axis=-1)  # Average over strands
        jsds[start:end] = jsd_batch_mean
    return jsds


def pearson_corr(arr1, arr2):
    """
    Computes the Pearson correlation in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the correlation of corresponding L-arrays will be
    computed and returned in an A x B array.
    """

    mean1 = np.mean(arr1, axis=-1, keepdims=True)
    mean2 = np.mean(arr2, axis=-1, keepdims=True)
    dev1, dev2 = arr1 - mean1, arr2 - mean2
    sqdev1, sqdev2 = np.square(dev1), np.square(dev2)
    numer = np.sum(dev1 * dev2, axis=-1)  # Covariance
    var1, var2 = np.sum(sqdev1, axis=-1), np.sum(sqdev2, axis=-1)  # Variances
    denom = np.sqrt(var1 * var2)
   
    # Divide numerator by denominator, but use NaN where the denominator is 0
    return np.divide(
        numer, denom, out=np.full_like(numer, np.nan), where=(denom != 0)
    )


def average_ranks(arr):
    """
    Computes the ranks of the elemtns of the given array along the last
    dimension. For ties, the ranks are _averaged_.
    Returns an array of the same dimension of `arr`. 
    """
    
    # 1) Generate the ranks for each subarray, with ties broken arbitrarily
    sorted_inds = np.argsort(arr, axis=-1)  # Sorted indices
    ranks, ranges = np.empty_like(arr), np.empty_like(arr)
    ranges = np.tile(np.arange(arr.shape[-1]), arr.shape[:-1] + (1,))
    # Put ranks by sorted indices; this creates an array containing the ranks of
    # the elements in each subarray of `arr`
    np.put_along_axis(ranks, sorted_inds, ranges, -1)
    ranks = ranks.astype(int)

    # 2) Create an array where each entry maps a UNIQUE element in `arr` to a
    # unique index for that subarray
    sorted_arr = np.take_along_axis(arr, sorted_inds, axis=-1)
    diffs = np.diff(sorted_arr, axis=-1)
    del sorted_arr  # Garbage collect
    # Pad with an extra zero at the beginning of every subarray
    pad_diffs = np.pad(diffs, ([(0, 0)] * (diffs.ndim - 1)) + [(1, 0)])
    del diffs  # Garbage collect
    # Wherever the diff is not 0, assign a value of 1; this gives a set of
    # small indices for each set of unique values in the sorted array after
    # taking a cumulative sum
    pad_diffs[pad_diffs != 0] = 1
    unique_inds = np.cumsum(pad_diffs, axis=-1).astype(int)
    del pad_diffs  # Garbage collect

    # 3) Average the ranks wherever the entries of the `arr` were identical
    # `unique_inds` contains elements that are indices to an array that stores
    # the average of the ranks of each unique element in the original array
    unique_maxes = np.zeros_like(arr)  # Maximum ranks for each unique index
    # Each subarray will contain unused entries if there are no repeats in that
    # subarray; this is a sacrifice made for vectorization; c'est la vie
    # Using `put_along_axis` will put the _last_ thing seen in `ranges`, which
    # result in putting the maximum rank in each unique location
    np.put_along_axis(unique_maxes, unique_inds, ranges, -1)
    # We can compute the average rank for each bucket (from the maximum rank for
    # each bucket) using some algebraic manipulation
    diff = np.diff(unique_maxes, prepend=-1, axis=-1)  # Note: prepend -1!
    unique_avgs = unique_maxes - ((diff - 1) / 2)
    del unique_maxes, diff  # Garbage collect

    # 4) Using the averaged ranks in `unique_avgs`, fill them into where they
    # belong
    avg_ranks = np.take_along_axis(
        unique_avgs, np.take_along_axis(unique_inds, ranks, -1), -1
    )

    return avg_ranks


def spearman_corr(arr1, arr2):
    """
    Computes the Spearman correlation in the last dimension of `arr1` and
    `arr2`. `arr1` and `arr2` must be the same shape. For example, if they are
    both A x B x L arrays, then the correlation of corresponding L-arrays will
    be computed and returned in an A x B array.
    """

    ranks1, ranks2 = average_ranks(arr1), average_ranks(arr2)
    return pearson_corr(ranks1, ranks2)


def mean_squared_error(arr1, arr2):
    """
    Computes the mean squared error in the last dimension of `arr1` and `arr2`.
    `arr1` and `arr2` must be the same shape. For example, if they are both
    A x B x L arrays, then the MSE of corresponding L-arrays will be computed
    and returned in an A x B array.
    """
    
    return np.mean(np.square(arr1 - arr2), axis=-1)


def profile_corr_mse(
    true_prof_probs, pred_prof_probs, prof_smooth_kernel_sigma,
    prof_smooth_kernel_width, smooth_true_profs=True, smooth_pred_profs=False,
    batch_size=200
):
    """
    Returns the correlations of the true and predicted PROFILE count
    probabilities (i.e. per base or per bin). If any of the entries are
    negative, the array is assumed to be log probabilities, and will be
    exponentiated. If counts are provided, normalization will happen.
    Arguments:
        true_prof_probs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, O is the output profile length;
            contains the true profiles for each task and strand, as RAW
            PROBABILITIES, LOG PROBABILITIES, or RAW COUNTS
        `pred_prof_probs`: N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as RAW PROBABILITIES, LOG
            PROBABILITIES, or RAW COUNTS
        `smooth_true_profs`: whether or not to smooth the true profiles before
            computing correlations/MSE
        `smooth_pred_profs`: whether or not to smooth the predicted profiles
            before computing correlations/MSE
        `batch_size`: performs computation in a batch size of this many samples
    Returns 3 N x T arrays, containing the Pearson correlation, Spearman
    correlation, and mean squared error of the profile predictions (as
    probabilities). Correlations/MSE are computed for each sample/task (strands
    are pooled together).
    """

    num_samples, num_tasks = true_prof_probs.shape[:2]
    pears = np.zeros((num_samples, num_tasks))
    spear = np.zeros((num_samples, num_tasks))
    mse = np.zeros((num_samples, num_tasks))

    if prof_smooth_kernel_width == 0:
        sigma, truncate = 1, 0
    else:
        sigma = prof_smooth_kernel_sigma
        truncate = (prof_smooth_kernel_width - 1) / (2 * sigma)

    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        true_batch = true_prof_probs[start:end]  # Shapes: B x T x O x 2
        pred_batch = pred_prof_probs[start:end]

        # Turn into probabilities from log probabilities, if needed
        if np.min(true_batch) < 0:
            true_batch = np.exp(true_batch)
        if np.min(pred_batch) < 0:
            pred_batch = np.exp(pred_batch)

        # Normalize into probabilities, if needed (keep 0 where all 0)
        true_batch_sum = np.sum(true_batch, axis=2, keepdims=True)
        if np.max(true_batch_sum) > 1.5:  # A little buffer
            true_batch = np.divide(
                true_batch, true_batch_sum,
                out=np.zeros_like(true_batch, dtype=float),
                where=(true_batch_sum != 0)
            )
        pred_batch_sum = np.sum(pred_batch, axis=2, keepdims=True)
        if np.max(pred_batch_sum) > 1.5:
            pred_batch = np.divide(
                pred_batch, pred_batch_sum,
                out=np.zeros_like(pred_batch, dtype=float),
                where=(pred_batch_sum != 0)
            )

        # Smooth along the output profile length
        if smooth_true_profs:
            true_batch = scipy.ndimage.gaussian_filter1d(
                true_batch, sigma, axis=2, truncate=truncate
            )
        if smooth_pred_profs:
            pred_batch = scipy.ndimage.gaussian_filter1d(
                pred_batch, sigma, axis=2, truncate=truncate
            )

        # Flatten by pooling strands
        new_shape = (true_batch.shape[0], num_tasks, -1)
        true_flat = np.reshape(true_batch, new_shape)
        pred_flat = np.reshape(pred_batch, new_shape)

        pears[start:end] = pearson_corr(true_flat, pred_flat)
        spear[start:end] = spearman_corr(true_flat, pred_flat)
        mse[start:end] = mean_squared_error(true_flat, pred_flat)

    return pears, spear, mse


def count_corr_mse(log_true_total_counts, log_pred_total_counts):
    """
    Returns the correlations of the true and predicted TOTAL counts.
    Arguments:
        `log_true_total_counts`: a N x T x 2 array, containing the true total
            LOG COUNTS for each task and strand
        `log_pred_total_counts`: a N x T x 2 array, containing the predicted
            total LOG COUNTS for each task and strand
    Returns 3 T-arrays, containing the Pearson correlation, Spearman
    correlation, and mean squared error of the total count predictions (as log
    counts). Correlations/MSE are computed for each task, over the samples and
    strands.
    """

    # Reshape inputs to be T x N * 2 (i.e. pool samples and strands)
    num_tasks = log_true_total_counts.shape[1]
    log_true_total_counts = np.reshape(
        np.swapaxes(log_true_total_counts, 0, 1), (num_tasks, -1)
    )
    log_pred_total_counts = np.reshape(
        np.swapaxes(log_pred_total_counts, 0, 1), (num_tasks, -1)
    )

    pears = pearson_corr(log_true_total_counts, log_pred_total_counts)
    spear = spearman_corr(log_true_total_counts, log_pred_total_counts)
    mse = mean_squared_error(log_true_total_counts, log_pred_total_counts)

    return pears, spear, mse

def compute_performance_metrics(
    true_profs, log_pred_profs, true_counts, log_pred_counts,
    prof_smooth_kernel_sigma=7, prof_smooth_kernel_width=81, smooth_true_profs=True,
    smooth_pred_profs=False
):
    """
    Computes some evaluation metrics on a set of positive examples, given the
    predicted profiles/counts, and the true profiles/counts.

    Arguments:
        `true_profs`: N x T x O x 2 array, where N is the number of
            examples, T is the number of tasks, and O is the output profile
            length; contains the true profiles for each for each task and
            strand, as RAW counts
        `log_pred_profs`: a N x T x O x 2 array, containing the predicted
            profiles for each task and strand, as LOG probabilities 
        `true_counts`: a N x T x 2 array, containing the true total counts
            for each task and strand
        `log_pred_counts`: a N x T x 2 array, containing the predicted LOG total
            counts for each task and strand
        `smooth_true_profs`: if True, smooth the true profiles before computing
            JSD and correlations; true profiles will not be smoothed for any
            other metric
        `smooth_pred_profs`: if True, smooth the predicted profiles before
            computing NLL, cross entropy, JSD, and correlations; predicted
            profiles will not be smoothed for any other metric
        `print_updates`: if True, print out updates and runtimes

    Returns a dictionary with the following:
        A N x T-array of the average negative log likelihoods for the profiles
            (given predicted probabilities, the likelihood for the true counts),
            for each sample/task (strands averaged)
        A N x T-array of the average cross entropy for the profiles (given
            predicted probabilities, the likelihood for the true counts), for
            each sample/task (strands averaged)
        A N x T array of average Jensen-Shannon divergence between the predicted
            and true profiles (strands averaged)
        A N x T array of the Pearson correlation of the predicted and true (log)
            counts, for each sample/task (strands pooled)
        A N x T array of the Spearman correlation of the predicted and true
            (log) counts, for each sample/task (strands pooled)
        A N x T array of the mean squared error of the predicted and true (log)
            counts, for each sample/task (strands pooled)
        A T-array of the Pearson correlation of the (log) total counts, over all
            strands and samples
        A T-array of the Spearman correlation of the (log) total counts, over
            all strands and samples
        A T-array of the mean squared error of the (log) total counts, over all
            strands and samples
    """

    # Multinomial NLL
    nll, ce = profile_multinomial_nll(
        true_profs, log_pred_profs, true_counts, prof_smooth_kernel_sigma,
        prof_smooth_kernel_width, smooth_pred_profs=smooth_pred_profs,
        return_cross_entropy=True
    )

    # Jensen-Shannon divergence
    # The true profile counts will be renormalized during JSD computation
    jsd = profile_jsd(
        true_profs, log_pred_profs, prof_smooth_kernel_sigma,
        prof_smooth_kernel_width, smooth_true_profs=smooth_true_profs,
        smooth_pred_profs=smooth_pred_profs
    )

    # Profile probability correlations/MSE 
    prof_pears, prof_spear, prof_mse = profile_corr_mse(
        true_profs, log_pred_profs, prof_smooth_kernel_sigma,
        prof_smooth_kernel_width, smooth_true_profs=smooth_true_profs,
        smooth_pred_profs=smooth_pred_profs
    )

    # Total count correlations/MSE
    log_true_counts = np.log(true_counts + 1)
    count_pears, count_spear, count_mse = count_corr_mse(
        log_true_counts, log_pred_counts
    )

    return {
        "nll": nll,
        "cross_ent": ce,
        "jsd": jsd,
        "profile_pearson": prof_pears,
        "profile_spearman": prof_spear,
        "profile_mse": prof_mse,
        "count_pearson": count_pears,
        "count_spearman": count_spear,
        "count_mse": count_mse
    }
