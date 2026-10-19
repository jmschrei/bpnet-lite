# test_basepairnet.py
# Tests for loading basepairmodels BPNet weights into BasePairNet.

import os
import tempfile
import unittest

import h5py
import numpy
import torch

from bpnetlite.bpnet import BasePairNet


def _write_bpnet_h5(filename, n_filters=8, n_layers=2, seed=0):
	"""Write basepairmodels-style weights whose trunk outputs one everywhere.

	The trunk's convolution weights are zero and the bias of the first
	convolution is one, so the trunk outputs one at every position and the
	expected outputs of both heads follow directly from their weights. The
	head weights and biases are random, so the mixing matrices are not
	symmetric.
	"""

	rng = numpy.random.default_rng(seed)

	shapes = {"main_conv_0": (21, 4, n_filters)}
	for i in range(1, n_layers+1):
		shapes["main_dil_conv_{}".format(i)] = (3, n_filters, n_filters)

	weights = {name: (numpy.zeros(shape, dtype='float32'),
		numpy.zeros(shape[-1], dtype='float32')) for name, shape in shapes.items()}
	weights["main_conv_0"][1][:] = 1

	for name, shape in [("main_profile_head", (75, n_filters, 2)),
		("profile_predictions", (1, 4, 2)), ("main_counts_head", (n_filters, 1)),
		("logcounts_predictions", (2, 1))]:
		weights[name] = (rng.normal(size=shape).astype('float32'),
			rng.normal(size=shape[-1]).astype('float32'))

	with h5py.File(filename, "w") as h5:
		for name, (kernel, bias) in weights.items():
			h5["model_weights/{0}/{0}/kernel:0".format(name)] = kernel
			h5["model_weights/{0}/{0}/bias:0".format(name)] = bias

	return weights


class TestBasePairNetFromBPNet(unittest.TestCase):
	def test_heads_match_basepairmodels(self):
		with tempfile.TemporaryDirectory() as tmpdir:
			filename = os.path.join(tmpdir, "model.h5")
			weights = _write_bpnet_h5(filename)
			model = BasePairNet.from_bpnet(filename)

		X = torch.zeros(1, 4, 2114)
		X_ctl = torch.zeros(1, 2, 2114)
		with torch.no_grad():
			y_profile, y_counts = model(X, X_ctl)

		# main_profile_head on the trunk, then profile_predictions with a zero
		# control track
		w_head, b_head = weights["main_profile_head"]
		w_pred, b_pred = weights["profile_predictions"]
		h = w_head.sum(axis=(0, 1)) + b_head
		profile = h.dot(w_pred[0, :2]) + b_pred

		numpy.testing.assert_allclose(y_profile[0].numpy(),
			numpy.tile(profile[:, None], (1, 1000)), rtol=1e-5, atol=1e-4)

		# main_counts_head on the mean of the trunk, then logcounts_predictions
		# with log(0 + 2) as the control counts input
		w_head, b_head = weights["main_counts_head"]
		w_pred, b_pred = weights["logcounts_predictions"]
		c = w_head.sum(axis=0) + b_head
		counts = c * w_pred[0] + numpy.log(2) * w_pred[1] + b_pred

		numpy.testing.assert_allclose(y_counts[0].numpy(), counts, rtol=1e-5,
			atol=1e-4)


if __name__ == '__main__':
	unittest.main()
