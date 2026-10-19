# test_bpnet.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import random_one_hot

from bpnetlite.bpnet import BPNet
from bpnetlite.bpnet import BasePairNet

from .chrombpnet_lite import write_chrombpnet_lite

from numpy.testing import assert_array_equal


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def X():
	return random_one_hot((8, 4, 100), random_state=0).type(torch.float32)


##


@pytest.mark.parametrize("cls, default", [(BPNet, 557), (BasePairNet, 256)])
def test_trimming_default(cls, default):
	model = cls(n_layers=8, verbose=False)
	assert model.trimming == default


@pytest.mark.parametrize("cls", [BPNet, BasePairNet])
@pytest.mark.parametrize("trimming", [0, 10, 40])
def test_trimming_explicit(cls, trimming, X):
	model = cls(n_filters=8, n_layers=2, n_outputs=1, n_control_tracks=0,
		trimming=trimming, verbose=False)
	assert model.trimming == trimming

	y_profile, y_counts = model(X)
	assert y_profile.shape == (8, 1, 100 - 2*trimming)
	assert y_counts.shape == (8, 1)


##


def _a_fraction_model(cls, trimming, n_control_tracks=0):
	"""A model whose predicted count is the fraction of A in the count window.

	Every weight is zero except the center tap of the first filter on A and
	the count head's weight on that filter, so the residual tower's first
	channel is 1 at each A and 0 elsewhere, and the count head returns its
	mean over the positions it averages.
	"""

	model = cls(n_filters=4, n_layers=2, n_outputs=1,
		n_control_tracks=n_control_tracks, trimming=trimming, verbose=False)
	with torch.no_grad():
		for parameter in model.parameters():
			parameter.zero_()
		model.iconv.weight[0, 0, 10] = 1
		model.linear.weight[0, 0] = 1
	return model


@pytest.fixture
def X_a30():
	X = torch.zeros(2, 4, 100)
	X[:, 0, :30] = 1
	X[:, 1, 30:] = 1
	return X


# The count window is the output window extended by 37 positions on each side
# and clipped to the input: trimming 0 and 10 average all 100 positions (30 of
# them A), and trimming 40 averages positions 3 to 96 (27 of 94 are A).
@pytest.mark.parametrize("cls", [BPNet, BasePairNet])
@pytest.mark.parametrize("trimming, fraction", [(0, 30/100), (10, 30/100),
	(40, 27/94)])
def test_count_window(cls, trimming, fraction, X_a30):
	model = _a_fraction_model(cls, trimming)
	_, y_counts = model(X_a30)
	assert torch.allclose(y_counts, torch.full((2, 1), fraction))


# The control track's sum covers the same window: 100 positions for trimming
# 0 and 10, 94 for trimming 40.
@pytest.mark.parametrize("trimming, total", [(0, 100), (10, 100), (40, 94)])
def test_count_window_control(trimming, total, X_a30):
	model = _a_fraction_model(BPNet, trimming, n_control_tracks=1)
	with torch.no_grad():
		model.linear.weight[0, 0] = 0
		model.linear.weight[0, 4] = 1

	_, y_counts = model(X_a30, torch.ones(2, 1, 100))
	assert torch.allclose(y_counts, torch.full((2, 1), float(total + 1)).log())


##


@pytest.mark.parametrize("layout", ["save", "save_weights"])
@pytest.mark.parametrize("accessibility", [False, True])
def test_from_chrombpnet_lite_weights(layout, accessibility, tmp_path):
	filename = str(tmp_path / "model.h5")
	weights = write_chrombpnet_lite(filename, accessibility, layout)
	model = BPNet.from_chrombpnet_lite(filename)

	assert model.n_filters == 8
	assert model.n_layers == 3

	convs = [model.iconv, *model.rconvs, model.fconv]
	for i, conv in enumerate(convs, start=1):
		kernel, bias = weights[i]
		assert_array_equal(conv.weight.detach().numpy(),
			kernel.transpose(2, 1, 0))
		assert_array_equal(conv.bias.detach().numpy(), bias)

	kernel, bias = weights['logcounts']
	assert_array_equal(model.linear.weight.detach().numpy(), kernel.T)
	assert_array_equal(model.linear.bias.detach().numpy(), bias)
