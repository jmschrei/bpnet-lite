# test_bpnet.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import random_one_hot

from bpnetlite.bpnet import BPNet
from bpnetlite.bpnet import BasePairNet


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
