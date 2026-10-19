# test_chrombpnet.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import torch
import pytest

from tangermeme.utils import random_one_hot

from bpnetlite.bpnet import BPNet
from bpnetlite.chrombpnet import ChromBPNet


torch.manual_seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


@pytest.fixture
def X():
	return random_one_hot((8, 4, 100), random_state=0).type(torch.float32)


def _bpnet(trimming):
	return BPNet(n_filters=8, n_layers=2, n_outputs=1, n_control_tracks=0,
		trimming=trimming, verbose=False)


##


# The bias model's profile is cropped to the accessibility model's length: by
# nothing when both trim the same amount, and by 10 positions on each side
# when the bias model trims 10 fewer.
@pytest.mark.parametrize("bias_trimming, accessibility_trimming",
	[(0, 0), (20, 20), (10, 20)])
def test_forward(bias_trimming, accessibility_trimming, X):
	bias = _bpnet(bias_trimming)
	accessibility = _bpnet(accessibility_trimming)
	model = ChromBPNet(bias, accessibility)

	with torch.no_grad():
		y_profile, y_counts = model(X)
		bias_profile, bias_counts = bias(X)
		accessibility_profile, accessibility_counts = accessibility(X)

	w = accessibility_trimming - bias_trimming
	n = 100 - 2*accessibility_trimming
	assert y_profile.shape == (8, 1, n)
	assert y_counts.shape == (8, 1)
	assert torch.allclose(y_profile, accessibility_profile
		+ bias_profile[:, :, w:w+n])
	assert torch.allclose(y_counts, torch.logsumexp(torch.cat([
		accessibility_counts, bias_counts], dim=-1), dim=-1, keepdim=True))
