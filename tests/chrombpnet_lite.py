# chrombpnet_lite.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>
#
# Helpers for testing the ChromBPNet-lite loaders: a writer of random weights
# in the h5 layout Keras uses, and a reference forward pass that follows the
# Keras model directly.

import h5py
import numpy
import torch


def write_chrombpnet_lite(filename, accessibility=False, layout='save_weights',
	n_filters=8, n_layers=3, random_state=0):
	"""Write random ChromBPNet-lite weights in the layout Keras uses.

	The architecture is `bpnet_seq` from ChromBPNet-lite's `model_arch.py`: a
	Conv1D of width 25, `n_layers` dilated Conv1D of width 3 with residual
	connections, a profile Conv1D of width 25 and a Dense layer named
	`logcounts`. A bias model's file stores layer `conv1d_3` as
	`conv1d_3/conv1d_3/kernel:0`. A ChromBPNet file stores only the
	accessibility model, nested as `model_1/conv1d_3_1/kernel:0`. Layers
	without weights are empty groups. `model.save_weights` writes these groups
	at the root of the file and `model.save` writes them under
	`model_weights`.


	Parameters
	----------
	filename: str
		The h5 file to write.

	accessibility: bool, optional
		Whether to write a ChromBPNet file, holding the accessibility model,
		rather than a bias model's file. Default is False.

	layout: 'save_weights' or 'save', optional
		Which Keras call the file imitates. Default is 'save_weights'.

	n_filters: int, optional
		The number of filters per convolution. Default is 8.

	n_layers: int, optional
		The number of dilated residual layers. Default is 3.

	random_state: int, optional
		The seed of the random weights. Default is 0.


	Returns
	-------
	weights: dict
		(kernel, bias) numpy arrays in the shapes Keras uses, keyed by layer
		number (1 to n_layers+2) and by 'logcounts'.
	"""

	rng = numpy.random.RandomState(random_state)
	shapes = {1: (25, 4, n_filters), n_layers+2: (25, n_filters, 1),
		'logcounts': (n_filters, 1)}
	shapes.update({i: (3, n_filters, n_filters) for i in range(2, n_layers+2)})

	weights = {key: (rng.normal(0, 0.1, size=shape).astype(numpy.float32),
		rng.normal(0, 0.1, size=shape[-1:]).astype(numpy.float32))
		for key, shape in shapes.items()}

	names = {key: 'logcounts' if key == 'logcounts' else 'conv1d_{}'.format(
		key) for key in weights}
	if accessibility:
		empty = ['input_1', 'input_2', 'input_3', 'concatenate_1',
			'logits_w_bias', 'logcounts_w_bias']
		paths = {key: 'model_1/{}_1'.format(name) for key, name in
			names.items()}
	else:
		empty = ['input_1', 'global_average_pooling1d_1', 'logits'] + [
			'add_{}'.format(i) for i in range(1, n_layers+1)]
		paths = {key: '{0}/{0}'.format(name) for key, name in names.items()}

	with h5py.File(filename, 'w') as f:
		root = f if layout == 'save_weights' else f.create_group(
			'model_weights')
		for name in empty:
			root.create_group(name)

		for key, path in paths.items():
			kernel, bias = weights[key]
			root.create_dataset(path + '/kernel:0', data=kernel)
			root.create_dataset(path + '/bias:0', data=bias)

	return weights


def bpnet_seq(X, weights, n_layers=3):
	"""The forward pass of `bpnet_seq` in ChromBPNet-lite's `model_arch.py`.

	Every Conv1D uses padding='same', so the profile logits are as long as the
	input, and the log counts come from a GlobalAvgPool1D over every position
	of the residual tower's output.


	Parameters
	----------
	X: torch.tensor, shape=(-1, 4, length)
		One-hot encoded sequences.

	weights: dict
		The weights returned by `write_chrombpnet_lite`.

	n_layers: int, optional
		The number of dilated residual layers. Default is 3.


	Returns
	-------
	logits: torch.tensor, shape=(-1, length)
		The profile logits.

	logcounts: torch.tensor, shape=(-1, 1)
		The log counts.
	"""

	def conv(X, key, dilation=1):
		kernel, bias = weights[key]
		return torch.nn.functional.conv1d(X, torch.from_numpy(kernel).permute(
			2, 1, 0), torch.from_numpy(bias), padding='same', dilation=dilation)

	X = torch.relu(conv(X, 1))
	for i in range(1, n_layers+1):
		X = X + torch.relu(conv(X, i+1, dilation=2**i))

	logits = conv(X, n_layers+2)[:, 0]

	kernel, bias = weights['logcounts']
	logcounts = X.mean(dim=-1) @ torch.from_numpy(kernel) + torch.from_numpy(
		bias)
	return logits, logcounts
