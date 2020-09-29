# bpnet.py
# Author: Jacob Schreiber
# Code adapted from Avanti Shrikumar and Ziga Avsec

import keras
import numpy

from keras.layers import Input, Dense, Conv1D, GlobalAvgPool1D, Conv2DTranspose
from keras.layers import add, concatenate, Reshape, Lambda

from keras.models import Model

import tensorflow as tf
import tensorflow_probability as tfp


def multinomial_nll(true_counts, logits):
	"""Compute the multinomial negative log-likelihood
	Args:
		true_counts: observed count values
		logits: predicted logit values
	"""
	counts_per_example = tf.reduce_sum(true_counts, axis=-1)
	dist = tfp.distributions.Multinomial(total_count=counts_per_example,
										 logits=logits)
	return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
			tf.cast(tf.shape(true_counts)[0], 'float32'))


#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
class MultichannelMultinomialNLL(object):
	def __init__(self, n):
		self.__name__ = "MultichannelMultinomialNLL"
		self.n = n

	def __call__(self, true_counts, logits):
		#return sum(multinomial_nll(true_counts[:, i], logits[:,i]) for i in range(self.n))
		for i in range(self.n):
			loss = multinomial_nll(true_counts[:, i], logits[:, i])
			if i == 0:
				total = loss
			else:
				total += loss
		return total

	def get_config(self):
		return {"n": self.n}


class BPNet(Model):
	"""A backpropoganda network."""

	def __init__(self, input_length, output_length):
		super(BPNet, self).__init__()

		self.input_length = input_length
		self.output_length = output_length

		self.n_filters = 64
		self.kernel_size = 21
		self.n_dilated_layers = 6
		self.tconv_kernel_size = 75
		self.lr = 0.004
		self.model = self._build_model()

	def _build_model(self):
		sequence = Input(shape=(self.input_length, 4), name="sequence")
		control_counts = Input(shape=(1,), name="control_logcount")
		control_profile = Input(shape=(self.output_length, 2), name="control_profile")

		x = Conv1D(self.n_filters, kernel_size=self.kernel_size, padding='same', activation='relu')(sequence)
		layers = [x]
		for i in range(1, self.n_dilated_layers+1):
			layer_sum = x if i == 1 else add(layers)
			x = Conv1D(self.n_filters, kernel_size=3, padding='same', activation='relu', dilation_rate=2**i)(layer_sum)
			layers.append(x)

		layer_sum = add(layers)
		average_conv = GlobalAvgPool1D()(layer_sum)

		# Predict counts
		x_with_count_bias = concatenate([average_conv, control_counts], axis=-1)
		y_count = Dense(2, name="task0_logcount")(x_with_count_bias)

		# Reshape from 1D to 2D
		layer_sum = Reshape((-1, 1, self.n_filters))(layer_sum)

		x_profile = Conv2DTranspose(2, kernel_size=(self.tconv_kernel_size, 1), padding='same')(layer_sum)
		x_profile = Reshape((-1, 2))(x_profile)

		x_with_profile_bias = concatenate([x_profile, control_profile], axis=-1)
		y_profile = Conv1D(2, kernel_size=1, name="task0_profile")(x_with_profile_bias)

		inputs = [sequence, control_counts, control_profile]
		outputs = [y_count, y_profile]
		model = Model(inputs=inputs, outputs=outputs)
		model.compile('adam', loss=['mse', MultichannelMultinomialNLL(2)],
			loss_weights=[1, 1])
		return model


	def fit_from_files(self, sequence, control_signal_plus, control_signal_neg, 
		output_signal_plus, output_signal_neg, peaks):

		