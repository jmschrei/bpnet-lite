# marginalize.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import time
import numpy
import torch
import pandas
import pyfaidx
import seaborn
import logomaker

from .io import one_hot_encode
from .attributions import calculate_attributions

import matplotlib.pyplot as plt


def read_meme(filename):
	motifs = {}

	with open(filename, "r") as infile:
		motif, width, i = None, None, 0

		for line in infile:
			if motif is None:
				if line[:5] == 'MOTIF':
					motif = line.split()[1]
				else:
					continue

			elif width is None:
				if line[:6] == 'letter':
					width = int(line.split()[5])
					pwm = numpy.zeros((width, 4))

			elif i < width:
				pwm[i] = list(map(float, line.split()))
				i += 1

			else:
				motifs[motif] = pwm
				motif, width, i = None, None, 0

	return motifs


def marginalize(model, motif, X):
	if isinstance(X, numpy.ndarray):
		X = torch.from_numpy(X)

	if model.n_control_tracks > 0:
		X_ctl = torch.zeros(X.shape[0], model.n_control_tracks, X.shape[-1],
			dtype=torch.float32)
		args = (X_ctl,)
	else:
		X_ctl = None
		args = None
		
	y_before_profile, y_before_counts = model.predict(X, X_ctl)
	y_before_profile = torch.nn.functional.softmax(y_before_profile, dim=-1)
	y_before = y_before_profile * y_before_counts.unsqueeze(-1)

	attr_before = calculate_attributions(model, X, args=args, n_shuffles=10)

	X_perturb = torch.clone(X)
	motif_ohe = one_hot_encode(motif, alphabet=['A', 'C', 'G', 'T', 'N'])
	motif_ohe = torch.from_numpy(motif_ohe)

	start = X.shape[-1] // 2 - len(motif) // 2
	for i in range(len(motif)):
		if motif_ohe[i].sum() > 0:
			X_perturb[:, :, start+i] = motif_ohe[i]

	y_after_profile, y_after_counts = model.predict(X_perturb, X_ctl)
	y_after_profile = torch.nn.functional.softmax(y_after_profile, dim=-1)
	y_after = y_after_profile * y_after_counts.unsqueeze(-1)

	attr_after = calculate_attributions(model, X_perturb, args=args, 
		n_shuffles=10)

	return y_before, y_after, attr_before, attr_after


def path_to_image_html(path):
	return '<img src="' + path + '" width="240" >'

def _plot_predictions(y, ylim, path, figsize=(10,3), **kwargs):
	plt.figure(figsize=figsize)
	plt.plot(y, color='r')
	#plt.plot(y_after, color='r')

	seaborn.despine()
	plt.xlim(0, y.shape[0])
	plt.ylim(*ylim)
	plt.yticks(fontsize=12)
	plt.savefig(path)
	plt.close()
	

def _plot_attributions(y, ylim, path, figsize=(10,3), **kwargs):
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111) 

	df = pandas.DataFrame(y, columns=['A', 'C', 'G', 'T'])
	df.index.name = 'pos'

	crp_logo = logomaker.Logo(df, ax=ax)
	crp_logo.style_spines(visible=False)

	plt.yticks(fontsize=12)
	plt.ylim(*ylim)
	plt.savefig(path)
	plt.close()


def marginalization_report(model, motifs, sequences, output_dir):
	motifs = list(read_meme(motifs).items())[:50]

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	results = {
		'name': [],
		'sequence': [],
		'predictions': [],
		'attributions': []
	}

	pred_diff, attr_diff = [], []

	for i, (name, pwm) in enumerate(motifs):
		motif = ''.join(numpy.array(['A', 'C', 'G', 'T'])[pwm.argmax(axis=1)])
		print(i, len(motifs), name, motif)

		pred_before, pred_after, attr_before, attr_after = marginalize(model, 
			motif, sequences)

		mid = attr_before.shape[-1] // 2
		w = 15
		s, e = mid - w, mid + w

		pred_diff_ = (pred_after - pred_before).mean(axis=0).T
		pred_diff.append(pred_diff_)

		attr_diff_ = (attr_after - attr_before).mean(axis=0)[:, s:e].T
		attr_diff.append(attr_diff_)

	pred_diff = torch.stack(pred_diff)
	attr_diff = torch.stack(attr_diff)

	pred_ylim = pred_diff.min() * 0.95, pred_diff.max() * 1.05
	attr_ylim = attr_diff.min() * 0.95, attr_diff.max() * 1.05

	idxs = pred_diff.max(axis=1).values.numpy()[:, 0].argsort()[::-1]
	for i, idx in enumerate(idxs):
		name, pwm = motifs[idx]

		pfname = output_dir + name + ".pred.png"
		afname = output_dir + name + ".attr.png"

		motif = ''.join(numpy.array(['A', 'C', 'G', 'T'])[pwm.argmax(axis=1)])

		_plot_predictions(pred_diff[idx], pred_ylim, pfname)
		_plot_attributions(attr_diff[idx], attr_ylim, afname)

		motif_ = motif[:25] + ('...' if len(motif) > 25 else '')

		results['name'].append(name)
		results['sequence'].append(motif_)
		results['predictions'].append(pfname)
		results['attributions'].append(afname)


	formatters = {
		'predictions': path_to_image_html,
		'attributions': path_to_image_html
	}

	results_df = pandas.DataFrame(results)
	results_df.to_html(open('marginalization.html'.format(output_dir), 'w'),
		escape=False, formatters=formatters, index=False)

