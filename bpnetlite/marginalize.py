# marginalize.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import numpy
import torch
import pandas
import pyfaidx
import seaborn
import logomaker

from .io import one_hot_encode
from .attributions import calculate_attributions

import matplotlib.pyplot as plt


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

	attr_before = calculate_attributions(model, X, args=args)

	X_perturb = torch.clone(X)
	motif_ohe = one_hot_encode(motif)
	motif_ohe = torch.from_numpy(motif_ohe)

	start = X.shape[-1] // 2 - len(motif) // 2
	for i in range(len(motif)):
		if motif_ohe[i].sum() > 0:
			X_perturb[:, :, start+i] = motif_ohe[i]

	y_after_profile, y_after_counts = model.predict(X_perturb, X_ctl)
	y_after_profile = torch.nn.functional.softmax(y_after_profile, dim=-1)
	y_after = y_after_profile * y_after_counts.unsqueeze(-1)

	attr_after = calculate_attributions(model, X_perturb, args=args)
	return y_before, y_after, attr_before, attr_after


def path_to_image_html(path):
	return '<img src="'+ path + '" width="240" >'

def _plot_predictions(y, path, figsize=(10,3), **kwargs):
	plt.figure(figsize=figsize)
	plt.plot(y)

	print(y.shape)

	seaborn.despine()
	plt.xlim(0, y.shape[0])
	plt.yticks(fontsize=12)
	plt.savefig(path)
	plt.close()
	

def _plot_attributions(y, path, figsize=(10,3), **kwargs):
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111) 

	df = pandas.DataFrame(y, columns=['A', 'C', 'G', 'T'])
	df.index.name = 'pos'

	crp_logo = logomaker.Logo(df, ax=ax)
	crp_logo.style_spines(visible=False)

	plt.yticks(fontsize=12)
	plt.savefig(path)
	plt.close()


def marginalization_report(model, motifs, sequences, output_dir):
	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	results = {
		'name': [],
		'sequence': [],
		'predictions (before)': [],
		'predictions (after)': [],
		'attributions (before)': [],
		'attributions (after)': []
	}

	for name in motifs.keys():
		motif = motifs[name][:].seq

		pfname = output_dir + name + ".pred."
		afname = output_dir + name + ".attr."


		y_before, y_after, attr_before, attr_after = marginalize(model, motif, 
			sequences)

		print(sequences.shape, y_before.shape, y_after.shape)

		_plot_predictions(y_before.mean(axis=0).T, pfname + "before.png")
		_plot_predictions(y_after.mean(axis=0).T, pfname + "after.png")


		mid = attr_before.shape[-1] // 2
		w = 15
		s, e = mid - w, mid + w

		_plot_attributions(attr_before.mean(axis=0)[:, s:e].T, afname + 
			"before.png")
		_plot_attributions(attr_after.mean(axis=0)[:, s:e].T, afname + 
			"after.png")

		results['name'].append(name)
		results['sequence'].append(motif)
		results['predictions (before)'].append(pfname + "before.png")
		results['predictions (after)'].append(pfname + "after.png")
		results['attributions (before)'].append(afname + "before.png")
		results['attributions (after)'].append(afname + "after.png")


	formatters = {
		'predictions (before)': path_to_image_html,
		'predictions (after)': path_to_image_html,
		'attributions (before)': path_to_image_html,
		'attributions (after)': path_to_image_html
	}

	results_df = pandas.DataFrame(results)
	results_df.to_html(open('marginalization.html'.format(output_dir), 'w'),
		escape=False, formatters=formatters, index=False)

