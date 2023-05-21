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

	attr_before = calculate_attributions(model, X, args=args, n_shuffles=10)

	X_perturb = torch.clone(X)
	motif_ohe = one_hot_encode(motif, alphabet=['A', 'C', 'G', 'T'])
	motif_ohe = torch.from_numpy(motif_ohe)

	start = X.shape[-1] // 2 - len(motif) // 2
	for i in range(len(motif)):
		if motif_ohe[i].sum() > 0:
			X_perturb[:, :, start+i] = motif_ohe[i]

	y_after_profile, y_after_counts = model.predict(X_perturb, X_ctl)
	y_after_profile = torch.nn.functional.softmax(y_after_profile, dim=-1)

	attr_after = calculate_attributions(model, X_perturb, args=args, 
		n_shuffles=10)

	return (y_before_profile, y_after_profile, y_before_counts, y_after_counts, 
		attr_before, attr_after)


def path_to_image_html(path):
	return '<img src="' + path + '" width="240" >'


def _plot_profiles(y, ylim, color, path, figsize=(10,3)):
	plt.figure(figsize=figsize)
	plt.plot(y, color=color)

	seaborn.despine()
	plt.xlim(0, y.shape[0])
	plt.xlabel("Relative Coordinate", fontsize=12)
	plt.ylabel("Predicted Signal", fontsize=12)
	plt.ylim(*ylim)
	plt.yticks(fontsize=12)
	plt.savefig(path)
	plt.close()


def _plot_counts(y_before, y_after, xlim, ylim, color, path, figsize=(10,3)):
	z = max(xlim[1], ylim[1])

	plt.figure(figsize=figsize)
	plt.plot([0, z], [0, z], color='0.5')
	plt.scatter(y_before, y_after, color=color, s=5)

	seaborn.despine()
	plt.xlabel("Pred Counts Before", fontsize=12)
	plt.ylabel("Pred Counts After", fontsize=12)
	plt.xlim(*xlim)
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


def marginalization_report(model, motifs, sequences, output_dir, minimal=False):
	motifs = list(read_meme(motifs).items())

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	results = {
		'name': [],
		'sequence': [],
		'profile (before)': [],
		'profile (after)': [],
		'profile (diff)': [],
		'counts': [],
		'attributions (before)': [],
		'attributions (after)': [],
		'attributions (diff)': []
	}

	prof_before, prof_after, prof_diff = [], [], []
	counts_before, counts_after, counts_diff = [], [], []
	attr_before, attr_after, attr_diff = [], [], []

	for i, (name, pwm) in enumerate(motifs):
		motif = ''.join(numpy.array(['A', 'C', 'G', 'T'])[pwm.argmax(axis=1)])
		print(i, len(motifs), name, motif)

		(y_profile_before, y_profile_after, y_counts_before, y_counts_after, 
			y_attr_before, y_attr_after) = marginalize(model, motif, sequences)

		mid = y_attr_before.shape[-1] // 2
		w = 15
		s, e = mid - w, mid + w

		prof_before.append(y_profile_before.mean(axis=0).T)
		prof_after.append(y_profile_after.mean(axis=0).T)
		profile_diff_ = (y_profile_after - y_profile_before).mean(axis=0).T
		prof_diff.append(profile_diff_)

		counts_before.append(y_counts_before)
		counts_after.append(y_counts_after)

		attr_before.append(y_attr_before.mean(axis=0)[:, s:e].T)
		attr_after.append(y_attr_after.mean(axis=0)[:, s:e].T)
		attr_diff_ = (y_attr_after - y_attr_before).mean(axis=0)[:, s:e].T
		attr_diff.append(attr_diff_)

	prof_before = torch.stack(prof_before)
	prof_after = torch.stack(prof_after)
	prof_diff = torch.stack(prof_diff)

	counts_before = torch.stack(counts_before)
	counts_after = torch.stack(counts_after)

	attr_before = torch.stack(attr_before)
	attr_after = torch.stack(attr_after)
	attr_diff = torch.stack(attr_diff)

	prof_before_ylim = prof_before.min() * 0.95, prof_before.max() * 1.05
	prof_after_ylim = prof_after.min() * 0.95, prof_after.max() * 1.05
	prof_diff_ylim = prof_diff.min() * 0.95, prof_diff.max() * 1.05

	counts_xlim = counts_before.min() * 0.95, counts_before.max() * 1.05
	counts_ylim = counts_after.min() * 0.95, counts_after.max() * 1.05

	attr_before_ylim = attr_before.min() * 0.95, attr_before.max() * 0.95
	attr_after_ylim = attr_after.min() * 0.95, attr_after.max() * 0.95
	attr_diff_ylim = attr_diff.min() * 0.95, attr_diff.max() * 0.95

	idxs = prof_diff.max(axis=1).values.numpy()[:, 0].argsort()[::-1]
	for i, idx in enumerate(idxs):
		name, pwm = motifs[idx]
		motif = ''.join(numpy.array(['A', 'C', 'G', 'T'])[pwm.argmax(axis=1)])

		if not minimal:
			_plot_profiles(prof_before[idx], prof_before_ylim, color='0.5', 
				path=(output_dir + name + ".profile.before.png"))
			_plot_profiles(prof_after[idx], prof_after_ylim, color='0.5', 
				path=(output_dir + name + ".profile.after.png"))
		_plot_profiles(prof_diff[idx], prof_diff_ylim, color='c', 
			path=(output_dir + name + ".profile.diff.png"))

		_plot_counts(counts_before[idx], counts_after[idx], counts_xlim, 
			counts_ylim, color='m', path=(output_dir + name + ".counts.png"))

		if not minimal:
			_plot_attributions(attr_before[idx], attr_before_ylim, 
				path=(output_dir + name + ".attr.before.png"))
			_plot_attributions(attr_after[idx], attr_after_ylim, 
				path=(output_dir + name + ".attr.after.png"))
		_plot_attributions(attr_diff[idx], attr_diff_ylim, 
			path=(output_dir + name + ".attr.diff.png"))

		motif_ = motif[:25] + ('...' if len(motif) > 25 else '')

		results['name'].append(name)
		results['sequence'].append(motif_)
		results['profile (before)'].append(output_dir + name + 
			".profile.before.png")
		results['profile (after)'].append(output_dir + name + 
			".profile.after.png")
		results['profile (diff)'].append(output_dir + name + 
			".profile.diff.png")
		results['counts'].append(output_dir + name + 
			".counts.png")
		results['attributions (before)'].append(output_dir + name + 
			".attr.before.png")
		results['attributions (after)'].append(output_dir + name + 
			".attr.after.png")
		results['attributions (diff)'].append(output_dir + name + 
			".attr.diff.png")

	formatters = {name: path_to_image_html for name in results.keys() 
		if name not in ('name', 'sequence')}

	results_df = pandas.DataFrame(results)
	if minimal:
		results_df = results_df[['name', 'sequence', 'profile (diff)', 
			'counts', 'attributions (diff)']]

	results_df.to_html(open('{}/marginalization.html'.format(output_dir), 'w'),
		escape=False, formatters=formatters, index=False)
