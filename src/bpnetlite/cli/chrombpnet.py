#!/usr/bin/env python
# ChromBPNet command-line tool
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import sys
import numpy
import torch
import argparse
import subprocess

from bpnetlite.bpnet import BPNet
from bpnetlite.chrombpnet import ChromBPNet

from bpnetlite.io import PeakGenerator
from bpnetlite.marginalize import marginalization_report

from tangermeme.io import extract_loci

import json


###
# Default Parameters
###

default_fit_parameters = {
	'n_filters': 256,
	'n_layers': 8,
	'profile_output_bias': True,
	'count_output_bias': True,
	'name': None,
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'max_jitter': 128,
	'reverse_complement': True,
	'max_epochs': 50,
	'validation_iter': 100,
	'lr': 0.001,
	'alpha': 10,
	'beta': 0.5,
	'early_stopping': None,
	'verbose': False,
	'bias_model': None,

	'min_counts': None,
	'max_counts': None,

	'training_chroms': ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7',
		'chr9', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17',
		'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
	'validation_chroms': ['chr8', 'chr10'],

	'sequences': None,
	'loci': None,
	'negatives': None,
	'signals': None,
	'random_state': None,

	# Fit bias model
	'bias_fit_parameters': {
		'n_filters': None,
		'n_layers': 4,
		'alpha': None,
		'max_counts': None,
		'loci': None,
		'verbose': None,
		'random_state': None
	}
}

default_predict_parameters = {
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr9', 'chr11',
		'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
		'chr20', 'chr21', 'chr22', 'chrX'],
	'sequences': None,
	'loci': None,
	'model': None,
	'profile_filename': 'y_profile.npz',
	'counts_filename': 'y_counts.npz'
}

default_attribute_parameters = {
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr9', 'chr11',
		'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
		'chr20', 'chr21', 'chr22', 'chrX'],
	'sequences': None,
	'loci': None,
	'model': None,
	'output': 'counts',
	'ohe_filename': 'ohe.npz',
	'attr_filename': 'attr.npz',
	'n_shuffles':20,
	'random_state':0,
	'warning_threshold':1e-4,
}

default_marginalize_parameters = {
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr9', 'chr11',
		'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
		'chr20', 'chr21', 'chr22', 'chrX'],
	'sequences': None,
	'motifs': None,
	'loci': None,
	'n_loci': None,
	'shuffle': False,
	'model': None,
	'output_filename':'marginalize/',
	'random_state':0,
	'minimal': True
}

default_pipeline_parameters = {
	# Model architecture parameters
	'n_filters': 256,
	'n_layers': 8,
	'profile_output_bias': True,
	'count_output_bias': True,
	'in_window': 2114,
	'out_window': 1000,
	'name': None,
	'model': None,
	'bias_model': None,
	'accessibility_model': None,
	'early_stopping': None,
	'verbose': False,

	# Data parameters
	'batch_size': 64,
	'max_jitter': 128,
	'reverse_complement': True,
	'max_epochs': 50,
	'validation_iter': 100,
	'lr': 0.001,
	'alpha': 10,
	'beta': 0.5,
	'min_counts': 0,
	'max_counts': 99999999,

	'sequences': None,
	'loci': None,
	'negatives': None,
	'signals': None,

	'training_chroms': ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7',
		'chr9', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16',
		'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
	'validation_chroms': ['chr8', 'chr10'],

	# Fit bias model
	'bias_fit_parameters': {
		'n_filters': None,
		'n_layers': 4,
		'alpha': None,
		'max_counts': None,
		'loci': None,
		'verbose': None,
		'random_state': None
	},

	# Fit accessibility model
	'chrombpnet_fit_parameters': {
		'batch_size': 64,
		'sequences': None,
		'loci': None,
		'signals': None,
		'verbose': None,
		'random_state': None,
	},

	# Predict parameters
	'predict_parameters': {
		'batch_size': 64,
		'chroms': ['chr8', 'chr10'],
		'profile_filename': None,
		'counts_filename': None,
		'sequences': None,
		'loci': None,
		'signals': None,
		'verbose': None,
	},


	# attribute parameters
	'attribute_parameters': {
		'batch_size': 64,
		'chroms': ['chr8', 'chr10'],
		'output': 'counts',
		'loci': None,
		'ohe_filename': None,
		'attr_filename': None,
		'n_shuffles': None,
		'warning_threshold':1e-4,
		'verbose': None,
		'random_state': None
	},

	# Modisco parameters
	'modisco_motifs_parameters': {
		'n_seqlets': 100000,
		'output_filename': None,
		'verbose': None
	},

	# Modisco report parameters
	'modisco_report_parameters': {
		'motifs': None,
		'output_folder': None,
		'verbose': None
	},

	# Marginalization parameters
	'marginalize_parameters': {
		'loci': None,
		'n_loci': 100,
		'shuffle': False,
		'output_folder': None,
		'motifs': None,
		'minimal': True,
		'verbose': None,
		'random_state': None
	}
}


###
# Commands
###


def merge_parameters(parameters, default_parameters):
	"""Merge the provided parameters with the default parameters.


	Parameters
	----------
	parameters: str
		Name of the JSON folder with the provided parameters

	default_parameters: dict
		The default parameters for the operation.


	Returns
	-------
	params: dict
		The merged set of parameters.
	"""

	with open(parameters, "r") as infile:
		parameters = json.load(infile)

	optional = ['bias_model', 'min_counts', 'max_counts', 'early_stopping']

	for parameter, value in default_parameters.items():
		if parameter not in parameters:
			if value is None and parameter not in optional:
				raise ValueError("Must provide value for '{}'".format(parameter))

			parameters[parameter] = value

	return parameters



def main():
    desc = """ChromBPNet is a neural network that builds off the original BPNet
        architecture by explicitly learning bias in the signal tracks themselves.
        Specifically, for ATAC-seq and DNAse-seq experiments, the cutting enzymes
        have a soft sequence bias (though this is much stronger for Tn5, the
        enzyme for ATAC-seq). Accordingly, ChromBPNet is a pair of neural networks
        where one models the bias explicitly and one models the accessibility
        explicitly. This tool provides functionality for training the combination
        of the bias model and accessibility model and making predictions using it.
        After training, the accessibility model can be used using the `bpnet`
        tool."""

    _help = """Must be either 'fit', 'predict', 'attribute', 'marginalize',
        or 'pipeline'."""

    # Read in the arguments
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(help=_help, required=True, dest='cmd')

    train_parser = subparsers.add_parser("fit", help="Fit a ChromBPNet model.")
    train_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for fitting the model.")

    predict_parser = subparsers.add_parser("predict",
        help="Make predictions using a trained ChromBPNet model.")
    predict_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for making predictions.")

    attribute_parser = subparsers.add_parser("attribute",
        help="Calculate attributions using a trained ChromBPNet model.")
    attribute_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for calculating attributions.")

    marginalize_parser = subparsers.add_parser("marginalize",
        help="Run marginalizations given motifs.")
    marginalize_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for calculating attributions.")

    pipeline_parser = subparsers.add_parser("pipeline",
        help="Run each step on the given files.")
    pipeline_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters used for each step.")

    # Pull the arguments
    args = parser.parse_args()

    ##########
    # FIT
    ##########

    if args.cmd == "fit":
        parameters = merge_parameters(args.parameters, default_fit_parameters)

        ###

        training_data = PeakGenerator(
            loci=parameters['loci'],
            sequences=parameters['sequences'],
            signals=parameters['signals'],
            controls=None,
            chroms=parameters['training_chroms'],
            in_window=parameters['in_window'],
            out_window=parameters['out_window'],
            max_jitter=parameters['max_jitter'],
            reverse_complement=parameters['reverse_complement'],
            min_counts=parameters['min_counts'],
            max_counts=parameters['max_counts'],
            random_state=parameters['random_state'],
            batch_size=parameters['batch_size'],
            verbose=parameters['verbose']
        )

        trimming = (parameters['in_window'] - parameters['out_window']) // 2

        if parameters['bias_model'] is None:
            counts = training_data.dataset.signals.sum(dim=(1, 2))
            min_counts = torch.quantile(counts, 0.01).item()

            name = 'bias_fit_parameters'
            bias_fit_parameters = {key: parameters[key] for key in
                default_fit_parameters}
            for parameter, value in bias_fit_parameters[name].items():
                if value is not None:
                    bias_fit_parameters[parameter] = value
                if parameter == 'loci' and value is None:
                    bias_fit_parameters[parameter] = parameters['negatives']

            del bias_fit_parameters['negatives'], bias_fit_parameters['beta']

            name = '{}.chrombpnet.bias.fit.json'.format(parameters['name'])
            bias_fit_parameters['max_counts'] = min_counts * parameters['beta']
            bias_fit_parameters['name'] = parameters['name'] + '.bias'
            parameters['bias_model'] = bias_fit_parameters['name'] + '.torch'

            with open(name, 'w') as outfile:
                outfile.write(json.dumps(bias_fit_parameters, sort_keys=True,
                    indent=4))

            subprocess.run(["bpnet", "fit", "-p", name], check=True)


        if parameters['negatives'] is not None:
            training_data = PeakGenerator(
                loci=[parameters['loci'], parameters['negatives']],
                sequences=parameters['sequences'],
                signals=parameters['signals'],
                controls=None,
                chroms=parameters['training_chroms'],
                in_window=parameters['in_window'],
                out_window=parameters['out_window'],
                max_jitter=parameters['max_jitter'],
                reverse_complement=parameters['reverse_complement'],
                min_counts=parameters['min_counts'],
                max_counts=parameters['max_counts'],
                random_state=parameters['random_state'],
                batch_size=parameters['batch_size'],
                verbose=parameters['verbose']
            )

        valid_sequences, valid_signals = extract_loci(
            sequences=parameters['sequences'],
            signals=parameters['signals'],
            in_signals=None,
            loci=parameters['loci'],
            chroms=parameters['validation_chroms'],
            in_window=parameters['in_window'],
            out_window=parameters['out_window'],
            max_jitter=0,
            ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
            verbose=parameters['verbose']
        )

        bias = torch.load(parameters['bias_model'], weights_only=False, map_location='cpu').cuda().eval()
        accessibility = BPNet(n_filters=parameters['n_filters'],
            n_layers=parameters['n_layers'], n_control_tracks=0, n_outputs=1,
            alpha=parameters['alpha'],
            name=parameters['name'] + '.accessibility',
            trimming=trimming).cuda()

        model = ChromBPNet(bias=bias, accessibility=accessibility,
            name=parameters['name'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['lr'])

        model.fit(training_data, optimizer, X_valid=valid_sequences,
            y_valid=valid_signals, max_epochs=parameters['max_epochs'],
            validation_iter=parameters['validation_iter'],
            batch_size=parameters['batch_size'])


    ##########
    # PREDICT
    ##########

    elif args.cmd == 'predict':
        subprocess.run(["bpnet", "predict", "-p", args.parameters], check=True)


    ##########
    # ATTRIBUTE
    ##########

    elif args.cmd == 'attribute':
        subprocess.run(["bpnet", "attribute", "-p", args.parameters], check=True)


    ##########
    # MARGINALIZE
    ##########

    elif args.cmd == 'marginalize':
        subprocess.run(["bpnet", "marginalize", "-p", args.parameters], check=True)


    ##########
    # PIPELINE
    ##########

    elif args.cmd == 'pipeline':
        parameters = merge_parameters(args.parameters, default_pipeline_parameters)
        model_name = parameters['name']

        # Step 1: Fit a BPNet model to the provided data
        if parameters['verbose']:
            print("Step 1: Fitting a ChromBPNet model")

        if parameters['model'] is None:
            name = '{}.chrombpnet.fit.json'.format(parameters['name'])
            parameters['model'] = parameters['name'] + '.torch'

            fit_parameters = {key: parameters[key] for key in
                default_fit_parameters}
            for parameter, value in parameters['chrombpnet_fit_parameters'].items():
                if value is not None:
                    fit_parameters[parameter] = value

            for parameter, value in parameters['bias_fit_parameters'].items():
                if value is not None:
                    fit_parameters['bias_fit_parameters'][parameter] = value

            with open(name, 'w') as outfile:
                outfile.write(json.dumps(fit_parameters, sort_keys=True, indent=4))

            subprocess.run(["chrombpnet", "fit", "-p", name], check=True)


        if parameters['bias_model'] is None:
            parameters['bias_model'] = model_name + '.bias.torch'

        if parameters['accessibility_model'] is None:
            parameters['accessibility_model'] = (model_name +
                '.accessibility.torch')

        del parameters['bias_fit_parameters']
        del parameters['chrombpnet_fit_parameters']

        # Run pipeline with ChromBPNet model
        name = '{}.chrombpnet.pipeline.json'.format(parameters['name'])
        with open(name, 'w') as outfile:
            outfile.write(json.dumps(parameters, sort_keys=True, indent=4))

        subprocess.run(["bpnet", "pipeline", "-p", name], check=True)


        # Run pipeline with accessibility model
        name = '{}.chrombpnet.accessibility.pipeline.json'.format(
            model_name)

        parameters['model'] = parameters['accessibility_model']
        parameters['name'] = model_name + '.accessibility'

        with open(name, 'w') as outfile:
            outfile.write(json.dumps(parameters, sort_keys=True, indent=4))

        subprocess.run(["bpnet", "pipeline", "-p", name], check=True)


        # Run pipeline with bias model
        name = '{}.chrombpnet.bias.pipeline.json'.format(model_name)

        parameters['model'] = parameters['bias_model']
        parameters['name'] = model_name + '.bias'

        with open(name, 'w') as outfile:
            outfile.write(json.dumps(parameters, sort_keys=True, indent=4))

        subprocess.run(["bpnet", "pipeline", "-p", name], check=True)
