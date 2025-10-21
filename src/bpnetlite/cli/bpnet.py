#!/usr/bin/env python
# BPNet command-line tool
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import sys
import json
import argparse

###
# Default Parameters
###

training_chroms = ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr9',
	'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18',
	'chr19', 'chr20', 'chr21', 'chr22', 'chrX']

validation_chroms = ['chr8', 'chr10']


default_fit_parameters = {
	'n_filters': 64,
	'n_layers': 8,
	'profile_output_bias': True,
	'count_output_bias': True,
	'name': None,
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'max_jitter': 128,
	'reverse_complement': True,
	'reverse_complement_average': True,
	'summits': False,
	'max_epochs': 50,
	'lr': 0.001,
	'negative_ratio': 0.333,
	'count_loss_weight': None,
	'dtype': 'float32',
	'device': 'cuda',
	'scheduler': True,
	'early_stopping': 10,
	'verbose': False,

	'min_counts': 0,
	'max_counts': 99999999,

	'training_chroms': training_chroms,
	'validation_chroms': validation_chroms,
	'sequences': None,
	'loci': None,
	'exclusion_lists': None,
	'negatives': None,
	'signals': None,
	'controls': None,
	'random_state': None,
	'performance_filename': 'performance.tsv'
}


default_predict_parameters = {
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': training_chroms,
	'reverse_complement_averaging': False,
	'device': 'cuda',
	'dtype': 'float32',
	'exclusion_lists': None,
	'sequences': None,
	'loci': None,
	'controls': None,
	'model': None,
	'profile_filename': 'predictions.profile.npz',
	'counts_filename': 'predictions.counts.npz',
	'idx_filename': 'predictions.idx.npy'
}


default_evaluate_parameters = {
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': training_chroms,
	'reverse_complement_average': False,
	'device': 'cuda',
	'dtype': 'float32',
	'exclusion_lists': None,
	'sequences': None,
	'loci': None,
	'controls': None,
	'model': None,
	'performance_filename': 'performance.tsv'
}


default_attribute_parameters = {
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': training_chroms,
	'exclusion_lists': None,
	'sequences': None,
	'loci': None,
	'model': None,
	'output': 'counts',
	'ohe_filename': 'attributions.ohe.npz',
	'attr_filename': 'attributions.attr.npz',
	'idx_filename': 'attributions.idx.npy',
	'n_shuffles': 20,
	'random_state': 0,
	'device': 'cuda',
	'warning_threshold': 1e-3
}


default_seqlet_parameters = {
	'threshold': 0.01,
	'min_seqlet_len': 4,
	'max_seqlet_len': 25,
	'additional_flanks': 3,
	'in_window': 2114,
	'chroms': training_chroms,
	'exclusion_lists': None,
	'verbose': False,
	'loci': None,
	'ohe_filename': None,
	'attr_filename': None,
	'idx_filename': None,
	'output_filename': 'seqlets.bed',
}


default_annotation_parameters = {
	'motifs': None,
	'sequences': None,
	'seqlet_filename': None,
	'n_score_bins': 100,
	'n_median_bins': 1000,
	'n_target_bins': 100,
	'n_cache': 250,
	'reverse_complement': True,
	'n_jobs': -1,
	'output_filename': 'seqlets_annotated.bed'
}


default_marginalize_parameters = {
	'batch_size': 64,
	'in_window': 2114,
	'out_window': 1000,
	'verbose': False,
	'chroms': training_chroms,
	'exclusion_lists': None,
	'sequences': None,
	'motifs': None,
	'loci': None,
	'attributions': False,
	'n_loci': 100,
	'shuffle': False,
	'model': None,
	'output_filename':'marginalize/',
	'random_state':0,
	'minimal': True,
	'device': 'cuda'
}


default_pipeline_parameters = {
	# Model architecture parameters
	'n_filters': 64,
	'n_layers': 8,
	'profile_output_bias': True,
	'count_output_bias': True,
	'in_window': 2114,
	'out_window': 1000,
	'name': None,
	'model': None,
	'dtype': 'float32',
	'device': 'cuda',
	'early_stopping': 10,

	# Data parameters
	'batch_size': 64,
	'max_jitter': 128,
	'reverse_complement': True,
	'reverse_complement_average': True,
	'max_epochs': 20,
	'lr': 0.001,
	'scheduler': True,
	'negative_ratio': 0.333,
	'count_loss_weight': None,
	'verbose': True,
	'min_counts': 0,
	'max_counts': 99999999,
	'random_state': None,

	'exclusion_lists': None,
	'sequences': None,
	'loci': None,
	'signals': None,
	'controls': None,
	'find_negatives': False,
	'unstranded': False,
	'fragments': False,


	# Fit parameters
	'fit_parameters': {
		'batch_size': 64,
		'training_chroms': training_chroms,
		'validation_chroms': validation_chroms,
		'sequences': None,
		'loci': None,
		'signals': None,
		'controls': None,
		'verbose': None,
		'random_state': None,
		'summits': False,
		'performance_filename': None
	},


	# Predict parameters
	'predict_parameters': {
		'batch_size': 64,
		'chroms': validation_chroms,
		'reverse_complement_average': False,
		'profile_filename': None,
		'counts_filename': None,
		'idx_filename': None,
		'sequences': None,
		'loci': None,
		'signals': None,
		'controls': None,
		'dtype': None,
		'device': None,
		'verbose': None
	},


	# Attribution parameters
	'attribute_parameters': {
		'batch_size': 64,
		'chroms': validation_chroms,
		'output': 'counts',
		'loci': None,
		'device': None,
		'ohe_filename': None,
		'attr_filename': None,
		'idx_filename': None,
		'n_shuffles': 20,
		'warning_threshold': 1e-3,
		'random_state': None,
		'verbose': None
	},


	# Seqlet Parameters
	'seqlet_parameters': {
		'threshold': 0.01,
		'min_seqlet_len': 4,
		'max_seqlet_len': 25,
		'additional_flanks': 3,
		'in_window': None,
		'chroms': None,
		'verbose': None,
		'loci': None,
		'ohe_filename': None,
		'attr_filename': None,
		'idx_filename': None,
		'output_filename': None
	},


	# Seqlet Annotation Parameters
	'annotation_parameters': {
		'motifs': None,
		'sequences': None,
		'seqlet_filename': None,
		'n_score_bins': 100,
		'n_median_bins': 1000,
		'n_target_bins': 100,
		'n_cache': 250,
		'reverse_complement': True,
		'n_jobs': -1,
		'output_filename': None
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
		'attributions': False,
		'batch_size': 64,
		'shuffle': False,
		'random_state': None,
		'output_folder': None,
		'motifs': None,
		'minimal': True,
		'device': None,
		'verbose': None
	}
}


##########
# COMMANDS
##########


def _extract_set(parameters, defaults, name):
	subparameters = {
		key: parameters.get(key, None) for key in defaults if key in parameters
	}

	for parameter, value in parameters[name].items():
		if value is not None:
			subparameters[parameter] = value

	return subparameters

def _check_set(parameters, parameter, value):
	if parameters.get(parameter, None) == None:
		parameters[parameter] = value


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

	if isinstance(parameters, str):
		with open(parameters, "r") as infile:
			parameters = json.load(infile)

	unset_parameters = ("controls", "warning_threshold", "early_stopping",
		"count_loss_weight", "exclusion_lists")
	for parameter, value in default_parameters.items():
		if parameter not in parameters:
			if value is None and parameter not in unset_parameters:
				raise ValueError("Must provide value for '{}'".format(parameter))

			parameters[parameter] = value

	return parameters


def main():
    desc = """BPNet is an neural network primarily composed of dilated residual
        convolution layers for modeling the associations between biological
        sequences and biochemical readouts. This tool will take in a fasta
        file for the sequence, a bed file for signal peak locations, and bigWig
        files for the signal to predict and the control signal, and train a
        BPNet model for you."""

    _help = """Must be either 'negatives', 'fit', 'predict', 'evaluate',
        'attribute', 'seqlets', 'marginalize', or 'pipeline'."""


    # Read in the arguments
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(help=_help, required=True, dest='cmd')

    #

    negatives_parser = subparsers.add_parser("negatives",
        help="Sample GC-matched negatives.")
    negatives_parser.add_argument("-i", "--peaks", required=True,
        help="Peak bed file.")
    negatives_parser.add_argument("-f", "--fasta", help="Genome FASTA file.")
    negatives_parser.add_argument("-b", "--bigwig", help="Optional signal bigwig.")
    negatives_parser.add_argument("-o", "--output", required=True,
        help="Output bed file.")
    negatives_parser.add_argument("-l", "--bin_width", type=float, default=0.02,
        help="GC bin width to match.")
    negatives_parser.add_argument("-n", "--max_n_perc", type=float, default=0.1,
        help="Maximum percentage of Ns allowed in each locus.")
    negatives_parser.add_argument("-a", "--beta", type=float, default=0.5,
        help="Multiplier on the minimum counts in peaks.")
    negatives_parser.add_argument("-w", "--in_window", type=int, default=2114,
        help="Width for calculating GC content.")
    negatives_parser.add_argument("-x", "--out_window", type=int, default=1000,
        help="Non-overlapping stride to use for loci.")
    negatives_parser.add_argument("-v", "--verbose", default=False,
        action='store_true')

    #

    json_parser = subparsers.add_parser("pipeline-json",
        help="Make a pipeline JSON file given the provided information.")
    json_parser.add_argument("-s", "--sequences", type=str,
        help="The FASTA file of sequences.")
    json_parser.add_argument("-i", "--inputs", type=str, action='append',
        help="A BAM or bigwig file. Repeatable.")
    json_parser.add_argument("-c", "--controls", type=str, action='append',
        help="A BAM or bigwig file. Repeatable.")
    json_parser.add_argument("-p", "--peaks", type=str, action='append',
        help="A BED-formatted file of peaks to use. Repeatable.")
    json_parser.add_argument("-neg", "--negatives", type=str, action='append',
        help="A BED-formatted file of negative loci to use. Repeatable.")
    json_parser.add_argument("-n", "--name", type=str,
        help="Name to use as a suffix in intermediary files.")
    json_parser.add_argument("-u", "--unstranded", action='store_true',
        default=False, help="Whether the input is stranded")
    json_parser.add_argument("-f", "--fragments", action='store_true',
        default=False, help='Whether the input are fragments or reads.')
    json_parser.add_argument("-m", "--motifs", type=str,
        default="A motif database for marginalization and TF-MoDISco.")
    json_parser.add_argument("-o", "--output", type=str,
        help="The filename for the pipeline JSON.")

    #

    fit_parser = subparsers.add_parser("fit", help="Fit a BPNet model.")
    fit_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for fitting the model.")

    #

    predict_parser = subparsers.add_parser("predict",
        help="Make predictions using a trained BPNet model.")
    predict_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for making predictions.")

    #

    predict_parser = subparsers.add_parser("evaluate",
        help="Evaluate a trained BPNet model.")
    predict_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for making predictions.")

    #

    attribute_parser = subparsers.add_parser("attribute",
        help="Calculate attributions using a trained BPNet model.")
    attribute_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for calculating attributions.")

    #

    seqlet_parser = subparsers.add_parser("seqlets",
        help="Identify seqlets from attributions.")
    seqlet_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for identifying seqlets.")

    #

    marginalize_parser = subparsers.add_parser("marginalize",
        help="Run marginalizations given motifs.")
    marginalize_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters for running marginalizations.")

    #

    pipeline_parser = subparsers.add_parser("pipeline",
        help="Run each step on the given files.")
    pipeline_parser.add_argument("-p", "--parameters", type=str, required=True,
        help="A JSON file containing the parameters used for each step.")


    # Pull the arguments
    args = parser.parse_args()


    ##########
    # NEGATIVES
    ##########


    if args.cmd == 'negatives':
        from tangermeme.match import extract_matching_loci

        # Extract regions that match the GC content of the peaks
        matched_loci = extract_matching_loci(
            loci=args.peaks,
            fasta=args.fasta,
            gc_bin_width=args.bin_width,
            max_n_perc=args.max_n_perc,
            bigwig=args.bigwig,
            signal_beta=args.beta,
            in_window=args.in_window,
            out_window=args.out_window,
            chroms=None,
            verbose=args.verbose
        )

        matched_loci.to_csv(args.output, header=False, sep='\t', index=False)


    ##########
    # PIPELINE-JSON
    ##########


    if args.cmd == 'pipeline-json':
        parameters = default_pipeline_parameters.copy()

        parameters['sequences'] = args.sequences
        parameters['loci'] = args.peaks
        parameters['negatives'] = args.negatives
        parameters['signals'] = args.inputs
        parameters['controls'] = args.controls
        parameters['name'] = args.name
        parameters['unstranded'] = args.unstranded
        parameters['motifs'] = args.motifs
        parameters['find_negatives'] = args.negatives is None
        parameters['fragments'] = args.fragments

        with open(args.output, 'w') as outfile:
            outfile.write(json.dumps(parameters, indent=4))

        sys.exit()


    ##########
    # IMPORTS
    ##########


    import os
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

    import torch
    torch.backends.cudnn.benchmark = True

    import numpy

    from bpnetlite.io import PeakGenerator
    from bpnetlite.bpnet import ControlWrapper

    from tangermeme.io import extract_loci
    from tangermeme.utils import example_to_fasta_coords
    from tangermeme.seqlet import recursive_seqlets


    ##########
    # FIT
    ##########


    if args.cmd == "fit":
        from bpnetlite.bpnet import BPNet

        parameters = merge_parameters(args.parameters, default_fit_parameters)

        if parameters['verbose']:
            print("Training Chroms: ", parameters['training_chroms'])
            print("Vaidation Chroms: ", parameters['validation_chroms'])

            print("\nLoading peaks from: ", parameters['loci'])
            print("Loading negatives from: ", parameters['negatives'], "\n")

        ###

        training_data = PeakGenerator(
            peaks=parameters['loci'],
            negatives=parameters['negatives'],
            sequences=parameters['sequences'],
            signals=parameters['signals'],
            controls=parameters['controls'],
            chroms=parameters['training_chroms'],
            in_window=parameters['in_window'],
            out_window=parameters['out_window'],
            max_jitter=parameters['max_jitter'],
            negative_ratio=parameters['negative_ratio'],
            reverse_complement=parameters['reverse_complement'],
            min_counts=parameters['min_counts'],
            max_counts=parameters['max_counts'],
            summits=parameters['summits'],
            exclusion_lists=parameters['exclusion_lists'],
            random_state=parameters['random_state'],
            batch_size=parameters['batch_size'],
            verbose=parameters['verbose']
        )

        valid_data = extract_loci(
            sequences=parameters['sequences'],
            signals=parameters['signals'],
            in_signals=parameters['controls'],
            loci=parameters['loci'],
            chroms=parameters['validation_chroms'],
            in_window=parameters['in_window'],
            out_window=parameters['out_window'],
            max_jitter=0,
            exclusion_lists=parameters['exclusion_lists'],
            ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
            verbose=parameters['verbose']
        )

        if parameters['verbose']:
            print("\nTraining Set Peaks: ", training_data.dataset.peak_sequences.shape[0])
            print("Training Set Negatives: ", training_data.dataset.negative_sequences.shape[0])
            print("Validation Set Size: ", valid_data[0].shape[0], "\n")


        if parameters['count_loss_weight'] is None:
            peak_read_count = training_data.dataset.peak_signals.sum(axis=-1)
            count_loss_weight = peak_read_count.mean(axis=(0, 1)).item()
            parameters['count_loss_weight'] = count_loss_weight

            if parameters['verbose']:
                print("Count loss weight set to {:4.4}".format(count_loss_weight))
                print("Negative Ratio: 1:{:4.4} pos:neg\n".format(
                    parameters['negative_ratio']))


        if parameters['controls'] is not None:
            valid_sequences, valid_signals, valid_controls = valid_data
            n_control_tracks = len(parameters['controls'])
        else:
            valid_sequences, valid_signals = valid_data
            valid_controls = None
            n_control_tracks = 0

        trimming = (parameters['in_window'] - parameters['out_window']) // 2

        model = BPNet(n_filters=parameters['n_filters'],
            n_layers=parameters['n_layers'],
            n_outputs=len(parameters['signals']),
            n_control_tracks=n_control_tracks,
            profile_output_bias=parameters['profile_output_bias'],
            count_output_bias=parameters['count_output_bias'],
            count_loss_weight=parameters['count_loss_weight'],
            trimming=trimming,
            name=parameters['name'],
            verbose=parameters['verbose']).to(parameters['device'])

        optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])

        if parameters['scheduler']:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                patience=5, threshold=0.0001, min_lr=0.0001)
        else:
            scheduler = None

        model.fit(training_data, optimizer, scheduler,
            X_valid=valid_sequences,
            X_ctl_valid=valid_controls, y_valid=valid_signals,
            max_epochs=parameters['max_epochs'],
            batch_size=parameters['batch_size'],
            early_stopping=parameters['early_stopping'],
            dtype=parameters['dtype'],
            device=parameters['device'])


        ### Evaluate Model

        evaluate_parameters = parameters.copy()
        evaluate_parameters['chroms'] = parameters['validation_chroms']
        evaluate_parameters['max_jitter'] = 0
        evaluate_parameters['reverse_complement'] = False
        evaluate_parameters['model'] = parameters['name'] + '.torch'
        evaluate_parameters['performance_filename'] = (parameters['name'] +
            '.performance.tsv')

        fname = "{}.evaluate.json".format(parameters['name'])
        with open(fname, "w") as outfile:
            outfile.write(json.dumps(evaluate_parameters, sort_keys=True, indent=4))

        os.system("bpnet evaluate -p {}".format(fname))


    ##########
    # PREDICT
    ##########


    elif args.cmd == 'predict':
        from tangermeme.predict import predict
        parameters = merge_parameters(args.parameters, default_predict_parameters)

        ###

        model = torch.load(parameters['model'], weights_only=False).to(
            parameters['device'])

        examples = extract_loci(
            sequences=parameters['sequences'],
            in_signals=parameters['controls'],
            loci=parameters['loci'],
            chroms=parameters['chroms'],
            in_window=parameters['in_window'],
            out_window=parameters['out_window'],
            exclusion_lists=parameters['exclusion_lists'],
            max_jitter=0,
            ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
            return_mask=True,
            verbose=parameters['verbose']
        )

        if parameters['controls'] == None:
            X, idxs = examples
            if model.n_control_tracks > 0:
                X_ctl = torch.zeros(X.shape[0], model.n_control_tracks, X.shape[-1])
            else:
                X_ctl = None
        else:
            X, X_ctl, idxs = examples

        if X_ctl is not None:
            X_ctl = (X_ctl,)

        y_profile, y_counts = predict(model, X, args=X_ctl,
            batch_size=parameters['batch_size'], device=parameters['device'],
            dtype=parameters['dtype'], verbose=parameters['verbose'])

        if parameters['reverse_complement_average']:
            X_rc = torch.flip(X, dims=(-1, -2))
            X_ctl_rc = None if X_ctl is None else (torch.flip(X_ctl[0], dims=(-1, -2)),)

            y_profile_rc, y_counts_rc = predict(model, X_rc, args=X_ctl_rc,
                batch_size=parameters['batch_size'], device=parameters['device'],
                dtype=parameters['dtype'], verbose=parameters['verbose'])

            y_profile_rc = torch.flip(y_profile_rc, dims=(-1, -2))

            y_profile = (y_profile + y_profile_rc) / 2
            y_counts = (y_counts + y_counts_rc) / 2

        numpy.savez_compressed(parameters['profile_filename'], y_profile)
        numpy.savez_compressed(parameters['counts_filename'], y_counts)
        numpy.savez_compressed(parameters['idx_filename'], idxs)


    ##########
    # EVALUATE
    ##########


    elif args.cmd == 'evaluate':
        from bpnetlite.performance import calculate_performance_measures

        parameters = merge_parameters(args.parameters, default_evaluate_parameters)
        measure_names = ['profile_mnll', 'profile_jsd', 'profile_pearson',
            'profile_spearman', 'count_pearson', 'count_spearman', 'count_mse']

        ###

        name = parameters['name']
        predict_parameters = parameters.copy()
        predict_parameters['profile_filename'] = '{}.eval.profile.npz'.format(name)
        predict_parameters['counts_filename'] = '{}.eval.counts.npz'.format(name)
        predict_parameters['idx_filename'] = '{}.eval.idx.npz'.format(name)

        with open("{}.bpnet.evaluate.predict.json".format(name), "w") as outfile:
            outfile.write(json.dumps(predict_parameters, sort_keys=True, indent=4))

        os.system("bpnet predict -p {}.bpnet.evaluate.predict.json".format(name))

        y_hat_logits = numpy.load('{}.eval.profile.npz'.format(name))['arr_0']
        y_hat_logits = torch.from_numpy(y_hat_logits)

        y_hat_logcounts = numpy.load('{}.eval.counts.npz'.format(name))['arr_0']
        y_hat_logcounts = torch.from_numpy(y_hat_logcounts)

        os.system("rm {}.eval.*.npz".format(name))

        _, y = extract_loci(
            sequences=parameters['sequences'],
            signals=parameters['signals'],
            loci=parameters['loci'],
            chroms=parameters['chroms'],
            in_window=parameters['in_window'],
            out_window=parameters['out_window'],
            exclusion_lists=parameters['exclusion_lists'],
            max_jitter=0,
            ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
            verbose=parameters['verbose']
        )

        measures = calculate_performance_measures(y_hat_logits, y,
            y_hat_logcounts, smooth_true=True)

        with open(parameters['performance_filename'], "w") as outfile:
            outfile.write("\t".join(measure_names))
            outfile.write("\n")
            outfile.write("\t".join([str(measures[name].mean().item())
                for name in measure_names]))


    ##########
    # ATTRIBUTE
    ##########


    elif args.cmd == 'attribute':
        from bpnetlite.bpnet import CountWrapper
        from bpnetlite.bpnet import ProfileWrapper
        from bpnetlite.chrombpnet import ChromBPNet
        from bpnetlite.attribute import deep_lift_shap

        parameters = merge_parameters(args.parameters, default_attribute_parameters)

        ###

        model = torch.load(parameters['model'], weights_only=False).to(
            parameters['device'])

        dtype = torch.float32
        if parameters['output'] == 'profile' or isinstance(model, ChromBPNet):
            dtype = torch.float64

        X, idxs = extract_loci(
            sequences=parameters['sequences'],
            loci=parameters['loci'],
            chroms=parameters['chroms'],
            max_jitter=0,
            ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
            return_mask=True,
            verbose=parameters['verbose']
        )

        n_idxs = X.sum(dim=(1, 2)) == X.shape[-1]
        X = X[n_idxs]

        idxs = idxs[n_idxs]

        model = ControlWrapper(model)
        if parameters['output'] == 'counts':
            wrapper = CountWrapper(model)
        elif parameters['output'] == 'profile':
            wrapper = ProfileWrapper(model)
        else:
            raise ValueError("output must be either `counts` or `profile`.")

        X_attr = deep_lift_shap(wrapper.type(dtype), X.type(dtype),
            hypothetical=True,
            n_shuffles=parameters['n_shuffles'],
            batch_size=parameters['batch_size'],
            device=parameters['device'],
            random_state=parameters['random_state'],
            verbose=parameters['verbose'],
            warning_threshold=parameters['warning_threshold'])

        numpy.savez_compressed(parameters['ohe_filename'], X)
        numpy.savez_compressed(parameters['attr_filename'], X_attr)
        numpy.save(parameters['idx_filename'], idxs)


    ##########
    # SEQLETS
    ##########


    elif args.cmd == 'seqlets':
        from tangermeme.io import _interleave_loci

        parameters = merge_parameters(args.parameters, default_seqlet_parameters)

        ###

        idxs = numpy.load(parameters['idx_filename'])

        loci = _interleave_loci(parameters['loci'], parameters['chroms'])
        loci = loci.iloc[idxs]

        X = numpy.load(parameters['ohe_filename'])['arr_0']
        X = torch.from_numpy(X)

        X_attr = numpy.load(parameters['attr_filename'])['arr_0']
        X_attr = torch.from_numpy(X_attr)
        X_attr = (X_attr * X).sum(dim=1)

        seqlets = recursive_seqlets(
            X_attr,
            threshold=parameters['threshold'],
            min_seqlet_len=parameters['min_seqlet_len'],
            max_seqlet_len=parameters['max_seqlet_len'],
            additional_flanks=parameters['additional_flanks']
        ).sort_values("attribution", ascending=False)

        seqlets = example_to_fasta_coords(seqlets, loci, parameters['in_window'])
        seqlets.to_csv(parameters['output_filename'], sep='\t', index=False,
            header=False)


    ##########
    # MARGINALIZE
    ##########


    elif args.cmd == 'marginalize':
        from bpnetlite.marginalize import marginalization_report

        parameters = merge_parameters(args.parameters,
            default_marginalize_parameters)

        ###

        model = torch.load(parameters['model'], weights_only=False).to(
            parameters['device'])
        model = ControlWrapper(model)

        X = extract_loci(
            sequences=parameters['sequences'],
            loci=parameters['loci'],
            chroms=parameters['chroms'],
            max_jitter=0,
            ignore=list('QWERYUIOPSDFHJKLZXVBNM'),
            n_loci=parameters['n_loci'],
            verbose=parameters['verbose']
        ).float()

        if parameters['shuffle'] == True:
            idxs = numpy.arange(X.shape[0])
            numpy.random.shuffle(idxs)
            X = X[idxs]

        if parameters['n_loci'] is not None:
            X = X[:parameters['n_loci']]

        marginalization_report(model, parameters['motifs'], X,
            parameters['output_filename'],
            attributions=parameters['attributions'],
            batch_size=parameters['batch_size'],
            minimal=parameters['minimal'],
            device=parameters['device'],
            verbose=parameters['verbose'])


    ##########
    # PIPLEINE
    ##########


    elif args.cmd == 'pipeline':
        import pandas
        import subprocess

        parameters = merge_parameters(args.parameters, default_pipeline_parameters)
        pname = parameters['name']


        ###
        # Step 0.1: Convert from SAM/BAMs to bigwigs if provided
        ###

        ftypes = '.sam', '.bam', '.tsv', '.tsv.gz'

        if parameters['signals'][0].endswith(ftypes):
            if parameters['verbose']:
                print("Step 0.1: Convert data to bigWigs")

            args = [
                "bam2bw",
                "-s", parameters['sequences'],
                "-n", pname,
            ]

            if parameters["unstranded"]:
                args += ["-u"]

            if parameters['fragments']:
                args += ["-f"]

            if parameters["verbose"]:
                args += ["-v"]

            args += parameters['signals']
            subprocess.run(args, check=True)

            if parameters["unstranded"]:
                parameters['signals'] = [pname + ".bw"]
            else:
                parameters['signals'] = [pname + ".+.bw", pname + ".-.bw"]


        if parameters['controls'] is not None:
            if parameters['controls'][0].endswith(ftypes):
                args = [
                    "bam2bw",
                    "-s", parameters['sequences'],
                    "-n", pname + ".control",
                ]

                if parameters["unstranded"]:
                    args += ["-u"]

                if parameters["fragments"]:
                    args += ["-f"]

                if parameters["verbose"]:
                    args += ["-v"]

                args += parameters['controls']
                subprocess.run(args, check=True)

                if parameters["unstranded"]:
                    parameters['controls'] = [pname + ".control.bw"]
                else:
                    parameters['controls'] = [pname + ".control.+.bw", pname + ".control.-.bw"]


        ###
        # Step 0.2: Identify GC-matched negative regions
        ###

        if parameters['find_negatives'] == True:
            if parameters['verbose']:
                print("\nStep 0.2: Find GC-matched negative regions.")

            args = [
                "bpnet", "negatives",
                "-i", parameters["loci"][0],
                "-f", parameters["sequences"],
                "-o", pname + ".negatives.bed"
            ]

            if parameters['verbose']:
                args += ['-v']

            parameters["negatives"] = [pname + ".negatives.bed"]

            subprocess.run(args, check=True)


        ###
        # Step 1: Fit a BPNet model to the provided data
        ###

        if parameters['verbose']:
            print("\nStep 1: Fitting a BPNet model")

        fit_parameters = _extract_set(parameters, default_fit_parameters,
            'fit_parameters')

        if parameters.get('model', None) == None:
            name = pname + '.bpnet.fit.json'
            parameters['model'] = pname + '.torch'
            _check_set(fit_parameters, 'performance_filename', pname + '.performance.tsv')

            with open(name, 'w') as outfile:
                outfile.write(json.dumps(fit_parameters, sort_keys=True, indent=4))

            subprocess.run(["bpnet", "fit", "-p", name], check=True)


        ###
        # Step 2: Make predictions for the entire validation set
        ###

        if parameters['verbose']:
            print("\nStep 2: Making predictions")

        predict_parameters = _extract_set(parameters,
            default_predict_parameters, 'predict_parameters')
        _check_set(predict_parameters, 'profile_filename', pname+'.predictions.profiles.npz')
        _check_set(predict_parameters, 'counts_filename',  pname+'.predictions.counts.npz')
        _check_set(predict_parameters, 'idx_filename',     pname+'.predictions.idxs.npy')

        name = '{}.bpnet.predict.json'.format(parameters['name'])
        with open(name, 'w') as outfile:
            outfile.write(json.dumps(predict_parameters, sort_keys=True, indent=4))

        subprocess.run(["bpnet", "predict", "-p", name], check=True)


        ###
        # Step 3: Calculate attributions
        ###

        if parameters['verbose']:
            print("\nStep 3: Calculating attributions")

        attribute_parameters = _extract_set(parameters,
            default_attribute_parameters, 'attribute_parameters')
        _check_set(attribute_parameters, 'ohe_filename',  pname+'.attributions.ohe.npz')
        _check_set(attribute_parameters, 'attr_filename', pname+'.attributions.attr.npz')
        _check_set(attribute_parameters, 'idx_filename',  pname+'.attributions.idxs.npy')

        name = '{}.bpnet.attribute.json'.format(parameters['name'])
        with open(name, 'w') as outfile:
            outfile.write(json.dumps(attribute_parameters, sort_keys=True, indent=4))

        subprocess.run(["bpnet", "attribute", "-p", name], check=True)


        ###
        # Step 4.1: Identify seqlets from attributions
        ###

        if parameters['verbose']:
            print("\nStep 4.1: Seqlet identification")

        seqlet_parameters = _extract_set(parameters,
            default_seqlet_parameters, 'seqlet_parameters')
        _check_set(seqlet_parameters, "ohe_filename",  pname+'.attributions.ohe.npz')
        _check_set(seqlet_parameters, "attr_filename", pname+'.attributions.attr.npz')
        _check_set(seqlet_parameters, "idx_filename",  pname+'.attributions.idxs.npy')
        _check_set(seqlet_parameters, "output_filename", pname+".seqlets.bed")
        _check_set(seqlet_parameters, "chroms", attribute_parameters['chroms'])

        name = '{}.bpnet.seqlets.json'.format(parameters['name'])
        with open(name, 'w') as outfile:
            outfile.write(json.dumps(seqlet_parameters, sort_keys=True, indent=4))

        subprocess.run(["bpnet", "seqlets", "-p", name], check=True)


        ###
        # Step 4.2: Annotate seqlets using motif database
        ###

        if parameters['verbose']:
            print("\nStep 4.2: Seqlet annotation")

        annotation_parameters = _extract_set(parameters,
            default_annotation_parameters, "annotation_parameters")
        _check_set(annotation_parameters, "seqlet_filename", pname+".seqlets.bed")
        _check_set(annotation_parameters, "output_filename", pname+".seqlets_annotated.bed")
        _check_set(annotation_parameters, "motifs", parameters["motifs"])

        annotation_parameters = merge_parameters(annotation_parameters,
            default_annotation_parameters)

        cmd = ["ttl"]
        cmd += ["-f", annotation_parameters["sequences"]]
        cmd += ["-b", annotation_parameters["seqlet_filename"]]
        cmd += ["-s", str(annotation_parameters["n_score_bins"])]
        cmd += ["-m", str(annotation_parameters["n_median_bins"])]
        cmd += ["-a", str(annotation_parameters["n_target_bins"])]
        cmd += ["-c", str(annotation_parameters["n_cache"])]
        cmd += ["-j", str(annotation_parameters["n_jobs"])]

        if not annotation_parameters["reverse_complement"]:
            cmd += ["-r"]

        if annotation_parameters['motifs'] is not None:
            cmd += ["-t", annotation_parameters["motifs"]]

        with open(annotation_parameters['output_filename'], "w") as f:
            subprocess.run(cmd, check=True, stdout=f)

        annotated_seqlets = pandas.read_csv(annotation_parameters['output_filename'],
            sep="\t", header=None, usecols=(3,), names=['motifs'])

        seqlet_count = annotated_seqlets.value_counts()
        seqlet_count.to_csv(pname+".motif_seqlet_count.tsv", sep="\t")


        ###
        # Step 5.1: Run TF-MoDISco
        ###

        if parameters['verbose']:
            print("\nStep 5.1: TF-MoDISco motifs")

        modisco_parameters = parameters['modisco_motifs_parameters']

        _check_set(modisco_parameters, "output_filename",
            pname+'_modisco_results.h5')
        _check_set(modisco_parameters, "verbose",
            parameters['verbose'])

        modisco_parameters = merge_parameters(modisco_parameters,
            default_pipeline_parameters['modisco_motifs_parameters'])

        cmd = "modisco motifs -s {} -a {} -n {} -o {}".format(
            attribute_parameters['ohe_filename'],
            attribute_parameters['attr_filename'],
            modisco_parameters['n_seqlets'],
            modisco_parameters['output_filename'])

        if 'verbose' in modisco_parameters and modisco_parameters['verbose']:
            cmd += ' -v'
        elif parameters['verbose']:
            cmd += ' -v'

        subprocess.run(cmd.split(), check=True)


        ###
        # Step 5.2: Generate the tf-modisco report
        ###

        report_parameters = parameters['modisco_report_parameters']
        _check_set(report_parameters, "verbose", parameters["verbose"])
        _check_set(report_parameters, "output_folder", pname+"_modisco/")
        _check_set(report_parameters, "motifs", parameters['motifs'])

        if report_parameters['verbose']:
            print("\nStep 5.2: TF-MoDISco reports")

        subprocess.run(["modisco", "report",
            "-i", modisco_parameters['output_filename'],
            "-o", report_parameters['output_folder'],
            "-s", './',
            "-m", report_parameters['motifs']
            ], check=True)


        ###
        # Step 6: Marginalization experiments
        ###

        if parameters['verbose']:
            print("\nStep 6: Run marginalizations")

        marginalize_parameters = _extract_set(parameters,
            default_marginalize_parameters, "marginalize_parameters")

        _check_set(marginalize_parameters, "loci", parameters["negatives"])
        _check_set(marginalize_parameters, 'output_filename', pname+"_marginalize/")
        _check_set(marginalize_parameters, 'motifs', parameters['motifs'])
        _check_set(marginalize_parameters, 'negatives', parameters['negatives'])

        name = '{}.bpnet.marginalize.json'.format(parameters['name'])

        with open(name, 'w') as outfile:
            outfile.write(json.dumps(marginalize_parameters, sort_keys=True,
                indent=4))

        subprocess.run(["bpnet", "marginalize", "-p", name], check=True)
