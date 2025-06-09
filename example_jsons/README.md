## Example JSON Parameters

Each of the steps for training and using BPNet and ChromBPNet models have many parameters. Most of the default parameters work well broadly but you may want to tweak them, or be operating in a special circumstance where changes to the defaults need to be made. Because there are so many parameters, it made sense to have these commands take in a JSON instead of forcing the user to specify many arguments for each step. In my experience, having a written record of what each command was is also immensely valuable when going back and documenting each of the steps or for debugging.

Below, you will find a copy of the example JSON for each command along with an explanation of what the parameter means. An important note is that you do not need to actually specify every parameter in each JSON: just the ones that are different than the defaults. The respective commands (`bpnet` and `chrombpnet`) store the default parameters for each command, so please check those if you are experiencing unexpected behavior when not specifying every command. 

> **Warning**: 
> Many of the JSONs in the repo above have spaces to make them visually more digestable. Your JSON parser may not like these spaces so, once you understand the parameters, please remove the spaces. 

### BPNet Pipeline Parameters

Likely, the most used command is `bpnet pipeline` which handles everything from processing the data to training the BPNet model and making predictions, attributions, identifying and annotating seqlets, running TF-MoDISco, and performing in silico marginalizations. Even if you do not need each step, it can be faster to simply run this because each step will have data filepaths programatically filled in from the previous step so you do not need to think too much.

This JSON in its entirety can be quite intimidating. Scroll past it for a minimal version that relies heavily on the defaults and for a command line tool that will automatically generate it given the data filenames.

```
{
	"n_filters": 64,               # Number of filters in the convolutions
	"n_layers": 8,                 # Number of dilated residual convolutions between the initial, and final, layers.
	"profile_output_bias": true,   # Whether to include a bias term in the profile head
	"count_output_bias": true,     # Whether to include a bias term in the count head
	"in_window": 2114,             # Length of the input window
	"out_window": 1000,            # Length of the output window
	"name": "test",                # Name of the model, primarily used to auto-generate output file names
	"model": null,                 # Name of the model to use for the remaining steps instead of fitting a new one
	                               # If null, train a new model
	"batch_size": 64,              # Batch size to use for training and validation
	"max_jitter": 128,             # Maximum amount of jitter when generating training examples
	"reverse_complement": true,    # Whether to randomly RC half of the training examples
	"max_epochs": 20,              # The maximum number of epochs to train for
	"validation_iter": 100,        # The number of batches to train on before calculating validation set performance
	"lr": 0.001,                   # Learning rate of the AdamW optimizer
	"alpha": 100,                  # Weight of the count-loss in the total loss.
	"device": "cuda",              # The device to use, usually `cuda` or `cpu`
	"dtype", "float32",            # The dtype to use, usually `float32` or `bfloat16`
	"verbose": true,               # Whether to print out a log to the terminal during training
	
	"min_counts": 0,               # Ensure that each training example has at least this number of counts
	"max_counts": 99999999,        # Ensure that each training example has no more than this number of counts
	"sequences": "hg38.fa",        # FASTA file of the genome to train on
	"loci": ['peaks.bed.gz'],      # Loci to use in other steps (must be a list)
	'find_negatives': true,        # Whether to find GC-matched negatives for the loci (will set the `negatives` field
	'unstranded': false,           # If processing BAMs/SAMs/tsv/tsv.gzs into bigWigs, if the signal is unstranded
	'fragments': false,            # If processing BAMs/SAMs/tsv/tsv.gzs into bigWigs, if the lines are fragments
	
	"signals": [                   # A list containing the BAM/SAM/tsv/tsv.gz or bigWig files with the signal
       "input1.bam", 
       "input2.bam"
    ],
	"controls":[                   # A list containing the BAM/SAM/tsv/tsv.gz or bigWig files with the controls
       "control1.bam",                 # Optional, should be `null` if not used.
       "control2.bam"
    ],
	
	"fit_parameters": {            # The parameters to use for the fit step. If `null`, will inherit from the upper level and then defaults.
		"batch_size": 64,
		"training_chroms": ["chr2", "chr3", "chr4", "chr5", "chr6", "chr7", 
			"chr9", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", 
			"chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"],
		"validation_chroms": ["chr8", "chr10"],
		"sequences": null,
		"loci": null,
		"signals": null,
		"controls": null,
		"verbose": null,
		"random_state": null
	},
	
	"predict_parameters": {        # The parameters to use for the predict step. If `null`, will inherit from the upper level and then defaults.
		"batch_size": 64,
		"chroms": ["chr8", "chr10"],
		"profile_filename": null,
		"counts_filename": null,
		"sequences": null,
		"loci": "peaks.bed.gz",
		"signals": null,
		"controls": null,
		"verbose": null
	},
	
	"attribute_parameters": {      # The parameters to use for the attribute step. If `null`, will inherit from the upper level and then defaults.
		"batch_size": 1,
		"output": "counts",
		"chroms": ["chr8"],
		"loci": "peaks.bed.gz",
		"ohe_filename": null,
		"attr_filename": null,
		"n_shuffles": 20,
		"random_state": null,
		"verbose": null
	},
	
	"seqlet_parameters": {         # The parameters to use for the seqlet step. If `null`, will inherit from the upper level and then defaults.
		"threshold": 0.01,
		"min_seqlet_len": 4,
		"max_seqlet_len": 25,
		"additional_flanks": 3,
		"in_window": null,
		"chroms": null,
		"verbose": null,
		"loci": null,
		"ohe_filename": null,
		"attr_filename": null,
		"idx_filename": null,
		"output_filename": null
	},

	"annotation_parameters": {      # The parameters to use for the seqlet annotation step. These are fed into `ttl`.
	        "motifs": null,
	        "sequences": null,
	        "seqlet_filename": null,
	        "n_score_bins": 100,
	        "n_median_bins": 1000,
	        "n_target_bins": 100,
	        "n_cache": 250,
	        "reverse_complement": true,
	        "n_jobs": -1,
	        "output_filename": null
	}

	"modisco_motifs_parameters": {  # The parameters to use for running TF-MoDISco step. If `null`, will inherit from the upper level and then defaults.
		"n_seqlets": 100000,
		"output_filename": null,
		"verbose": null
	},
	
	"modisco_report_parameters": {  # The parameters to use for generating the TF-MoDISco report. If `null`, will inherit from the upper level and then defaults.
		"motifs": "motifs.meme",
		"output_folder": null,
		"verbose": null
	},
	
	"marginalize_parameters": {     # The parameters to use for the marginalization step. If `null`, will inherit from the upper level and then defaults.
		"loci": null,
		"n_loci": 100,
		"shuffle": false,
		"random_state": null,
		"output_folder": null,
		"motifs": "motifs.meme",
		"minimal": true,
		"verbose": null
	} 
}
```

As mentioned, this JSON can be quite intimidating because it contains parameters for *every* step, including the data preprocessing and all of the downstream analyses. However, not all parameters have to be specified. When a value is set ot `null` or not included in the JSON, the tool reverts back to the default for that parameter. Some parameters are required, and you will be prompted to pass them in if this is the case. If you strip out all of the parameters that are not required, here is a minimal BPNet pipeline JSON.

```
```



If you just want to get started, it may seem a lot to have to consider all of these parameters. This is why we included the `bpnet pipeline-json` command, which will automatically format the JSON for you given the provided data filenames. That way you can just poke the specific parameters you think need to be changed.

Regardless, not all parameters have to be specified. When a value is set ot `null` or not included in the JSON, the tool reverts back to the default for that parameter. Some parameters are required, and you will be prompted to pass them in if this is the case. However, if you strip out all of the parameters that are not required, here is a minimal BPNet pipeline JSON.




### BPNet Fit Parameters

```
{
   "n_filters": 64,              # Number of filters in the convolutions
   "n_layers": 8,                # Number of dilated residual convolutions between the initial, and final, layers.
   "profile_output_bias": true,  # Whether to include a bias term in the profile head
   "count_output_bias": true,    # Whether to include a bias term in the count head
   "name": "example",            # Name of the model, primarily used to auto-generate output names if not provided
   
   "batch_size": 64,             # Batch size to use for training and validation
   "in_window": 2114,            # Length of the input window
   "out_window": 1000,           # Length of the output window
   "max_jitter": 128,            # Maximum amount of jitter when generating training examples
   "reverse_complement": true,   # Whether to randomly RC half of the training examples
   "max_epochs": 50,             # The maximum number of epochs to train for
   "validation_iter": 100,       # The number of batches to train on before calculating validation set performance
   "lr": 0.001,                  # Learning rate of the AdamW optimizer
   "alpha": 1,                   # Weight of the count-loss in the total loss.
   "verbose": false,             # Whether to print out a log to the terminal during training

   "min_counts": 0,              # Ensure that each training example has at least this number of counts
   "max_counts": 99999999,       # Ensure that each training example has no more than this number of counts

   # Chromosomes to train on
   "training_chroms": ["chr2", "chr3", "chr4" "chr5", "chr6", "chr7", 
      "chr9", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17", 
      "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"],
   
   # Chromosomes to validate on
   "validation_chroms": ["chr8", "chr10"],

   "sequences":"../../oak/common/hg38/hg38.fa",                      # FASTA file of the genome to train on
   "loci":"../../tfatlas/processed_data/ENCSR000BGW/peaks.bed.gz",   # Loci to train and validate on (can be a list)
   
   # A list of bigWig files to extract signal from -- each element in the list corresponds to one model output
   "signals":[ 
      "../../tfatlas/processed_data/ENCSR000BGW/ENCSR000BGW_plus.bigWig", 
      "../../tfatlas/processed_data/ENCSR000BGW/ENCSR000BGW_minus.bigWig"
   ],
   
   # An optional list of bigWig files containing control signals 
   "controls":[
      "../../tfatlas/processed_data/ENCSR000BGW/ENCSR000BGW_control_plus.bigWig", 
      "../../tfatlas/processed_data/ENCSR000BGW/ENCSR000BGW_control_minus.bigWig"
   ],
   "random_state": 0  # A seed to control parameter initialization and data generation
}
```

### BPNet Predict

```
{
   "batch_size":64,             # Batch size to use for training and validation
   "in_window":2114,            # Length of the input window
   "out_window":1000,           # Length of the output window
   "verbose":true,              # Whether to print out a log to the terminal during training
   "chroms":["chr8", "chr10"],  # Chromosomes whose peaks to make predictions for
   "model":"bpnet.64.8.torch",  # Model to use to make the predictions

   "sequences":"../../oak/common/hg38/hg38.fa", # FASTA file of the genome to train on
   "loci":"../../tfatlas/processed_data/ENCSR000BGW/peaks.bed.gz", # Loci to train and validate on (can be a list)
   
   # An optional list of bigWig files containing control signals 
   "controls": [
      "../../tfatlas/processed_data/ENCSR000BGW/ENCSR000BGW_control_plus.bigWig", 
      "../../tfatlas/processed_data/ENCSR000BGW/ENCSR000BGW_control_minus.bigWig"
   ],

   "profile_filename": "y_profile.npz",  # The name of the file to store profile predictions
   "counts_filename": "y_counts.npz"     # The name of the file to store count predictions
}
```

### BPNet Attribute

```
{
   "batch_size": 64,             # Batch size to use for training and validation
   "in_window": 2114,            # Length of the input window
   "out_window": 1000,           # Length of the output window
   "verbose": true,              # Whether to print out a progress bar during attribution
   "chroms": ["chr8", "chr10"],  # Chromosomes whose peaks to make predictions for
   "model":"bpnet.64.8.torch",   # Model to use for calculating attributions

   "sequences":"../../oak/common/hg38/hg38.fa",  # FASTA file of the genome to train on
   "loci":"../../tfatlas/processed_data/ENCSR000BGW/peaks.bed.gz",  # Loci to attribute (can be a list)

   "output": "profile",          # Which head to calculate attributions for
   "ohe_filename": "ohe.npz",    # Filename to store one-hot encodings of the sequences
   "attr_filename": "attr.npz",  # Filename to store DeepLIFT/SHAP values for the sequences
   "n_shuffles":20,              # Number of GC-matched shuffles to perform
   "random_state":0              # A seed to control the shuffles
}
```

### BPNet Marginalize

```
{
	"batch_size": 64,             # Batch size to use for training and validation
	"in_window": 2114,            # Length of the input window
	"out_window": 1000,           # Length of the output window
	"verbose": true,              # Whether to print out a log as motifs are inserted
	"chroms": ["chr8", "chr10"],  # Chromosomes whose peaks to make predictions for
   
	"sequences": "../../oak/common/hg38/hg38.fa",  # FASTA file of the genome to train on
	"loci": "../../tfatlas/processed_data/ENCSR000BGW/gc_neg_only.bed.gz", # Loci to use as background
   
   "motifs": "motifs.meme",           # MEME file of motifs to insert into the sequences
	"n_loci": 100,                     # Number of background loci to use
	"shuffle": false,                  # Whether to shuffle the loci extracted before choosing `n_loci`
	"model": "bpnet.64.8.torch",       # Model to use to make the predictions
	"output_filename":"marginalize/",  # Folder to store the results including the summary
	"random_state":0,                  # A seed to control the shuffling
	"minimal": true                    # Whether to produce a minimal report or report all columns
}
```

### BPNet Pipeline

When running the pipeline command, JSONs for each of the steps in the pipeline will be constructed from merging the parameters in the first layer with those in the nested JSONs, e.g., in the `fit_parameters`. If provided, parameters in the nested JSON will always override those in the first layer. For example, if you want to make predictions on a different genome than one has trained on, you can pass a different `sequences` and `loci` keyword in the `predict_parameters` JSON, or if you want to use a different MEME file in the marginalization and TF-MoDISco step you can. 

```
{
	"n_filters": 64,              # Number of filters in the convolutions
	"n_layers": 8,                # Number of dilated residual convolutions between the initial, and final, layers.
	"profile_output_bias": true,  # Whether to include a bias term in the profile head
	"count_output_bias": true,    # Whether to include a bias term in the count head
	"in_window": 2114,            # Length of the input window
	"out_window": 1000,           # Length of the output window
	"name": "spi1",               # Name of the model, primarily used to auto-generate output names if not provided
	"model": "spi1.torch",        # Name of the model to use for the remaining steps instead of fitting a new one
	"batch_size": 64,             # Batch size to use for training and validation
	"max_jitter": 128,            # Maximum amount of jitter when generating training examples
	"reverse_complement": true,   # Whether to randomly RC half of the training examples
	"max_epochs": 50,             # The maximum number of epochs to train for
	"validation_iter": 100,       # The number of batches to train on before calculating validation set performance
	"lr": 0.001,                  # Learning rate of the AdamW optimizer
	"alpha": 10,                  # Weight of the count-loss in the total loss.
	"verbose": true,              # Whether to print out a log to the terminal during training
	"min_counts": 0,              # Ensure that each training example has at least this number of counts
	"max_counts": 99999999,       # Ensure that each training example has no more than this number of counts
	"sequences": "../../../oak/common/hg38/hg38.fa",  # FASTA file of the genome to train on
	"loci": null,                 # Loci to use in other steps (can be a list)
   
   # A list of bigWig files to extract signal from -- each element in the list corresponds to one model output
	"signals": [
       "../../../tfatlas/processed_data/ENCSR000BGQ/ENCSR000BGQ_plus.bigWig", 
       "../../../tfatlas/processed_data/ENCSR000BGQ/ENCSR000BGQ_minus.bigWig"
    ],
    
   # An optional list of bigWig files containing control signals 
	"controls":[
       "../../../tfatlas/processed_data/ENCSR000BGQ/ENCSR000BGQ_control_plus.bigWig", 
       "../../../tfatlas/processed_data/ENCSR000BGQ/ENCSR000BGQ_control_minus.bigWig"
    ],
    
   # Parameters to pass into the fit step
	"fit_parameters": {
		"batch_size": 64,  # Batch size to use for training and validation
      
		# Chromosomes to train on
      "training_chroms": ["chr2", "chr3", "chr4", "chr5", "chr6", "chr7", 
			"chr9", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", 
			"chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"],
      
      # Chromosomes to validate on
		"validation_chroms": ["chr8", "chr10"],
		"sequences": null,  # FASTA file of the genome to train on
      
      # Loci to train and validate on (can be a list)
		"loci": [
			"../../../tfatlas/processed_data/ENCSR000BGQ/peaks.bed.gz",
			"../../../tfatlas/processed_data/ENCSR000BGQ/gc_neg_only.bed.gz"
		],
		"signals": null,      # A list of bigWig files to extract signal from
		"controls": null,	    # An optional list of bigWig files containing control signals 
		"verbose": null,      # Whether to print out a log to the terminal during training
		"random_state": null  # A seed to control parameter initialization and data generation
	},
   
   # Parameters to pass into the predict step
	"predict_parameters": {
		"batch_size": 64,             # Batch size to use for training and validation
		"chroms": ["chr8", "chr10"],  # Chromosomes to make predictions for
		"profile_filename": null,     # The name of the file to store profile predictions
		"counts_filename": null,      # The name of the file to store count predictions
		"sequences": null,            # FASTA file of the genome to make predictions for
      
      # Loci to make predictions for (can be a list)
		"loci": "../../../tfatlas/processed_data/ENCSR000BGQ/peaks.bed.gz", 
      
		"sequences": null,   # A list of bigWig files to extract signal from
		"controls": null,  # An optional list of bigWig files containing control signals 
		"verbose": null    # Whether to print out a log to the terminal during training
	},
   
   # Parameters to pass into the attribute step
	"attribute_parameters": {
		"batch_size": 1,      # Batch size to use for training and validation
		"output": "profile",  # Which head to calculate attributions for
		"chroms": ["chr8"],   # Chromosomes to make predictions for
      
      # Loci to make attributions for (can be a list)
		"loci": "../../../tfatlas/processed_data/ENCSR000BGQ/peaks.bed.gz",
      
		"ohe_filename": null,   # Filename to store one-hot encodings of the sequences
		"attr_filename": null,  # Filename to store DeepLIFT/SHAP values for the sequences
		"n_shuffles": 20,       # Number of GC-matched shuffles to perform
		"random_state": null,   # A seed to control the shuffles
		"verbose": null         # Whether to print out a progress bar during attribution

	},
   
   # Parameters to pass into the `modisco motifs` subcommand
	"modisco_motifs_parameters": {
		"n_seqlets": 100000,      # Maximum number of seqlets to use
		"output_filename": null,  # Filename to store the clustering results
		"verbose": null           # Whether to print out a progress bar during TF-MoDISco
	},
   
   # Parameters to pass into the `modisco report` subcommand
	"modisco_report_parameters": {
		"motifs": "motifs.meme",  # MEME file of motifs to compare the found motifs to
		"output_folder": null,    # Folder to store image outputs and report
	},
   
   # Parameters to pass into the marginalization step
	"marginalize_parameters": {
      # Loci to make predictions for (can be a list)
		"loci": "../../../tfatlas/processed_data/ENCSR000BGQ/gc_neg_only.bed.gz",
		"n_loci": 100,                    # Number of background loci to use
		"shuffle": false,                 # Whether to shuffle the loci extracted before choosing `n_loci`
		"random_state": null,             # A seed to control the shuffling
		"output_folder": null,            # Folder to store the results including the summary
		"motifs": "ctcf.gata2.sp1.meme",  # MEME file of motifs to insert into the sequences
		"minimal": true,                  # Whether to produce a minimal report or report all columns
		"verbose": null                   # Whether to print out a log to the terminal as motifs are inserted
	} 
}
```

## ChromBPNet Fit

```
{
   "n_filters": 64,              # Number of filters in the convolutions
   "n_layers": 8,                # Number of dilated residual convolutions between the initial, and final, layers.
   "profile_output_bias": true,  # Whether to include a bias term in the profile head
   "count_output_bias": true,    # Whether to include a bias term in the count head
   "name": "atac"                # Name of the model

   "batch_size": 64,             # Batch size to use for training and validation
   "in_window": 2114,            # Length of the input window
   "out_window": 1000,           # Length of the output window
   "max_jitter": 128,            # Maximum amount of jitter when generating training examples
   "reverse_complement": true,   # Whether to randomly RC half of the training examples
   "max_epochs": 50,             # The maximum number of epochs to train for
   "validation_iter": 100,       # The number of batches to train on before calculating validation set performance
   "lr": 0.001,                  # Learning rate of the AdamW optimizer
   "alpha": 10,                  # Weight of the count-loss in the total loss.
   "beta": 0.5,                  # Multiplier on the minimum read count in peaks to use when training a bias model

   "min_counts": 0,              # Ensure that each training example has at least this number of counts
   "max_counts": 99999999,       # Ensure that each training example has no more than this number of counts

   # Loci to train and validate on (can be a list)
   "loci": "../../../chromatin-atlas/ATAC/ENCSR637XSC/preprocessing/downloads/peaks.bed.gz",
   
   # Negatives to train on
   "negatives": "../../../chromatin-atlas/ATAC/ENCSR637XSC/negatives_data/negatives.bed",
   
   # FASTA file of the genome to train on
   "sequences": "../../../oak/common/hg38/hg38.fa",
   
   # A list of bigWig files to extract signal from -- each element in the list corresponds to one model output
   "signals": [
        "../../../chromatin-atlas/ATAC/ENCSR637XSC/preprocessing/bigWigs/ENCSR637XSC.bigWig"
   ],
   
   # If you've already trained a bias model, put the filepath in here
   "bias_model": null,

   # Parameters for training the bias model
   "bias_fit_parameters": {
        "alpha": null,
        "loci": "../../../chromatin-atlas/ATAC/ENCSR637XSC/negatives_data/negatives.bed",
        "max_counts": null,
        "n_filters": null,
        "n_layers": 4,
        "random_state": null,
        "verbose": null
    },
    

    # Chromosomes to train on
    "training_chroms": ["chr2", "chr3", "chr4", "chr5", "chr6", "chr7", 
			"chr9", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", 
			"chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"],
      
    # Chromosomes to validate on
    "validation_chroms": ["chr8", "chr10"],
    
    "verbose": true,   # # Whether to print out a log to the terminal during training
    "random_state": 0  # A random state to control parameter initialization and data generation 
}
```

## ChromBPNet Pipeline

When running the pipeline command, JSONs for each of the steps in the pipeline will be constructed from merging the parameters in the first layer with those in the nested JSONs, e.g., in the `fit_parameters`. If provided, parameters in the nested JSON will always override those in the first layer. For example, if you want to make predictions on a different genome than one has trained on, you can pass a different `sequences` and `loci` keyword in the `predict_parameters` JSON, or if you want to use a different MEME file in the marginalization and TF-MoDISco step you can. 

```
{
   "n_filters": 64,              # Number of filters in the convolutions
   "n_layers": 8,                # Number of dilated residual convolutions between the initial, and final, layers.
   "profile_output_bias": true,  # Whether to include a bias term in the profile head
   "count_output_bias": true,    # Whether to include a bias term in the count head

   "batch_size": 64,             # Batch size to use for training and validation
   "in_window": 2114,            # Length of the input window
   "out_window": 1000,           # Length of the output window
   "max_jitter": 128,            # Maximum amount of jitter when generating training examples
   "reverse_complement": true,   # Whether to randomly RC half of the training examples
   "max_epochs": 50,             # The maximum number of epochs to train for
   "validation_iter": 100,       # The number of batches to train on before calculating validation set performance
   "lr": 0.001,                  # Learning rate of the AdamW optimizer
   "alpha": 10,                  # Weight of the count-loss in the total loss.
   "beta": 0.5,                  # Multiplier on the minimum read count in peaks to use when training a bias model

   "min_counts": 0,              # Ensure that each training example has at least this number of counts
   "max_counts": 99999999,       # Ensure that each training example has no more than this number of counts

   "name": "atac",                                     # Name to use for default filenames
   "model": "atac.torch",                              # Name of the full model if already trained
   "bias_model": "atac.bias.torch",                    # Name of the bias model if already trained
   "accessibility_model": "atac.accessibility.torch",  # Name of the accessibility model if already trained
    
   "verbose": true,    # Whether to print logs to the terminal
   "random_state": 0,  # Seed to use for the steps

   # Loci to train and validate on (can be a list)
   "loci": "../../../chromatin-atlas/ATAC/ENCSR637XSC/preprocessing/downloads/peaks.bed.gz",
   
   # Negatives to train on
   "negatives": "../../../chromatin-atlas/ATAC/ENCSR637XSC/negatives_data/negatives.bed",
   
   # FASTA file of the genome to train on
   "sequences": "../../../oak/common/hg38/hg38.fa",
   
   # A list of bigWig files to extract signal from -- each element in the list corresponds to one model output
   "signals": [
        "../../../chromatin-atlas/ATAC/ENCSR637XSC/preprocessing/bigWigs/ENCSR637XSC.bigWig"
   ],

   # Chromosomes to train on
   "training_chroms": ["chr2", "chr3", "chr4", "chr5", "chr6", "chr7", 
            "chr9", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", 
            "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"],
	    
   # Chromosomes to validate on
   "validation_chroms": ["chr8", "chr10"],
    
   # Parameters to use in the bias fit step 
   "bias_fit_parameters": {
        "n_filters": null,   # Number of filters in the convolutions
        "n_layers": 4,       # Number of dilated residual convolutions between the initial, and final, layers.
        "alpha": null,       # Weight of the count-loss in the total loss.
        "max_counts": null,  # Ensure that each training example has no more than this number of counts
	
	# Loci to train and validate on (can be a list)
        "loci": "../../../chromatin-atlas/ATAC/ENCSR637XSC/negatives_data/negatives.bed",
        
	"verbose": null,      # Whether to print logs to the terminal
        "random_state": null  # A random state to control parameter initialization and data generation
    },
    
    # Parameters to use in the fit step
    "fit_parameters": {
        "batch_size": null,   # Batch size to use for training and validation
        "sequences": null,    # FASTA file of the genome to train on
        "loci": null,         # Loci to train and validate on (can be a list)
        "signals": null,      # A list of bigWig files to extract signal from
        "verbose": null,      # Whether to print logs to the terminal
        "random_state": null  # A random state to control parameter initialization and data generation
    },
    
   # Parameters to pass into the predict step
	"predict_parameters": {
		"batch_size": 64,             # Batch size to use for training and validation
		"chroms": ["chr8", "chr10"],  # Chromosomes to make predictions for
		"profile_filename": null,     # The name of the file to store profile predictions
		"counts_filename": null,      # The name of the file to store count predictions
		"sequences": null,            # FASTA file of the genome to make predictions for
      
      # Loci to make predictions for (can be a list)
		"loci": "../../../chromatin-atlas/ATAC/ENCSR637XSC/preprocessing/downloads/peaks.bed.gz", 
      
		"signals": null,   # A list of bigWig files to extract signal from
		"controls": null,  # An optional list of bigWig files containing control signals 
		"verbose": null    # Whether to print out a log to the terminal during training
	},
   
   # Parameters to pass into the attribute step
	"attribute_parameters": {
		"batch_size": 1,      # Batch size to use for training and validation
		"output": "profile",  # Which head to calculate attributions for
		"chroms": ["chr8"],   # Chromosomes to make predictions for
      
      # Loci to make attributions for (can be a list)
		"loci": "../../../tfatlas/processed_data/ENCSR000BGQ/peaks.bed.gz",
      
		"ohe_filename": null,   # Filename to store one-hot encodings of the sequences
		"attr_filename": null,  # Filename to store DeepLIFT/SHAP values for the sequences
		"n_shuffles": 20,       # Number of GC-matched shuffles to perform
		"random_state": null,   # A seed to control the shuffles
		"verbose": null         # Whether to print out a progress bar during attribution

	},
   
   # Parameters to pass into the `modisco motifs` subcommand
	"modisco_motifs_parameters": {
		"n_seqlets": 100000,      # Maximum number of seqlets to use
		"output_filename": null,  # Filename to store the clustering results
		"verbose": null           # Whether to print out a progress bar during TF-MoDISco
	},
   
   # Parameters to pass into the `modisco report` subcommand
	"modisco_report_parameters": {
		"motifs": "motifs.meme",  # MEME file of motifs to compare the found motifs to
		"output_folder": null,    # Folder to store image outputs and report
	},
   
   # Parameters to pass into the marginalization step
	"marginalize_parameters": {
      # Loci to make predictions for (can be a list)
		"loci": "../../../chromatin-atlas/ATAC/ENCSR637XSC/negatives_data/negatives.bed",
		"n_loci": 100,                    # Number of background loci to use
		"shuffle": false,                 # Whether to shuffle the loci extracted before choosing `n_loci`
		"random_state": null,             # A seed to control the shuffling
		"output_folder": null,            # Folder to store the results including the summary
		"motifs": "ctcf.gata2.sp1.meme",  # MEME file of motifs to insert into the sequences
		"minimal": true,                  # Whether to produce a minimal report or report all columns
		"verbose": null                   # Whether to print out a log to the terminal as motifs are inserted
	} 
}
```
