# Example JSON Parameters

BPNet and ChromBPNet have many parameters for defining the architecture, learning process, and data. Rather than passing all of these parameters in through the command-line one should pass them in through a JSON. An additional benefit of this format is that, after running the command one has a complete log of the parameters used to generate the results.

Default parameters are specified in the command-line tools themselves. When a value is not provided in the parameter JSON the subcommand will fall back on those default values. Potentially, this can significantly reduce the size of the JSONs you need to specify.

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

### BPNet Interpret

```
{
   "batch_size": 64,             # Batch size to use for training and validation
   "in_window": 2114,            # Length of the input window
   "out_window": 1000,           # Length of the output window
   "verbose": true,              # Whether to print out a log to the terminal during training
   "chroms": ["chr8", "chr10"],  # Chromosomes whose peaks to make predictions for
   "model":"bpnet.64.8.torch",   # Model to use to make the predictions

   "sequences":"../../oak/common/hg38/hg38.fa",  # FASTA file of the genome to train on
   "loci":"../../tfatlas/processed_data/ENCSR000BGW/peaks.bed.gz",  # Loci to train and validate on (can be a list)

   "output": "profile",          # Which head to calculate attributions for
   "ohe_filename": "ohe.npz",    # Filename to store one-hot encodings of the sequences
   "attr_filename": "attr.npz",  # Filename to store DeepLIFT/SHAP values for the sequences
   "n_shuffles":20,              # Number of GC-matched shuffles to perform
   "random_state":0              # A seed to control the shuffles
}
```
