# bpnet-lite

[![PyPI Downloads](https://static.pepy.tech/badge/bpnet-lite)](https://pepy.tech/projects/bpnet-lite)

> **Note**
> IMPORTANT: bpnet-lite is not meant to replace the full service implementations of BPNet or ChromBPNet. Please see the official repositories for those projects for TensorFlow/Keras implementations of those models along with complete tutorials on how to use them effectively.

bpnet-lite is a lightweight version of BPNet [[paper](https://www.nature.com/articles/s41588-021-00782-6) | [code](https://github.com/kundajelab/basepairmodels)] and ChromBPNet [paper | [code](https://github.com/kundajelab/chrombpnet)], containing PyTorch reference implementations of both models. Additionally, it contains efficient data loaders and common operations one would do with these trained models including calculating attributions, running TF-MoDISco, and performing marginalization experiments. These operations are wrapped in command-line tools for ease-of-use and organized in a pipeline command representing the standard workflow. This package is meant to be used for quickly exploring data sets using BPNet or ChromBPNet and as a springboard for prototyping new ideas that involve modifying the code. 

### Installation

You can install bpnet-lite with `pip install bpnet-lite`.

### Data Processing

BPNet and ChromBPNet models are both trained on read ends that have been mapped at basepair resolution (hence, the name). Accordingly, the data used for training is made up of integer counts with one count per read in the file. Once you have used your favorite tool to align your FASTQ of reads to your genome of interest (we recommend ChroMAP), you should then use the following command line arguments to get bigwigs of integer counts.

If you are using stranded data, e.g., ChIP-seq:

```
bedtools genomecov -5 -bg -strand + -g <your genome>.chrom.sizes -ibam <your bam>.bam | sort -k1,1 -k2,2n > pos_strand.bedGraph
./bedGraphToBigWig pos_strand.bedGraph <your genome>.chrom.sizes pos_strand.bw

bedtools genomecov -5 -bg -strand - -g <your genome>.chrom.sizes -ibam <your bam>.bam | sort -k1,1 -k2,2n > neg_strand.bedGraph
./bedGraphToBigWig neg_strand.bedGraph <your genome>.chrom.sizes neg_strand.bw
```

If you are using unstranded data, e.g., DNase- or ATAC-seq.

```
bedtools genomecov -5 -bg -g <your genome>.chrom.sizes -ibam <your bam>.bam | sort -k1,1 -k2,2n > data.bedGraph
./bedGraphToBigWig data.bedGraph <your genome>.chrom.sizes data.bw
```

## BPNet

![image](https://github.com/jmschrei/bpnet-lite/assets/3916816/5c6e6f73-aedd-4256-8776-5ef57a728d5e)

BPNet is a convolutional neural network that has been used to map nucleotide sequences to experimental readouts, e.g. ChIP-seq, ChIP-nexus, and ChIP-exo, and identify the driving motifs underlying these assays. Although these models achieve high predictive accuracy, their main purpose is to be interpreted using feature attribution methods to inspect the cis-regulatory code underlying the readouts being modeled. Specifically, when paired with a method like DeepLIFT/SHAP, they can be used to explain the driving motifs and syntax of those motifs underlying each signal peak in a readout. When looking across all peaks these attributions can be clustered using an algorithm like TF-MoDISco to identify repeated patterns. Finally, one can construct a variety of perturbations to reference sequence to identify variant effect or marginalize out background. 

### BPNet Command Line Tools

bpnet-lite comes with a command-line tool, `bpnet`, that supports the steps necessary for training and using BPNet models. Except for extracting GC-matched negatives, each command requires a JSON that contains the parameters, with examples of each in the `example_jsons` folder. See the README in that folder for exact parameters for each JSON.

```
bpnet negatives -i <peaks>.bed -f <fasta>.fa -b <bigwig>.bw -o matched_loci.bed -l 0.02 -w 2114 -v
bpnet fit -p bpnet_fit_example.json
bpnet predict -p bpnet_predict_example.json
bpnet attribute -p bpnet_attribute_example.json
bpnet marginalize -p bpnet_marginalize_example.json
```

Alternatively, one can run the entire pipeline of commands specified above (except for the calculation of negatives) in addition to also running TF-MoDISco and generating a report on the found motifs.

```
bpnet pipeline -p bpnet_pipeline_example.json
```

## ChromBPNet

![image](https://github.com/jmschrei/bpnet-lite/assets/3916816/e6f9bbdf-f107-4b3e-8b97-dc552af2239c)

ChromBPNet extends the original modeling framework to DNase-seq and ATAC-seq experiments. A separate framework is necessary because the cutting enzymes used in these experiments, particularly the hyperactive Tn5 enzyme used in ATAC-seq experiments, have soft sequences preferences that can distort the observed readouts. Hence, it becomes necessary to train a small BPNet model to explicitly capture this soft sequence (the "bias model") bias before subsequently training a second BPNet model jointly with the frozen bias model to capture the true drivers of accessibility (the "accessibiity model"). Together, these models and the manner in which their predictions are combined are referred to as ChromBPNet. 

Generally, one can perform the same analyses using ChromBPNet as one can using BPNet. However, an important note is that the full ChromBPNet model faithfully represents the experimental readout -- bias and all -- and so for more inspection tasks, e.g. variant effect prediction and interpretation, one should use only the accessibility model. Because the accessibiity model itself is conceptually, and also literally implemented as, a BPNet model, one can run the same procedure and use the BPNet command-line tool using it.

###

bpnet-lite comes with a second command-line tool, `chrombpnet`, that supports the steps necessary for training and using ChromBPNet models. These commands are used exactly the same way as the `bpnet` command-line tool with only minor changes to the parameters in the JSON. Note that the `predict`, `attribute` and `marginalize` commands will internally run their `bpnet` counterparts, but are still provided for convenience.

```
chrombpnet fit -p chrombpnet_fit_example.json
chrombpnet predict -p chrombpnet_predict_example.json
chrombpnet attribute -p chrombpnet_attribute_example.json
chrombpnet marginalize -p chrombpnet_marginalize_example.json
```

Similarly to `bpnet`, one can run the entire pipeline of commands specified above in addition to also running TF-MoDISco and generating a report on the found motifs. Unlike `bpnet`, this command will run each of those steps for (1) the full ChromBPNet model, (2) the accessibility model alone, and (3) the bias model. 

```
chrombpnet pipeline -p chrombpnet_pipeline_example.json
```

## Python API

If you'd rather train and use BPNet/ChromBPNet models programmatically, you can use the Python API. The command-line tool is made up of wrappers around these methods and functions, so please take a look if you'd like additional documentation on how to get started.

The first step is loading data. Much like with the command-line tool, if you're using the built-in data loader then you need to specify where the FASTA containing sequences, a BED file containing loci and bigwig files to train on are. The signals need to be provided in a list and the index of each bigwig in the list will correspond to a model output. Optionally, you can also provide control bigwigs. See the BPNet paper for how these control bigwigs get used during training. 

```python
import torch

from tangermeme.io import extract_loci
from bpnetlite.io import PeakGenerator
from bpnetlite import BPNet

peaks = 'test/CTCF.peaks.bed' # A set of loci to train on.
seqs = '../../oak/common/hg38/hg38.fa' # A set of sequences to train on
signals = ['test/CTCF.plus.bw', 'test/CTCF.minus.bw'] # A set of bigwigs
controls = ['test/CTCF.plus.ctl.bw', 'test/CTCF.minus.ctl.bw'] # A set of bigwigs
```

After specifying filepaths for each of these, you can create the data generator. If you have a set of chromosomes you'd like to use for training, you can pass those in as well. They must match exactly with the names of chromsomes given in the BED file. 

```python
training_chroms = ['chr{}'.format(i) for i in range(1, 17)]

training_data = PeakGenerator(peaks, seqs, signals, controls, chroms=training_chroms)
```

The `PeakGenerator` function is a wrapper around several functions that extract data, pass them into a generator that applies shifts and shuffling, and pass that generator into a PyTorch data loader object for use during training. The end result is an object that can be directly iterated over while training a bpnet-lite model. 

Although wrapping all that functionality is good for the training set, the validation set should remain constant during training. Hence, one should only use the `extract_loci` function that is the first step when handling the training data.

```python
valid_chroms = ['chr{}'.format(i) for i in range(18, 23)]

X_valid, y_valid, X_ctl_valid = extract_peaks(peaks, seqs, signals, controls, chroms=valid_chroms, max_jitter=0)
```
Note that this function can be used without control tracks and, in that case, will only return two arguments. Further, it can used with only a FASTA and will only return one argument in that case: the extracted sequences. 

Now, we can define the model. If you want to change the architecture, check out the documentation.

```python
model = BPNet(n_outputs=2, n_control_tracks=2, trimming=(2114 - 1000) // 2).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

And, finally, we can call the `fit_generator` method to train the model. This function is largely just a training loop that trains the profile head using the multinomial log-likelihood loss and the count head using the mean-squared error loss, but a benefit of this built-in method is that it outputs a tsv of the training statistics that you can redirect to a log file.

```python
model.fit(training_data, optimizer, X_valid=X_valid, 
	X_ctl_valid=X_ctl_valid, y_valid=y_valid)
```

Because `model` is a PyTorch object, it can be trained using a custom training loop in the same way any base PyTorch model can be trained if you'd prefer to do that. Likewise, if you'd prefer to use a custom data generator you can write your own and pass that into the `fit` function. 
