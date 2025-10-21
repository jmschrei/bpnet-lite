# bpnet-lite

[![PyPI Downloads](https://static.pepy.tech/badge/bpnet-lite)](https://pepy.tech/projects/bpnet-lite)

> [!IMPORTANT]
> bpnet-lite is not meant to replace the full service implementations of BPNet or ChromBPNet and is still under development. Please see the official repositories for those projects for complete TensorFlow/Keras implementations of those models along with tutorials on how to use them effectively. Although bpnet-lite is capable of loading models trained using these TensorFlow/Keras repositories into PyTorch and perfectly reproducing their outputs, and can train BPNet models to similar performance as the official BPNet repository, its fitting procedure for ChromBPNet models does not yet match and can sometimes significantly underperform those trained using the official ChromBPNet repository.

bpnet-lite is a lightweight version of BPNet [[paper](https://www.nature.com/articles/s41588-021-00782-6) | [code](https://github.com/kundajelab/basepairmodels)] and ChromBPNet [[preprint](https://www.biorxiv.org/content/10.1101/2024.12.25.630221v2) | [code](https://github.com/kundajelab/chrombpnet)], containing PyTorch reference implementations of both models. Additionally, it contains efficient data loaders and common operations one would do with these trained models including calculating attributions, running TF-MoDISco, and performing marginalization experiments. These operations are wrapped in command-line tools for ease-of-use and organized in a pipeline command representing the standard workflow. This package is primarily meant to be used for prototyping new ideas that involve modifying the code and for loading models trained using the official repositories into PyTorch. 

#### Installation

`pip install bpnet-lite`

#### Data Preprocessing

> [!NOTE]
> As of v0.9.0 you can now include BAM/SAM and .tsv/.tsv.gz files in the JSONs for the bpnet-lite command-line tool and the conversion to bigWigs will be automatically performed using bam2bw. Because bam2bw is fast (around ~500k records/second) it is not always necessary to separately preprocess your data anymore.

BPNet and ChromBPNet models are both trained on read ends that have been mapped at basepair resolution (hence, the name). Accordingly, the data used for training is made up of integers with one count per read in the file (or two counts per fragment). Once you have used your favorite tool to align your FASTQ of reads to your genome of interest (we recommend ChroMAP), you can either use [bam2bw](https://github.com/jmschrei/bam2bw) to convert your BAM/SAM or fragment tsv/tsv.gz files to bigWig files, or put these raw data files in the JSON and have bpnet-lite automatically do the conversion for you.

If you are using stranded data, e.g., ChIP-seq:

```
bam2bw <bam1>.bam <bam2>.bam ...  -s <genome>.chrom.sizes/<genome>.fa -n <name> -v 
```

This command will create two bigWig files, one for the + strand and one for the - strand, using the name provided as the suffix.

If you are using unstranded data:

```
bam2bw <bam1>.bam <bam2>.bam ...  -s <genome>.chrom.sizes/<genome>.fa -n <name> -v -u
```

If you have a file of fragments, usually formatted as a .tsv or .tsv.gz and coming from ATAC-seq or scATAC-seq data, you can use the `-f` flag to map both the start and end (end-1, specifically) instead of just the 5' end. You will probably also want the `-u` flag because the underlying data is unstranded.

```
bam2bw <frag1>.tsv.gz <frag2>.tsv.gz  ...  -s <genome>.chrom.sizes/<genome>.fa -n <name> -v -u -f
```

These tools require positive loci (usually peaks for the respective activity or elements like promoters) and negative loci (usually GC-matched background sequences) for training. One or more BED files of positive loci are required from the user, potentially acquired by applying a tool like MACS2 to your .BAM files. The negative loci can be calculated using a command-line tool in this package, described later, or by specifying in the JSON that `find_negatives: true`. 

## BPNet

![image](https://github.com/jmschrei/bpnet-lite/assets/3916816/5c6e6f73-aedd-4256-8776-5ef57a728d5e)

BPNet is a convolutional neural network that maps nucleotide sequences to experimental readouts, e.g. ChIP-seq, ChIP-nexus, and ChIP-exo. It is composed of one big convolution layer, a series of dilated residual layers that mix information across distances, and another big convolution layer. Importantly, BPNet makes predictions for the total (log) read count in the region and also for the basepair resolution profiles, with these profiles being a probability vector over each position. 

Although these models achieve high predictive accuracy, their main purpose is to estimate the influence of non-coding variants and to extract principles of the cis-regulatory code underlying the readouts being modeled. Specifically, when paired with a feature attribution algorithm like DeepLIFT/SHAP or in silico saturation mutagenesis, these models can assign to each nucleotide an importance in the model's predictions. These attributions can shed insight into how individual loci work, and when considered genome-wide, algorithms like TF-MoDISco can identify the repeated high-attribution patterns. 

### BPNet Command Line Tools

bpnet-lite comes with a command-line tool, `bpnet`, that supports the steps necessary for training and using BPNet models. The fastest way to go from your raw data to results is to use the `bpnet pipeline-json` command followed by the `bpnet pipeline` command. 

```
bpnet pipeline-json -s hg38.fa -p peaks.bed.gz -i input1.bam -i input2.bam -c control1.bam -c control2.bam -n test -o pipeline.json -m JASPAR_2024.meme
bpnet pipeline -p pipeline.json
```

The `bpnet-json` cmmand takes in pointers to your data files and produces a properly formatted `pipeline.json` file. These data files usually include a reference genome, some number of input (and optionally control) BAM/SAM/tsv/tsv.gz files (the `-i` and `-c` arguments can be repeated) a BED file of positive loci, and a MEME formatted motif database used for evaluation of the model.

The `bpnet pipeline` command takes in the JSON and (0) optionally preprocesses your BAM/SAM/tsv/tsv.gz files and identifies GC-matched negatives (you can provide your own bigWigs and/or negatives and skip the respective portions of this), (1) trains a BPNet model, (2) makes predictions on the provided loci, (3) calculates DeepLIFT/SHAP attributions on the provided loci, (4) calls seqlets and annotates them using `ttl`, (5) runs TF-MoDISco and generates a report, and (6) runs in silico marginalizations using the provided motif database.

These commands are separated because, although the first command produces a valid JSON that the second command can immediately use (no need to copy/paste JSONs from this GitHub anymore!), one may wish to modify some of the many parameters in the JSON. These parameters include the number of filters and layers in the model, the training and validation chromosomes, and the even very technical ones like the number of shuffles to use when calculating attributions and the p-value threshold for calling seqlets. The defaults for most of these steps seem reasonable in practice but there is immense flexibility there, e.g., the ability to train the model using a reference genome and then make predictions or attributions on synthetic sequences or the reference genome from another species. In this manner, the JSON serves as documentation for the experiments that have been performed.

When running the pipeline, a JSON is produced for each one of the steps (except for running TF-MoDISco and annotating the seqlets, which uses `ttl`). Each of these JSON can be run by themselves using the appropriate built-in command. Because some of the values in the JSONs for these steps are set programmatically when running the file pipeline, e.g., the filenames to read in and save to, being able to inspect every one of the JSONs can be handy for debugging.

```
bpnet fit -p bpnet_fit_example.json
bpnet predict -p bpnet_predict_example.json
bpnet attribute -p bpnet_attribute_example.json
bpnet seqlets -p bpnet_seqlet_example.json
bpnet marginalize -p bpnet_marginalize_example.json
```

For a complete description of each of the JSONs and the command-line tools, see the `example_jsons` folder.

## ChromBPNet

![image](https://github.com/jmschrei/bpnet-lite/assets/3916816/e6f9bbdf-f107-4b3e-8b97-dc552af2239c)

> [!Warning]
> Several users have reported that the performance of ChromBPNet models trained using bpnet-lite significantly underperforms those trained using the official ChromBPNet repo. We are currently looking into this. Until we resolve the differences, please consider using the official repository for training your ChromBPNet models and then bpnet-lite for loading them into PyTorch.

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

> [!Warning]
> This is no longer accurate as of v0.9.2 with the switch to the PeakNegativeSampler. I will update soon.

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

X_valid, y_valid, X_ctl_valid = extract_loci(peaks, seqs, signals, controls, chroms=valid_chroms, max_jitter=0)
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
