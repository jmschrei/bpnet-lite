# bpnet-lite

[![PyPI Downloads](https://static.pepy.tech/badge/bpnet-lite)](https://pepy.tech/projects/bpnet-lite)


> [!IMPORTANT]
> bpnet-lite is still under development. It is currently capable of loading models trained using the official Chrom/BPNet TensorFlow/Keras repositories into PyTorch and perfectly reproducing their outputs. It can also train BPNet models to parity with the official BPNet repository. However, it does not currently support training ChromBPNet models.

[docs/tutorials](https://bpnet-lite.readthedocs.io/en/latest/)

bpnet-lite is a lightweight version of BPNet [[paper](https://www.nature.com/articles/s41588-021-00782-6) | [code](https://github.com/kundajelab/basepairmodels)] and ChromBPNet [[preprint](https://www.biorxiv.org/content/10.1101/2024.12.25.630221v2) | [code](https://github.com/kundajelab/chrombpnet)], containing PyTorch reference implementations of both models. It has both a Python API and a set of command-line tools for training, using, and interpreting these models. This includes a complete pipeline that goes from preprocessing data, training and evaluating a model, calculating DeepLIFT/SHAP attributions using the model, running TF-MoDISco and seqlet calling/annotations on these attributions, and performing in silico marginalizations on a motif database.

#### Installation

`pip install bpnet-lite`

### Data Preprocessing

> [!NOTE]
> As of v0.9.0 you can now include BAM/SAM and .tsv/.tsv.gz files in the JSONs for the bpnet-lite command-line tool and the conversion to bigWigs will be automatically performed using bam2bw. Because bam2bw is fast (around ~500k records/second) it is not always necessary to separately preprocess your data anymore.

BPNet and ChromBPNet models are trained on read ends that have been mapped at basepair resolution (hence, the name) in peak regions and on GC-matched negatives. To facilitate getting your data into the format expected by these models, bpnet-lite has a built-in pipeline to preprocess your data once it has been aligned to your genome of interest (we recommend you use ChroMAP to do this step). First, bpnet-lite will run MACS3 on your input data (and optionally the controls) to call peaks and will then identify GC-matched negatives given those peaks and the provided genome. Then, it will convert SAM/BAM/tsv/bed/etc formatted files into the required bigWig files.  Note that if you can also provide any of these files (peak calls, peaks + negatives, signal and control tracks in bigWig format) instead. 

See the [MACS3](https://macs3-project.github.io/MACS/docs/callpeak.html) and [bam2bw](https://github.com/jmschrei/bam2bw) documenation for further details if you would like to manually do these steps.


## BPNet

![image](https://github.com/jmschrei/bpnet-lite/assets/3916816/5c6e6f73-aedd-4256-8776-5ef57a728d5e)

BPNet is a dense convolutional neural network that maps nucleotide sequences to experimental readouts, e.g. ChIP-seq, ChIP-nexus, and ChIP-exo. It is composed of a convolution layer with a large kernel width, a series of dilated residual layers that mix information across distances and channels, and another convolution layer with an even larger kernel width. BPNet models factorize the observed reads into the strength of the signal, by predicting the total log counts at each region, and the shape of the signal, by predicting a probability distribution over the nucleotides in the region.

A key advantage of BPNet models is that they are lightweight, and so can run forward/backward passes very quickly. This makes them much more suitable for downstream tasks, such as feature interpretation and design, than larger models that are significantly slower in these passes.

### BPNet Command Line Tools

bpnet-lite comes with a command-line tool, `bpnet`, that implements commands for training and using BPNet models. The documentation contains more details on each of the available commands but, likely, the most useful command is `pipeline`. This command handles preprocessing the data (see above), training and evaluating the model, calculating attributions on the provided sequences, calling/annotating seqlets, running TF-MoDISco, and performing in silico marginalizations. In essence, it is a one-stop shop for converting raw data into an understanding of the relevant cis-regulatory features governing the observed readouts.

A multi-step pipeline like this has many hyperparameters that can be customized at each step (e.g., number of filters in the model, number of seqlets to use for TF-MoDISco) and several pointers to input and output files. Rather than using a giant command-line call, bpnet-lite uses JSONs to manage this. An advantage of using JSONs is that they create a permanent record of the exact command that was run. The fastest way to produce this JSON is using the `pipeline-json` command, which takes in pointers to your data files and produces a valid JSON. These data files usually include a reference genome, some number of input (and optionally control) BAM/SAM/tsv/tsv.gz files (the `-i` and `-c` arguments can be repeated) a BED file of positive loci, and a MEME formatted motif database used for evaluation of the model.

For example:

```
bpnet pipeline-json -s hg38.fa -p peaks.bed.gz -i input1.bam -i input2.bam -c control1.bam -c control2.bam -n test -o pipeline.json -m JASPAR_2024.meme
```

If you are working with ATAC-seq data, which is unstranded and comes in the form of paired-end fragmnents, and would like to shift the reads to be +4/-4 (as they do in the ChromBPNet work) you can use the following:

```
bpnet pipeline-json -s hg38.fa -p peaks.bed.gz -i input1.bam -i input2.bam -n atac-test -o atac-pipeline.json -m JASPAR_2024.meme -ps 4 -ns -4 -u -f -pe
```

Note that any of these data pointers can point to remote files. This will stream the data through bam2bw and read the peak files remotely. Processing speed will then be dependant on the speed of your internet connection and whether the hosting site throttles your connection.

The JSON stores at `pipeline.json` or `atac-pipeline.json` can then be executed using the `pipeline` command. These commands are separated because, although the first command produces a valid JSON that the second command can immediately use (no need to copy/paste JSONs from this GitHub anymore!), one may wish to modify some of the many parameters in the JSON. These parameters include the number of filters and layers in the model, the training and validation chromosomes, and the even very technical ones like the number of shuffles to use when calculating attributions and the p-value threshold for calling seqlets. The defaults for most of these steps seem reasonable in practice but there is immense flexibility there, e.g., the ability to train the model using a reference genome and then make predictions or attributions on synthetic sequences or the reference genome from another species. In this manner, the JSON serves as documentation for the experiments that have been performed.

```
bpnet pipeline -p pipeline.json
```

When running the pipeline, a JSON is produced for each one of the steps (except for running TF-MoDISco and annotating the seqlets, which uses `ttl`). Each of these JSON can be run by themselves using the appropriate built-in command. Because some of the values in the JSONs for these steps are set programmatically when running the file pipeline, e.g., the filenames to read in and save to, being able to inspect every one of the JSONs can be handy for debugging.


### BPNet Python API

BPNet models trained using the pipeline described above are saved in the standard PyTorch format. This means that loading them in Python is quite simple. Because the pipeline above used the name "test", there will be a model named "test.torch" which is the best performing model on the validation data. We can load it into Python as follows:

```python
import torch

bpnet = torch.load("test.torch", weights_only=False)
```

This is now a PyTorch model that can be used exactly the same way as any other model. You can use it with tangermeme to efficiently make predictions:

```python
from tangermeme.predict import predict
from tangermeme.utils import random_one_hot

X = random_one_hot((10, 4, 2114), random_state=0)

y_logits, y_logcounts = predict(bpnet, X)
```

... or to calculate attributions.

```
from tangermeme.deep_lift_shap import deep_lift_shap

X_attr = deep_lift_shap(bpnet, X)
```

## ChromBPNet

![image](https://github.com/jmschrei/bpnet-lite/assets/3916816/e6f9bbdf-f107-4b3e-8b97-dc552af2239c)

> [!Warning]
> Several users have reported that the performance of ChromBPNet models trained using bpnet-lite underperforms those trained using the official ChromBPNet repo. Until we resolve these differences, we are removing support for *training* ChromBPNet models so as to not give a misleading impression of the performance these models can achieve. You can still load models trained using the official repository and use them as normal.

ChromBPNet extends the original basepair-resolution modeling framework of BPNet to DNase-seq and ATAC-seq experiments. In these experiments, a more involved framework is necessary because the cutting enzymes themselves have a soft sequence preference that can distort the observed readouts. This means that if you care about the true basepair resolution shape of the profiles, e.g. you want to look at footprinting, you need to adjust the positioning of the reads by removing this bias. This is done in ChromBPNet through the initial training of a smaller BPNet model on background regions where the sequence bias is still present (the "bias model"), followed by freezing it and training a second larger model (the "accessibility model"), to predict the residual between the observed readouts in peaks and the predictions from the bias model. Together, the bias and accessibility models form a "ChromBPNet" model. Usually, the bias model is discarded after training the accessibility model, whose readouts can be viewed as de-biased versions of the experiments.

Generally, one can perform the same analyses using ChromBPNet as one can using BPNet. This is because the accessibility model *is* a BPNet model and most analyses, such as feature attributions or design, mostly care about the de-noised predictions. However, an important note is that the full ChromBPNet model faithfully represents the experimental readout -- bias and all -- and so for more inspection tasks, e.g. variant effect prediction and interpretation, one should use only the accessibility model. Because the accessibiity model itself is conceptually, and also literally implemented as, a BPNet model, one can run the same procedure and use the BPNet command-line tool using it.


#### ChromBPNet Python API

Depending on the format of your ChromBPNet models, there are several ways that one could load them into Python for subsequent analyses. See the tutorial on loading models for more detail. Here is an example of loading a ChromBPNet model directly from a h5 downloaded from the ENCODE Portal.

```python
import tarfile

from io import BytesIO
from bpnetlite.chrombpnet import ChromBPNet

with tarfile.open("ENCFF142IOR.tar.gz", "r:gz") as tar:
    bias_tar = tar.extractfile("./fold_0/model.bias_scaled.fold_0.ENCSR637XSC.h5").read()
    accessibility_tar = tar.extractfile("./fold_0/model.chrombpnet_nobias.fold_0.ENCSR637XSC.h5").read()

chrombpnet = ChromBPNet.from_chrombpnet(
    BytesIO(bias_tar),
    BytesIO(accessibility_tar)
)

chrombpnet
```

`chrombpnet` is now a PyTorch model that can be used the same as any other model and produces identical predictions to the original TensorFlow model. For example, here is an example of using this model to design an accessible site:

```python
import torch
from bpnetlite.bpnet import CountWrapper
from tangermeme.design import greedy_substitution
from tangermeme.utils import random_one_hot

X = random_one_hot((1, 4, 2114), random_state=0)
y_bar = torch.tensor([[10.0]])

X_bar = greedy_substitution(CountWrapper(chrombpnet), X, y_bar, max_iter=5)
```

