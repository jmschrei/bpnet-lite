# bpnet-lite

![bpnet-schematic](https://user-images.githubusercontent.com/3916816/215882453-873d2835-c639-47d5-a84b-b57a7922fce0.png)

bpnet-lite is a lightweight version of [BPNet](https://www.nature.com/articles/s41588-021-00782-6) that contains a reference implementation in PyTorch, efficient data loaders, and a command-line tool for training, making predictions for, and calculating attributions from, a model. This implementation is meant to be used for quickly exploring data sets using BPNet and as a springboard for prototyping new ideas that involve modifying the code. Important: bpnet-lite does not include all of the features that have been developed for BPNet: see https://github.com/kundajelab/basepairmodels for that.

### Installation

You can install bpnet-lite with `pip install bpnet-lite`.

### BPNet Command Line Tools

bpnet-lite comes with a command-line tool, `bpnet`, that supports the steps necessary for training and using BPNet models. Specifically, it can be used to calculate GC-matched negatives from a peak file, train a BPNet model with flexibility as to the inputs, use a trained model to make predictions, use a trained model to calculate DeepLIFT/DeepSHAP attribution scores, and to perform marginalization experiments. Each comment, except for calculating GC-matched negatives, requires a JSON that contains the parameters, with examples of each in the `example_jsons` folder. 

#### Calculating GC-matched negatives
`bpnet negatives -i <peaks>.bed -f <fasta>.fa -b <bigwig>.bw -o matched_loci.bed -l 0.02 -w 2114 -v`

This command takes in a bed file of loci, some information about the genome being considered, the GC bin size, and the width of the window to calculate GC content for (usually the same size as the model input window), and returns a set of loci that are GC-matched and not within the provided bed file coordinates. If a FASTA file is provided, the first step is to calculate a rolling average of the GC content using the provided window size, and store that average at the provided bigwig path. If the GC content has already been calculated and stored as a bigwig, a FASTA file does not need to be provided and the bigwig will be used directly.

#### Training a BPNet model 
`bpnet train -p example_train_bpnet.json`

This command takes in a [JSON](https://github.com/jmschrei/bpnet-lite/blob/master/example_jsons/example_train_bpnet.json) that has model architecture, training hyperparameters, and locations of the data, and outputs two models: one that is the best model found during training according to the validation set loss ({name}.torch), and one that is the final state of the model after all training, regardless of if it's the best model ({name}.final.torch). 

#### Making predictions with a BPNet model

`bpnet predict -p predict_example.json`

This command takes in a [JSON](https://github.com/jmschrei/bpnet-lite/blob/master/example_jsons/predict_example.json) that specifies the loci and locations of the data and saves a numpy array for the profile head and a second numpy array for the count head.

#### Calculating attributions with a BPNet model

`bpnet interpret -p interpret_json` 

Similarly to the predict command, this command takes in a [JSON](https://github.com/jmschrei/bpnet-lite/blob/master/example_jsons/interpret_example.json) that species the loci and the locations of the data and saves a numpy array for the one-hot encoded sequence and a second numpy array for the SHAP scores. Importantly, these SHAP scores will include the hypothetical contributions, i.e., the contributions for all possible characters at each position, not just the actual character. 


### Python API

The Python API is made up of a small number of components that can be customized or re-organized. Start off by specifying where your data will be.

```python
import torch

from bpnetlite.io import extract_loci
from bpnetlite.io import PeakGenerator
from bpnetlite import BPNet

peaks = 'test/CTCF.peaks.bed'
seqs = '../../oak/common/hg38/hg38.fa'
signals = ['test/CTCF.plus.bw', 'test/CTCF.minus.bw']
controls = ['test/CTCF.plus.ctl.bw', 'test/CTCF.minus.ctl.bw']

training_chroms = ['chr{}'.format(i) for i in range(1, 17)]
valid_chroms = ['chr{}'.format(i) for i in range(18, 23)]
```

Next, you should load up your peaks by passing in the peak loci, sequences, signals, and control tracks. If `controls=None` then no controls are passed into the model. Likewise, the length of `signals` should be equal to the number of outputs in your model and can be just one if dealing with unstranded data.

```python
training_data = PeakGenerator(peaks, seqs, signals, controls, chroms=training_chroms)
```

The `PeakGenerator` function is a wrapper around several functions which extract data, pass it into a data generator, and pass that into a PyTorch data loader object. The end result is an object that can be directly iterated over while training a model. However, this great for a validation set because we want that to be fixed and the `PeakGenerator` object has options to jitter the data, randomly reverse complement it, and randomly sample from the peaks. Instead, we want to just use the `extract_peaks` function to extract the raw data at these locations. Note that, if there are no controls, you should exclude `X_ctl_valid`. 

```python
X_valid, y_valid, X_ctl_valid = extract_peaks(peaks, seqs, signals, controls, chroms=valid_chroms, max_jitter=0)
```

Now, we can define the model. If you want to change the architecture, check out the documentation.

```python
model = BPNet(n_outputs=2, n_control_tracks=2, trimming=(2114 - 1000) // 2).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

And, finally, we can call the `fit_generator` method to train the model. This function is largely just a training loop that trains the profile head using the multinomial log-likelihood loss and the count head using the mean-squared error loss, but a benefit of this built-in method is that it outputs a tsv of the training statistics that you can redirect to a log file.

```python
model.fit_generator(training_data, optimizer, X_valid=X_valid, 
	X_ctl_valid=X_ctl_valid, y_valid=y_valid)
```

Because `model` is a PyTorch object, it can be trained using a custom training loop in the same way any base PyTorch model can be trained if you'd prefer to do that. Likewise, if you'd prefer to use a custom data generator you can write your own and pass that into the `fit_generator` function. 
