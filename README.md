# bpnet-lite

bpnet-lite is a lightweight version of BPNet that contains a reference implementation in PyTorch, efficient data loaders, and a command-line tool that accepts a JSON as input. This implementation is meant to be used for quickly exploring data sets using BPNet and as a springboard for prototyping new ideas that involve modifying the code. bpnet-lite does not include all of the features that have been developed for BPNet: see https://github.com/kundajelab/basepairmodels for that.

### Installation

You can install bpnet-lite with `pip install bpnet-lite`.

### Usage

There are two main ways to use bpnet-lite. The first is through the Python API, where you interact with a BPNet object and the associated data loaders. Here is a complete example Python script that could be used: 

```python
import numpy
import torch

from bpnetlite import BPNet
from bpnetlite.io import DataGenerator
from bpnetlite.io import extract_peaks

n_filters = 64
n_layers = 8

batch_size = 64

in_window = 2114
out_window = 1000
trimming = (2114 - 1000) // 2
max_jitter = 128

###

training_chroms = ['chr1', 'chr2', 'chr3', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9',
	'chr10', 'chr12', 'chr13', 'chr14', 'chr16', 'chr18', 'chr19', 'chr20', 
	'chr22']

valid_chroms = ['chr4', 'chr15', 'chr21']

###

sequence_path = ... 
peak_path = ...
plus_bw_path = ...
minus_bw_path = ...
plus_ctl_bw_path = ...
minus_ctl_bw_path = ...

train_sequences, train_signals, train_controls = extract_peaks(sequence_path, 
	plus_bw_path, minus_bw_path, plus_ctl_bw_path, minus_ctl_bw_path, 
	peak_path, training_chroms, verbose=True)

valid_sequences, valid_signals, valid_controls = extract_peaks(sequence_path, 
	plus_bw_path, minus_bw_path, plus_ctl_bw_path, minus_ctl_bw_path, 
	peak_path, test_chroms, max_jitter=0, verbose=True)

###

training_peaks = DataGenerator(
	sequences=train_sequences,
	signals=train_signals,
	controls=train_controls,
	in_window=in_window,
	out_window=out_window,
	random_state=0)

training_data = torch.utils.data.DataLoader(training_peaks, 
	pin_memory=True, 
	batch_size=batch_size)

model = BPNet(n_filters=n_filters, n_layers=n_layers, trimming=trimming).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001563)

model.fit_generator(training_data, optimizer, X_valid=valid_sequences, 
	X_ctl_valid=valid_controls, y_valid=valid_signals, max_epochs=250, 
	validation_iter=100, batch_size=batch_size)
```

The second is through the command-line API. The command is `BPNet` and it currently accepts all of its arguments through a JSON. See `example.json` in the repository for an example JSON with all the parameters that can be set. Once the JSON is set, you can run the command as `bpnet -p example.json`.
