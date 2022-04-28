# bpnet-lite

bpnet-lite is a lightweight version of BPNet that contains a reference implementation in PyTorch, efficient data loaders, and a command-line tool that accepts a JSON as input. This implementation is meant to be used for quickly exploring data sets using BPNet and as a springboard for prototyping new ideas that involve modifying the code. bpnet-lite does not include all of the features that have been developed for BPNet: see https://github.com/kundajelab/basepairmodels for that.

### Installation

You can install bpnet-lite with `pip install bpnet-lite`.

### Usage

There are two main ways to use bpnet-lite. The first is through the Python API, where you interact with a BPNet object and the associated data loaders. Here is a complete example Python script that could be used: 

```python
import torch

from bpnetlite.io import extract_peaks
from bpnetlite.io import PeakGenerator
from bpnetlite import BPNet

peaks = 'test/CTCF.peaks.bed'
seqs = '../../oak/common/hg38/hg38.fa'
signals = ['test/CTCF.plus.bw', 'test/CTCF.minus.bw']
controls = ['test/CTCF.plus.ctl.bw', 'test/CTCF.minus.ctl.bw']

training_chroms = ['chr{}'.format(i) for i in range(1, 17)]
valid_chroms = ['chr{}'.format(i) for i in range(18, 23)]

training_data = PeakGenerator(peaks, seqs, signals, controls, 
	chroms=training_chroms)

X_valid, y_valid, X_ctl_valid = extract_peaks(peaks, seqs, signals, controls, 
	chroms=valid_chroms, max_jitter=0)

model = BPNet(n_outputs=2, n_control_tracks=2, trimming=(2114 - 1000) // 2).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.fit_generator(training_data, optimizer, X_valid=X_valid, 
	X_ctl_valid=X_ctl_valid, y_valid=y_valid)
```

The second is through the command-line API. The command is `BPNet` and it currently accepts all of its arguments through a JSON. See `example.json` in the repository for an example JSON with all the parameters that can be set. Once the JSON is set, you can run the command as `bpnet -p example.json`.
