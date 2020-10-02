# bpnet-lite
This repository hosts a minimal version of a Python API for BPNet.

### Installation

You can install bpnet-lite with `pip install bpnet-lite`.

### Usage

bpnet-lite has two main functions. The first is the `BPNet` function, which returns a TensorFlow / Keras model with the desired architectural parameters set. The second is the `io.extract_subset` function, which replaces the many command-line tools that needed to be used to preprocess the data. This function takes an entire FASTA file, entire bigWig files for signal, and a bedgraph file for coordinates (such as peaks) and extracts the subset for you.

See below an example of the current usage.

```Python
# Extract training data
training_chroms = ['chr{}'.format(i) for i in range(1, 23) if i not in (1, 8, 21, 22)]
(X_sequences, X_control_positives, X_control_negatives, y_output_positives, 
	y_output_negatives) = extract_subset("hg38.genome.fa", 
	"control_neg_strand.bw", "control_pos_strand.bw", "neg_strand.bw", 
	"pos_strand.bw", "peaks.bed", chroms=training_chroms)

X_control_profiles = numpy.array([X_control_negatives, X_control_positives]).transpose([1, 2, 0])
X_control_counts = numpy.log(X_control_profiles.sum(axis=1).sum(axis=1) + 1)

y_control_profiles = numpy.array([y_output_negatives, y_output_positives]).transpose([1, 2, 0])
y_control_counts = numpy.log(y_control_profiles.sum(axis=1) + 1)

X_train = {
	'sequence': X_sequences, 
	'control_logcount': X_control_counts, 
	'control_profile': X_control_profiles, 
}

y_train = {
	'task0_logcount': y_control_counts, 
	'task0_profile': y_control_profiles
}

# Extract validation data
(X_sequences, X_control_positives, X_control_negatives, y_output_positives, 
	y_output_negatives) = extract_subset("hg38.genome.fa", 
	"control_neg_strand.bw", "control_pos_strand.bw", "neg_strand.bw", 
	"pos_strand.bw", "peaks.bed", chroms=['chr22'])

X_control_profiles = numpy.array([X_control_negatives, X_control_positives]).transpose([1, 2, 0])
X_control_counts = numpy.log(X_control_profiles.sum(axis=1).sum(axis=1) + 1)

y_control_profiles = numpy.array([y_output_negatives, y_output_positives]).transpose([1, 2, 0])
y_control_counts = numpy.log(y_control_profiles.sum(axis=1) + 1)

X_valid = {
	'sequence': X_sequences, 
	'control_logcount': X_control_counts, 
	'control_profile': X_control_profiles, 
}

y_valid = {
	'task0_logcount': y_control_counts, 
	'task0_profile': y_control_profiles
}

model = BPNet()
model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))
```
