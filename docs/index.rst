.. bpnet-lite documentation master file, created by
   sphinx-quickstart on Tue Feb 20 13:46:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


bpnet-lite
==========

bpnet-lite is a re-implementation of the BPNet and ChromBPNet models developed by the Kundaje Lab in PyTorch, along with useful functionality for using these models in practice. It has been developed with a focus on making the usage of these models as simple as possible and minimize the barrier to entry. Accordingly, care has been taken to support loading models trained using the official repositories, it is integrated with other packages in this ecosystem (bam2bw, TF-MoDISco, and tangermeme), and has command-line tools supporting the training and downstream usage of these models. The training of BPNet models is supported, both in the Python and command-line interface, but the training of ChromBPNet models should be done using the official TensorFlow repository to ensure consistent performance.

In this documentation, you will find a narrative introduction to the models supported here, practical guides for "how-to" use this code to apply Chrom/BPNet models in your setting, and tutorials on how to use the command-line API to go completely end-to-end from raw data to analysis results. Additionally, we have included a notebook confirming that the predictions made when loading the models into PyTorch exactly match those made from the official repositories.

Likely, the fastest way to get started with BPNet models is to use the command-line tool with the `pipeline` argument. This handles the preprocessing of your data, training and evaluation of a BPNet model, calculation of attributions using DeepLIFT/SHAP, seqlet calling+annotation, running of TF-MoDISCo, and running of in silico marginalizations. Ultimately, you end up with an evaluated model and multiple results hinting at the types of cis-regulatory features it has learned.

Installation is as simple as `pip install bpnet-lite`.

.. toctree::
	:maxdepth: 1
	:hidden:
	:caption: bpnet-lite
	
	self
	whats_new.rst


.. toctree::
	:maxdepth: 1
	:hidden:
	:caption: Introductions
	
	tutorials/Intro_What_Is_BPNet.ipynb
	tutorials/Intro_What_Is_ChromBPNet.ipynb
	tutorials/Intro_What_Is_ProCapNet.ipynb
	tutorials/Confirming_Chrom+BPnet_Predictions_Identical.ipynb


.. toctree::
	:maxdepth: 1
	:hidden:
	:caption: How-Tos

	tutorials/How-To-Save_Load_Convert.ipynb
	tutorials/How-To-Train-BPNet-Python.ipynb
	tutorials/How-To-Use-Pipeline.ipynb
	tutorials/How-To-Evaluate-a-Model.ipynb
	tutorials/How-To-Design-Simple.ipynb


.. toctree::
	:maxdepth: 1
	:hidden:
	:caption: API
	
	api/attribute.rst
	api/bpnet.rst
	api/io.rst
	api/losses.rst
	api/performance.rst
