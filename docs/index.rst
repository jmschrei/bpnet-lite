.. bpnet-lite documentation master file, created by
   sphinx-quickstart on Tue Feb 20 13:46:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


bpnet-lite
==========

bpnet-lite is a re-implementation of BPNet and ChromBPNet in PyTorch along with useful functions for using these models in practice. It has been developed to make training and using these models as simple as possible, with the overall goal being to make the sorts of ML-based analyses common in research labs as simple to use in practice as 

In this documentation you will find an introduction to the BPNet and ChromBPNet models, tutorials on how to use this code to perform your own analyses in Python, and vignettes for how you can use Chrom/BPNet models in practice to explore genomic phenomena.

Installation is as simple as `pip install bpnet-lite`.


.. toctree::
	:maxdepth: 1
	:hidden:
	:caption: Getting Started
	
	self
	tutorials/Intro_What_Is_BPNet.ipynb
	tutorials/Intro_What_Is_ChromBPNet.ipynb
	tutorials/Intro_What_Is_ProCapNet.ipynb
	tutorials/Confirming_Chrom+BPnet_Predictions_Identical.ipynb

	whats_new.rst

.. toctree::
	:maxdepth: 1
	:hidden:
	:caption: How-Tos

	tutorials/How-To-Save_Load_Convert.ipynb
	tutorials/How-To-Train-BPNet-Python.ipynb
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
