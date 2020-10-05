from setuptools import setup

setup(
    name='bpnet-lite',
    version='0.0.2',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['bpnetlite'],
    url='https://github.com/jmschrei/bpnet-lite',
    license='LICENSE.txt',
    description='bpnet-lite is a minimal implementation of BPNet, a neural network aimed at interpreting regulatory activity of the genome.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "tensorflow >= 2.0.0",
        "tensorflow_probability",
        "tqdm >= 4.24.0"
    ],
)
