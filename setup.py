from setuptools import setup

setup(
    name='bpnet-lite',
    version='0.1.1',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['bpnetlite'],
    scripts=['bpnet'],
    url='https://github.com/jmschrei/bpnet-lite',
    license='LICENSE.txt',
    description='bpnet-lite is a minimal implementation of BPNet, a neural network aimed at interpreting regulatory activity of the genome.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "pandas >= 1.3.3",
        "pyBigWig >= 0.3.17",
        "torch >= 1.9.0",
    ],
)
