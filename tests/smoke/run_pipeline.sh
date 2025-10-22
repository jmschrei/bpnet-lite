#!/bin/bash

set -euxo pipefail

if [ ! -f chr21_22.fa ]; then
    wget https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/chr21.fa.gz
    wget https://hgdownload.cse.ucsc.edu/goldenpath/hg38/chromosomes/chr22.fa.gz
    cat chr21.fa.gz chr22.fa.gz > chr21_22.fa.gz
    gzip -dc chr21_22.fa.gz > chr21_22.fa
fi

bpnet pipeline -p pipeline.json
