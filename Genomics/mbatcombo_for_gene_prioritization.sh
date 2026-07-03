#!/bin/bash

pheno=$1
input_file="${pheno}.tsv"

# Run mBAT analysis for specified chromosome
for i in {1..22}; do
    gcta-1.94.1 \
        --bfile ../1kgpanel/EUR/chr${i} \
        --maf 0.01 \
        --mBAT-gene-list glist_ensgid_hg38_v40.txt \
        --mBAT-combo ${input_file} \
        --mBAT-print-all-p \
        --out ${pheno}_${i} \
        --thread-num 60
done