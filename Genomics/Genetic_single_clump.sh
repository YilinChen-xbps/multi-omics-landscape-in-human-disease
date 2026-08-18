#!/bin/bash
pheno=$1
input_file="${pheno}.tsv"

# Clumping Analysis for Single-variant association results ===========================================================
clump_r2="0.1"
clump_kb="1000"
clump_p1="5e-8"

# Loop through chromosomes 1 to 22 for clumping
for chr in {1..22}; do
plink2 \
--bfile Q0_unre_Caucasian_c${chr} \
--clump ${input_file} \
--clump-p1 ${clump_p1} \
--clump-r2 ${clump_r2} \
--clump-kb ${clump_kb} \
--chr ${chr} \
--clump-field Pvalue \
--clump-snp-field ID \
--threads 64 \
--out ${pheno}_chr${chr}
done