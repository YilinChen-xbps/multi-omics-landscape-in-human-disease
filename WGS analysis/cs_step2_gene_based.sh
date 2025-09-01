#!/bin/bash
conda activate regenie_env

options=$(getopt -o t:c:g:s:e:p --long type:,chrom:,genebased:,start:,end:,print -- "$@")
eval set -- "$options"
 
while true; do
  case $1 in 
  	-t | --type) shift; type=$1 ; shift ;;
    -c | --chrom) shift; chrom=$1 ; shift ;;
    -g | --genebased) shift; genebased=$1 ; shift ;;
    -s | --start) shift; start=$1 ; shift ;;
    -e | --end) shift; end=$1 ; shift ;;
    -p | --print) print=true; shift ;;
    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done
 
if [ -z "$type" ]; then
    echo "Error: type is required"
    exit 1
fi
if [ -z "$chrom" ]; then
    echo "Error: chrom is required"
    exit 1
fi
if [ -z "$genebased" ]; then
    echo "Error: genebased is required"
    exit 1
fi
if [ -z "$start" ]; then
    echo "Error: start is required"
    exit 1
fi
if [ -z "$end" ]; then
    echo "Error: end is required"
    exit 1
fi

if [ "$print" = true ]; then
    echo "Population: $type; Chromosome: $chrom; Genebased: $genebased; Start_num: $start; End_num: $end";
fi

outpath="~/WGS_cross_section/result_regenie/pipline_2_gene_based"
eventColList=`sed -n "${start},${end}p" ~/cyl_disease_list.txt | sed 's/^/Incident_/' | paste -sd ','`

mkdir -p ${outpath}/${type}_result/${genebased}

regenie \
  --step 2 \
  --chr ${chrom} \
  --bed ~/WGS_35W/Q0_unre_Caucasian_c${chrom} \
  --phenoFile ~/disease_Data_all_wgs_pheno_${type}.txt \
  --covarFile ~/disease_Data_all_wgs_cov_${type}.txt \
  --catCovarList sex,gene_batch,center,sequencing_provider \
  --maxCatLevels 26 \
  --anno-file ~/Anno_New/chr${chrom}/Main/OLD${genebased}_chr${chrom}.txt \
  --set-list ~/Anno_New/chr${chrom}/Main/chr${chrom}_${genebased}.setlist \
  --mask-def ~/Anno_New/Mask/Mask_${genebased}.txt \
  --aaf-bins 0.001 \
  --vc-tests skato,acato-full \
  --vc-maxAAF 0.001 \
  --joint minp,acat \
  --rgc-gene-p \
  --check-burden-files \
  --write-mask \
  --bt \
  --firth --approx \
  --firth-se \
  --pThresh 0.01 \
  --phenoColList ${eventColList} \
  --pred ~/WGS_cross_section/result_regenie/pipline_1/${type}_result/pipline_${type}_pred.list \
  --bsize 200 \
  --threads 150 \
  --write-samples \
  --write-mask-snplist \
  --print-pheno \
  --out ${outpath}/${type}_result/${genebased}/step2_${type}_chr${chrom}_${genebased}

