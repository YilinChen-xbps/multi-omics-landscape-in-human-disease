#!/bin/bash
conda activate regenie_env

options=$(getopt -o t:c:s:e:p --long type:,chrom:,start:,end:,print -- "$@")
eval set -- "$options"
 
while true; do
  case $1 in 
  	-t | --type) shift; type=$1 ; shift ;;
    -c | --chrom) shift; chrom=$1 ; shift ;;
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
if [ -z "$start" ]; then
    echo "Error: start is required"
    exit 1
fi
if [ -z "$end" ]; then
    echo "Error: end is required"
    exit 1
fi

if [ "$print" = true ]; then
    echo "Population: $type; Chromosome: $chrom; Start_num: $start; End_num: $end";
fi

outpath="~/WGS_cox/result_regenie/pipline_2_single"
eventColList=`sed -n "${start},${end}p" ~/disease_list.txt | sed 's/^/Incident_/' | paste -sd ','`
phenoColList=`sed -n "${start},${end}p" ~/disease_list.txt | sed 's/^/Fuduration_/' | paste -sd ','`

mkdir -p ${outpath}/${type}_result/

regenie \
  --step 2 \
  --chr ${chrom} \
  --bed ~/WGS_35W/Q0_unre_Caucasian_c${chrom} \
  --phenoFile ~/disease_Data_all_wgs_pheno_${type}.txt \
  --covarFile ~/disease_Data_all_wgs_cov_${type}.txt \
  --catCovarList sex,gene_batch,center,sequencing_provider \
  --maxCatLevels 26 \
  --bt \
  --firth --approx \
  --firth-se \
  --pThresh 0.01 \
  --t2e \
  --eventColList ${eventColList} \
  --phenoColList ${phenoColList} \
  --pred ~/WGS_cox/result_regenie/pipline_1/${type}_result/pipline_${type}_pred.list \
  --minMAC 20 \
  --bsize 1000 \
  --threads 150 \
  --write-samples \
  --print-pheno \
  --out ${outpath}/${type}_result/pipline2_${type}_chr${chrom}

if [ $? -ne 0 ]; then
    echo "${type} ${chrom} step2 single assocaition analysis failed"
else
    echo "${type} ${chrom} step2 single assocaition analysis succeed"
fi