#!/bin/bash
conda activate regenie_env

options=$(getopt -o t:s:e:p --long type:,start:,end:,print -- "$@")
eval set -- "$options"
 
while true; do
  case $1 in 
  	-t | --type) shift; type=$1 ; shift ;;
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
if [ -z "$start" ]; then
    echo "Error: start is required"
    exit 1
fi
if [ -z "$end" ]; then
    echo "Error: end is required"
    exit 1
fi

if [ "$print" = true ]; then
    echo "Population: $type; Start_num: $start; End_num: $end";
fi

outpath="~/WGS_cross_section/result_regenie/pipline_1"
eventColList=`sed -n "${start},${end}p" ~/cyl_disease_list.txt | sed 's/^/Incident_/' | paste -sd ','`

mkdir -p ${outpath}/${type}_result
mkdir -p ${outpath}/${type}_tmp

regenie \
  --step 1 \
  --bed ~/ukb_array_call_hg38/ukb_cal_allChrs_hg38 \
  --extract ~/ukb_array_call_hg38/qc_pass.snplist \
  --phenoFile ~/disease_Data_all_wgs_pheno_${type}.txt \
  --covarFile ~/disease_Data_all_wgs_cov_${type}.txt \
  --catCovarList sex,gene_batch,center,sequencing_provider \
  --maxCatLevels 26 \
  --bt \
  --phenoColList ${eventColList} \
  --bsize 1000 \
  --lowmem \
  --lowmem-prefix ${outpath}/${type}_tmp/${type}_tmp_preds_${start}_${end} \
  --out ${outpath}/${type}_result/pipline_${type}_${start}_${end}

if [ $? -ne 0 ]; then
    echo "${type} for pheno index from ${start} to ${end} step1 null model failed"
else
    echo "${type} for pheno index from ${start} to ${end} step1 null model succeed"
fi
