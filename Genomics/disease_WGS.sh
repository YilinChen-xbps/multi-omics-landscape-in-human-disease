##############################################################################survival model####################################################################################
####################################
#regenie step1 
####################################
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



mkdir -p ${HOME}/pipline_1/${type}_result
mkdir -p ${HOME}/pipline_1/${type}_tmp

regenie \
  --step 1 \
  --bed ${HOME}/ukb_cal_allChrs_hg38 \
  --extract ${HOME}/qc_pass.snplist \
  --phenoFile ${HOME}/disease_Data_all_wgs_pheno_${type}.txt \
  --covarFile ${HOME}/disease_Data_all_wgs_cov_${type}.txt \
  --catCovarList sex,gene_batch,center,sequencing_provider \
  --maxCatLevels 26 \
  --bt \
  --t2e \
  --eventColList ${eventColList} \
  --phenoColList ${phenoColList} \
  --bsize 1000 \
  --niter 10000 \
  --lowmem \
  --lowmem-prefix ${HOME}/pipline_1/${type}_tmp/${type}_tmp_preds_${start}_${end} \
  --out ${HOME}/pipline_1/${type}_result/pipline_${type}_${start}_${end}

if [ $? -ne 0 ]; then
    echo "${type} for pheno index from ${start} to ${end} step1 null model failed"
else
    echo "${type} for pheno index from ${start} to ${end} step1 null model succeed"
fi

####################################
#regenie step2 single variant analysis
####################################
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


mkdir -p ${HOME}/pipline_2_single/${type}_result/

regenie \
  --step 2 \
  --chr ${chrom} \
  --bed ${HOME}/Q0_unre_Caucasian_c${chrom} \
  --phenoFile ${HOME}/disease_Data_all_wgs_pheno_${type}.txt \
  --covarFile ${HOME}/disease_Data_all_wgs_cov_${type}.txt \
  --catCovarList sex,gene_batch,center,sequencing_provider \
  --maxCatLevels 26 \
  --bt \
  --firth --approx \
  --firth-se \
  --pThresh 0.01 \
  --t2e \
  --eventColList ${eventColList} \
  --phenoColList ${phenoColList} \
  --pred ${HOME}/${type}_result/pipline_${type}_pred.list \
  --minMAC 20 \
  --bsize 1000 \
  --threads 150 \
  --write-samples \
  --print-pheno \
  --out ${HOME}/${type}_result/pipline2_${type}_chr${chrom}

if [ $? -ne 0 ]; then
    echo "${type} ${chrom} step2 single assocaition analysis failed"
else
    echo "${type} ${chrom} step2 single assocaition analysis succeed"
fi



####################################
#regenie step2 gene-based analysis
####################################
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


mkdir -p ${HOME}/pipline_2_gene_based/${type}_result/${genebased}

regenie \
  --step 2 \
  --chr ${chrom} \
  --bed ${HOME}/Q0_unre_Caucasian_c${chrom} \
  --phenoFile ${HOME}/disease_Data_all_wgs_pheno_${type}.txt \
  --covarFile ${HOME}/disease_Data_all_wgs_cov_${type}.txt \
  --catCovarList sex,gene_batch,center,sequencing_provider \
  --maxCatLevels 26 \
  --anno-file ${HOME}/Anno_New/chr${chrom}/Main/OLD${genebased}_chr${chrom}.txt \
  --set-list ${HOME}/Anno_New/chr${chrom}/Main/chr${chrom}_${genebased}.setlist \
  --mask-def ${HOME}/Anno_New/Mask/Mask_${genebased}.txt \
  --aaf-bins 0.001 \
  --vc-maxAAF 0.001 \
  --check-burden-files \
  --write-mask \
  --bt \
  --firth --approx \
  --firth-se \
  --pThresh 0.01 \
  --t2e \
  --eventColList ${eventColList} \
  --phenoColList ${phenoColList} \
  --pred ${HOME}/pipline_1/${type}_result/pipline_${type}_pred.list \
  --bsize 1000 \
  --threads 150 \
  --write-samples \
  --write-mask-snplist \
  --print-pheno \
  --out ${HOME}/pipline_2_gene_based/${type}_result/${genebased}/step2_${type}_chr${chrom}_${genebased}

if [ $? -ne 0 ]; then
    echo "${type} ${chrom} step2 gene-based assocaition analysis failed"
else
    echo "${type} ${chrom} step2 gene-based assocaition analysis succeed"
fi



##############################################################################cross-section model####################################################################################
####################################
#regenie step1 
####################################
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


regenie \
  --step 1 \
  --bed ${HOME}/ukb_cal_allChrs_hg38 \
  --extract ${HOME}/qc_pass.snplist \
  --phenoFile ${HOME}/disease_Data_all_wgs_pheno_${type}.txt \
  --covarFile ${HOME}/disease_Data_all_wgs_cov_${type}.txt \
  --catCovarList sex,gene_batch,center,sequencing_provider \
  --maxCatLevels 26 \
  --bt \
  --phenoColList ${eventColList} \
  --bsize 1000 \
  --lowmem \
  --lowmem-prefix ${HOME}/pipline_1/${type}_tmp/${type}_tmp_preds_${start}_${end} \
  --out ${HOME}/pipline_1/${type}_result/pipline_${type}_${start}_${end}

if [ $? -ne 0 ]; then
    echo "${type} for pheno index from ${start} to ${end} step1 null model failed"
else
    echo "${type} for pheno index from ${start} to ${end} step1 null model succeed"
fi


####################################
#regenie step2 single variant analysis
####################################
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



regenie \
  --step 2 \
  --chr ${chrom} \
  --bed ${HOME}/Q0_unre_Caucasian_c${chrom} \
  --phenoFile ${HOME}/disease_Data_all_wgs_pheno_${type}.txt \
  --covarFile ${HOME}/disease_Data_all_wgs_cov_${type}.txt \
  --catCovarList sex,gene_batch,center,sequencing_provider \
  --maxCatLevels 26 \
  --bt \
  --firth --approx \
  --firth-se \
  --pThresh 0.01 \
  --phenoColList ${eventColList} \
  --pred ${HOME}/pipline_1/${type}_result/pipline_${type}_pred.list \
  --minMAC 20 \
  --bsize 1000 \
  --threads 150 \
  --write-samples \
  --print-pheno \
  --out ${HOME}/pipline_2_single/${type}_result/pipline2_${type}_chr${chrom}

if [ $? -ne 0 ]; then
    echo "${type} ${chrom} step2 single assocaition analysis failed"
else
    echo "${type} ${chrom} step2 single assocaition analysis succeed"
fi

####################################
#regenie step2 gene-based analysis
####################################
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


regenie \
  --step 2 \
  --chr ${chrom} \
  --bed ${HOME}/Q0_unre_Caucasian_c${chrom} \
  --phenoFile ${HOME}/disease_Data_all_wgs_pheno_${type}.txt \
  --covarFile ${HOME}/disease_Data_all_wgs_cov_${type}.txt \
  --catCovarList sex,gene_batch,center,sequencing_provider \
  --maxCatLevels 26 \
  --anno-file ${HOME}/Anno_New/chr${chrom}/Main/OLD${genebased}_chr${chrom}.txt \
  --set-list ${HOME}/Anno_New/chr${chrom}/Main/chr${chrom}_${genebased}.setlist \
  --mask-def ${HOME}/Anno_New/Mask/Mask_${genebased}.txt \
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
  --pred ${HOME}/pipline_1/${type}_result/pipline_${type}_pred.list \
  --bsize 200 \
  --threads 150 \
  --write-samples \
  --write-mask-snplist \
  --print-pheno \
  --out ${HOME}/pipline_2_gene_based/${type}_result/${genebased}/step2_${type}_chr${chrom}_${genebased}

if [ $? -ne 0 ]; then
    echo "${type} ${chrom} step2 gene-based assocaition analysis failed"
else
    echo "${type} ${chrom} step2 gene-based assocaition analysis succeed"
fi
