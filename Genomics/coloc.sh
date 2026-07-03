rm(list = ls())
pacman::p_load(data.table, openxlsx, dplyr, tidyr, stringr, coloc, plyr)

args <- commandArgs(trailingOnly = TRUE)
INPUT_DIR <- args[1]
pheno <- args[2]
clump_r2 <- args[3]
clump_kb <- args[4]
QTL_num <- as.integer(args[5])

cat('Input Directory:', INPUT_DIR, '\n')
cat('Phenotype:', pheno, '\n')
cat('Clump R2:', clump_r2, '\n')
cat('Clump KB:', clump_kb, '\n')
cat('QTL Number:', QTL_num, '\n')

# file path
coloc_outpath <- paste0(INPUT_DIR, "/leadSNP_5e-8_", clump_r2, "_", clump_kb, "/coloc_output/")
output_file <- paste0(INPUT_DIR, "/leadSNP_5e-8_", clump_r2, "_", clump_kb, "/coloc_input/", pheno, "_leadsnp_2mb.tsv")
probelist_file <- paste0(INPUT_DIR, "/leadSNP_5e-8_", clump_r2, "_", clump_kb, "/candidate_gene/", pheno,"_probe_list.tsv")

if (!dir.exists(coloc_outpath)) {  dir.create(coloc_outpath, recursive = TRUE)}

window <- 1e6  # 1Mb window
pp_threshold <- 0.8

# Filter lead SNPs
filter_lead_snvs <- function(lead_snvs_data, window) {
  lead_snvs_data <- lead_snvs_data[order(lead_snvs_data$Pvalue),]
  filtered_lead_snvs <- data.frame()
  for (j in 1:nrow(lead_snvs_data)) {
    current_snv <- lead_snvs_data[j,]
    if (nrow(filtered_lead_snvs) == 0) {
      filtered_lead_snvs <- rbind(filtered_lead_snvs, current_snv)
      next
    }
    is_far_enough <- all(abs(current_snv$GENPOS - filtered_lead_snvs$GENPOS) > window | current_snv$CHR != filtered_lead_snvs$CHR)
    if (is_far_enough) filtered_lead_snvs <- rbind(filtered_lead_snvs, current_snv)
  }
  return(filtered_lead_snvs)
}

probelist <- fread(probelist_file) %>%
  as.data.frame() %>%
  mutate(
    symbol_count = str_count(probe_SYMBOL, ";") + 1,
    ensg_count = str_count(probe_ENSG, ";") + 1
  ) %>%
  filter(symbol_count == ensg_count) %>%
  select(-symbol_count, -ensg_count) %>%
  separate_rows(probe_SYMBOL, probe_ENSG, sep = ";")  %>%
  mutate(ENSG = gsub("\\..*", "", probe_ENSG))

final_probelist <- unique(probelist$ENSG)
ENSG_SYMBOL <- probelist %>% dplyr::select(ENSEMBL=ENSG,SYMBOL=probe_SYMBOL) %>% distinct(ENSEMBL,.keep_all = T)

# load WGS summary data
wgs_file_normed <- fread(output_file) %>% as.data.frame() %>% mutate(
    ID_NEW=paste(CHROM,POS_NEW,REF_NEW,ALT_NEW,sep = ":"),
    POS_NEW = ifelse(grepl("NA", ID_NEW), GENPOS, POS_NEW),
    REF_NEW = ifelse(grepl("NA", ID_NEW), ALLELE0, REF_NEW),
    ALT_NEW = ifelse(grepl("NA", ID_NEW), ALLELE1, ALT_NEW),
    ID_NEW = ifelse(grepl("NA", ID_NEW), ID, ID_NEW),
    LOG10P = as.numeric(LOG10P),
    Pvalue = 10^(-LOG10P)
  ) %>% 
  dplyr::select(CHROM, GENPOS, POS = POS_NEW, ID, ID_NEW, Allele1 = ALT_NEW, Allele2 = REF_NEW, A1FREQ, BETA, SE, Pvalue)

lead_snvs_data <- wgs_file_normed %>% dplyr::filter(ID %in% probelist$ID)
filtered_lead_snvs <- filter_lead_snvs(lead_snvs_data, window ) %>% 
  mutate(identifier = paste(CHROM, GENPOS, sep = ":")) %>% 
  dplyr::rowwise() %>% 
  dplyr::mutate(region_start = min(wgs_file_normed$POS[wgs_file_normed$CHROM == CHROM & wgs_file_normed$POS >= (POS - (window / 2))], na.rm = TRUE), 
                region_end = max(wgs_file_normed$POS[wgs_file_normed$CHROM == CHROM & wgs_file_normed$POS <= (POS + (window / 2))], na.rm = TRUE)) %>%
  ungroup() %>% as.data.frame()
snv_list <- filtered_lead_snvs$ID_NEW

# Load xQTL data
QTL_index <- openxlsx::read.xlsx("~/coloc/data/QTL_Data_Summary.xlsx", sheet = 1) %>% 
  filter(USE == 1, Index != 9)

tissue_name <- QTL_index[QTL_num, "Tissue"]
N_xQTL <- QTL_index[QTL_num, "Sample_Size"]
PMID <- QTL_index[QTL_num, "PMID"]
xQTL_used <- QTL_index[QTL_num, "xQTL_used"]

cat("xQTL file --- PMID:", PMID, "; Tissue:", tissue_name, "; Sample_Size:", N_xQTL, "\n")
xQTL_data_normed <- QTL_index[QTL_num, 'Server_Location'] %>% 
  fread() %>% as.data.frame() %>% dplyr::select(SNP, CHROM, POS, Allele1, Allele2, BETA, SE, Pvalue, ENSG, SYMBOL)

# Coloc analysis function
run_coloc_analysis <- function(xQTL_data, wgs_file, snv_list, N_xQTL) {
  coloc_results <- list()
  for (snv in snv_list) { 
    print(paste("SNP:", snv))
    lead_snv <- wgs_file_normed[which(wgs_file_normed$ID_NEW == snv), ]
    if (nrow(lead_snv) == 0) next
    lead_snv_position <- lead_snv$POS
    lead_snv_chromosome <- lead_snv$CHROM
    
    # Extract xQTL and GWAS data in window
    wgs_sub <- wgs_file_normed %>% filter(CHROM == lead_snv_chromosome, GENPOS >= (lead_snv_position - (window / 2)), GENPOS <= (lead_snv_position + (window / 2)),
                                          is.finite(BETA),  is.finite(SE),!is.na(BETA),!is.na(SE)) %>% 
      mutate(identifier = paste(CHROM, GENPOS, sep = ":"))
    xQTL_sub <- xQTL_Gene %>% dplyr::filter(CHROM == lead_snv_chromosome, POS >= (lead_snv_position - (window / 2)), POS <= (lead_snv_position + (window / 2)),
                                            is.finite(BETA),  is.finite(SE),!is.na(BETA),!is.na(SE)) %>% 
      mutate(identifier = paste(CHROM, POS, sep = ":")  )
    common_snps <- intersect(wgs_sub$identifier, xQTL_sub$identifier)
    if (length(common_snps) == 0) next
    cat("Number of common SNPs:", length(common_snps), '\n')
    
    xQTL_snvs_subset_clean <- xQTL_sub %>% filter(identifier %in% common_snps) %>% arrange(Pvalue) %>% 
      distinct(identifier, .keep_all = TRUE) %>% arrange(identifier)
    wgs_snvs_subset_clean <- wgs_sub %>% filter(identifier %in% common_snps) %>% arrange(Pvalue) %>% 
      distinct(identifier, .keep_all = TRUE) %>% arrange(identifier)
    
    # Align alleles
    keep_rows <- rep(TRUE, nrow(xQTL_snvs_subset_clean))
    for (r in 1:nrow(xQTL_snvs_subset_clean)) {
      if (xQTL_snvs_subset_clean$Allele1[r] == wgs_snvs_subset_clean$Allele1[r] && xQTL_snvs_subset_clean$Allele2[r] == wgs_snvs_subset_clean$Allele2[r]) next
      else if (xQTL_snvs_subset_clean$Allele1[r] == wgs_snvs_subset_clean$Allele2[r] && xQTL_snvs_subset_clean$Allele2[r] == wgs_snvs_subset_clean$Allele1[r]) {
        xQTL_snvs_subset_clean$Allele1[r] <- wgs_snvs_subset_clean$Allele1[r]
        xQTL_snvs_subset_clean$Allele2[r] <- wgs_snvs_subset_clean$Allele2[r]
        xQTL_snvs_subset_clean$BETA[r] <- -xQTL_snvs_subset_clean$BETA[r]
      } else keep_rows[r] <- FALSE
    }
    
    xQTL_snvs_subset_clean <- xQTL_snvs_subset_clean[keep_rows, ]
    wgs_snvs_subset_clean <- wgs_snvs_subset_clean[keep_rows, ]
    if (nrow(xQTL_snvs_subset_clean) == 0 || nrow(wgs_snvs_subset_clean) == 0) next
    
    # Run coloc analysis
    coloc_result <- coloc.abf(
      dataset1 = list(snp = wgs_snvs_subset_clean$ID, beta = wgs_snvs_subset_clean$BETA, varbeta = (wgs_snvs_subset_clean$SE)^2, 
                      p = wgs_snvs_subset_clean$Pvalue, N = wgs_snvs_subset_clean$N[1], type = "cc"),
      dataset2 = list(snp = wgs_snvs_subset_clean$ID, 
                      beta = xQTL_snvs_subset_clean$BETA, varbeta = (xQTL_snvs_subset_clean$SE)^2, 
                      N = as.numeric(N_xQTL), type = "quant"),
      MAF = wgs_snvs_subset_clean$A1FREQ, p1 = 1e-4, p2 = 1e-4, p12 = 1e-5)
    coloc_results[[snv]] <- coloc_result 
  }
  return(coloc_results) 
}

# Process coloc results
coloc_results_ensg <- list()
for (i in 1:length(final_probelist)) {
  Gene_ensg <- final_probelist[i]
  print(paste("Processing Gene:", Gene_ensg))
  
  xQTL_Gene <- xQTL_data_normed %>% dplyr::filter(ENSG == Gene_ensg)
  coloc_results <- run_coloc_analysis(xQTL_Gene, wgs_file_normed, snv_list, N_xQTL)
  
  if (length(coloc_results) == 0) { cat(Gene_ensg, "No common SNPs in xQTL data. \n"); next }
  
  coloc_results_ensg[[Gene_ensg]] <- data.frame()
  for (snp_name in names(coloc_results)) {
    results_df <- coloc_results[[snp_name]]$results
    summary_df <- coloc_results[[snp_name]]$summary
    snp_filtered <- data.frame(sig_SNPs = results_df$snp, PPH4_abf = summary_df["PP.H4.abf"], 
                               PPH4_SNP = results_df$SNP.PP.H4, row.names = NULL) %>% 
      dplyr::filter(PPH4_abf > pp_threshold,PPH4_SNP  > pp_threshold)
    if (nrow(snp_filtered) == 0) { cat("filtered_lead_SNPs,",snp_name,",  no significant SNPs with PPH4 >",pp_threshold,"\n"); next }
    snp_filtered$filtered_lead_SNPs <- snp_name
    coloc_results_ensg[[Gene_ensg]] <- rbind(coloc_results_ensg[[Gene_ensg]], snp_filtered)
  }
  
  if (nrow(coloc_results_ensg[[Gene_ensg]]) == 0) next 
  
  coloc_results_ensg[[Gene_ensg]] <- coloc_results_ensg[[Gene_ensg]] %>% 
    tidyr::separate(sep = ':', col = 'sig_SNPs', into = c("CHROM", "POS"), remove = F, extra = 'drop', convert = T) %>% 
    mutate(ENSG = Gene_ensg,Target = Gene_ensg,Tissue=tissue_name,xQTL=xQTL_used)
}

# Combine coloc results and save
coloc_results_PMID <- plyr::rbind.fill(coloc_results_ensg) 
if (nrow(coloc_results_PMID) == 0) {
  cat("No significant coloc results found. Exiting script.\n")
  quit(save = "no", status = 0) # Exit the script without saving any results
}

coloc_results_PMID$SYMBOL <- ENSG_SYMBOL$SYMBOL[match(coloc_results_PMID$ENSG, ENSG_SYMBOL$ENSEMBL)]

coloc_results_PMID <- coloc_results_PMID %>% dplyr::mutate(identifier = stringr::str_extract(filtered_lead_SNPs, "^[^:]+:[^:]+")) 
coloc_results_PMID$region_start <- filtered_lead_snvs$region_start[match(coloc_results_PMID$identifier, filtered_lead_snvs$identifier)]
coloc_results_PMID$region_end <- filtered_lead_snvs$region_end[match(coloc_results_PMID$identifier, filtered_lead_snvs$identifier)]
coloc_results_PMID <- coloc_results_PMID %>% dplyr::mutate(PMID = PMID, identifier = NULL   )
                
output_file <- paste0(coloc_outpath, pheno,'_',tissue_name,'_', xQTL_used, '_PMID', PMID, '.tsv')
fwrite(coloc_results_PMID, file = output_file, sep = '\t')
cat(paste("Done, Coloc results saved to:", output_file, "\n"))
