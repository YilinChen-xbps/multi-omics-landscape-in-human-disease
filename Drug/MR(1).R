## TwoSampleMR ##
library(data.table)
library(TwoSampleMR)
Disease <- read.csv("Disease.csv",header = T)
BloodDict <- read.csv("BloodDict.csv",header = T)
Pro <- BloodDict[which(BloodDict$Omics_group=="Proteome"),]
Met <- BloodDict[which(BloodDict$Omics_group=="Metabolome"),]
Bio <- BloodDict[which(BloodDict$Omics_group=="Biochemistry"),]
#Metabolome/Biochemistry-Disease#
M_IV <- read.csv("met_IV.csv",header = T)
B_IV <- read.csv("BC_IV.csv",header = T)
ins <- format_data(M_IV,type = "exposure",header = T,phenotype_col = "PHENO",snp_col = "ID",beta_col = "BETA",se_col = "SE",eaf_col = "A1_FREQ",effect_allele_col = "A1",other_allele_col = "AX",pval_col = "P")
#ins <- format_data(B_IV,type = "exposure",header = T,phenotype_col = "PHENO",snp_col = "ID",beta_col = "BETA",se_col = "SE",eaf_col = "A1_FREQ",effect_allele_col = "A1",other_allele_col = "AX",pval_col = "P")
F11 <- Disease$NAME
for (i in 1:length(F11)) {
  pheno <- as.data.frame(fread(paste0("Finngen_GWAS_summary/R11/finngen_R11_",F11[i]),header = T))
  pheno$pheno <- F11[i]
  out <- format_data(pheno,type = "outcome",header = T,snps = ins$SNP,phenotype_col = "pheno",snp_col = "rsids",beta_col = "beta",se_col = "sebeta",effect_allele_col ="alt",other_allele_col = "ref",eaf_col = "af_alt",pval_col = "pval")
  harmo <- harmonise_data(exposure_dat = ins,outcome_dat = out)
  mr_res <- mr(harmo,method_list = c("mr_wald_ratio","mr_ivw"))
  mr_pleio <- mr_pleiotropy_test(harmo)
  mr_hetero <- mr_heterogeneity(harmo)
  results_OR <- generate_odds_ratios(mr_res)
  results_OR <- results_OR[,-c(1,2)]
  write.csv(results_OR,paste0("M_D_MR/",F11[i],".csv"),row.names = F)
  #write.csv(results_OR,paste0("B_D_MR/",F11[i],".csv"),row.names = F)
}

#Proteome-Metabolome/Biochemistry#
pro_ukb_IV <- as.data.frame(fread("cispQTL_IV_UKB.txt",header = T))
ins <- format_data(pro_ukb_IV,type = "exposure",header = T,phenotype_col = "Pro_code",snp_col = "rsid",beta_col = "BETA",se_col = "SE",eaf_col = "A1FREQ",effect_allele_col = "ALLELE1",other_allele_col = "ALLELE0",pval_col = "P")
list <- Met$Omics_feature
#list <- Bio$Omics_feature
for (i in 1:length(list)) {
  pheno <- data.frame()
  for (j in 1:22) {
    dt <- as.data.frame(fread(paste0("UKB_Met_ex_Pro_GWAS/raw/chr",j,".",list[i],".glm.linear"),header = T))
    #dt <- as.data.frame(fread(paste0("UKB_BC_ex_Pro_GWAS/raw/",list[i],"/chr",j,".",list[i],".glm.linear"),header = T))
    pheno <- rbind(pheno,dt)
  }
  pheno$PHENO <- list[i]
  out <- format_data(pheno,type = "outcome",header = T,snps = ins$SNP,phenotype_col = "PHENO",snp_col = "ID",beta_col = "BETA",se_col = "SE",effect_allele_col ="A1",other_allele_col = "AX",eaf_col = "A1_FREQ",pval_col = "P")
  harmo <- harmonise_data(exposure_dat = ins,outcome_dat = out)
  mr_res <- mr(harmo,method_list = c("mr_wald_ratio","mr_ivw"))
  mr_pleio <- mr_pleiotropy_test(harmo)
  mr_hetero <- mr_heterogeneity(harmo)
  results_OR <- generate_odds_ratios(mr_res)
  results_OR <- results_OR[,-c(1,2)]
  write.csv(results_OR,paste0("P_M_MR/",list[i],".csv"),row.names = F)
  #write.csv(results_OR,paste0("P_B_MR/",list[i],".csv"),row.names = F)
}

#Proteome-Disease#
pro_ukb_IV <- as.data.frame(fread("cispQTL_IV_UKB.txt",header = T))
ins <- format_data(pro_ukb_IV,type = "exposure",header = T,phenotype_col = "Pro_code",snp_col = "rsid",beta_col = "BETA",se_col = "SE",eaf_col = "A1FREQ",effect_allele_col = "ALLELE1",other_allele_col = "ALLELE0",pval_col = "P")
F11 <- Disease$NAME
for (i in 1:length(F11)) {
  pheno <- as.data.frame(fread(paste0("Finngen_GWAS_summary/R11/finngen_R11_",F11[i]),header = T))
  pheno$pheno <- F11[i]
  out <- format_data(pheno,type = "outcome",header = T,snps = ins$SNP,phenotype_col = "pheno",snp_col = "rsids",beta_col = "beta",se_col = "sebeta",effect_allele_col ="alt",other_allele_col = "ref",eaf_col = "af_alt",pval_col = "pval")
  harmo <- harmonise_data(exposure_dat = ins,outcome_dat = out)
  mr_res <- mr(harmo,method_list = c("mr_wald_ratio","mr_ivw"))
  mr_pleio <- mr_pleiotropy_test(harmo)
  mr_hetero <- mr_heterogeneity(harmo)
  results_OR <- generate_odds_ratios(mr_res)
  results_OR <- results_OR[,-c(1,2)]
  write.csv(results_OR,paste0("P_D_MR/",F11[i],".csv"),row.names = F)
}