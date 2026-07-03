pacman::p_load(data.table,dplyr,plyr,magrittr,stringr,DEswan)  
rm(list=ls())  

# 基础设置  
omics <- c('Metabolomic', 'Proteomic', 'Biochemistry')  

new_names <- BloodDict$Omics_feature  
new_names <- ifelse(grepl("-0.0", new_names),   
                    paste0("X", gsub("-0.0", "", new_names)),   
                    new_names)  
BloodDict <- BloodDict %>%   
  mutate(Omics_feature = gsub('-', '_', new_names))  


# 读取协变量数据  
categorical <- c("Sex", "Race_imp", "Season", "Smoke_imp", "Statin_imp")  

# 设置协变量组合  
cyl_c1 <- c( "Sex", "Race_imp")  
cyl_cbackup <- c("Race_imp")  

# 读取组学数据  
omic_data <- list()  
for (omic in omics) {  
  omic_data[[omic]] <- file.path(omic_filepath, paste0(omic, 'Data.csv')) %>%   
    fread() %>%   
    as.data.frame()  
  new_names <- names(omic_data[[omic]])  
  new_names <- ifelse(grepl("-0.0", new_names),   
                      paste0("X", gsub("-0.0", "", new_names)),   
                      new_names)  
  omic_data[[omic]] <- omic_data[[omic]] %>%  
    magrittr::set_colnames(new_names) %>%  
    filter(eid %in% multiomics_id)  
}  

# 合并所有组学数据  
omics_dat <- Reduce(function(x, y) merge(x, y, by='eid'), omic_data) %>%   
  as.data.frame()  


# 准备年龄轨迹分析数据  
oriV1_omics <- omics_dat %>%  
  inner_join(covar, by = 'eid') %>%  
  dplyr::select(eid, Year = Age, BloodDict$Omics_feature, cyl_c1) %>%  
  as.data.frame()  

# 检查性别变量  
deswan_c <- cyl_c1  
if (any(table(oriV1_omics$Sex) %>% as.data.frame() %>% pull(Freq) == 0)) {  
  deswan_c <- cyl_cbackup  
}  

# 对不同的parcel_width进行DESWAN分析  
res_DEswan <- res_p <- list()  
for (parcel_width in c(2,3,4,5)) {
  print(paste("Processing parcel_width:", parcel_width))  
  character_parcel_width <- parcel_width %>% as.character()  
  
  # DESWAN分析  
  res_DEswan[[character_parcel_width]] <- DEswan(  
    data.df = oriV1_omics[, BloodDict$Omics_feature],  
    covariates = oriV1_omics[, deswan_c],  
    qt = oriV1_omics[, 'Year'],  
    window.center = seq(40, 70, 1),  # 根据您的年龄范围调整  
    buckets.size = parcel_width  
  )  
  
  # 重塑结果  
  res_p[[character_parcel_width]] = res_DEswan[[character_parcel_width]] %>%   
    reshape.DEswan(parameter = 1, factor = "qt")  
}  
