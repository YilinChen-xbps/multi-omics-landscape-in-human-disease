pacman::p_load(data.table, dplyr, plyr, magrittr, stringr, MatchIt, openxlsx, tidyr)

# ==============================================================================
# Configuration & File Paths (Relative to project root)
# ==============================================================================
dir_omic_data     <- "./Data/BloodData/"
dir_covariates    <- "./Data/Covariates/"
dir_endpoints     <- "./Data/UKB_Disease/"
dir_output        <- "./RESULTS/multiomics_trajectory/diseasespan_trajectory/"

file_caucasian    <- "./Data/Caucasian_eid.csv"
file_blood_dict   <- file.path(dir_omic_data, "BloodDict.csv")
file_covar        <- file.path(dir_covariates, "Covariates.csv")
file_protein_cov  <- file.path(dir_omic_data, "ProteinData_n_Cov.csv")
file_disease      <- "./DATA/disease_filter_merge.xlsx"

# Create output directory if it doesn't exist
dir.create(dir_output, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Data Loading & Preprocessing
# ==============================================================================
omics <- c('Metabolomic', 'Proteomic', 'Biochemistry')

# Load target population IDs
multiomics_id <- fread(file_caucasian) %>% 
  as.data.frame() %>% 
  filter(Caucasian == 1) %>% 
  pull(eid)

# Load Omics Dictionary
BloodDict <- fread(file_blood_dict) %>% as.data.frame()

# Load Covariates
categorical <- c("Sex", "Race_imp", "Season", "Smoke_imp", "Statin_imp")
covar <- fread(file_covar) %>% 
  as.data.frame() %>%
  mutate_at(vars(all_of(categorical)), as.factor) %>% 
  mutate(Age2 = Age^2)

# Pre-load Proteomics covariate data 
pro_cov <- fread(file_protein_cov) %>%  
  as.data.frame() %>%  
  dplyr::select(-c(2:13)) %>%  
  set_colnames(c('eid', colnames(.)[-1] %>% gsub('-', '_', .))) 

# Define covariate models for residual calculation
cyl_c1           <- c('Age', 'Age2', "Sex", "Race_imp", "TDI_imp", "FastingTime_imp", "Season", "BMI_imp", "Smoke_imp")
cyl_c2           <- 'Sex*Age + Age2 * Sex'
cyl_noSex        <- c('Age', 'Age2', "Race_imp", "TDI_imp", "FastingTime_imp", "Season", "BMI_imp", "Smoke_imp")
cyl_noSex_noRace <- c('Age', 'Age2', "TDI_imp", "FastingTime_imp", "Season", "BMI_imp", "Smoke_imp")
cyl_noRace_c1    <- c('Age', 'Age2', "Sex", "TDI_imp", "FastingTime_imp", "Season", "BMI_imp", "Smoke_imp")

# Load and filter disease endpoints
final_disease <- openxlsx::read.xlsx(file_disease)

# ==============================================================================
# Main Analysis Loop
# ==============================================================================
for (omic in omics) {
  for (endpoint in final_disease$NAME) {
    
    print(paste("Processing -> Omic:", omic, "| Endpoint:", endpoint))
    
    # 1. Load Outcome Data & Merge
    outcome <- file.path(dir_endpoints, paste0(endpoint, '.csv')) %>% 
      fread() %>% 
      as.data.frame()
    
    if (omic == 'Proteomic') {  
      omic_data <- pro_cov 
    } else {  
      omic_data <- file.path(dir_omic_data, paste0(omic, 'Data.csv')) %>%   
        fread() %>% 
        as.data.frame() 
    }
    
    # Format column names
    omic_data <- omic_data %>% 
      rename_with(~ifelse(grepl("-0.0", .), paste0("X", gsub("-0.0", "", .)), .))  
    
    outcome_status <- 'target_y'
    outcome_years <- 'BL2Target_yrs'
    
    data_merged <- omic_data %>%  
      inner_join(covar, by = 'eid') %>% 
      inner_join(outcome, by = 'eid') %>% 
      filter(eid %in% multiomics_id)  
    
    # 1.2 Propensity Score Matching (Case vs Control)
    mt_out1 <- matchit(as.formula(paste(outcome_status, "~ Age + Sex + Race_imp + BMI_imp + TDI_imp")),
                       method = "nearest", distance = "mahalanobis", ratio = 5, link = "logit", data = data_merged)
    
    mt_data <- match.data(mt_out1)
    
    # 1.3 Adjust Time Scale for Matched Pairs
    mt_final <- mt_data %>% arrange(subclass) 
    time_scale <- mt_final[[outcome_years]][which(mt_final[[outcome_status]] == 1)]
    subclass_sizes <- mt_final %>% dplyr::group_by(subclass) %>% dplyr::summarise(n = n())
    repeated_time_scale <- rep(time_scale, times = subclass_sizes$n)
    
    mt_final <- mt_final %>% mutate(time_scale = repeated_time_scale * sign(-1))  
    
    # ==========================================================================
    # 2. Residual Calculation
    # ==========================================================================
    resid_mt <- mt_final %>% magrittr::set_rownames(.$eid)
    
    for (i in BloodDict$Omics_feature[which(BloodDict$Omics_group == omic)]) {
      i_SampAge <- paste0(i, '_SampAge')
      
      # Determine base formula by omic type
      if (omic == 'Proteomic') {
        FML_str <- paste0(i, " ~ ", i_SampAge, " + ", paste(cyl_c1, collapse = ' + '), " + ", cyl_c2)
        sub_data <- resid_mt[, c('eid', i, i_SampAge, cyl_c1)] %>% na.omit() %>% magrittr::set_rownames(.$eid)
      } else {
        FML_str <- paste0(i, " ~ ", paste(c(cyl_c1, 'Statin_imp'), collapse = ' + '), " + ", cyl_c2)
        sub_data <- resid_mt[, c('eid', i, cyl_c1, 'Statin_imp')] %>% na.omit() %>% magrittr::set_rownames(.$eid)
      }
      
      # Handle rank-deficient fits (missing variance in Sex or Race)
      missing_sex <- any(table(sub_data$Sex) %>% as.data.frame() %>% pull(Freq) == 0)
      missing_race <- any(table(sub_data$Race_imp) %>% as.data.frame() %>% pull(Freq) == 0)
      
      if (missing_sex & missing_race) {
        if (omic == 'Proteomic') {
          FML_str <- paste0(i, " ~ ", i_SampAge, " + ", paste(cyl_noSex_noRace, collapse = ' + '))
          sub_data <- resid_mt[, c('eid', i, i_SampAge, cyl_noSex_noRace)] %>% na.omit() %>% magrittr::set_rownames(.$eid)
        } else {
          FML_str <- paste0(i, " ~ ", paste(c(cyl_noSex_noRace, 'Statin_imp'), collapse = ' + '))
          sub_data <- resid_mt[, c('eid', i, cyl_noSex_noRace, 'Statin_imp')] %>% na.omit() %>% magrittr::set_rownames(.$eid)
        }
      } else if (missing_sex) {
        if (omic == 'Proteomic') {
          FML_str <- paste0(i, " ~ ", i_SampAge, " + ", paste(cyl_noSex, collapse = ' + '))
          sub_data <- resid_mt[, c('eid', i, i_SampAge, cyl_noSex)] %>% na.omit() %>% magrittr::set_rownames(.$eid)
        } else {
          FML_str <- paste0(i, " ~ ", paste(c(cyl_noSex, 'Statin_imp'), collapse = ' + '))
          sub_data <- resid_mt[, c('eid', i, cyl_noSex, 'Statin_imp')] %>% na.omit() %>% magrittr::set_rownames(.$eid)
        }
      } else if (missing_race) {
        if (omic == 'Proteomic') {
          FML_str <- paste0(i, " ~ ", i_SampAge, " + ", paste(cyl_noRace_c1, collapse = ' + '), " + ", cyl_c2)
          sub_data <- resid_mt[, c('eid', i, i_SampAge, cyl_noRace_c1)] %>% na.omit() %>% magrittr::set_rownames(.$eid)
        } else {
          FML_str <- paste0(i, " ~ ", paste(c(cyl_noRace_c1, 'Statin_imp'), collapse = ' + '), " + ", cyl_c2)
          sub_data <- resid_mt[, c('eid', i, cyl_noRace_c1, 'Statin_imp')] %>% na.omit() %>% magrittr::set_rownames(.$eid)
        }
      }
      
      # Extract residuals
      sub_data[, i] <- resid(lm(as.formula(FML_str), data = sub_data))
      resid_mt[, i] <- NA  
      resid_mt[rownames(sub_data), i] <- as.numeric(sub_data[, i])   
    }
    
    save(resid_mt, file = file.path(dir_output, paste0("resid_", endpoint, '_', omic, ".RData")))
    
    # ==========================================================================
    # 3. Z-score Normalization (Case relative to Control)
    # ==========================================================================
    df_group <- resid_mt 
    variable.names <- BloodDict$Omics_feature[which(BloodDict$Omics_group == omic)]
    
    df_group_case <- df_group %>% ungroup() %>% filter(.data[[outcome_status]] == 1)
    df_group_control <- df_group %>% ungroup() %>% filter(.data[[outcome_status]] == 0)
    
    # Calculate control baseline statistics
    stats_control <- df_group_control %>%
      summarise_at(vars(all_of(variable.names)), 
                   list(mean = ~ mean(., na.rm = TRUE), sd = ~ sd(., na.rm = TRUE))) %>%
      tidyr::pivot_longer(cols = everything(), names_to = c("Omics_feature", ".value"), names_pattern = "(.*)_(mean|sd)") %>% 
      as.data.frame() %>% 
      set_rownames(.$Omics_feature) 
    
    # Apply Z-score transformation
    df_z_scores <- df_group_case
    for (var_name in variable.names) {
      df_z_scores[[var_name]] <- (df_group_case[[var_name]] - stats_control[var_name, 'mean']) / stats_control[var_name, 'sd']
    }
    
    # Export trajectory data
    final_trajectory_df <- df_z_scores %>% dplyr::select(eid, subclass, Year = time_scale, all_of(variable.names))
    
    fwrite(final_trajectory_df, 
           file = file.path(dir_output, paste0('trajectories_Zscore_', endpoint, '_', omic, '.tsv')), 
           sep = '\t', row.names = FALSE, col.names = TRUE, quote = FALSE)
  }
}


# DESWAN
pacman::p_load(data.table, dplyr, plyr, magrittr, stringr, DEswan, openxlsx)

# ==============================================================================
# Configuration & File Paths (Relative to project root)
# ==============================================================================
omics <- c('Metabolomic', 'Proteomic', 'Biochemistry')

# Directories (Matching the output of the previous script)
dir_omic_data     <- "./Data/BloodData/"
dir_zscore_input  <- "./RESULTS/multiomics_trajectory/diseasespan_trajectory/"
dir_deswan_output <- paste0("./RESULTS/multiomics_trajectory/DESWAN_Zscore/")

# Files
file_blood_dict   <- file.path(dir_omic_data, "BloodDict.csv")
file_disease      <- "./DATA/disease_filter_merge_1031_final.xlsx"

# Create output directory
dir.create(dir_deswan_output, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# Data Loading & Preprocessing
# ==============================================================================

# Load and clean Omics Dictionary
BloodDict <- fread(file_blood_dict) %>% as.data.frame()

# Load and filter disease endpoints
final_disease <- openxlsx::read.xlsx(file_disease)

# ==============================================================================
# DEswan Analysis 
# ==============================================================================
for (endpoint in final_disease$NAME) {
  
  # 1. Load and merge Z-score trajectories for all omics layers
  Zscore_omics <- list()
  for (omic in omics) {
    file_zscore <- file.path(dir_zscore_input, paste0("trajectories_Zscore_", endpoint, '_', omic, ".tsv"))
    
    t <- fread(file_zscore) %>% as.data.frame() 
    Zscore_omics[[omic]] <- t %>% 
      dplyr::select(eid, Year, all_of(BloodDict$Omics_feature[which(BloodDict$Omics_group == omic)]))
  }
  
  # Merge list into a single dataframe
  Zscore_omics <- Reduce(function(x, y) merge(x, y, by = c('eid', 'Year')), Zscore_omics) %>% 
    as.data.frame() %>% 
    mutate(across(-eid, as.numeric))
  
  # Print time distribution 
  print(quantile(Zscore_omics$Year, probs = seq(.0, 0.9, .1)))
  
  # 2. Run DEswan across varying parcel widths
  res_DEswan <- list()
  res_p <- list()
  
  for (parcel_width in 1:8) {
    character_parcel_width <- as.character(parcel_width)
    
    res_DEswan[[character_parcel_width]] <- DEswan(
      data.df = Zscore_omics[, BloodDict$Omics_feature], 
      qt = Zscore_omics[, 'Year'], 
      window.center = seq(-13, -3, 1), 
      buckets.size = parcel_width
    )
    
    res_p[[character_parcel_width]] <- res_DEswan[[character_parcel_width]] %>% 
      reshape.DEswan(parameter = 1, factor = "qt")
  }
  
  # Save DEswan results
  save(res_DEswan, res_p, file = file.path(dir_deswan_output, paste0(endpoint, ".RData")))
}
