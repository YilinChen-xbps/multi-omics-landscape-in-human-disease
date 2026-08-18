# Load necessary packages
pacman::p_load(data.table, openxlsx, survival, dplyr, stringr)

# ==============================================================================
# Configuration & File Paths (Relative to project root)
# ==============================================================================
# Define omics model names
vec_omics <- c('Biochemistry', 'Metabolome', 'Proteome', 'Genome', 'MultiOmics', 'MultiOmicsCov')

# Directories
dir_base_results <- "./Results/"
dir_endpoints    <- "./UKB_Disease/"
dir_summary_out  <- "./RESULTS/Prediction/"

# Files
file_disease     <- "./Multiomics/merged_disease_listV2_250316.xlsx"
file_chapter     <- "./DATA/disease_chapter_mapping.csv"

outcome_status <- 'target_y'
outcome_years  <- 'BL2Target_yrs'

# ==============================================================================
# Data Preprocessing: Endpoint & Chapter Mapping
# ==============================================================================
inc_endpoints <- openxlsx::read.xlsx(file_disease, sheet = 1) %>% select(NAME, LONGNAME, Chapter)
prev_endpoints <- openxlsx::read.xlsx(file_disease, sheet = 2) %>% select(NAME, LONGNAME, Chapter)

endpointlist_raw <- rbind(inc_endpoints, prev_endpoints)
endpointlist_dist <- distinct(endpointlist_raw, NAME, .keep_all = TRUE) 

chapter_mapping <- fread(file_chapter) %>% 
  as.data.frame() %>% 
  arrange(chapter_code) %>% 
  select(1:3)

endpointlist <- merge(endpointlist_dist, chapter_mapping, by = 'Chapter')
merge_cols <- c('Chapter', 'NAME', 'LONGNAME')
endpointlist_sub <- endpointlist[, merge_cols, drop = FALSE]

# ==============================================================================
# 1. Compute HR (Incident Data - Cox Proportional Hazards)
# ==============================================================================
cat("\n--- Starting Incident Analysis (HR) ---\n")
type_inc <- 'Incident'
dir_pred_inc <- paste0(dir_base_results, type_inc)
out_dir_inc <- paste0(dir_summary_out, type_inc, '/summary/')
dir.create(out_dir_inc, recursive = TRUE, showWarnings = FALSE)

disease_names_inc <- inc_endpoints$NAME

res_list_inc <- lapply(disease_names_inc, function(endpoint) {
  
  pred_paths <- paste0(dir_pred_inc, vec_omics, '/', endpoint, "/Pred_prob.csv")
  models_to_run <- vec_omics[file.exists(pred_paths)]
  
  if (length(models_to_run) == 0) return(NULL)
  
  # Load and filter outcome data (> 0 years for Incident)
  outcome_df <- as.data.frame(fread(file.path(dir_endpoints, paste0(endpoint, '.csv'))))
  outcome_v1 <- outcome_df[outcome_df[[outcome_years]] > 0, , drop = FALSE]
  
  # Load predictions, split into tertiles
  pred_prob_list <- lapply(models_to_run, function(model) {
    tmp <- as.data.frame(fread(paste0(dir_pred_inc, model, '/', endpoint, "/Pred_prob.csv")))
    tmp$group3 <- factor(ntile(tmp$Logits, 3), levels = 1:3)
    tmp$Phenotype <- endpoint
    tmp$Model <- model
    return(tmp[, c("Phenotype", "Model", "eid", "group3")])
  })
  
  # Merge predictions with outcomes
  pred_prob <- do.call(rbind, pred_prob_list)
  pred_prob <- merge(pred_prob, outcome_v1[, c('eid', outcome_status, outcome_years)], by = 'eid')
  
  # Fit Cox models
  model_results <- lapply(unique(pred_prob$Model), function(m) {
    sub_data <- pred_prob[pred_prob$Model == m, ]
    fml <- as.formula(paste0('Surv(', outcome_years, ', ', outcome_status, ' == 1) ~ group3'))
    sfit <- summary(coxph(fml, data = sub_data))
    
    # Extract metrics for group 2 (Middle) and group 3 (High)
    data.frame(
      Characteristics = c("Middle", "High"),
      nevent_N = sprintf("%s / %s", format(sfit$nevent, big.mark=","), format(sfit$n, big.mark=",")),
      Pvalue = sfit$coefficients[, 5],
      HR = sfit$coefficients[, 2],
      LCI = sfit$conf.int[, 3],
      UCI = sfit$conf.int[, 4],
      Phenotype = endpoint,
      Model = m,
      stringsAsFactors = FALSE
    )
  })
  
  return(do.call(rbind, model_results))
})

# Finalize and export Incident HR summary
df_hr_all <- do.call(rbind, res_list_inc)
df_hr_all <- merge(df_hr_all, endpointlist_sub, by.x = 'Phenotype', by.y = 'NAME', all.x = TRUE)

final_cols_inc <- c("Phenotype", "Chapter", "LONGNAME", "Model", "Characteristics", 
                    "nevent_N", "HR", "LCI", "UCI", "Pvalue")
openxlsx::write.xlsx(df_hr_all[, final_cols_inc], file = paste0(out_dir_inc, 'ST_3group_HR.xlsx'), asTable = FALSE)


# ==============================================================================
# 2. Compute OR (Prevalent Data - Logistic Regression)
# ==============================================================================
cat("\n--- Starting Prevalent Analysis (OR) ---\n")
type_prev <- 'Prevalent'
dir_pred_prev <- paste0(dir_base_results, type_prev)
out_dir_prev <- paste0(dir_summary_out, type_prev, '/summary/')
dir.create(out_dir_prev, recursive = TRUE, showWarnings = FALSE)

disease_names_prev <- prev_endpoints$NAME

res_list_prev <- lapply(disease_names_prev, function(endpoint) {
  
  pred_paths <- paste0(dir_pred_prev, vec_omics, '/', endpoint, "/Pred_prob.csv")
  models_to_run <- vec_omics[file.exists(pred_paths)]
  
  if (length(models_to_run) == 0) return(NULL)
  
  # Load and filter outcome data (<= 0 years for Prevalent)
  outcome_df <- as.data.frame(fread(file.path(dir_endpoints, paste0(endpoint, '.csv'))))
  outcome_v1 <- outcome_df[outcome_df[[outcome_years]] <= 0, , drop = FALSE]
  
  # Load predictions, split into tertiles
  pred_prob_list <- lapply(models_to_run, function(model) {
    tmp <- as.data.frame(fread(paste0(dir_pred_prev, model, '/', endpoint, "/Pred_prob.csv")))
    tmp$group3 <- factor(ntile(tmp$Logits, 3), levels = 1:3)
    tmp$Phenotype <- endpoint
    tmp$Model <- model
    return(tmp[, c("Phenotype", "Model", "eid", "group3")])
  })
  
  # Merge predictions with outcomes
  pred_prob <- do.call(rbind, pred_prob_list)
  pred_prob <- merge(pred_prob, outcome_v1[, c('eid', outcome_status, outcome_years)], by = 'eid')
  
  # Fit Logistic models
  model_results <- lapply(unique(pred_prob$Model), function(m) {
    sub_data <- pred_prob[pred_prob$Model == m, ]
    fml <- as.formula(paste(outcome_status, '~ group3'))
    
    fit_model <- glm(fml, data = sub_data, family = binomial(link = "logit"))
    sfit <- summary(fit_model)
    
    odds_ratios <- exp(coef(fit_model))
    conf_int_exp <- suppressMessages(exp(confint(fit_model))) 
    
    nevent <- sum(sub_data[[outcome_status]] == 1)
    ntotal <- nrow(sub_data)
    
    # Extract metrics for group 2 (Middle) and group 3 (High), skipping the intercept
    data.frame(
      Characteristics = c("Middle", "High"),
      nevent_N = sprintf("%s / %s", format(nevent, big.mark=","), format(ntotal, big.mark=",")),
      Pvalue = sfit$coefficients[2:3, 4],
      OR = odds_ratios[2:3],
      LCI = conf_int_exp[2:3, 1],
      UCI = conf_int_exp[2:3, 2],
      Phenotype = endpoint,
      Model = m,
      stringsAsFactors = FALSE
    )
  })
  
  return(do.call(rbind, model_results))
})

# Finalize and export Prevalent OR summary
df_or_all <- do.call(rbind, res_list_prev)
df_or_all <- merge(df_or_all, endpointlist_sub, by.x = 'Phenotype', by.y = 'NAME', all.x = TRUE)

final_cols_prev <- c("Phenotype", "Chapter", "LONGNAME", "Model", "Characteristics", 
                     "nevent_N", "OR", "LCI", "UCI", "Pvalue")
openxlsx::write.xlsx(df_or_all[, final_cols_prev], file = paste0(out_dir_prev, 'ST_3group_OR.xlsx'), asTable = FALSE)
