# Load necessary packages
pacman::p_load(data.table, openxlsx, survival, pROC, dplyr, compareC)

# ==============================================================================
# Configuration
# ==============================================================================
type <- 'Incident'

# Define omics model names
vec_omics <- c('Biochemistry', 'Metabolome', 'Proteome', 'Genome', 'MultiOmics', 
               'Biochemistry_Cov', 'Metabolome_Cov', 'Proteome_Cov', 'Genome_Cov', 'Cov', 'MultiOmicsCov')

# Directories
dir_base_results <- "./Results/"
dir_pred_input   <- paste0(dir_base_results, type)
dir_compareC     <- paste0("./RESULTS/Prediction/", type, "/compareC/")
dir_endpoints    <- "./UKB_Disease/"
file_disease     <- "./Multiomics/merged_disease_listV2_250316.xlsx"

dir.create(dir_compareC, recursive = TRUE, showWarnings = FALSE)

# Load target diseases
sheet_num <- ifelse(type == 'Incident', 1, 2)
final_disease <- openxlsx::read.xlsx(file_disease, sheet = sheet_num)
disease_name_list <- final_disease$NAME

outcome_status <- 'target_y'
outcome_years  <- 'BL2Target_yrs'

# ==============================================================================
# C-index Calculation Loop
# ==============================================================================
lapply(disease_name_list, function(endpoint) {
  
  # Check which models exist for the current endpoint
  pred_paths <- paste0(dir_pred_input, vec_omics, '/', endpoint, "/Pred_prob.csv")
  exists_flag <- file.exists(pred_paths)
  var4compare <- vec_omics[exists_flag]
  
  if (length(var4compare) < 2) return(NULL)
  
  # Load outcome data
  outcome <- as.data.frame(fread(file.path(dir_endpoints, paste0(endpoint, '.csv'))))
  outcome_v1 <- if (type == 'Incident') outcome[outcome[[outcome_years]] > 0, ] else outcome[outcome[[outcome_years]] <= 0, ]
  
  # Load prediction probabilities for available models
  pred_prob_list <- lapply(var4compare, function(model) {
    tmp <- as.data.frame(fread(paste0(dir_pred_input, model, '/', endpoint, "/Pred_prob.csv")))
    tmp <- tmp[, c("eid", "Logits")]
    names(tmp)[2] <- model
    return(tmp)
  })
  
  # Merge all prediction probabilities and outcome data
  pred_prob <- Reduce(function(x, y) merge(x, y, by = "eid"), pred_prob_list)
  pred_prob <- merge(pred_prob, outcome_v1[, c('eid', outcome_status, outcome_years)], by = 'eid')
  
  # Pairwise comparison
  results <- list()
  for (i in 1:(length(var4compare) - 1)) {
    for (j in (i + 1):length(var4compare)) {
      
      m1 <- var4compare[i]
      m2 <- var4compare[j]
      
      fitC <- compareC(pred_prob[[outcome_years]], pred_prob[[outcome_status]], 
                       pred_prob[[m1]], pred_prob[[m2]])
      
      results[[length(results) + 1]] <- data.frame(
        Phenotype = endpoint,
        Model1 = m1,
        Model2 = m2,
        Zvalue = fitC$zscore,
        Pvalue = fitC$pval
      )
    }
  }
  
  # Save comparative results
  final_df <- do.call(rbind, results)
  fwrite(final_df, file = file.path(dir_compareC, paste0(endpoint, '.tsv')), 
         sep = '\t', quote = FALSE, row.names = FALSE)
  
  return(TRUE)
})