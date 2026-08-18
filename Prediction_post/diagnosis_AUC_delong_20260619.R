# Load necessary packages
pacman::p_load(data.table, openxlsx, survival, pROC, dplyr)

# ==============================================================================
# Configuration
# ==============================================================================
type <- 'Prevalent'

# Define omics model names (matching folder names)
vec_omics <- c(
  'Biochemistry', 'Metabolome', 'Proteome', 'Genome', 'MultiOmics',
  'Biochemistry_Cov', 'Metabolome_Cov', 'Proteome_Cov',
  'Genome_Cov', 'Cov', 'MultiOmicsCov'
)

# Directories
dir_base_results <- "./Results/"
dir_pred_input   <- paste0(dir_base_results, type)
dir_auc_out      <- paste0("./RESULTS/Prediction/v2_20260530/", type, "/AUC/")
dir_endpoints    <- "./UKB_Disease/"
file_disease     <- "./Multiomics/merged_disease_listV2_250316.xlsx"

dir.create(dir_auc_out, recursive = TRUE, showWarnings = FALSE)

# Load target diseases
sheet_num <- ifelse(type == 'Incident', 1, 2)
final_disease <- openxlsx::read.xlsx(file_disease, sheet = sheet_num)
disease_name_list <- final_disease$NAME

outcome_status <- 'target_y'
outcome_years  <- 'BL2Target_yrs'


# ==============================================================================
# DeLong's Test Loop
# ==============================================================================
lapply(disease_name_list, function(endpoint) {
  
  cat("\nProcessing endpoint:", endpoint, "\n")
  
  # --------------------------------------------------------------------------
  # 1. Check available models
  # --------------------------------------------------------------------------
  pred_paths <- paste0(
    dir_pred_input, vec_omics, '/', endpoint, "/Pred_prob.csv"
  )
  
  exists_flag <- file.exists(pred_paths)
  var4compare <- vec_omics[exists_flag]
  
  # DeLong requires at least two models
  if (length(var4compare) < 2) {
    cat("  Fewer than 2 models available. Skip DeLong test.\n")
    return(NULL)
  }
  
  
  # --------------------------------------------------------------------------
  # 2. Load outcome data
  # --------------------------------------------------------------------------
  outcome <- as.data.frame(
    fread(file.path(dir_endpoints, paste0(endpoint, '.csv')))
  )
  
  outcome_v1 <- if (type == 'Incident') {
    outcome[outcome[[outcome_years]] > 0, ]
  } else {
    outcome[outcome[[outcome_years]] <= 0, ]
  }
  
  
  # --------------------------------------------------------------------------
  # 3. Load prediction probabilities
  # --------------------------------------------------------------------------
  pred_prob_list <- lapply(var4compare, function(model) {
    
    tmp <- as.data.frame(
      fread(
        paste0(
          dir_pred_input,
          model,
          '/',
          endpoint,
          "/Pred_prob.csv"
        )
      )
    )
    
    tmp <- tmp[, c("eid", "Logits")]
    names(tmp)[2] <- model
    
    return(tmp)
  })
  
  
  # --------------------------------------------------------------------------
  # 4. Merge prediction probabilities and outcome data
  # --------------------------------------------------------------------------
  pred_prob <- Reduce(
    function(x, y) merge(x, y, by = "eid"),
    pred_prob_list
  )
  
  pred_prob <- merge(
    pred_prob,
    outcome_v1[, c('eid', outcome_status, outcome_years)],
    by = 'eid'
  )
  
  
  # ==========================================================================
  # 5. Generate ROC objects for each model
  # ==========================================================================
  fit_test <- list()
  
  for (model in var4compare) {
    
    cat("  Generating ROC:", model, endpoint, "\n")
    
    fit_test[[model]] <- pROC::roc(
      response  = pred_prob[[outcome_status]],
      predictor = pred_prob[[model]],
      ci         = TRUE,
      auc        = TRUE,
      quiet      = TRUE
    )
  }
  
  
  # ==========================================================================
  # 6. Pairwise Comparison: DeLong's Test
  # ==========================================================================
  results_delong <- list()
  
  for (i in 1:(length(var4compare) - 1)) {
    
    for (j in (i + 1):length(var4compare)) {
      
      m1 <- var4compare[i]
      m2 <- var4compare[j]
      
      roc_test_res <- pROC::roc.test(
        fit_test[[m1]],
        fit_test[[m2]],
        method = "delong",
        paired = TRUE
      )
      
      results_delong[[length(results_delong) + 1]] <- data.frame(
        Phenotype = endpoint,
        Model1     = m1,
        Model2     = m2,
        AUC_Model1 = as.numeric(pROC::auc(fit_test[[m1]])),
        AUC_Model2 = as.numeric(pROC::auc(fit_test[[m2]])),
        Zvalue     = if (!is.null(roc_test_res$statistic)) {
          as.numeric(roc_test_res$statistic)
        } else {
          NA_real_
        },
        Pvalue     = roc_test_res$p.value,
        stringsAsFactors = FALSE
      )
    }
  }
  
  
  # --------------------------------------------------------------------------
  # 7. Save results
  # --------------------------------------------------------------------------
  if (length(results_delong) > 0) {
    
    df_delong <- do.call(rbind, results_delong)
    
    fwrite(
      df_delong,
      file = file.path(
        dir_auc_out,
        paste0('delong_', endpoint, '.tsv')
      ),
      sep = '\t',
      quote = FALSE,
      row.names = FALSE
    )
  }
  
  return(TRUE)
})