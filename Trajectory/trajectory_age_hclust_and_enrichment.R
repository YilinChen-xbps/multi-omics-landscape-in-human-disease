# Load necessary packages
pacman::p_load(data.table, dplyr, plyr, magrittr, tidyr, tibble)

# Configuration parameters
span <- 0.5
omics <- c("Metabolomic", "Proteomic", "Biochemistry")
endpoint <- 'age'
optimal_clusters <- 9

# Set relative data and output paths
# Ensure these folders exist relative to your project working directory
omic_filepath <- './Data/BloodData/' 
loess_dat_path <- paste0('./Data/LoessFit/AgeSpan/LoessFit_', span, '/')
output_dir <- './RESULTS/multiomics_trajectory/hclust_Cluster_age/'

print(paste("Endpoint:", endpoint))

# Load data dictionary
BloodDict <- file.path(omic_filepath, 'BloodDict.csv') %>% 
  fread() %>% 
  as.data.frame()

# Initialize list for Loess fit data
loess_omics <- list()

# Read and process Loess fit data for each omic layer
for (omic in omics) {
  df_t <- paste0(loess_dat_path, omic, '_alpha1.csv') %>% 
    fread() %>% 
    as.data.frame() %>% 
    select(Year = timeline, paste0(BloodDict$Omics_feature[which(BloodDict$Omics_group == omic)], '_loess_fit')) %>% 
    magrittr::set_colnames(gsub('_loess_fit', '', colnames(.)))
  
  df_loess <- df_t %>% 
    tidyr::pivot_longer(
      cols = BloodDict$Omics_feature[which(BloodDict$Omics_group == omic)], 
      names_to = "Characteristics", 
      values_to = 'Estimate_loess'
    ) 
  
  loess_omics[[omic]] <- df_loess
}

# Combine all omics data
loess_omics <- plyr::rbind.fill(loess_omics)

# Convert to wide-format matrix for clustering
df_wide <- loess_omics %>% 
  tidyr::pivot_wider(names_from = Characteristics, values_from = Estimate_loess) %>% 
  tibble::column_to_rownames("Year") %>% 
  t() %>% 
  as.matrix()

# Setup output directory
cluster_path <- file.path(output_dir, span)
dir.create(cluster_path, showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# 1. Hierarchical Clustering (hclust)
# ==============================================================================

# Compute distance matrix and apply Ward's method
dist_omics <- dist(df_wide, method = "euclidean")  
fit_omics <- hclust(dist_omics, method = "ward.D") 

# Cut tree into defined optimal clusters
cluster_assign <- cutree(fit_omics, k = optimal_clusters)

# Record cluster assignment for each omics feature
cluster_info <- data.frame(
  Omics_feature = rownames(df_wide),
  cluster = cluster_assign,
  stringsAsFactors = FALSE
)

# Calculate the mean trajectory (center) for each cluster
cluster_center <- lapply(1:optimal_clusters, function(cl) {
  subset_data <- df_wide[cluster_info$cluster == cl, , drop = FALSE]
  center_vals <- colMeans(subset_data)
  
  data.frame(
    cluster = cl,
    time = as.numeric(colnames(df_wide)),
    value = center_vals,
    stringsAsFactors = FALSE
  )
})

# Reshape cluster-specific feature data to long format for visualization
cluster_specific <- lapply(1:optimal_clusters, function(cl) {
  tmp_features <- cluster_info$Omics_feature[cluster_info$cluster == cl]
  tmp <- df_wide[tmp_features, , drop = FALSE]
  
  tmp_long <- as.data.frame(tmp) %>%
    tibble::rownames_to_column(var = "Omics_feature") %>%
    tidyr::pivot_longer(cols = -Omics_feature, names_to = "time", values_to = "value") %>%
    dplyr::mutate(
      time = as.numeric(time),
      cluster = cl,
      membership = 1 
    ) %>%
    dplyr::left_join(BloodDict[, c("Omics_feature", "Omics_group")], by = "Omics_feature") %>%
    dplyr::arrange(desc(Omics_group)) %>%
    dplyr::mutate(Omics_feature = factor(Omics_feature, levels = unique(Omics_feature)))
  
  return(tmp_long)
})

# Save clustering results
save(cluster_info, cluster_specific, cluster_center, 
     file = file.path(cluster_path, paste0(endpoint, "_Cluster_", optimal_clusters, ".RData")))




# Load necessary packages
pacman::p_load(dplyr, tidyr, data.table, ReactomePA, clusterProfiler, org.Hs.eg.db)

# Configuration parameters
span <- '0.5'
endpoint <- 'age'
optimal_clusters <- 9

# Set relative paths
# Ensure directory structure matches the output of the previous script
cluster_input_dir <- './RESULTS/multiomics_trajectory/hclust_Cluster_age/'
output_dir <- './RESULTS/Enrichment/'

# Load clustering results
load(file.path(cluster_input_dir, span, paste0(endpoint, "_Cluster_", optimal_clusters, ".RData")))

# Load and preprocess blood dictionary
BloodDict <- fread('./Data/BloodData/BloodDict_v2.csv') %>% as.data.frame()
new_names <- BloodDict$Omics_feature
new_names <- ifelse(grepl("-0.0", new_names), paste0("X", gsub("-0.0", "", new_names)), new_names)
BloodDict <- BloodDict %>% dplyr::mutate(Omics_feature = gsub('-', '_', new_names))

# Prepare gene list for enrichment per cluster
final_cluster_info <- cluster_info %>% 
  merge(BloodDict, by = 'Omics_feature', all.x = TRUE) %>% 
  filter(Omics_group == 'Proteome') %>% 
  separate_rows(Omics_code, sep = "[_]") %>% 
  mutate(Omics_code = trimws(Omics_code))

universe_geneList <- bitr(geneID = unique(final_cluster_info$Omics_code), 
                          fromType = "SYMBOL", toType = c("ENTREZID"), OrgDb = org.Hs.eg.db) %>% 
  select(Omics_code = SYMBOL, everything())

final_cluster_info <- final_cluster_info %>% 
  merge(universe_geneList, by = 'Omics_code', all.x = TRUE) %>% 
  filter(!is.na(ENTREZID)) %>% 
  group_by(cluster) %>% 
  summarise(genes = list(unique(ENTREZID)))

clusters <- setNames(final_cluster_info$genes, paste0("Cluster", final_cluster_info$cluster))

# Initialize result containers
merge_enrich_reactome <- NULL
merge_enrich_GO <- NULL

# Run enrichment analysis
for (i in seq_along(clusters)) {
  current_genes <- clusters[[i]]
  
  if (length(current_genes) < 3) next
  
  # Reactome enrichment
  proteomics_reactome <- enrichPathway(gene = current_genes,
                                       organism = "human",
                                       pAdjustMethod = "fdr",
                                       pvalueCutoff = 0.05,
                                       qvalueCutoff = 0.05,
                                       readable = TRUE)
  
  if (!is.null(proteomics_reactome)) {
    e <- proteomics_reactome@result %>%
      mutate(cluster = names(clusters[i]), Phenotype = 'Lifespan')
    merge_enrich_reactome <- rbind(merge_enrich_reactome, e)
  }
  
  # GO enrichment
  ego <- enrichGO(gene = current_genes, 
                  keyType = "ENTREZID",
                  OrgDb = org.Hs.eg.db,
                  ont = "ALL",
                  pAdjustMethod = 'fdr', 
                  qvalueCutoff = 0.05, 
                  pvalueCutoff = 0.05, 
                  minGSSize = 1,
                  readable = TRUE)
  
  if (!is.null(ego)) {
    ego_simplified <- simplify(ego, cutoff = 0.7, by = "p.adjust", select_fun = min)
    go <- data.frame(ego_simplified) %>% 
      mutate(cluster = names(clusters[i]), Phenotype = 'Lifespan')
    merge_enrich_GO <- rbind(merge_enrich_GO, go)
  }
}

# Save results
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

if (!is.null(merge_enrich_reactome)) {
  fwrite(merge_enrich_reactome %>% select(Phenotype, cluster, everything()),
         file = file.path(output_dir, paste0('Reactome_span', span, '_', optimal_clusters, '.tsv')),
         sep = '\t', col.names = TRUE, row.names = FALSE, quote = FALSE, na = NA)
}

if (!is.null(merge_enrich_GO)) {
  fwrite(merge_enrich_GO %>% select(Phenotype, cluster, everything()),
         file = file.path(output_dir, paste0('GO_span', span, '_', optimal_clusters, '.tsv')),
         sep = '\t', col.names = TRUE, row.names = FALSE, quote = FALSE, na = NA)
}