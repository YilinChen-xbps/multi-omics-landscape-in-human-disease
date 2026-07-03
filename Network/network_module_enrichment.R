# Load necessary packages
pacman::p_load(dplyr, tidyr, data.table, ReactomePA, clusterProfiler, org.Hs.eg.db)

# ==============================================================================
# Configuration & File Paths (Relative to project root)
# ==============================================================================
network_version <- 'Louvain'

# Directories
dir_network_base  <- "./RESULTS/Network/"
dir_summary       <- paste0(dir_network_base, "summary/", network_version, "/")
dir_enrichment    <- paste0(dir_network_base, "Enrichment/", network_version, "/")

# Files
file_node_enrich  <- paste0(dir_summary, "Network_node_summary.tsv")

# Create output directory
dir.create(dir_enrichment, recursive = TRUE, showWarnings = FALSE)

# ==============================================================================
# Data Loading & ID Mapping
# ==============================================================================
# Load and clean network endpoints
network_endpoints <- fread(file_node_enrich) %>% 
  as.data.frame() %>% 
  filter(Omics_group %in% c('WGS_SNP', 'WGS_gene', 'Proteome')) %>%
  separate_rows(Omics4enrichment, sep = "[;_]") %>%
  mutate(Omics4enrichment = trimws(Omics4enrichment))

# Separate genes by ENSEMBL vs SYMBOL prefixes for proper bitr mapping
idx_ensg <- grep("^ENSG", network_endpoints$Omics4enrichment)
ensg_ids <- unique(network_endpoints$Omics4enrichment[idx_ensg])
sym_ids  <- unique(network_endpoints$Omics4enrichment[-idx_ensg])

# Map ENSEMBL to ENTREZID
ensg_geneList <- bitr(geneID = ensg_ids, 
                      fromType = "ENSEMBL", toType = "ENTREZID", OrgDb = org.Hs.eg.db) %>% 
  select(Omics4enrichment = ENSEMBL, everything())

# Map SYMBOL to ENTREZID and combine
universe_geneList <- bitr(geneID = sym_ids, 
                          fromType = "SYMBOL", toType = "ENTREZID", OrgDb = org.Hs.eg.db) %>% 
  select(Omics4enrichment = SYMBOL, everything()) %>% 
  rbind(ensg_geneList)

# Merge mapped IDs back to endpoint data and group by module
network_mapped <- network_endpoints %>% 
  merge(universe_geneList, by = 'Omics4enrichment', all.x = TRUE) %>% 
  filter(!is.na(ENTREZID)) %>% 
  group_by(Phenotype, module) %>% 
  summarise(genes = list(unique(ENTREZID)), .groups = 'drop')

final_disease <- unique(network_mapped$Phenotype)

# ==============================================================================
# Enrichment Analysis Loop (Reactome & GO)
# ==============================================================================
for (i in seq_along(final_disease)) {
  endpoint <- final_disease[i]
  
  # Extract modules for current endpoint
  network_endpoint <- network_mapped %>% filter(Phenotype == endpoint)
  modules <- setNames(network_endpoint$genes, paste0("MFM", network_endpoint$module))
  
  genes_all <- unique(unlist(modules))
  if (length(genes_all) < 3) next
  
  # Initialize lists for accumulating results
  list_reactome <- list()
  list_go <- list()
  
  for (mod_name in names(modules)) {
    mod_genes <- modules[[mod_name]]
    if (length(mod_genes) < 3) next
    
    # --------------------------------------------------------------------------
    # 1. Reactome Pathway Enrichment
    # --------------------------------------------------------------------------
    proteomics_reactome <- enrichPathway(gene = mod_genes,
                                         organism = "human",
                                         pAdjustMethod = "fdr",
                                         pvalueCutoff = 0.05,
                                         qvalueCutoff = 0.05,
                                         readable = TRUE)
    
    if (!is.null(proteomics_reactome)) {
      e <- proteomics_reactome@result %>%
        mutate(module = mod_name, Phenotype = endpoint)
      list_reactome[[length(list_reactome) + 1]] <- e
    }
    
    # --------------------------------------------------------------------------
    # 2. Gene Ontology (GO) Enrichment
    # --------------------------------------------------------------------------
    ego <- enrichGO(gene = mod_genes, 
                    keyType = "ENTREZID",
                    OrgDb = org.Hs.eg.db,
                    ont = "ALL",
                    pAdjustMethod = 'fdr',  
                    qvalueCutoff = 0.05, 
                    pvalueCutoff = 0.05, 
                    minGSSize = 1,
                    readable = TRUE)
    
    if (!is.null(ego)) {
      # Simplify redundant GO terms
      ego2 <- simplify(ego, cutoff = 0.7, by = "p.adjust", select_fun = min)
      go_df <- data.frame(ego2) %>% 
        mutate(module = mod_name, Phenotype = endpoint)
      list_go[[length(list_go) + 1]] <- go_df
    }
  }
  
  # ============================================================================
  # Save Aggregated Results
  # ============================================================================
  
  # Save Reactome results
  if (length(list_reactome) > 0) {
    df_reactome_final <- do.call(rbind, list_reactome) %>% select(Phenotype, module, everything())
    fwrite(df_reactome_final, 
           file = file.path(dir_enrichment, paste0('Reactome_', endpoint, '.tsv')),
           sep = '\t', col.names = TRUE, row.names = FALSE, quote = FALSE, na = NA)
  } else {
    cat('  - No Reactome pathway enriched\n')
  }
  
  # Save GO results
  if (length(list_go) > 0) {
    df_go_final <- do.call(rbind, list_go) %>% select(Phenotype, module, everything())
    fwrite(df_go_final, 
           file = file.path(dir_enrichment, paste0('GO_', endpoint, '.tsv')),
           sep = '\t', col.names = TRUE, row.names = FALSE, quote = FALSE, na = NA)
  } else {
    cat('  - No GO pathway enriched\n')
  }
}
