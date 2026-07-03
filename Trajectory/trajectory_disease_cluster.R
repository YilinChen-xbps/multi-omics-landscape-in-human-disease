
library(Seurat)
library(tidyverse)
library(qs)
# df <- t(alldata$mutliomics[,-1])
# head(df)[1:4,1:4]
sce.all <- CreateSeuratObject(counts = (alldata$mutilomic2))
sce.all

library(Seurat)
library(tidyverse)
library(qs)
sce.all <- qread("LoessFit_alpha1/sce.all.qs")

sce.all$Col2 <- factor(sce.all$Col2,levels = seq(0, 15, by = 0.5))

sce.all=sce.all%>%
  #NormalizeData()%>%
  FindVariableFeatures(nfeatures = 2000)%>% #
  ScaleData(layer="counts")%>% 
  RunPCA()%>%
  FindNeighbors(dims=1:15)%>%
  FindClusters(resolution=3)%>% 
  RunUMAP(dims=1:15 ,min.dist = 0.3,spread = 3,
          n.neighbors = 50,
          metric = "correlation"
  ) 
sce.all <- RunTSNE(sce.all,dims=1:10)

