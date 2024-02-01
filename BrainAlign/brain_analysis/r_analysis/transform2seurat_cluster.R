#@Time : 2023/7/23 21:23
#@Author : Biao Zhang
#@Email : littlebiao@outlook.com
#@File : transform2seurat.r
#@Description: This file is used to ...


library(Matrix)
library(Seurat)

library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%


adata_path_th <- "../../data/srrsc_mouse_human_binary/results_20_1000genes_all_came_selfloop/2023-06-23_20-31-14/embeds/figs/4_spatial_analysis/3_experiment_spatial_clusters/"

output_for_R_path <- "R_TH_mouse"

save_dir<- paste0(adata_path_th, output_for_R_path)

counts<-readMM(paste0(save_dir,'/counts.mtx'))
dim(counts)
cellMeta<-read.csv(paste0(save_dir,'/counts_cellMeta.csv'))
head(cellMeta)
geneMeta<-read.csv(paste0(save_dir,'/counts_geneMeta.csv'))
dim(geneMeta)
head(geneMeta)
### Set the rownames and colnames
rownames(counts)<-cellMeta$Barcode
colnames(counts)<-geneMeta$GeneName

seo <- CreateSeuratObject(counts = t(counts))
### Set the meta data
seo@meta.data<-cbind(cellMeta,seo@meta.data)
rownames(seo@meta.data)<-colnames(seo)
### Normalize the data
#seo <- NormalizeData(seo)
groupby <- "cluster_name_acronym"

Idents(seo) <- "cluster_name_acronym"

adata_markers <- FindAllMarkers(seo, only.pos = TRUE, logfc.threshold=0.11) #, min.pct = 0.25, logfc.threshold = 0.25
#adata_markers %>% group_by(cluster_name_acronym) %>% slice_max(n = 1, order_by = avg_log2FC)
#adata_markers %>% group_by("cluster_name_acronym") %>% top_n(2, avg_logFC)

print(adata_markers)

saveRDS(seo, file.path(save_dir,"mouse_th.rds"))



# ------------------------------------------------------------
# human----------------
print('----------------------------------human----------------------------------------------')
output_for_R_path <- "R_TH_human"

save_dir<- paste0(adata_path_th, output_for_R_path)

counts<-readMM(paste0(save_dir,'/counts.mtx'))
dim(counts)
cellMeta<-read.csv(paste0(save_dir,'/counts_cellMeta.csv'))
head(cellMeta)
geneMeta<-read.csv(paste0(save_dir,'/counts_geneMeta.csv'))
dim(geneMeta)
head(geneMeta)
### Set the rownames and colnames
rownames(counts)<-cellMeta$Barcode
colnames(counts)<-geneMeta$GeneName

seo <- CreateSeuratObject(counts = t(counts))
### Set the meta data
seo@meta.data<-cbind(cellMeta,seo@meta.data)
rownames(seo@meta.data)<-colnames(seo)
### Normalize the data
#seo <- NormalizeData(seo)
groupby <- "cluster_name_acronym"

Idents(seo) <- "cluster_name_acronym"


adata_markers <- FindAllMarkers(seo, only.pos = TRUE, logfc.threshold=0.15) #, min.pct=0.1, return.thresh=0.05, thresh.use = 0.15,
#adata_markers %>% group_by(cluster_name_acronym) %>% slice_max(n = 1, order_by = avg_log2FC)
#adata_markers %>% group_by("cluster_name_acronym") %>% top_n(2, avg_logFC)
print(adata_markers)

saveRDS(seo, file.path(save_dir,"human_th.rds"))
