#@Time : 2023/7/23 21:23
#@Author : Biao Zhang
#@Email : littlebiao@outlook.com
#@File : transform2seurat.r
#@Description: This file is used to ...


library(Matrix)
library(Seurat)



adata_path_isocortex <- "../../data/srrsc_mouse_human_binary/results_20_1000genes_all_came_selfloop/2023-06-23_20-31-14/embeds/figs/4_spatial_analysis/2_experiment_spatial_isocortex/"

output_for_R_path <- "R_isocortex_mouse"

save_dir<- paste0(adata_path_isocortex, output_for_R_path)

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

seo <- CreateSeuratObject(counts = t(counts), project = "min", min.cells = 2, min.features = 5)
### Set the meta data
seo@meta.data<-cbind(cellMeta,seo@meta.data)
rownames(seo@meta.data)<-colnames(seo)
### Normalize the data
#seo <- NormalizeData(seo)
groupby <- "region_name"

Idents(seo) <- "region_name"

adata_markers <- FindAllMarkers(seo, only.pos = TRUE, logfc.threshold=0.15) #, min.pct = 0.25, logfc.threshold = 0.25
#adata_markers %>% group_by(groupby) %>% slice_max(n = 1, order_by = avg_log2FC)
print(adata_markers)

saveRDS(seo, file.path(save_dir,'mouse_isocortex.rds'))



# ------------------------------------------------------------
# human----------------
output_for_R_path <- "R_isocortex_human"

save_dir<- paste0(adata_path_isocortex, output_for_R_path)

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

seo <- CreateSeuratObject(counts = t(counts), project = "min", min.cells = 2, min.features = 5)
### Set the meta data
seo@meta.data<-cbind(cellMeta,seo@meta.data)
rownames(seo@meta.data)<-colnames(seo)
### Normalize the data
#seo <- NormalizeData(seo)
groupby <- "region_name"

Idents(seo) <- "region_name"

adata_markers <- FindAllMarkers(seo, only.pos = TRUE, logfc.threshold=0.15) #, min.pct = 0.25, logfc.threshold = 0.25
#adata_markers %>% group_by(groupby) %>% slice_max(n = 1, order_by = avg_log2FC)
print(adata_markers)

saveRDS(seo, file.path(save_dir,'human_isocortex.rds'))