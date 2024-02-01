#@Time : 2023/7/22 20:38
#@Author : Biao Zhang
#@Email : littlebiao@outlook.com
#@File : genomic_findmarkers.r
#@Description: This file is used to find marker genes with Seurat findMarker
#source('./genomic_functions.R', local = TRUE)

library(sceasy)
library(anndata)
library(Seurat)
library(Matrix)

library(reticulate)
#use_condaenv('pad')
#loompy <- reticulate::import('loompy')

print("No error")

# convert_scanpy_seurat
convert_scanpy2seurat <- function(adata_path, save_path){
 #ad <- zellkonverter::readH5AD(adata_path)
  #ad <- zellkonverter::readH5AD(adata_path)
  #ad <- read_h5ad(adata_path)
 # end with .rds file
  #print(ad)
 sceasy::convertFormat(adata_path, from="anndata", to="seurat", outFile=save_path)
 #if_success <- TRUE
 #return(if_success)
}

seurat_findmarker <- function(input_data, groupby)
{
  adata_markers <- FindAllMarkers(input_data, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
  adata_markers %>% group_by(groupby) %>% slice_max(n = 1, order_by = avg_log2FC)
  #return(NULL)
}


# isocortex
adata_mouse_path_isocortex <- "D:/Research_programs/HeCo/BrainAlign/data/srrsc_mouse_human_binary/results_20_1000genes_all_came_selfloop/2023-06-23_20-31-14/embeds/figs/4_spatial_analysis/2_experiment_spatial_isocortex/adata_mouse_exp_isocortex.h5ad"
save_path <- "D:/Research_programs/HeCo/BrainAlign/data/srrsc_mouse_human_binary/results_20_1000genes_all_came_selfloop/2023-06-23_20-31-14/embeds/figs/4_spatial_analysis/2_experiment_spatial_isocortex/"
mouse_data_path <- paste0(save_path, 'adata_mouse_path_isocortex.rds')
print(mouse_data_path)
adata <- read_h5ad(adata_mouse_path_isocortex)
print(adata)

#adata$X <- as(adata$X, "dgCMatrix")

#print(adata$X)
#seurat_object <- Seurat::Convert(adata)
# Create a Seurat object
seurat_object <- CreateSeuratObject(counts = adata$X)

# Transfer feature names (genes)
rownames(seurat_object) <- rownames(adata$X)

# Transfer cell names and metadata
colnames(seurat_object) <- colnames(adata$X)
seurat_object$meta.data <- adata$obs
# Transfer additional metadata columns (obs)
seurat_object$meta.data$additional_metadata_column <- adata$obs$additional_metadata_column

# Transfer additional metadata columns (var)
seurat_object$var$additional_metadata_column <- adata$var$additional_metadata_column


saveRDS(seurat_object, mouse_data_path)

#convert_scanpy2seurat(adata_mouse_path_isocortex, mouse_data_path) #mouse_data_path
mouse_isocortex_data <- readRDS(file = mouse_data_path)
Idents(mouse_isocortex_dat) <- "region_name"
seurat_findmarker(mouse_isocortex_data, groupby="region_name")

print("Mouse finished")

# adata_human_path_isocortex <- "../../data/srrsc_mouse_human_binary/results_20_1000genes_all_came_selfloop/2023-06-23_20-31-14/embeds/figs/4_spatial_analysis/2_experiment_spatial_isocortex/adata_mouse_exp_isocortex.h5ad"
# human_data_path <- paste0(save_path, 'adata_human_path_isocortex.rds')
# convert_scanpy2seurat(adata_human_path_isocortex, paste(save_path, human_data_path))
# human_isocortex_data <- readRDS(file = human_data_path)
# Idents(human_isocortex_dat) <- "region_name"
# seurat_findmarker(human_isocortex_data, groupby="region_name")
#
# print("Human finished.")

