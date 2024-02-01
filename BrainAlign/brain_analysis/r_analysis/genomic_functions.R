#@Time : 2023/7/23 11:09
#@Author : Biao Zhang
#@Email : littlebiao@outlook.com
#@File : genomic_functions.r
#@Description: This file is used to ...
#source('./genomic_functions.R', local = TRUE)

library(Seurat)
library(anndata)
library(sceasy)

# convert_scanpy_seurat
if_success <- convert_scanpy2seurat(adata_path, save_path)
{
 ad <- anndata::read_h5ad(adata_path)
 # end with .rds file
 sceasy::convertFormat(ad, from="anndata", to="seurat", outFile=save_path)
 if_success <- TRUE
 #return(if_success)
}


NULL <- seurat_findmarker(input_data, groupby)
{
  adata_markers <- FindAllMarkers(input_data, only.pos = TRUE, min.pct = 0.25, logfc.threshold = 0.25)
  adata_markers %>% group_by(groupby) %>% slice_max(n = 1, order_by = avg_log2FC)
  #return(NULL)
}
