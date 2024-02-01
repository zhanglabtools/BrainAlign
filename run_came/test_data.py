import scanpy as sc


fp1 = './came/sample_data/raw-Baron_human.h5ad'
anndata_1 = sc.read_h5ad(fp1)

print(anndata_1)

print(type(anndata_1.X))