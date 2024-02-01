
The source code of paper 'Whole Brain Alignment of Spatial Transcriptomics between Humans and Mice with BrainAlign'

# BrainAlign

[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIT-2.0)

`BrainAlign` is a Python package containing tools for independence testing using multiscale graph correlation and other statistical tests, that is capable of dealing with high dimensional and multivariate data.

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Setting up the development environment](#setting-up-the-development-environment)
- [License](#license)
- [Issues](https://github.com/neurodata/mgcpy/issues)

# Overview
``BrainAlign`` aims to integrate human and mouse whole brain spatial transcriptomics spots and analyze mouse and human brains in a common space. The package utilizes a simple class structure to enhance usability while also allowing easy extension of the package for developers. The package can be installed on all major platforms (e.g. BSD, GNU/Linux, OS X, Windows)from Python Package Index (PyPI) and GitHub.


# System Requirements
## Hardware requirements
`BrainAlign` package requires a standard computer with enough RAM (~100GB RAM for CPU running on datasets in the paper) to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *Linux* and *Windows*. The package has been tested on the following systems:
+ Linux: Ubuntu >= 18.04
+ Windows: Windows >= 11

### Python Dependencies
`BrainAlign` mainly depends on the Python (>= 3.8.16) scientific stack.

```
 colorcet V3.0.1
 colormap V1.0.4
 dgl V1.0.1
 fonttools V4.38.0
 gseapy V1.0.4
 leidenalg V0.9.1
 matplotlib V3.7.0
 pandas V1.5.3
 plotly V5.14.0
 scanpy V1.9.2
 scikit-learn V1.2.1
 scipy V1.10.1
 seaborn V0.12.2
 statannot V0.2.3
 statsmodels V0.13.5
 torch V1.13.1
 torchaudio V0.13.1, 
 torchvision V0.14.1
 yacs V0.1.8
```

# Installation Guide:

### Install Python 
We advice to create a enviroment in Miniconda or Anaconda. 

### Clone code from Github
```
git clone https://github.com/zhanglabtools/BrainAlign
```
### Download or process datasets
The processed mouse and human datasets is avaible online on Google Drive: https://drive.google.com/drive/folders/1XLoReIZf_MzRryOvGe24A8fp3UoJLw19?usp=sharing
The datasets file path/name and introduction are:
- `mouse_2020sa_64regions.h5ad`: Mouse brain spot expression and spatial coordinates data from [https://www.molecularatlas.org/download-data](https://www.molecularatlas.org/download-data).
- `human_brain_region_88_sparse_with3d.h5ad`: Human brain spot expression and spatial coordinates data from The data were 
downloaded from the Allen Institute’s API (http://api.brain-map.org) and pre-processed using the 
abagen package in Python (https://abagen.readthedocs.io/en/stable/).
  The preprocessing procedures are identical to the steps in [Beauchamp, Antoine, et al. "Whole-brain comparison of rodent and human brains using spatial transcriptomics." elife 11 (2022): e79418.](https://github.com/abeaucha/MouseHumanTranscriptomicSimilarity/)
- `gene_matches_mouse2human.csv`: gene many-to-namy orthologs. 
- `gene_matches_1v1_mouse2human.csv`: gene one-to-one orthologs. These two file are downloaded from [https://www.ensembl.org/biomart/martview/47867ae5bb8d4dc7bd104770d7735869](https://www.ensembl.org/biomart/martview/47867ae5bb8d4dc7bd104770d7735869) 
### Running steps
1. Run CAME for embedding initialization. Run the python script in `./BrainAlign/run_came/analysis_script/run_came.py`
```python
import sys
sys.path.append('../')

from analysis_utils import ttest_plot_utils
from analysis_utils import homo_random_config as config
import os


if __name__ == '__main__':

    cfg = config._C
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #cfg.CAME.visible_device
    #cfg.CAME.n_top_genes = 1000
    cfg.CAME.visible_device = '-1'
    n_top_genes_list = [2000]
    #quantile_gene_list = [0.8]
    #quantile_sample_list = [0.9]
    #cfg.CAME.quantile_gene = quantile_gene_list[0]
    #cfg.CAME.quantile_sample = quantile_sample_list[0]
    #for n_top_genes in n_top_genes_list:
    cfg.CAME.n_top_genes = n_top_genes_list[0]
    cfg.CAME.sparse = False
    cfg.CAME.do_normalize = [False, True]
    cfg.CAME.ROOT = '../analysis_results/mouse_2020sa/'
    cfg.CAME.path_rawdata1 = '../brain_mouse_2020sa/mouse_2020sa_64regions.h5ad'
    cfg.CAME.path_rawdata2 = '../../Brain_ST_human_mouse/data/human_brain_region_88_sparse_with3d.h5ad'

    cfg.CAME.path_mouse_labels = '../brain_mouse_2020sa/mouse_region_list_64.csv'
    cfg.CAME.path_human_labels = '../brain_human_mouse/human_88_label_origin.csv'

    cfg.CAME.human_mouse_homo_region = '../brain_human_mouse/MouseHumanMatches_H88M67_all.csv'
    #    ttest_plot_utils.run_came_homo_random(cfg)

    cfg.PROCESS.path_rawdata1 = cfg.CAME.path_rawdata1
    cfg.PROCESS.path_rawdata2 = cfg.CAME.path_rawdata2

    #cfg.PROCESS.path_mouse_labels = '../brain_human_mouse/mouse_67_label_10regions.csv'
    #cfg.PROCESS.path_human_labels = '../brain_human_mouse/human_88_label_10regions.csv'

    #cfg.PROCESS.path_rawdata1_part = '../../Brain_ST_human_mouse/data/10regions_mouse_brain_region_67_sagittal.h5ad'
    #cfg.PROCESS.path_rawdata2_part = '../../Brain_ST_human_mouse/data/10regions_human_brain_region_88_sparse_with3d.h5ad'

    ttest_plot_utils.run_came_homo_random(cfg)
```
After running CAME, we get a result folder similar to `./BrainAlign/run_came/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-06-03_14.38.27/` 
2. Run BrainAlign and analyze the results for self-superwised alignment. The script file is `./BrainAlign/BrainAlign/data/srrsc_mouse_human_binary/Run_BrainAlign.py`:
```python
import sys
sys.path.append('../../../')
from BrainAlign.code.utils import set_params
from BrainAlign.brain_analysis.configs import sr_rsc_config_binary
from BrainAlign.brain_analysis.data_utils import load_dimensions_binary
from BrainAlign.brain_analysis import process
from BrainAlign.SR_RSC import main_sr_rsc
from BrainAlign.brain_analysis.analysis import load_srrsc_embeddings

from BrainAlign.brain_analysis.analysis_main import alignment_STs, alignment_STs_analysis,\
    gene_comparison
from BrainAlign.brain_analysis.analysis_anatomical import anatomical_STs_analysis
from BrainAlign.brain_analysis.analysis_genomic import genomic_analysis
from BrainAlign.brain_analysis.process import get_spatial_relation
from BrainAlign.brain_analysis.analysis_spatial import spatial_analysis
import time
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    cfg = sr_rsc_config_binary._C
    cfg.SRRSC_args.if_pretrained = False
    #cfg.freeze()
    cfg.CAME.path_rawdata1 = '../../../run_came/brain_mouse_2020sa/mouse_2020sa_64regions.h5ad'
    cfg.CAME.path_rawdata2 = '../../../run_came/brain_human_mouse/human_brain_region_88_sparse_with3d.h5ad'
    cfg.CAME.ROOT = '../../../run_came/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-06-03_14.38.27/'#Baron_mouse-Baron_human-05-23_21.33.44/ #Baron_mouse-Baron_human-01-03_19.40.04
    cfg.CAME.figdir = '../../../run_came/analysis_results/mouse_2020sa/Baron_mouse-Baron_human-06-03_14.38.27/figs/'

    cfg.CAME.homo_region_file_path = '../../../run_came/brain_human_mouse/MouseHumanMatches_H88M67_all.csv'  # '../../../../CAME/brain_human_mouse/MouseHumanMatches_H88M67.csv'
    cfg.CAME.labels_dir = '../../../run_came/brain_mouse_2020sa/'
    cfg.CAME.labels_mouse_file = 'mouse_region_list_64.csv'
    cfg.CAME.labels_human_file = 'human_88_label.csv'

    cfg.CAME.parent_labels_mouse_file = 'mouse_parent_region_list_15.csv'
    cfg.CAME.parent_labels_human_file = 'human_16_label.csv'

    cfg.BrainAlign.homo_region_num = 20

    cfg.BrainAlign.embedding_type = 'came'
    cfg.BrainAlign.embedding_pca_dim = 30


    cfg.BrainAlign.dataset = 'mouse_human_binary'
    cfg.BrainAlign.result_save_folder = './results_{}_1000genes_all_came_selfloop/'.format(cfg.BrainAlign.homo_region_num)
    cfg.BrainAlign.experiment_time = '2023-06-23_20-31-14'#'2023-06-02_21-34-15' #'2023-05-11_10-08-35'#'2023-05-10_20-36-57'#time.strftime("%Y-%m-%d_%H-%M-%S")#'2023-04-30_23-00-05'#'2023-04-20_11-02-23'#'2023-02-12_10-57-19'#time.strftime("%Y-%m-%d_%H-%M-%S")  #'2023-01-04_10-15-59'
    cfg.BrainAlign.result_save_path = cfg.BrainAlign.result_save_folder + cfg.BrainAlign.experiment_time
    cfg.BrainAlign.embeddings_file_path = cfg.BrainAlign.result_save_path + "/embeds/"
    cfg.BrainAlign.DATA_PATH = cfg.BrainAlign.result_save_path + '/data/'

    cfg.SRRSC_args.save_path = "./results_{}_1000genes_all_came_selfloop/".format(
        cfg.BrainAlign.homo_region_num) + cfg.BrainAlign.experiment_time + '/'  # "../data/{}/results/".format(_C.HECO_args.dataset)+_C.HECO.experiment_time+'/'
    cfg.SRRSC_args.data_path = "./results_{}_1000genes_all_came_selfloop/".format(cfg.BrainAlign.homo_region_num) + cfg.BrainAlign.experiment_time + '/data/'

    S, M, binary_S, binary_M, binary_H, binary_V = load_dimensions_binary(cfg.CAME.ROOT + 'datapair_init.pickle')

    cfg.BrainAlign.S = S
    cfg.BrainAlign.M = M

    cfg.BrainAlign.binary_S = binary_S
    cfg.BrainAlign.binary_M = binary_M
    cfg.BrainAlign.binary_H = binary_H
    cfg.BrainAlign.binary_V = binary_V

    cfg.BrainAlign.node_relation = 'spatial'  # 'spatial' # knn
    cfg.BrainAlign.spatial_node_neighbor = 5

    cfg.ANALYSIS.umap_neighbor = 30  # 30
    cfg.ANALYSIS.mouse_umap_neighbor = 30  # 40
    cfg.ANALYSIS.human_umap_neighbor = 30

    cfg.ANALYSIS.umap_marker_size = 5  # 40
    cfg.ANALYSIS.mouse_umap_marker_size = 5  # 40
    cfg.ANALYSIS.human_umap_marker_size = 10  # 40

    cfg.ANALYSIS.genes_umap_neighbor = 15#15

    cfg.ANALYSIS.sample_cluster_num = 60

    cfg.ANALYSIS.cut_ov = 0

    cfg.SRRSC_args.lr = 0.02
    cfg.SRRSC_args.patience = 30

    cfg.SRRSC_args.out_ft = 128

    cfg.SRRSC_args.drop_prob = 0.7
    cfg.SRRSC_args.lamb = 0.5  # coefficient for the losses in node task
    cfg.SRRSC_args.lamb_lp = 1.0  # coefficient for the losses in link task
    cfg.SRRSC_args.margin = 0.5 #0.8

    cfg.BrainAlign.fig_format = 'png'

    cfg.SRRSC_args.isAtt = True#True#True
    cfg.SRRSC_args.isBias = True

    cfg.SRRSC_args.lamb = 0.5  # coefficient for the losses in node task
    cfg.SRRSC_args.lamb_lp = 1.0  # coefficient for the losses in link task
    cfg.SRRSC_args.margin = 0.8  # coefficient for the margin loss

    cfg.SRRSC_args.train_ratio = 0.5
    cfg.SRRSC_args.validation_ratio = 0.45
    cfg.SRRSC_args.test_ratio = 0.05

    cfg.SRRSC_args.lr_stepsize = 10
    cfg.SRRSC_args.lr_gamma = 0.5

    cfg.SRRSC_args.nb_epochs = 100

    print('--------------------------Config:', cfg)
    process.get_srrsc_input(cfg)
    print('Training BrainAlign...')
    main_sr_rsc.run_srrsc(cfg)

    get_spatial_relation(cfg)

    print('Analysis of BrainAlign embeddings...')


    load_srrsc_embeddings(cfg)  # (21749, 128)
    #alignment_STs(cfg)

    alignment_STs_analysis_obj = alignment_STs_analysis(cfg)
    #alignment_STs_analysis_obj.forward()
    alignment_STs_analysis_obj.experiment_1_cross_species_clustering()
    alignment_STs_analysis_obj.experiment_2_umap_evaluation()
    alignment_STs_analysis_obj.experiment_2_umap_diagram()
    alignment_STs_analysis_obj.experiment_3_homo_random()
    alignment_STs_analysis_obj.experiment_3_1_homo_random_beforeAlign()
    # self.experiment_3_homo_random()
    alignment_STs_analysis_obj.experiment_4_cross_species_genes_analysis()
    alignment_STs_analysis_obj.experiment_5_genes_homo_random()
    alignment_STs_analysis_obj.experiment_6_brain_region_classfier()
    alignment_STs_analysis_obj.experiment_7_align_cross_species()
    alignment_STs_analysis_obj.experiment_7_1_align_cross_species_split()
    alignment_STs_analysis_obj.experiment_7_2_align_cross_species_alignementscore()

    alignment_STs_analysis_obj.experiment_7_3_beforealign_cross_species_split()

    alignment_STs_analysis_obj.experiment_8_cross_evaluation_aligment_cluster()
    alignment_STs_analysis_obj.experiment_8_1_cross_aligment_cluster_merge_lines()
    alignment_STs_analysis_obj.experiment_9_name_clusters()
    alignment_STs_analysis_obj.experiment_9_1_region_cluster_heatmap()
    alignment_STs_analysis_obj.experiment_10_check_alignment()
    alignment_STs_analysis_obj.experiment_10_1_check_alignment_plot()
    alignment_STs_analysis_obj.experiment_11_paga()


    #alignment_STs_analysis_obj.forward()
    genomic_analysis_obj = genomic_analysis(cfg)
    genomic_analysis_obj.experiment_1_th_clusters_deg()
    genomic_analysis_obj.experiment_2_clusters_deg()
    genomic_analysis_obj.experiment_2_1_homoregions_deg()

    genomic_analysis_obj.experiment_2_3_homoregions_deg_proportion()

    genomic_analysis_obj.experiment_2_1_homoregions_deg_homo()

    genomic_analysis_obj.experiment_2_2_homoregions_multiple_deg()

    genomic_analysis_obj.experiment_3_regions_gene_module()
    genomic_analysis_obj.experiment_3_1_regions_gene_module_analysis()
    genomic_analysis_obj.experiment_4_clusters_gene_module()
    genomic_analysis_obj.experiment_5_deg_homo_distribution()
    genomic_analysis_obj.experiment_5_1_gene_module_homogenes_ratio()

    genomic_analysis_obj.experiment_6_regions_gene_module_average_exp()
    genomic_analysis_obj.experiment_6_1_regions_gene_module_cross()
    genomic_analysis_obj.experiment_7_clusters_gene_module_average_exp()
    genomic_analysis_obj.experiment_7_1_clusters_gene_module_cross()
    genomic_analysis_obj.experiment_8_homoregions_go_terms_overlap()
    genomic_analysis_obj.experiment_8_1_plot_heatmap()

    genomic_analysis_obj.experiment_9_clusters_go_terms_overlap()
    genomic_analysis_obj.experiment_9_1_plot_heatmap()
    genomic_analysis_obj.experiment_10_binary_clustering()
    genomic_analysis_obj.experiment_10_1_binary_clustering_multiple_genesets()

    #gene_comparison(cfg)
    anatomical_STs_analysis_obj = anatomical_STs_analysis(cfg)
    anatomical_STs_analysis_obj.forward()
    spatial_analysis_obj = spatial_analysis(cfg)
    spatial_analysis_obj.experiment_1_spatial_hippocampal()
    #spatial_analysis_obj.forward()
    spatial_analysis_obj.experiment_2_spatial_isocortex()
    spatial_analysis_obj.experiment_3_spatial_clusters()


```

# Reproduction of Analysis
The code after `print('Analysis of BrainAlign embeddings...')` are for reproduction of analysis results in the paper.


# License

This project is covered under the **MIT 2.0 License**.
