#@Time : 2022/12/18 22:47
#@Author : Biao Zhang
#@Email : littlebiao@outlook.com
#@File : load_mouse_region_tree.r
#@Description: This file is used to ...


# Packages -------------------------------------------------------------------

suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library(data.tree))
suppressPackageStartupMessages(library(rjson))
suppressPackageStartupMessages(library(optparse))

working_dir <- getwd()

path_tree_tools <- '../analysis_utils/tree_tools.R'
fileTree <- '../brain_mouse_2020sa/DSURQE_tree.json'



