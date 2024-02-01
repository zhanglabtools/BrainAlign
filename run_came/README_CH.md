# CAME

[English](README.md) | **简体中文**

CAME 是一个基于异质图神经网络的 **细胞类型注释和基因模块提取** 的工具。

有关详细用法，请参阅 [CAME-文档](https://xingyanliu.github.io/CAME/index.html).

<img src="docs/_images/Fig1ABC.png" width="600"/>

对查询数据集的每个细胞，CAME 可以输出其细胞类型的量化估计，即对应于参考数据中的细胞类型概率，
从而能帮助够识别查询数据中的潜在未知细胞类型。

此外，CAME 还提供了跨物种对齐的细胞和基因嵌入表示，有助于进一步的低维可视化和联合基因模块提取。
（如图所示）

<img src="docs/_images/Fig1D.png" width="600"/>


### 安装

推荐使用 Conda 新建一个 Python 环境来运行 CAME：

```shell
conda create -n env_came python=3.8
conda activate env_came
```

安装必要的依赖包：

```shell
pip install "scanpy[leiden]"
pip install torch  # >=1.8 
pip install dgl  # tested on 0.7.2
```

See [Scanpy](https://scanpy.readthedocs.io/en/stable/), 
[PyTorch](https://pytorch.org/) and [DGL](https://www.dgl.ai/) 
for detailed installation guide (especially for GPU version).

从 PyPI 安装 CAME:

```shell
pip install came
```

从源代码中安装 CAME 的开发版本:

```shell
git clone https://github.com/XingyanLiu/CAME.git
cd CAME
python setup.py install
```

### 样例数据

测试代码和文档中的分析示例都基于CAME包附带的样例数据，初始时以压缩形式保存 (`./came/sample_data.zip`)，
在需要的时候会自动解压到默认目录下 (`./came/sample_data/`)，其中包含以下文件:

- gene_matches_1v1_human2mouse.csv (非必要)
- gene_matches_1v1_mouse2human.csv (非必要)
- gene_matches_human2mouse.csv
- gene_matches_mouse2human.csv
- raw-Baron_mouse.h5ad
- raw-Baron_human.h5ad 

如果你需要用 CAME 来分析你自己的数据集，同物种的跨数据（跨组学）分析至少要准备以上后两个文件（基因表达count）。

对于跨物种分析，您需要提供另一个“.csv”文件，其中第一列包含参考物种中的基因，第二列包含相应的查询同源基因。

> 注意:
> 数据文件 “raw-Baron_human.h5ad” 仅用于代码测试，是原始数据的子样本 （20%），
> 因此结果的注释精度可能不如使用完整数据集作为参考。

### 测试 CAME 的分析流程 (非必要)

可以直接运行 `test_pipeline.py` 来测试 CAME 的分析流程：

```python
# test_pipeline.py
import came

if __name__ == '__main__':
    came.__test1__(6, batch_size=2048)
    came.__test2__(6, batch_size=None)
```

```shell
python test_pipeline.py 
```

如果测试过程中没有报错，那就放心使用CAME吧～

### Contribute

* 问题追踪: https://github.com/XingyanLiu/CAME/issues
* 源代码:
  * https://github.com/zhanglabtools/CAME
  * https://github.com/XingyanLiu/CAME (the developmental version)

### Support

如果你有其他问题，也可以通过邮箱联系我们:

* xingyan@amss.ac.cn
* 544568643@qq.com


### 引用

如果 CAME 对你的研究有帮助，可以引用我们的预印本哦～

> Cross-species cell-type assignment of single-cell RNA-seq by a heterogeneous graph neural network.
   Xingyan Liu, Qunlun Shen, Shihua Zhang.
   bioRxiv 2021.09.25.461790; doi: https://doi.org/10.1101/2021.09.25.461790

