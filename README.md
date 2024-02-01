# Brain Alignment

## Data Process

## Run Commands

## install packages
```shell
conda create -n brainalign python=3.8
conda activate brainalign

pip install "scanpy[leiden]"
pip install torch  # >=1.8
pip install dgl

pip install yacs --upgrade

pip install xgboost

pip install scGeneFit

pip install colorcet
README.md
pip install imblearn

pip install gseapy
```

# Modification of the other packages
## gseapy
- Add fig to gseapy.dotplot's return, i.e., change
```python
 if ofname is None:
        return ax
    dot.fig.savefig(ofname, bbox_inches="tight", dpi=300)
```
to:
```python
 if ofname is None:
        return ax
    dot.fig.savefig(ofname, bbox_inches="tight", dpi=300)
```
- Change dotplot colorbar title pad, i.e., change
```python
cbar.ax.set_title(self.cbar_title, loc="left", fontweight="bold")
```
```python
cbar.ax.set_title(self.cbar_title, loc="left", fontweight="bold", pad=20)
```

- delete '\n' in dotplot colorbar title
```python
ax.legend(
            handles,
            labels,
            title="% Genes in set",
            bbox_to_anchor=(1.02, 0.9),
            loc="upper left",
            frameon=False,
            labelspacing=1.0,
        )
```

## Scanpy
### UMAP
Add legend position in line 1109
```python
    if legend_loc == 'lower center':
        for label in cats:
            ax.scatter([], [], c=palette[label], label=label)
        ax.legend(
            frameon=False,
            loc='lower center',
            bbox_to_anchor=(0.45, -0.3),
            ncol=(1 if len(cats) <= 14 else 2 if len(cats) <= 30 else 3),
            fontsize=legend_fontsize,
        )
```