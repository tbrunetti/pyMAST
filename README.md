# pyMAST

**Pure Python implementation of the MAST R/Bioconductor package for single-cell differential expression analysis.**

[![CI](https://github.com/tbrunetti/pyMAST/actions/workflows/ci.yml/badge.svg)](https://github.com/tbrunetti/pyMAST/actions)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

pyMAST is a **100% pure Python**, no-R reimplementation of the
[MAST R/Bioconductor package](https://bioconductor.org/packages/release/bioc/html/MAST.html)
(Finak et al. 2015, *Genome Biology*).

Every statistical function is a line-by-line verified Python equivalent of the R source code
in [RGLab/MAST](https://github.com/RGLab/MAST) (main branch, pinned to v1.38.0), implemented
exclusively using `scipy`, `numpy`, `statsmodels`, and `patsy`. Zero R dependencies.

### Key features

| Feature | Description |
|---|---|
| **Two-part hurdle model** | Jointly models discrete (detection) + continuous (expression) components |
| **AnnData-native API** | Accepts `AnnData` objects directly — no custom containers |
| **Scanpy-compatible output** | Drop-in replacement for `sc.tl.rank_genes_groups(method='mast')` |
| **Empirical Bayes** | Variance shrinkage for the continuous component (limma-style) |
| **Bayesian GLM** | Cauchy-prior logistic regression (Gelman et al. `arm::bayesglm`) |
| **Bootstrap GSEA** | Gene set enrichment via bootstrap (Subramanian et al. 2005) |
| **No R** | Zero R dependencies — pure `scipy`/`numpy`/`statsmodels`/`patsy` |

---

## Installation

### From GitHub (recommended for UV environments)

```bash
uv add git+https://github.com/tbrunetti/pyMAST.git

# Pinned version:
uv add "pymast @ git+https://github.com/tbrunetti/pyMAST.git@v0.1.0"
```

### With pip

```bash
pip install git+https://github.com/tbrunetti/pyMAST.git
```

### For development

```bash
git clone https://github.com/tbrunetti/pyMAST.git
cd pyMAST
uv pip install -e ".[dev]"
```

### Docker

```bash
docker build -t pymast:latest .
docker run --rm pymast:latest pytest tests/ -v
```

---

## Quick Start

```python
import scanpy as sc
import pymast

# Load your data as AnnData (any loader: 10x, BD Rhapsody, Parse, etc.)
adata = sc.read_h5ad("my_data.h5ad")

# Run MAST — drop-in replacement for sc.tl.rank_genes_groups
pymast.tl.rank_genes_groups(
    adata,
    groupby="leiden",          # obs column with group labels
    groups="all",              # test all groups vs. rest
    reference="rest",
    layer=None,                # use adata.X (pass layer name for a specific layer)
    ebayes=True,               # empirical Bayes variance shrinkage
    method="bayesglm",         # "bayesglm" | "glm"
    n_jobs=-1,                 # parallelism (-1 = all cores)
)

# Results are stored in exactly the same format as sc.tl.rank_genes_groups:
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
sc.pl.rank_genes_groups_dotplot(adata, n_genes=10)

# Helper DataFrame:
import scanpy as sc
df = sc.get.rank_genes_groups_df(adata, group="0")

# MAST-specific decomposition (disc + cont component p-values):
import pandas as pd
disc_df = pd.DataFrame(adata.uns["mast"]["disc_pvals"], index=adata.var_names)
cont_df = pd.DataFrame(adata.uns["mast"]["cont_pvals"], index=adata.var_names)
```

---

## Output Structure

### `adata.uns['rank_genes_groups']` (Scanpy-compatible)

Identical format to `sc.tl.rank_genes_groups` — all `sc.pl.*` plotting functions work out of the box.

| Key | Type | Description |
|---|---|---|
| `params` | `dict` | Run parameters (`groupby`, `method='mast'`, etc.) |
| `names` | structured `np.ndarray[object]` | Gene names ranked by hurdle p-value |
| `scores` | structured `np.ndarray[float]` | Hurdle LRT chi-sq statistic |
| `pvals` | structured `np.ndarray[float]` | Raw hurdle p-values |
| `pvals_adj` | structured `np.ndarray[float]` | BH-adjusted p-values |
| `logfoldchanges` | structured `np.ndarray[float]` | Log2 fold changes (continuous component) |
| `pts` | `pd.DataFrame` | Fraction expressing per group |

### `adata.uns['mast']` (MAST-specific)

```python
adata.uns["mast"] = {
    "hurdle_pvals":  ...,  # combined hurdle p-values (genes × groups)
    "disc_pvals":    ...,  # discrete component p-values
    "cont_pvals":    ...,  # continuous component p-values
    "disc_logfc":    ...,  # log fold change in detection rate
    "cont_logfc":    ...,  # log fold change in expression | detected
    "coef_C":        ...,  # continuous component coefficients
    "coef_D":        ...,  # discrete component coefficients
    "cdr":           ...,  # per-cell cellular detection rate covariate
    "params":        ...,  # full parameter dict
}
```

---

## Mathematical Foundation

pyMAST faithfully reimplements the MAST hurdle model described in:

> Finak G, McDavid A, Yajima M, et al. (2015). **MAST: a flexible statistical framework for
> assessing transcriptional changes and characterizing heterogeneity in single-cell RNA sequencing data.**
> *Genome Biology* 16:278. DOI: [10.1186/s13059-015-0844-5](https://doi.org/10.1186/s13059-015-0844-5)

Key statistical components and their references:

| Component | Python Module | Reference |
|---|---|---|
| Hurdle model (two-part GLM) | `pymast/zlm.py` | Finak et al. 2015 |
| Empirical Bayes shrinkage | `pymast/ebayes.py` | Smyth 2004 (limma) |
| Bayesian logistic regression | `pymast/bayes_glm.py` | Gelman et al. 2008 (arm) |
| Likelihood ratio test | `pymast/lrtest.py` | Neyman & Pearson 1933 |
| Bootstrap GSEA | `pymast/gsea.py` | Subramanian et al. 2005 |
| Log fold change | `pymast/log_fc.py` | Love et al. 2014 (DESeq2) |

All functions contain inline comments in the format `# R: <file>.R L<start>-<end>` referencing the
exact R source lines in [RGLab/MAST main branch](https://github.com/RGLab/MAST/tree/main/R).

---

## API Reference

### `pymast.tl.rank_genes_groups()`

```python
pymast.tl.rank_genes_groups(
    adata: AnnData,
    groupby: str,
    groups: str | list[str] = "all",
    reference: str = "rest",
    layer: str | None = None,
    formula_extra: str | None = None,
    use_highly_variable: bool = True,
    ebayes: bool = True,
    method: str = "bayesglm",     # "bayesglm" | "glm"
    n_jobs: int = -1,
    key_added: str = "rank_genes_groups",
    copy: bool = False,
) -> AnnData | None
```

### `pymast.tl.zlm()`

```python
fit = pymast.tl.zlm(
    adata,
    formula="~ group + cdr",
    layer=None,
    method="bayesglm",
    ebayes=True,
    n_jobs=-1,
)
# fit is a ZlmFit object
lr_results = fit.lr_test(contrast="group")
```

---

## Development

```bash
# Run tests
pytest tests/ -v --cov=pymast --cov-report=html

# Lint
ruff check pymast/

# Docker test
docker build -t pymast:latest . && docker run --rm pymast:latest
```

---

## Citation

If you use pyMAST, please cite the original MAST paper:

```bibtex
@article{Finak2015,
  title={MAST: a flexible statistical framework for assessing transcriptional changes
         and characterizing heterogeneity in single-cell RNA sequencing data},
  author={Finak, Greg and McDavid, Andrew and Yajima, Masanao and others},
  journal={Genome Biology},
  volume={16},
  pages={278},
  year={2015},
  doi={10.1186/s13059-015-0844-5}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
