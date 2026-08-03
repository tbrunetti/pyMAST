# pyMAST: Pure Python Implementation of the MAST R Package

**Primary Publication:** Finak et al. (2015). *Genome Biology* — DOI: [10.1186/s13059-015-0844-5](https://doi.org/10.1186/s13059-015-0844-5)  
**Bioconductor DOI:** [10.18129/B9.bioc.MAST](https://doi.org/10.18129/B9.bioc.MAST)  
**R Source:** https://github.com/RGLab/MAST (branch: **main**)  
**R Source Version Pinned:** MAST v1.38.0 (Bioconductor Release 3.23)  
**Target Python:** ≥ 3.14 (per UV lock file)  
**UV project compatibility:** `scanpy-sci-universe-04022026` with frozen dependencies

---

## Background

MAST (Model-based Analysis of Single-cell Transcriptomics) is an R/Bioconductor package by McDavid, Finak, and Yajima for differential expression analysis of zero-inflated single-cell assay data. It implements a **two-part hurdle model** that jointly models:

1. **Discrete component** — logistic regression on binary detection (gene expressed or not)
2. **Continuous component** — Gaussian/linear regression on log-expression level, conditional on detection

The combined "hurdle" p-value is derived from both components via likelihood ratio testing. Key features also include:
- Cellular Detection Rate (CDR) covariate computation
- Empirical Bayes variance shrinkage (limma-style) for the continuous component
- Bayesian logistic regression (Cauchy prior, as in `arm::bayesglm`) for the discrete component
- Gene Set Enrichment Analysis (GSEA) via bootstrap
- Multiple data container classes (`SingleCellAssay`, `FluidigmAssay`, `RNASeqAssay`)

---

## User Review Required

> [!IMPORTANT]
> **UV Compatibility Constraint:** The user's frozen pyproject.toml requires Python ≥ 3.14 and pins `anndata==0.12.16`, `pandas>=3.0.3`. All pyMAST dependencies must be compatible with these versions.

> [!IMPORTANT]
> **Mathematical Fidelity:** Every function must be a line-by-line verified Python equivalent of the R source in `RGLab/MAST` (**main** branch, pinned to v1.38.0). Comments in every function will reference the R file and line number in format `# R: <file>.R L<start>-<end>`.

> [!WARNING]
> **Bayesian GLM:** The R package uses `arm::bayesglm` (Gelman et al., Cauchy prior). We implement this from scratch using iteratively reweighted least squares (IRLS) with the Cauchy prior augmentation — **no R bridge**.

> [!CAUTION]
> **No R dependencies whatsoever.** Zero R code. All statistical routines (GLM, logistic regression, Chi-squared CDF, Beta function, digamma, etc.) implemented exclusively with `scipy`, `numpy`, `statsmodels`, and `patsy`. This is critical and non-negotiable.

---

## Open Questions

> [!IMPORTANT]
> **RESOLVED — Input Format:** pyMAST accepts **AnnData objects directly** as the sole input format. No R data structures. No SingleCellAssay container exposed to users. All technologies supported (10x Chromium, BD Rhapsody, Parse Biosciences) since all output to AnnData via existing loaders (`sc.read_10x_h5`, `sc.read_10x_mtx`, etc.).

> [!IMPORTANT]
> **RESOLVED — GSEA:** Bootstrap GSEA (`GSEA-by-boot.R`) will be included and fully documented. Version compatible with the workflow will be determined during implementation.

> [!IMPORTANT]
> **RESOLVED — Fluidigm:** Thin wrappers for `computeEtFromCt` and `FromFlatDF` included as utility functions only. **Not a priority.** All pyMAST functions accept AnnData as input — Fluidigm wrappers simply convert to AnnData internally before processing.

---

## Mathematical Foundation (Line-by-Line Verification Targets)

### R Source → Python Module Mapping

| R File | R Functions | Python Module | Python Class/Function | Key References |
|---|---|---|---|---|
| `zeroinf.R` | `zlm()`, `.zlm()`, `methodDict` | `pymast/zlm.py` | `zlm()`, `ZlmFitter` | [Finak et al. 2015](https://doi.org/10.1186/s13059-015-0844-5); [R src](https://github.com/RGLab/MAST/blob/main/R/zeroinf.R) |
| `ZlmFit.R` | `collectSummaries()`, `.lrtZlmFit()`, `lrTest`, `waldTest`, `summary` | `pymast/zlm_fit.py` | `ZlmFit`, `collect_summaries()` | [Finak et al. 2015](https://doi.org/10.1186/s13059-015-0844-5); [R src](https://github.com/RGLab/MAST/blob/main/R/ZlmFit.R) |
| `LmWrapper.R` | `.fit()`, `makeChiSqTable()`, `.waldTest()`, `.lrTest()`, `.rotateMM()` | `pymast/lm_wrapper.py` | `LMlike`, `make_chisq_table()`, `wald_test()` | [McDavid et al. 2013](https://doi.org/10.1093/bioinformatics/btt087); [R src](https://github.com/RGLab/MAST/blob/main/R/LmWrapper.R) |
| `lrtest.R` | `lrtest()`, `logProd()`, `lrt()`, `LRT()` | `pymast/lrtest.py` | `lrtest()`, `log_prod()`, `lrt()`, `LRT()` | [Neyman & Pearson 1933](https://doi.org/10.1098/rsta.1933.0009); [R src](https://github.com/RGLab/MAST/blob/main/R/lrtest.R) |
| `ebayes-helpers.R` | `ebayes()`, `getMarginalHyperLikelihood()`, `solveMoM()`, `getSSg_rNg()` | `pymast/ebayes.py` | `ebayes()`, `get_marginal_hyperlikelihood()`, `solve_mom()` | [Smyth 2004 (limma)](https://doi.org/10.2202/1544-6115.1027); [R src](https://github.com/RGLab/MAST/blob/main/R/ebayes-helpers.R) |
| `lmWrapper-bayesglm.R` | `BayesGLMlike` class | `pymast/bayes_glm.py` | `BayesGLMLike` | [Gelman et al. 2008 (arm)](https://doi.org/10.1214/08-AOAS161); [R src](https://github.com/RGLab/MAST/blob/main/R/lmWrapper-bayesglm.R) |
| `lmWrapper-glm.R` | `GLMlike` class | `pymast/glm_wrapper.py` | `GLMLike` | [Nelder & Wedderburn 1972 (GLM)](https://doi.org/10.2307/2344614); `statsmodels.GLM`; [R src](https://github.com/RGLab/MAST/blob/main/R/lmWrapper-glm.R) |
| `AllClasses.R` | `SingleCellAssay`, `ZlmFit` S4 classes | `pymast/anndata_utils.py` | Internal AnnData wrapper *(user faces AnnData directly)* | [AnnData docs](https://anndata.readthedocs.io); [R src](https://github.com/RGLab/MAST/blob/main/R/AllClasses.R) |
| `Readers.R` | `FromMatrix()`, `FromFlatDF()`, `SceToSingleCellAssay()` | `pymast/readers.py` | `from_anndata()`, `compute_et_from_ct()` | [Scanpy read docs](https://scanpy.readthedocs.io/en/stable/api/io.html); [R src](https://github.com/RGLab/MAST/blob/main/R/Readers.R) |
| `filterEval.R` | `mast_filter()`, `burdenOfFiltering()` | `pymast/filter.py` | `mast_filter()`, `burden_of_filtering()` | [McDavid et al. 2013](https://doi.org/10.1093/bioinformatics/btt087); [R src](https://github.com/RGLab/MAST/blob/main/R/filterEval.R) |
| `UtilityFunctions.R` | `computeEtFromCt()`, `freq()`, `thresholdSCRNA()` | `pymast/utils.py` | `compute_et_from_ct()`, `freq()`, `compute_cdr()` | [Fluidigm qPCR protocol](https://doi.org/10.1126/science.1100301); [R src](https://github.com/RGLab/MAST/blob/main/R/UtilityFunctions.R) |
| `ZlmFit-logFC.R` | `getLogFC()` | `pymast/log_fc.py` | `get_log_fc()` | [Love et al. 2014 (DESeq2 logFC)](https://doi.org/10.1186/s13059-014-0550-8); [R src](https://github.com/RGLab/MAST/blob/main/R/ZlmFit-logFC.R) |
| `ZlmFit-bootstrap.R` | `bootZlm()` | `pymast/bootstrap.py` | `boot_zlm()` | [Efron & Tibshirani 1994](https://doi.org/10.1007/978-1-4899-4541-9); [R src](https://github.com/RGLab/MAST/blob/main/R/ZlmFit-bootstrap.R) |
| `GSEA-by-boot.R` | `gseaAfterBoot()` | `pymast/gsea.py` | `gsea_after_boot()` | [Subramanian et al. 2005](https://doi.org/10.1073/pnas.0506580102); [R src](https://github.com/RGLab/MAST/blob/main/R/GSEA-by-boot.R) |
| `Hypothesis.R` | `CoefficientHypothesis`, `Hypothesis` | `pymast/hypothesis.py` | `CoefficientHypothesis`, `Hypothesis` | [R src](https://github.com/RGLab/MAST/blob/main/R/Hypothesis.R) |

---

## Core Mathematics — Verified Against R Source

### 1. Hurdle Model (`zeroinf.R`, `LmWrapper.R`) — AnnData Input

All pyMAST functions receive an `AnnData` object directly. Internally, the code extracts expression matrices from `adata.X` or `adata.layers[layer]`.

**Discrete component** — logistic regression on `Z_gi = 1{X_gi > 0}`:
```
logit P(Z_gi = 1) = X_i @ beta_D
```
R: `object@fitD <- glm(family=binomial)` or `bayesglm(family=binomial)`  
Python: Bayesian logit via IRLS (Cauchy prior) in `bayes_glm.py`, or standard `statsmodels.Logit`

**Continuous component** — linear regression on `X_gi | Z_gi = 1`:
```
X_gi | Z_gi = 1 = X_i[pos] @ gamma_C + eps,  eps ~ N(0, sigma^2)
```
R: `object@fitC <- lm()` with optional empirical Bayes dispersion  
Python: `numpy.linalg.lstsq` + empirical Bayes shrinkage (`ebayes.py`)

### 2. Empirical Bayes Variance Shrinkage (`ebayes-helpers.R`)

The **marginal hyperlikelihood** for gene g (R lines 6-27):
```python
# Li = -lbeta(rNg/2, a0) - rNg/2 * log(b0) - log(1 + SSg/(2*b0)) * (rNg/2 + a0)
from scipy.special import betaln, digamma
Li = -betaln(rNg/2, a0) - rNg/2 * np.log(b0) - np.log(1 + SSg/(2*b0)) * (rNg/2 + a0)
```

Score functions (R lines 21-23):
```python
score_a0_i = digamma(rNg/2 + a0) - digamma(a0) - np.log(1 + SSg/(2*b0))
score_b0_i = (a0 * SSg - rNg * b0) / (SSg * b0 + 2 * b0**2)
```

MLE via L-BFGS-B (R line 93 -> Python `scipy.optimize.minimize`):
```python
result = minimize(neg_ll, x0=[1.0, 1.0], jac=neg_grad,
                  method='L-BFGS-B', bounds=[(0.001, np.inf), (0.001, np.inf)])
v = max(th_b0 / th_a0, 0)   # prior variance
df = max(2 * th_a0, 0)       # prior degrees of freedom
```

### 3. LRT - Simple Two-Group Test (`lrtest.R`, lines 5-50)

The `lrtest()` function (R -> Python `lrtest()`):
```python
# Discrete (binomial) component log-likelihood ratio
p0 = (e_x + e_y) / (n_x + n_y)  # pooled proportion
p_x = e_x / n_x; p_y = e_y / n_y

binom = log_prod(e_x, p0/p_x) + log_prod(e_y, p0/p_y) \
      + log_prod(n_x - e_x, (1-p0)/(1-p_x)) + log_prod(n_y - e_y, (1-p0)/(1-p_y))

# Continuous (Gaussian) component log-likelihood ratio
T_star = 1 + e_x*e_y/(e_x+e_y) * (mu_x - mu_y)**2 / (SS_x + SS_y)
norm_lr = -(e_x + e_y)/2 * np.log(T_star)

# Combined hurdle statistic: -2 * (binom + norm_lr) ~ chi2(df=2)
log_LR = binom + norm_lr
```

R `logProd(prod, logand)` -> Python `log_prod(n, ratio)`:
```python
def log_prod(n: float, ratio: float) -> float:
    return 0.0 if n == 0 else n * np.log(ratio)
```

### 4. LRT - Full Model (`ZlmFit.R`, `LmWrapper.R`)

The `lrTest` on `ZlmFit` (`.lrtZlmFit`, R lines 30-51):
```python
# lambda = -2 * (loglik_reduced - loglik_full)
lambda_ = -2 * (loglik_0 - loglik_1)
df = df_resid_0 - df_resid_1  # degrees of freedom difference
p_value = chi2.sf(lambda_, df=df)
```

`makeChiSqTable` (R lines 105-127):
```python
lambda_hurdle = lambda_C + lambda_D
df_hurdle = df_C + df_D
p_hurdle = chi2.sf(lambda_hurdle, df=df_hurdle)
```

### 5. Wald Test (`LmWrapper.R`, lines 142-167)

```python
# contrC = contrast_matrix.T @ coef_C
# contrCovC = contrast_matrix.T @ vcov_C @ contrast_matrix
# lambda_C = contrC.T @ solve(contrCovC, contrC)
lambda_C = contrC @ np.linalg.solve(contrCovC, contrC)
lambda_D = contrD @ np.linalg.solve(contrCovD, contrD)
```

### 6. Bayesian GLM - Cauchy Prior (`lmWrapper-bayesglm.R`, `bayesglm.R`)

Cauchy prior on logistic regression coefficients (matching `arm::bayesglm`):
```
Prior: beta_j ~ Cauchy(0, scale_j)
Augmented IRLS with prior weight correction per coefficient
Default scale: 2.5 for all predictors (Gelman et al. default)
```

---

## Proposed Repository Structure

```
pyMAST/
|-- pymast/                         # Main package
|   |-- __init__.py                 # Public API exports
|   |-- single_cell_assay.py        # SingleCellAssay, FluidigmAssay, RNASeqAssay
|   |-- readers.py                  # from_matrix(), from_flat_df(), from_anndata()
|   |-- filter.py                   # mast_filter(), burden_of_filtering()
|   |-- zlm.py                      # zlm() - main entry point, gene-wise loop
|   |-- zlm_fit.py                  # ZlmFit result container
|   |-- lm_wrapper.py               # LMlike abstract base, make_chisq_table()
|   |-- bayes_glm.py                # BayesGLMLike - Cauchy-prior logistic regression
|   |-- glm_wrapper.py              # GLMLike - standard logistic regression
|   |-- ebayes.py                   # ebayes(), empirical Bayes variance shrinkage
|   |-- lrtest.py                   # lrtest(), lrt(), LRT() - two-group LRT
|   |-- log_fc.py                   # get_log_fc() - log fold change computation
|   |-- hypothesis.py               # CoefficientHypothesis, Hypothesis classes
|   |-- utils.py                    # compute_et_from_ct(), freq(), CDR
|   |-- bootstrap.py                # boot_zlm() - v1.1 stub
|   `-- gsea.py                     # GSEA - v1.1 stub
|-- tests/                          # pytest test suite
|   |-- conftest.py                 # Synthetic vbeta-like dataset fixtures
|   |-- test_single_cell_assay.py
|   |-- test_zlm.py
|   |-- test_lrtest.py              # Verified vs known R output
|   |-- test_ebayes.py
|   |-- test_wald_test.py
|   |-- test_filter.py
|   `-- test_readers.py
|-- examples/
|   |-- 01_basic_workflow.ipynb
|   |-- 02_anndata_integration.ipynb
|   `-- 03_comparison_with_r.ipynb
|-- antigravity_docs/               # Version-controlled documentation (gittracked)
|   |-- implementation_plan.md      # This file (copy)
|   |-- task.md                     # Execution checklist
|   `-- walkthrough.md              # Post-implementation summary
|-- pyproject.toml
|-- README.md
|-- Dockerfile
|-- docker-compose.yml
|-- .github/
|   `-- workflows/
|       |-- ci.yml
|       `-- release.yml
|-- CHANGELOG.md
`-- LICENSE
```

---

## Proposed Changes

### [NEW] pyproject.toml
UV-compatible, pip-installable from GitHub:
```toml
[project]
name = "pymast"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "numpy>=2.0",
    "scipy>=1.14",
    "pandas>=3.0.3",
    "statsmodels>=0.14",
    "anndata==0.12.16",
    "scikit-learn>=1.5",
    "patsy>=0.5",
    "tqdm>=4.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov", "ruff", "black"]
jupyter = ["jupyter", "marimo", "matplotlib", "seaborn"]

[project.urls]
Homepage = "https://github.com/<user>/pyMAST"
Repository = "https://github.com/<user>/pyMAST"
```

UV install from GitHub:
```bash
uv add git+https://github.com/<user>/pyMAST.git
# Or pinned version:
uv add "pymast @ git+https://github.com/<user>/pyMAST.git@v0.1.0"
```

### [NEW] Dockerfile
```dockerfile
FROM python:3.14-slim
RUN pip install uv
WORKDIR /app
COPY . .
RUN uv pip install --system ".[dev,jupyter]"
EXPOSE 8888
CMD ["pytest", "tests/", "-v"]
```

### [NEW] .github/workflows/ci.yml
- Triggers on push/PR to main and develop
- Matrix: Python 3.14
- Steps: `uv sync`, `ruff check`, `pytest --cov`

---

## Verification Plan

### Automated Tests
```bash
cd /home/tonya/Downloads/github/pyMAST
pytest tests/ -v --cov=pymast --cov-report=html
ruff check pymast/
```

### Math Verification Strategy
- Each Python function has inline comment: `# R: <filename>.R L<start>-<end>`
- Test fixtures include hardcoded expected values from R MAST vignette output
- `test_lrtest.py` verifies numerical agreement to 1e-6 rtol on deterministic inputs

### Docker Verification
```bash
docker build -t pymast:latest .
docker run --rm pymast:latest pytest tests/ -v
```

### UV Environment Verification
```bash
uv add git+https://github.com/<user>/pyMAST.git
python -c "import pymast; print(pymast.__version__)"
```

---

## Output Structure — Scanpy-Compatible (`adata.uns['mast']`)

pyMAST is designed as a **drop-in replacement for `sc.tl.rank_genes_groups(method='mast')`**. Results are stored in `adata.uns` using the same structured NumPy recarray format that Scanpy uses, so all downstream Scanpy plotting functions work without modification.

### API Usage

```python
import pymast

# Drop-in replacement for:
# sc.tl.rank_genes_groups(adata, groupby='leiden', method='mast')
pymast.tl.rank_genes_groups(
    adata,
    groupby='leiden',          # obs column with group labels
    groups='all',              # groups to test, or list e.g. ['0','1']
    reference='rest',          # reference group
    layer=None,                # use adata.X by default, or specify layer name
    formula_extra=None,        # optional extra covariates e.g. '+ n_genes_by_counts'
    use_highly_variable=True,  # filter to HVGs if flagged
    ebayes=True,               # empirical Bayes variance shrinkage
    method='bayesglm',         # 'bayesglm' | 'glm'
    n_jobs=-1,                 # parallelism (-1 = all cores)
    key_added='rank_genes_groups',  # key in adata.uns
    copy=False,                # return copy of adata if True
)

# Access results exactly as you would from sc.tl.rank_genes_groups:
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names   # ('0', '1', '2', ...)

# Or use Scanpy's helper:
import scanpy as sc
df = sc.get.rank_genes_groups_df(adata, group='0')
```

### `adata.uns['rank_genes_groups']` Structure

Follows the **exact same format** as `sc.tl.rank_genes_groups` — fully compatible with `sc.pl.rank_genes_groups`, `sc.pl.rank_genes_groups_dotplot`, etc.

| Key | Type | Description | MAST Source |
|---|---|---|---|
| `params` | `dict` | Run parameters (`groupby`, `method='mast'`, `reference`, etc.) | Metadata |
| `names` | structured `np.ndarray[object]` | Gene names ranked by hurdle p-value per group | `ZlmFit.R` summary |
| `scores` | structured `np.ndarray[float]` | Hurdle LRT chi-sq statistic (−log10 hurdle p-value for ranking) | `lrtest.R` `lambda` |
| `pvals` | structured `np.ndarray[float]` | Raw hurdle p-values (combined disc + cont) | `makeChiSqTable()` |
| `pvals_adj` | structured `np.ndarray[float]` | BH-adjusted p-values | `p.adjust('BH')` |
| `logfoldchanges` | structured `np.ndarray[float]` | Log2 fold changes from continuous component | `ZlmFit-logFC.R` |
| `pts` | `pd.DataFrame` | Fraction of cells expressing gene per group | Computed from `adata.X` |

### MAST-Specific Extended Results

Additionally stored in `adata.uns['mast']` (separate key for MAST-specific data not present in `rank_genes_groups`):

```python
adata.uns['mast'] = {
    'hurdle_pvals':   # raw hurdle p-values (genes × groups)
    'disc_pvals':     # discrete component p-values
    'cont_pvals':     # continuous component p-values
    'disc_logfc':     # log fold change in detection rate
    'cont_logfc':     # log fold change in expression level | detected
    'coef_C':         # continuous component coefficients
    'coef_D':         # discrete component coefficients
    'cdr':            # per-cell cellular detection rate used as covariate
    'params':         # full parameter dict
}
```

This lets you use Scanpy's standard plotting pipeline **and** access MAST-specific decomposition:
```python
# Standard Scanpy plots work out of the box:
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)
sc.pl.rank_genes_groups_dotplot(adata, n_genes=10)

# MAST-specific decomposition:
disc_df  = pd.DataFrame(adata.uns['mast']['disc_pvals'], index=adata.var_names)
cont_df  = pd.DataFrame(adata.uns['mast']['cont_pvals'], index=adata.var_names)
```

---

## Git Strategy

- **Branches:** `main` (stable), `develop` (active dev), feature branches `feat/<name>`
- **Commits:** Conventional commits (`feat:`, `fix:`, `docs:`, `test:`)
- **Tags:** Semantic versions (`v0.1.0`, `v0.1.1`, etc.)
- **Changelog:** CHANGELOG.md updated on each version tag
- **CI:** GitHub Actions runs on every push/PR

---

## Compatibility Matrix

| Concern | Solution |
|---|---|
| anndata==0.12.16 | Use only `.X`, `.obs`, `.var`, `.layers` stable API |
| pandas >=3.0.3 | Use `pd.concat()`, `.iloc`, `.loc`; avoid deprecated `.append()` |
| Python >=3.14 | Use `match` statements, `X | Y` union types, `f-strings` |
| No R bridge | 100% scipy/numpy/statsmodels/patsy — zero R |
| UV lockfile | pyMAST added as git dependency; no conflicts with frozen deps |
| 10x / BD Rhapsody / Parse | All load to AnnData via existing scanpy readers; pyMAST is agnostic |
| Scanpy plotting compat | Output structure mirrors `sc.tl.rank_genes_groups`; all `sc.pl.*` functions work |

---

*Plan created: 2026-08-03 | Version: 2.0.0 (revised per user feedback)*
