# pyMAST: Implementation Plan v1.0.0 — ORIGINAL (Pre-Review)
> **Status:** Submitted for user review 2026-08-03. See `implementation_plan_v2_final.md` for the approved plan.
> Inline `🗒 REVIEW:` callouts show the user's annotated feedback on each section.

**R Source:** https://github.com/RGLab/MAST (branch: **devel**)

> 🗒 **REVIEW:** *"not branch devel; use a stable branch/release or main"*

**DOI Reference:** 10.18129/B9.bioc.MAST
**Target Python:** ≥ 3.14 | **UV project:** `scanpy-sci-universe-04022026`

---

## Background

MAST implements a **two-part hurdle model** jointly modeling:
1. **Discrete component** — logistic regression on binary detection
2. **Continuous component** — Gaussian/linear regression on log-expression | detected

Combined hurdle p-value from likelihood ratio testing. Key features: CDR covariate, empirical Bayes
variance shrinkage, Bayesian logistic regression (Cauchy prior via `arm::bayesglm`), bootstrap GSEA,
multiple data containers (`SingleCellAssay`, `FluidigmAssay`, `RNASeqAssay`).

---

## User Review Required

> [!IMPORTANT]
> **Mathematical Fidelity:** Line-by-line verified Python equivalent of R source in `RGLab/MAST`
> (devel branch). Comments reference R file and line number.
>
> 🗒 **REVIEW:** *"Again, not devel branch; pick a stable release or main branch"*

> [!WARNING]
> **No R dependencies whatsoever.** All statistical routines implemented using `scipy`, `numpy`, `statsmodels` only.
>
> 🗒 **REVIEW:** *"Yes, this is critical. No R whatsoever"* → Escalated to CAUTION in v2.

---

## Open Questions

> [!IMPORTANT]
> **Q1 — Input format:** `SingleCellAssay` container + AnnData bridge? OR AnnData only?
> *Recommendation (v1):* Both — `SingleCellAssay` + `from_anndata()` converters.
>
> 🗒 **REVIEW:** *"AnnData direct only, all devices now use AnnData"*
> → **Resolved in v2:** AnnData is sole input; `SingleCellAssay` moved to internal `anndata_utils.py`.

> [!IMPORTANT]
> **Q2 — GSEA:** Include `GSEA-by-boot.R` in v1.0 or defer to v1.1 stub?
> *Recommendation (v1):* Stub with `NotImplementedError`.
>
> 🗒 **REVIEW:** *"Yes include it and make sure it is well documented"*
> → **Resolved in v2:** Fully included with Subramanian et al. 2005 reference.

> [!IMPORTANT]
> **Q3 — Fluidigm:** Include `computeEtFromCt` / `FromFlatDF` thin wrappers?
> *Recommendation (v1):* Include since simple arithmetic.
>
> 🗒 **REVIEW:** *"Yes include it but keep it in AnnData form; lowest priority but still wanted"*
> → **Resolved in v2:** Wrappers included, lowest priority, all paths convert to AnnData first.

> [!IMPORTANT]
> **Q4 (implicit) — Output format:** Custom or Scanpy-compatible?
>
> 🗒 **REVIEW:** User confirmed drop-in replacement for `sc.tl.rank_genes_groups(method='mast')`,
> results stored in `adata.uns['rank_genes_groups']` (exact scanpy recarray format),
> plus MAST-specific extras in `adata.uns['mast']`.

---

## R Source → Python Module Mapping (v1 — no References column)

> 🗒 **REVIEW:** *"Can we add references to the math so when I look at the table I have the papers too"*
> → References column added in v2.

| R File | Python Module | Python Class/Function |
|---|---|---|
| `zeroinf.R` | `pymast/zlm.py` | `zlm()`, `ZlmFitter` |
| `ZlmFit.R` | `pymast/zlm_fit.py` | `ZlmFit`, `collect_summaries()` |
| `LmWrapper.R` | `pymast/lm_wrapper.py` | `LMlike`, `make_chisq_table()`, `wald_test()` |
| `lrtest.R` | `pymast/lrtest.py` | `lrtest()`, `log_prod()`, `lrt()`, `LRT()` |
| `ebayes-helpers.R` | `pymast/ebayes.py` | `ebayes()`, `get_marginal_hyperlikelihood()` |
| `lmWrapper-bayesglm.R` | `pymast/bayes_glm.py` | `BayesGLMLike` |
| `lmWrapper-glm.R` | `pymast/glm_wrapper.py` | `GLMLike` |
| `AllClasses.R` | `pymast/single_cell_assay.py` | `SingleCellAssay`, `FluidigmAssay`, `RNASeqAssay` |
| `Readers.R` | `pymast/readers.py` | `from_matrix()`, `from_flat_df()`, `from_anndata()` |
| `filterEval.R` | `pymast/filter.py` | `mast_filter()`, `burden_of_filtering()` |
| `UtilityFunctions.R` | `pymast/utils.py` | `compute_et_from_ct()`, `freq()` |
| `ZlmFit-logFC.R` | `pymast/log_fc.py` | `get_log_fc()` |
| `ZlmFit-bootstrap.R` | `pymast/bootstrap.py` | `boot_zlm()` (v1.1 stub) |
| `GSEA-by-boot.R` | `pymast/gsea.py` | stub → v1.1 |
| `Hypothesis.R` | `pymast/hypothesis.py` | `CoefficientHypothesis`, `Hypothesis` |

---

## Core Mathematics (same in v1 and v2 — devel branch refs updated to main in v2)

### Hurdle Model
- Discrete: `logit P(Z_gi = 1) = X_i @ beta_D` via `glm(family=binomial)` or `bayesglm`
- Continuous: `X_gi | Z_gi = 1 = X_i[pos] @ gamma_C + eps` via `lm()`

### Empirical Bayes (`ebayes-helpers.R` L6-27)
```python
Li = -betaln(rNg/2, a0) - rNg/2 * np.log(b0) - np.log(1 + SSg/(2*b0)) * (rNg/2 + a0)
score_a0_i = digamma(rNg/2 + a0) - digamma(a0) - np.log(1 + SSg/(2*b0))
score_b0_i = (a0 * SSg - rNg * b0) / (SSg * b0 + 2 * b0**2)
```

### LRT Simple (`lrtest.R` L5-50)
```python
p0 = (e_x + e_y) / (n_x + n_y)
binom = log_prod(e_x, p0/p_x) + log_prod(e_y, p0/p_y) + ...
T_star = 1 + e_x*e_y/(e_x+e_y) * (mu_x - mu_y)**2 / (SS_x + SS_y)
norm_lr = -(e_x + e_y)/2 * np.log(T_star)
log_LR = binom + norm_lr   # -2*log_LR ~ chi2(df=2)
```

### LRT Full (`ZlmFit.R`)
```python
lambda_ = -2 * (loglik_0 - loglik_1); p_value = chi2.sf(lambda_, df=df)
lambda_hurdle = lambda_C + lambda_D; p_hurdle = chi2.sf(lambda_hurdle, df=df_hurdle)
```

### Wald Test (`LmWrapper.R` L142-167)
```python
lambda_C = contrC @ np.linalg.solve(contrCovC, contrC)
```

### Bayesian GLM Cauchy Prior
```
Prior: beta_j ~ Cauchy(0, 2.5); IRLS with Cauchy augmentation (Gelman et al. default)
```

---

## Repository Structure (v1)

```
pyMAST/
├── pymast/
│   ├── __init__.py
│   ├── single_cell_assay.py   # ← changed to anndata_utils.py in v2
│   ├── readers.py
│   ├── filter.py
│   ├── zlm.py
│   ├── zlm_fit.py
│   ├── lm_wrapper.py
│   ├── bayes_glm.py
│   ├── glm_wrapper.py
│   ├── ebayes.py
│   ├── lrtest.py
│   ├── log_fc.py
│   ├── hypothesis.py
│   ├── utils.py
│   ├── bootstrap.py           # ← stub only in v1; implemented in v2
│   └── gsea.py                # ← stub only in v1; implemented in v2
├── tests/
├── examples/
├── antigravity_docs/
├── pyproject.toml
├── README.md
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/
├── CHANGELOG.md
└── LICENSE
```

---

## pyproject.toml (same v1 and v2)

```toml
[project]
name = "pymast"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "numpy>=2.0", "scipy>=1.14", "pandas>=3.0.3",
    "statsmodels>=0.14", "anndata==0.12.16",
    "scikit-learn>=1.5", "patsy>=0.5", "tqdm>=4.0",
]
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov", "ruff", "black"]
jupyter = ["jupyter", "marimo", "matplotlib", "seaborn"]
```

---

## Verification Plan

```bash
pytest tests/ -v --cov=pymast --cov-report=html
ruff check pymast/
docker build -t pymast:latest . && docker run --rm pymast:latest pytest tests/ -v
```

---

*Plan Version: 1.0.0 (original pre-review) | Created: 2026-08-03*
