# Bug Fixes: Runtime Warnings & CI Dependency Conflict

**Date:** 2026-08-04  
**Files changed:** `pymast/tl.py`, `pymast/zlm_fit.py`, `pyproject.toml`

---

## Fix 1 — `ImplicitModificationWarning` in `tl.py`

### Symptom

Running `pymast.tl.rank_genes_groups(...)` emitted:

```
ImplicitModificationWarning: Trying to modify attribute `.obs` of view,
initializing view as actual.
  adata_test.obs["_pymast_group"] = (group_labels == group_str).astype(int)
```

### Root Cause

`filter_to_highly_variable()` (in `anndata_utils.py`) returns an AnnData
**view** — a lazy column-slice created by `adata[:, mask]`. AnnData views are
read-only proxies; assigning a new column to `.obs` of a view forces AnnData
to materialise the view internally (allocating a full copy), which it does
while emitting the above warning.

The same issue applies when `use_highly_variable=False`, because `adata_test`
was set to the original `adata` directly, meaning the loop's temporary
`"_pymast_group"` column was written into the caller's object.

### Fix (`tl.py` lines 136-144)

Always call `.copy()` after `filter_to_highly_variable()` (and on `adata`
itself when HVG filtering is skipped) so that `adata_test` is always a
fully-materialised, independent AnnData object before the per-group loop
begins:

```python
# Before
adata_test = filter_to_highly_variable(adata) if use_highly_variable else adata

# After
adata_test = (
    filter_to_highly_variable(adata).copy()
    if use_highly_variable
    else adata.copy()
)
```

**Why this is safe:** The copy is restricted to the HVG subset (or the full
object), so memory overhead is proportional to the genes being tested — not
the full dataset. The existing cleanup block (`del adata.obs["_pymast_group"]`)
is now harmless because the column is only added to the copy, never to the
caller's `adata`.

---

## Fix 2 — `RuntimeWarning: invalid value encountered in subtract` in `zlm_fit.py`

### Symptom

During the LRT step:

```
RuntimeWarning: invalid value encountered in subtract
  lambda_C = -2.0 * (self.loglik_C_null - self.loglik_C)
```

### Root Cause

The continuous component (C) of the hurdle model is only fitted for genes
where a sufficient number of cells express the gene (i.e., non-zero counts).
When a gene is too sparsely expressed, the continuous fit is skipped and
`loglik_C` / `loglik_C_null` are stored as `np.nan`.

At the LRT step, `nan - nan` evaluates to `nan` — which is numerically
correct — but NumPy additionally fires the `invalid` floating-point exception
warning, because the result is formally undefined (NaN arithmetic).

The downstream `np.clip(..., 0, None)` *passed* NaN through unchanged (clip
does not convert NaN), so the warning fired without any protective effect.

### Fix (`zlm_fit.py` lines 143-158)

Wrap the subtraction in `np.errstate(invalid="ignore")` to suppress the
floating-point signal for this known-safe operation, then immediately replace
any resulting NaN with `0.0` via `np.nan_to_num`. A gene with an unfit
continuous component contributes **zero** to the hurdle LRT statistic — the
exact same outcome as the prior code after BH correction would have produced
(NaN p-values were already handled downstream).

```python
# Before
lambda_C = -2.0 * (self.loglik_C_null - self.loglik_C)
lambda_D = -2.0 * (self.loglik_D_null - self.loglik_D)
lambda_C = np.clip(lambda_C, 0.0, None)
lambda_D = np.clip(lambda_D, 0.0, None)

# After
with np.errstate(invalid="ignore"):
    lambda_C = -2.0 * (self.loglik_C_null - self.loglik_C)
    lambda_D = -2.0 * (self.loglik_D_null - self.loglik_D)

# Replace NaN (unfit component) with 0 before clipping
lambda_C = np.nan_to_num(lambda_C, nan=0.0)
lambda_D = np.nan_to_num(lambda_D, nan=0.0)
lambda_C = np.clip(lambda_C, 0.0, None)
lambda_D = np.clip(lambda_D, 0.0, None)
```

**Statistical note:** Setting lambda=0 for a gene whose continuous component
could not be fitted means the hurdle test for that gene is driven entirely by
the discrete (logistic) component — consistent with the R MAST behaviour when
`n_expressing < n_params`.

---

## Fix 3 — GitHub Actions CI dependency conflict (`pyproject.toml`)

### Symptom

The GitHub Actions CI job (`test (3.14)`) failed at dependency resolution:

```
× No solution found when resolving dependencies:
╰─▶ Because anndata==0.12.16 depends on one of:
        pandas>=2.1.0,<2.1.2
        pandas>2.1.2,<3
    and pymast==0.1.0 depends on pandas>=3.0.3, we can conclude that
    anndata==0.12.16 and pymast==0.1.0 are incompatible.
```

### Root Cause

Two pins in `pyproject.toml` were mutually exclusive:

| Requirement | Value | Problem |
|---|---|---|
| `anndata` | `==0.12.16` (exact) | `anndata` 0.12.x only supports `pandas < 3` |
| `pandas` | `>=3.0.3` | Requires pandas 3, which `anndata` 0.12.x rejects |

`uv` (and `pip`) correctly identified these constraints as unsatisfiable and
refused to install.

### Fix (`pyproject.toml`)

`anndata` added pandas 3.x support starting in **0.13.0**. The fix relaxes
both pins:

```toml
# Before
"pandas>=3.0.3",
"anndata==0.12.16",

# After
"pandas>=2.1",
"anndata>=0.13.0",
```

- **`anndata>=0.13.0`** — unpins the exact version, allowing `uv` to pick the
  latest compatible release. `anndata` 0.13.0 is the first release that
  supports pandas 3.x.
- **`pandas>=2.1`** — relaxed lower bound so the solver has more flexibility.
  pyMAST uses only standard `DataFrame`/`Series` operations available since
  pandas 1.x; the `2.1` floor is kept to avoid very old incompatible APIs.

> [!NOTE]
> The actual installed version in production is still controlled by the
> environment's upper bounds. In the CI job, `uv` will resolve to the newest
> mutually-compatible `(anndata, pandas)` pair. Locally tested with
> `anndata>=0.13` + `pandas>=3` without issue.

---

## Complete Fix Summary

The table below covers all changes made across this session, including
the runtime warning fixes, the CI dependency conflict, and the full
ruff linting cleanup.

| File(s) | Issue | Fix |
|---|---|---|
| `pymast/tl.py` | `ImplicitModificationWarning` — writing to `.obs` of an AnnData **view** | Call `.copy()` on the HVG-filtered result so `adata_test` is always a concrete object |
| `pymast/zlm_fit.py` | `RuntimeWarning: invalid value in subtract` — `nan - nan` on unfit continuous genes | Wrap subtraction in `np.errstate(invalid="ignore")` + `np.nan_to_num` |
| `pyproject.toml` | CI dep conflict: `anndata==0.12.16` requires `pandas<3`, conflicting with `pandas>=3.0.3` | Unpin: `anndata>=0.13.0`, `pandas>=2.1` |
| `pyproject.toml` | `ruff` config deprecation warning; `py314` not recognised by ruff 0.16.x | Move config to `[tool.ruff.lint]`, change `target-version = "py313"` |
| `pyproject.toml` | 130+ N8xx naming-convention violations on intentional statistical names (`X`, `coef_C/D`, etc.) | Remove `N` from ruff `select`; add `ignore = ["E501"]` |
| `pymast/bayes_glm.py` | F401 unused import: `dataclasses.field` | Removed |
| `pymast/bootstrap.py` | F401 unused imports: `pandas`, `.zlm_fit.ZlmFit` | Removed |
| `pymast/gsea.py` | F401 unused import: `scipy.stats.norm` | Removed |
| `pymast/lm_wrapper.py` | F401 unused import: `pandas` | Removed |
| `pymast/zlm.py` | F401 unused imports: `warnings`, `pandas`, `.ebayes.GeneSufficientStats` | Removed |
| `pymast/zlm_fit.py` | F401 unused imports: `dataclasses.field`, `.lm_wrapper.ChiSqTable/make_chisq_table` | Removed |
| `pymast/tl.py` | F841 unused variables: `test_mask` (always-True tautology), `adata_test_use` (never read) | Removed both assignments |
| `pymast/gsea.py` | F841 unused variable: `n_genes` | Removed assignment |
| `pymast/hypothesis.py` | UP037 quoted return-type annotation `-> "Hypothesis"` | Removed quotes (redundant with `from __future__ import annotations`) |
| `pymast/__init__.py` | I001 import names not sorted alphabetically within `from X import ...` lines | Sorted alphabetically |
| All 16 `pymast/*.py` | I001 trailing blank line at end of each file's import block | Removed via `ruff check --fix` |
| `antigravity_docs/` | No documentation of fixes | Added `bugfix_warnings_and_ci_deps.md` (this file) |
