# pyMAST Execution Task List

> Approved plan: `implementation_plan_v2_final.md`  
> Revision log: `implementation_revisions.md`  
> Branch strategy: work on `develop`, merge to `main` via PR  
> Started: 2026-08-03

---

## Phase 0 — Repo Scaffold & Infrastructure

- [x] Create `antigravity_docs/implementation_plan_v1_original.md` (with user review annotations)
- [x] Create `antigravity_docs/implementation_plan_v2_final.md` (approved final plan)
- [x] Create `antigravity_docs/task.md` (this file)
- [x] Create `antigravity_docs/implementation_revisions.md` (revision log + prompt traceability)
- [x] Initialize git repo, set up `main` + `develop` branches
- [x] Create `pyproject.toml`
- [x] Create `Dockerfile`
- [x] Create `docker-compose.yml`
- [ ] Create `.github/workflows/ci.yml`
- [ ] Create `.github/workflows/release.yml`
- [x] Create `CHANGELOG.md`
- [x] Create `LICENSE` (MIT)
- [x] Create `README.md`

---

## Phase 1 — Core Package Skeleton

- [x] `pymast/__init__.py` — public API exports, `__version__`
- [x] `pymast/anndata_utils.py` — internal AnnData validation helpers
- [x] `pymast/readers.py` — `from_anndata()`, Fluidigm wrappers (lowest priority)
- [x] `pymast/utils.py` — `compute_cdr()`, `freq()`, `compute_et_from_ct()`
- [x] `pymast/filter.py` — `mast_filter()`, `burden_of_filtering()`
- [x] `pymast/hypothesis.py` — `CoefficientHypothesis`, `Hypothesis`

---

## Phase 2 — Statistical Core (Highest Fidelity)

All functions reference R source as `# R: <file>.R L<start>-<end>` (main branch, v1.38.0)

- [x] `pymast/lrtest.py` — `log_prod()`, `lrtest()`, `lrt()`, `LRT()`
- [x] `pymast/ebayes.py` — `get_ss_g_r_ng()`, `get_marginal_hyperlikelihood()`, `solve_mom()`, `ebayes()`
- [x] `pymast/bayes_glm.py` — `BayesGLMLike` (IRLS + Cauchy prior, scale=2.5)
- [x] `pymast/glm_wrapper.py` — `GLMLike` (statsmodels logit wrapper)
- [x] `pymast/lm_wrapper.py` — `LMlike` ABC, `make_chisq_table()`, `wald_test()`, `rotate_model_matrix()`

---

## Phase 3 — ZLM Fitting Engine

- [x] `pymast/zlm.py` — `zlm()`, `ZlmFitter`, CDR integration, `n_jobs` parallelism
- [x] `pymast/zlm_fit.py` — `ZlmFit`, `collect_summaries()`, `lr_test()`, `wald_test()`
- [x] `pymast/log_fc.py` — `get_log_fc()`

---

## Phase 4 — Bootstrap & GSEA

- [x] `pymast/bootstrap.py` — `boot_zlm()`
- [x] `pymast/gsea.py` — `gsea_after_boot()` (Subramanian et al. 2005)

---

## Phase 5 — High-Level Scanpy-Compatible API

- [x] `pymast/tl.py` — `rank_genes_groups()` writing to `adata.uns['rank_genes_groups']` + `adata.uns['mast']`

---

## Phase 6 — Test Suite

- [x] `tests/conftest.py` — synthetic AnnData fixture (200 cells × 50 genes, vbeta-like)
- [x] `tests/test_lrtest.py` — 13 tests; numerical to 1e-6 rtol ✅
- [x] `tests/test_ebayes.py` — 13 tests ✅
- [x] `tests/test_zlm.py` — 8 tests (7 failed on first run, all fixed) ✅
- [x] `tests/test_filter.py` — 5 tests ✅
- [x] `tests/test_readers.py` — 3 tests ✅
- [x] `tests/test_utils.py` — 5 tests ✅
- [ ] `tests/test_rank_genes_groups.py` — end-to-end scanpy compat test (v1.1)
- [ ] `tests/test_bayes_glm.py` — Bayesian GLM unit tests (v1.1)
- [ ] `tests/test_wald_test.py` — Wald test unit tests (v1.1)

**Current test result: 49/49 passed, 0 warnings**

### Post-Implementation Bug Fixes (see `implementation_revisions.md`)

- [x] Fix `"BH"` → `"fdr_bh"` in `lrtest.py`, `zlm_fit.py`, `tl.py`, `gsea.py`
- [x] Fix `patsy.dmatrices` → `patsy.dmatrix` in `zlm.py` (one-sided formula support)
- [x] Move `ZlmFitter` instantiation before formula parsing for correct error ordering
- [x] Cast `obs_df.index.astype(str)` in `readers.py` to suppress AnnData warning

---

## Phase 7 — Examples & Documentation

- [ ] `examples/01_basic_workflow.ipynb`
- [ ] `examples/02_anndata_integration.ipynb`
- [ ] `examples/03_comparison_with_r.ipynb` (numerical parity vs R MAST)
- [ ] `antigravity_docs/walkthrough.md`

---

## Phase 8 — Docker, CI, Release

- [ ] Docker build + pytest verification (`docker build -t pymast:latest . && docker run --rm pymast:latest`)
- [ ] `.github/workflows/ci.yml` — matrix Python 3.14, pytest + ruff
- [ ] `.github/workflows/release.yml` — publish on `v*` tag
- [ ] `git tag v0.1.0` on `main` (after Docker verified)
- [ ] `CHANGELOG.md` update for v0.1.0

---

*Last updated: 2026-08-03 | v0.1.0-rc (all tests passing, pending Docker + tag)*
