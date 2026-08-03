# pyMAST Execution Task List

> Approved plan: `implementation_plan_v2_final.md`
> Branch strategy: work on `develop`, merge to `main` via PR
> Started: 2026-08-03

---

## Phase 0 — Repo Scaffold & Infrastructure

- [x] Create `antigravity_docs/implementation_plan_v1_original.md` (with user review annotations)
- [x] Create `antigravity_docs/implementation_plan_v2_final.md` (approved final plan)
- [x] Create `antigravity_docs/task.md` (this file)
- [ ] Initialize git repo, set up `main` + `develop` branches
- [ ] Create `pyproject.toml`
- [ ] Create `Dockerfile`
- [ ] Create `docker-compose.yml`
- [ ] Create `.github/workflows/ci.yml`
- [ ] Create `.github/workflows/release.yml`
- [ ] Create `CHANGELOG.md`
- [ ] Create `LICENSE` (MIT)
- [ ] Create `README.md`

---

## Phase 1 — Core Package Skeleton

- [ ] `pymast/__init__.py` — public API exports, `__version__`
- [ ] `pymast/anndata_utils.py` — internal AnnData validation helpers
- [ ] `pymast/readers.py` — `from_anndata()`, Fluidigm wrappers (lowest priority)
- [ ] `pymast/utils.py` — `compute_cdr()`, `freq()`, `compute_et_from_ct()`
- [ ] `pymast/filter.py` — `mast_filter()`, `burden_of_filtering()`
- [ ] `pymast/hypothesis.py` — `CoefficientHypothesis`, `Hypothesis`

---

## Phase 2 — Statistical Core (Highest Fidelity)

All functions reference R source as `# R: <file>.R L<start>-<end>` (main branch, v1.38.0)

- [ ] `pymast/lrtest.py` — `log_prod()`, `lrtest()`, `lrt()`, `LRT()`
- [ ] `pymast/ebayes.py` — `get_ss_g_r_ng()`, `get_marginal_hyperlikelihood()`, `solve_mom()`, `ebayes()`
- [ ] `pymast/bayes_glm.py` — `BayesGLMLike` (IRLS + Cauchy prior, scale=2.5)
- [ ] `pymast/glm_wrapper.py` — `GLMLike` (statsmodels logit wrapper)
- [ ] `pymast/lm_wrapper.py` — `LMlike` ABC, `make_chisq_table()`, `wald_test()`, `rotate_model_matrix()`

---

## Phase 3 — ZLM Fitting Engine

- [ ] `pymast/zlm.py` — `zlm()`, `ZlmFitter`, CDR integration, `n_jobs` parallelism
- [ ] `pymast/zlm_fit.py` — `ZlmFit`, `collect_summaries()`, `lr_test()`, `wald_test()`
- [ ] `pymast/log_fc.py` — `get_log_fc()`

---

## Phase 4 — Bootstrap & GSEA

- [ ] `pymast/bootstrap.py` — `boot_zlm()`
- [ ] `pymast/gsea.py` — `gsea_after_boot()` (Subramanian et al. 2005)

---

## Phase 5 — High-Level Scanpy-Compatible API

- [ ] `pymast/tl.py` — `rank_genes_groups()` writing to `adata.uns['rank_genes_groups']` + `adata.uns['mast']`

---

## Phase 6 — Test Suite

- [ ] `tests/conftest.py` — synthetic AnnData fixture
- [ ] `tests/test_lrtest.py` — numerical to 1e-6 rtol
- [ ] `tests/test_ebayes.py`
- [ ] `tests/test_bayes_glm.py`
- [ ] `tests/test_wald_test.py`
- [ ] `tests/test_zlm.py`
- [ ] `tests/test_filter.py`
- [ ] `tests/test_readers.py`
- [ ] `tests/test_rank_genes_groups.py` — scanpy compat output format

---

## Phase 7 — Examples & Documentation

- [ ] `examples/01_basic_workflow.ipynb`
- [ ] `examples/02_anndata_integration.ipynb`
- [ ] `examples/03_comparison_with_r.ipynb`
- [ ] `antigravity_docs/walkthrough.md`

---

## Phase 8 — Docker, CI, Release

- [ ] Docker build + pytest verification
- [ ] `git tag v0.1.0` on `main`
- [ ] `CHANGELOG.md` update

---
*Last updated: 2026-08-03*
