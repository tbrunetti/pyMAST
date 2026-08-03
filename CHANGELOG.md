# Changelog

All notable changes to pyMAST will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Pure Python implementation of MAST two-part hurdle model
- Mathematical fidelity to RGLab/MAST v1.38.0 (main branch)
- AnnData-native API: `pymast.tl.rank_genes_groups()`
- Scanpy-compatible output in `adata.uns['rank_genes_groups']`
- MAST-specific extended results in `adata.uns['mast']`
- Empirical Bayes variance shrinkage (`ebayes.py`)
- Bayesian GLM with Cauchy prior (`bayes_glm.py`)
- Bootstrap GSEA (`gsea.py`)
- Fluidigm utility wrappers (`utils.py`)
- Docker support + GitHub Actions CI
- UV-installable from GitHub

---

## [0.1.0] — TBD

*Initial release*
