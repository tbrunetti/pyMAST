"""
pymast — Pure Python implementation of the MAST R/Bioconductor package.

MAST: Model-based Analysis of Single-cell Transcriptomics
R/Bioconductor source: https://github.com/RGLab/MAST (main branch, v1.38.0)
Primary publication: Finak et al. (2015) Genome Biology 16:278
DOI: 10.1186/s13059-015-0844-5

pyMAST is a 100% pure Python, no-R reimplementation designed to be used
as a drop-in replacement for ``sc.tl.rank_genes_groups(method='mast')``.

Zero R dependencies — all statistical routines implemented with
scipy, numpy, statsmodels, and patsy.

Quick Start
-----------
>>> import pymast
>>> pymast.tl.rank_genes_groups(adata, groupby="leiden")
>>> import scanpy as sc
>>> sc.pl.rank_genes_groups(adata, n_genes=25)

Or use the low-level ZLM API:
>>> fit = pymast.zlm(adata, formula="~ condition + cdr")
>>> lr_results = fit.lr_test(contrast="condition")
"""

from __future__ import annotations

from . import tl
from .bootstrap import boot_zlm, BootZlmResult
from .ebayes import ebayes, EBayesResult, GeneSufficientStats
from .filter import burden_of_filtering, mast_filter
from .gsea import gsea_after_boot, GSEAResult
from .hypothesis import CoefficientHypothesis, Hypothesis
from .log_fc import get_log_fc
from .lrtest import LRT, lrt, lrtest, log_prod
from .readers import from_anndata, from_flat_df, from_matrix
from .utils import compute_cdr, compute_et_from_ct, freq, threshold_scRNA
from .zlm import zlm, ZlmFitter
from .zlm_fit import ZlmFit

__version__ = "0.1.0"
__author__ = "pyMAST Contributors"
__license__ = "MIT"
__doi__ = "10.1186/s13059-015-0844-5"  # MAST publication DOI

__all__ = [
    # Sub-modules
    "tl",
    # Main fitting
    "zlm",
    "ZlmFitter",
    "ZlmFit",
    # Hypothesis / testing
    "CoefficientHypothesis",
    "Hypothesis",
    "lrtest",
    "lrt",
    "LRT",
    "log_prod",
    # Empirical Bayes
    "ebayes",
    "EBayesResult",
    "GeneSufficientStats",
    # Log fold change
    "get_log_fc",
    # Bootstrap & GSEA
    "boot_zlm",
    "BootZlmResult",
    "gsea_after_boot",
    "GSEAResult",
    # Readers
    "from_anndata",
    "from_flat_df",
    "from_matrix",
    # Filtering
    "mast_filter",
    "burden_of_filtering",
    # Utilities
    "compute_cdr",
    "freq",
    "compute_et_from_ct",
    "threshold_scRNA",
    # Version
    "__version__",
]
