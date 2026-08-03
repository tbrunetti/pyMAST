"""
pymast/readers.py
=================
Data reader utilities: AnnData validation entry point and Fluidigm-specific
data loaders.

This module replaces the R ``Readers.R`` file which provides ``FromMatrix()``,
``FromFlatDF()``, and ``SceToSingleCellAssay()`` constructors for the
``SingleCellAssay`` S4 class.  In pyMAST, all analysis functions accept
``AnnData`` directly, so this module is primarily used for:

1. Validating and preparing an ``AnnData`` object for pyMAST.
2. Constructing AnnData objects from flat DataFrames (e.g. Fluidigm plate data).

R reference: RGLab/MAST main branch, R/Readers.R
             https://github.com/RGLab/MAST/blob/main/R/Readers.R
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from .anndata_utils import validate_anndata
from .utils import compute_cdr, compute_et_from_ct


# ---------------------------------------------------------------------------
# Primary entry point: from_anndata
# R: Readers.R — SceToSingleCellAssay(), FromMatrix()
# ---------------------------------------------------------------------------

def from_anndata(
    adata: AnnData,
    layer: str | None = None,
    *,
    compute_cdr_key: str | None = "cdr",
    inplace: bool = True,
) -> AnnData:
    """Validate an AnnData object for use with pyMAST and optionally compute CDR.

    This is the primary entry point for preparing data. It validates the object
    and ensures a CDR column is present in ``adata.obs`` for use as a nuisance
    covariate.

    R: Readers.R ``SceToSingleCellAssay()`` — converts SingleCellExperiment to
       SingleCellAssay and computes ``ngeneson`` (CDR) covariate.

    Parameters
    ----------
    adata : AnnData
        Input single-cell data.
    layer : str or None
        Layer to use for expression. ``None`` uses ``adata.X``.
    compute_cdr_key : str or None
        If not ``None``, compute CDR and store in ``adata.obs[compute_cdr_key]``.
        Set to ``None`` to skip CDR computation (if already present).
    inplace : bool
        If ``True``, modify ``adata`` in-place and return it.
        If ``False``, work on a copy.

    Returns
    -------
    AnnData
        The validated (and optionally CDR-annotated) AnnData.
    """
    validate_anndata(adata)

    if not inplace:
        adata = adata.copy()

    if compute_cdr_key is not None and compute_cdr_key not in adata.obs.columns:
        compute_cdr(adata, layer=layer, key_added=compute_cdr_key, inplace=True)

    return adata


# ---------------------------------------------------------------------------
# Fluidigm flat DataFrame → AnnData
# R: Readers.R — FromFlatDF()
# ---------------------------------------------------------------------------

def from_flat_df(
    df: pd.DataFrame,
    obs_cols: list[str],
    var_cols: list[str] | None = None,
    *,
    ct_threshold: float = 40.0,
    et_threshold: float = 0.0,
    is_ct: bool = True,
) -> AnnData:
    """Build an AnnData from a flat (tidy) Fluidigm plate DataFrame.

    Converts Ct values to Et values using :func:`compute_et_from_ct` and stores
    them in the resulting AnnData's ``.X``.

    R: Readers.R ``FromFlatDF()`` — constructs ``SingleCellAssay`` from a
       data.frame with cell-level and gene-level metadata columns interleaved.

    Parameters
    ----------
    df : pd.DataFrame
        Flat tidy DataFrame where rows are cell × gene observations.
        Must contain all columns listed in ``obs_cols`` and at least one
        gene column not in ``obs_cols``.
    obs_cols : list[str]
        Column names in ``df`` that contain cell-level metadata (e.g.
        ``["cell_id", "condition", "batch"]``).
    var_cols : list[str] or None
        Gene/feature column names. If ``None``, inferred as all columns
        not in ``obs_cols``.
    ct_threshold : float
        Ct threshold for ``compute_et_from_ct()``. Default 40.
    et_threshold : float
        Et floor for ``compute_et_from_ct()``. Default 0.
    is_ct : bool
        If ``True``, treat gene columns as Ct values and convert to Et.
        If ``False``, use values as-is.

    Returns
    -------
    AnnData
        ``adata.X`` contains Et (or raw) values (cells × genes).
        ``adata.obs`` contains cell metadata from ``obs_cols``.
        ``adata.var`` contains gene names as index.
    """
    # R: Readers.R FromFlatDF L~30 — separate cell metadata from expression
    obs_df = df[obs_cols].copy()

    if var_cols is None:
        var_cols = [c for c in df.columns if c not in obs_cols]

    expr_df = df[var_cols]

    if is_ct:
        # Convert Ct → Et; R: Readers.R calls computeEtFromCt internally
        expr_arr = compute_et_from_ct(
            expr_df,
            ct_threshold=ct_threshold,
            et_threshold=et_threshold,
        )
        if isinstance(expr_arr, pd.DataFrame):
            expr_arr = expr_arr.values
    else:
        expr_arr = np.asarray(expr_df, dtype=np.float64)

    var_df = pd.DataFrame(index=pd.Index(var_cols, name="gene"))

    adata = AnnData(X=expr_arr.astype(np.float32), obs=obs_df, var=var_df)
    return adata


# ---------------------------------------------------------------------------
# Matrix → AnnData
# R: Readers.R — FromMatrix()
# ---------------------------------------------------------------------------

def from_matrix(
    expression_matrix: np.ndarray,
    obs: pd.DataFrame | None = None,
    var: pd.DataFrame | None = None,
    *,
    gene_names: list[str] | None = None,
    cell_names: list[str] | None = None,
) -> AnnData:
    """Build an AnnData from a dense or sparse expression matrix.

    Convenience wrapper around ``AnnData()`` that mirrors R MAST's
    ``FromMatrix()`` constructor.

    R: Readers.R ``FromMatrix()`` — constructs ``SingleCellAssay`` from a
       matrix of expression values, with optional cell and gene metadata.

    Parameters
    ----------
    expression_matrix : np.ndarray
        Expression matrix, shape (n_cells, n_genes). For Ct-transformed
        Fluidigm data, use :func:`from_flat_df` instead.
    obs : pd.DataFrame or None
        Cell-level metadata. Index should be cell identifiers.
    var : pd.DataFrame or None
        Gene-level metadata. Index should be gene identifiers.
    gene_names : list[str] or None
        Gene names (used if ``var`` is ``None``).
    cell_names : list[str] or None
        Cell names (used if ``obs`` is ``None``).

    Returns
    -------
    AnnData
    """
    expression_matrix = np.asarray(expression_matrix, dtype=np.float32)
    n_cells, n_genes = expression_matrix.shape

    if obs is None:
        index = cell_names or [str(i) for i in range(n_cells)]
        obs = pd.DataFrame(index=pd.Index(index, name="cell"))

    if var is None:
        index = gene_names or [str(i) for i in range(n_genes)]
        var = pd.DataFrame(index=pd.Index(index, name="gene"))

    return AnnData(X=expression_matrix, obs=obs, var=var)
