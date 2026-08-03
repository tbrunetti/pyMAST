"""
pymast/filter.py
================
Cell and gene filtering utilities for MAST.

Provides ``mast_filter()`` and ``burden_of_filtering()`` which replicate the
R MAST filtering functions for removing low-quality cells/genes from
single-cell assay data.

R reference: RGLab/MAST main branch, R/filterEval.R
             https://github.com/RGLab/MAST/blob/main/R/filterEval.R

Citation: McDavid A, Finak G, et al. (2013). Data exploration, quality control
and testing in single-cell qPCR-based gene expression experiments.
*Bioinformatics* 29(4):461–467. DOI: 10.1093/bioinformatics/btt087
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from .anndata_utils import get_expression_matrix


# ---------------------------------------------------------------------------
# mast_filter
# R: filterEval.R — mast_filter() / filterEval()
# ---------------------------------------------------------------------------

def mast_filter(
    adata: AnnData,
    layer: str | None = None,
    *,
    min_cells_expressing: int | None = None,
    min_expressing_fraction: float | None = None,
    min_mean_expression: float | None = None,
    min_genes_per_cell: int | None = None,
    min_detection_fraction: float | None = None,
    inplace: bool = True,
    verbose: bool = True,
) -> AnnData | None:
    """Filter cells and genes based on detection thresholds.

    Replicates the R MAST ``mast_filter()`` function which filters out genes
    expressed in too few cells and cells expressing too few genes.

    R: filterEval.R ``filterEval()``
       - Gene filter: ``rowSums(assay > 0) >= min_cells_expressing``
       - Cell filter: ``colSums(assay > 0) >= min_genes_per_cell``

    Parameters
    ----------
    adata : AnnData
        Input data.
    layer : str or None
        Expression layer. ``None`` uses ``adata.X``.
    min_cells_expressing : int or None
        Minimum number of cells that must express a gene for it to be kept.
        Equivalent to R ``nCell`` filter.
    min_expressing_fraction : float or None
        Minimum fraction of cells expressing a gene. Alternative to
        ``min_cells_expressing``. If both provided, both are applied.
    min_mean_expression : float or None
        Minimum mean expression across all cells for a gene to be kept.
    min_genes_per_cell : int or None
        Minimum number of genes a cell must express to be kept.
        Equivalent to R ``nGene`` filter.
    min_detection_fraction : float or None
        Minimum fraction of genes detected per cell. Alternative to
        ``min_genes_per_cell``.
    inplace : bool
        If ``True``, filter ``adata`` in-place.
    verbose : bool
        Print summary of how many cells/genes were removed.

    Returns
    -------
    AnnData or None
        Filtered AnnData if ``inplace=False``, else ``None``.
    """
    X = get_expression_matrix(adata, layer=layer)  # (n_obs, n_vars)
    detected = (X > 0)

    # ---- Gene-level filters ----
    gene_mask = np.ones(adata.n_vars, dtype=bool)

    # R: filterEval.R — rowSums(assay > 0) >= nCell
    if min_cells_expressing is not None:
        gene_mask &= detected.sum(axis=0) >= min_cells_expressing

    if min_expressing_fraction is not None:
        gene_mask &= detected.mean(axis=0) >= min_expressing_fraction

    if min_mean_expression is not None:
        gene_mask &= X.mean(axis=0) >= min_mean_expression

    # ---- Cell-level filters ----
    cell_mask = np.ones(adata.n_obs, dtype=bool)

    # R: filterEval.R — colSums(assay > 0) >= nGene
    if min_genes_per_cell is not None:
        cell_mask &= detected.sum(axis=1) >= min_genes_per_cell

    if min_detection_fraction is not None:
        cell_mask &= detected.mean(axis=1) >= min_detection_fraction

    n_genes_removed = int((~gene_mask).sum())
    n_cells_removed = int((~cell_mask).sum())

    if verbose:
        print(
            f"[mast_filter] Removing {n_genes_removed} genes "
            f"({adata.n_vars - n_genes_removed} kept) | "
            f"Removing {n_cells_removed} cells "
            f"({adata.n_obs - n_cells_removed} kept)"
        )

    if inplace:
        # Filter genes first (columns), then cells (rows)
        adata._inplace_subset_var(gene_mask)
        adata._inplace_subset_obs(cell_mask)
        return None
    else:
        return adata[cell_mask, :][:, gene_mask].copy()


# ---------------------------------------------------------------------------
# burden_of_filtering
# R: filterEval.R — burdenOfFiltering()
# ---------------------------------------------------------------------------

def burden_of_filtering(
    adata: AnnData,
    layer: str | None = None,
    *,
    min_cells_list: list[int] | None = None,
) -> pd.DataFrame:
    """Compute the 'burden of filtering' across a range of thresholds.

    Returns a summary DataFrame showing how many genes and cells would be
    retained at each threshold value, allowing the user to pick an appropriate
    filter cutoff.

    R: filterEval.R ``burdenOfFiltering()`` — sweeps thresholds and reports
       numbers of retained features.

    Parameters
    ----------
    adata : AnnData
        Input data.
    layer : str or None
        Expression layer. ``None`` uses ``adata.X``.
    min_cells_list : list[int] or None
        Thresholds to evaluate. Default: ``[1, 2, 5, 10, 20, 50]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``min_cells_expressing``, ``n_genes_retained``,
        ``n_cells_retained`` (cells with ≥ 1 gene after gene filter).
    """
    if min_cells_list is None:
        min_cells_list = [1, 2, 5, 10, 20, 50]

    X = get_expression_matrix(adata, layer=layer)
    detected = (X > 0)

    rows = []
    for threshold in min_cells_list:
        gene_mask = detected.sum(axis=0) >= threshold
        # After gene filtering, how many cells still have ≥ 1 expressed gene?
        X_filtered = detected[:, gene_mask]
        cell_mask = X_filtered.sum(axis=1) >= 1
        rows.append({
            "min_cells_expressing": threshold,
            "n_genes_retained": int(gene_mask.sum()),
            "n_cells_retained": int(cell_mask.sum()),
        })

    return pd.DataFrame(rows)
