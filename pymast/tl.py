"""
pymast/tl.py
============
High-level Scanpy-compatible API for pyMAST.

Provides ``rank_genes_groups()`` as a drop-in replacement for
``scanpy.tl.rank_genes_groups(method='mast')``.  Results are stored in
``adata.uns`` using the exact same structured NumPy recarray format that
Scanpy uses, so all downstream ``sc.pl.*`` plotting functions work without
modification.

Additionally stores MAST-specific decomposition in ``adata.uns['mast']``.

R reference: RGLab/MAST main branch — overall workflow orchestration.
Scanpy reference: https://scanpy.readthedocs.io/en/stable/generated/
                  scanpy.tl.rank_genes_groups.html

Usage
-----
>>> import pymast
>>> pymast.tl.rank_genes_groups(adata, groupby="leiden")
>>> # Then use sc.pl.rank_genes_groups(adata) as usual
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from statsmodels.stats.multitest import multipletests

from .anndata_utils import (
    filter_to_highly_variable,
    get_expression_matrix,
    validate_anndata,
)
from .utils import compute_cdr
from .zlm import zlm


# ---------------------------------------------------------------------------
# rank_genes_groups — Scanpy-compatible MAST differential expression
# ---------------------------------------------------------------------------

def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    *,
    groups: str | list[str] = "all",
    reference: str = "rest",
    layer: str | None = None,
    formula_extra: str | None = None,
    use_highly_variable: bool = True,
    ebayes: bool = True,
    method: Literal["bayesglm", "glm"] = "bayesglm",
    n_jobs: int = -1,
    key_added: str = "rank_genes_groups",
    copy: bool = False,
    cdr_key: str = "cdr",
) -> AnnData | None:
    """Run MAST differential expression, storing results in Scanpy-compatible format.

    Drop-in replacement for ``sc.tl.rank_genes_groups(method='mast')``.

    Results are stored in:

    - ``adata.uns[key_added]``: Scanpy-compatible structured array (same as
      ``sc.tl.rank_genes_groups``), so all ``sc.pl.*`` plotting functions work.
    - ``adata.uns['mast']``: MAST-specific decomposition (discrete + continuous
      component p-values, log fold changes, CDR, etc.).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str
        Column in ``adata.obs`` that defines the groups to test.
    groups : str or list[str]
        Groups to test. ``"all"`` tests all unique values in ``groupby``.
    reference : str
        Reference group. ``"rest"`` uses all other cells as the reference.
        Or specify a specific group name from ``groupby``.
    layer : str or None
        Expression layer to use. ``None`` uses ``adata.X``.
    formula_extra : str or None
        Additional covariates to add to the formula, e.g. ``"+ batch"``.
        The full formula will be ``"~ group + cdr {formula_extra}"``.
    use_highly_variable : bool
        If ``True`` and ``adata.var["highly_variable"]`` exists, restrict
        testing to highly variable genes.
    ebayes : bool
        Apply empirical Bayes variance shrinkage to continuous component.
    method : str
        Discrete component fitter: ``"bayesglm"`` (default) or ``"glm"``.
    n_jobs : int
        Parallelism (``-1`` = all cores, ``1`` = single-threaded).
    key_added : str
        Key in ``adata.uns`` where Scanpy-compatible results are stored.
        Default: ``"rank_genes_groups"`` (same as scanpy).
    copy : bool
        Return a copy of ``adata`` rather than modifying in-place.
    cdr_key : str
        Column name for the CDR covariate in ``adata.obs``.

    Returns
    -------
    AnnData or None
        Returns ``adata`` copy if ``copy=True``, else ``None``.

    Examples
    --------
    >>> import pymast
    >>> pymast.tl.rank_genes_groups(adata, groupby="leiden")
    >>> import scanpy as sc
    >>> sc.pl.rank_genes_groups(adata, n_genes=25)
    >>> df = sc.get.rank_genes_groups_df(adata, group="0")
    """
    validate_anndata(adata)

    if copy:
        adata = adata.copy()

    # Compute CDR if not already present
    if cdr_key not in adata.obs.columns:
        compute_cdr(adata, layer=layer, key_added=cdr_key, inplace=True)

    # Determine groups to test
    all_groups = list(adata.obs[groupby].astype(str).unique())
    if groups == "all":
        test_groups = all_groups
    else:
        test_groups = [str(g) for g in groups]

    # Restrict to HVGs if requested.
    # Always materialise as a copy so that writing to adata_test.obs never
    # triggers AnnData's ImplicitModificationWarning (which fires when you
    # assign to a column of a *view* rather than a real AnnData object).
    adata_test = (
        filter_to_highly_variable(adata).copy()
        if use_highly_variable
        else adata.copy()
    )
    gene_names = list(adata_test.var_names)
    n_genes = len(gene_names)

    # Pre-allocate output arrays (Scanpy recarray format)
    # Structured dtype with one field per group
    group_dtype = np.dtype([(str(g), object) for g in test_groups])
    float_dtype = np.dtype([(str(g), np.float32) for g in test_groups])

    names_arr       = np.empty(n_genes, dtype=group_dtype)
    scores_arr      = np.zeros(n_genes, dtype=float_dtype)
    pvals_arr       = np.ones(n_genes, dtype=float_dtype)
    pvals_adj_arr   = np.ones(n_genes, dtype=float_dtype)
    logfc_arr       = np.zeros(n_genes, dtype=float_dtype)

    # Storage for MAST-specific decomposition
    mast_results: dict[str, dict] = {}

    # ---- Per-group MAST fits ----
    for group in test_groups:
        group_str = str(group)

        # Build reference mask
        group_labels = adata_test.obs[groupby].astype(str)
        if reference == "rest":
            # Create binary group column
            adata_test.obs["_pymast_group"] = (group_labels == group_str).astype(int)
        else:
            ref_str = str(reference)
            if ref_str not in all_groups:
                raise ValueError(
                    f"Reference group '{reference}' not found in '{groupby}'. "
                    f"Available: {all_groups}"
                )
            # Only include focal group vs reference
            mask = (group_labels == group_str) | (group_labels == ref_str)
            adata_sub = adata_test[mask].copy()
            adata_sub.obs["_pymast_group"] = (
                adata_sub.obs[groupby].astype(str) == group_str
            ).astype(int)
            adata_test.obs["_pymast_group"] = (group_labels == group_str).astype(int)

        # Build formula
        formula = f"~ _pymast_group + {cdr_key}"
        if formula_extra:
            formula = formula + " " + formula_extra.lstrip("+ ").lstrip()

        # Fit ZLM
        fit = zlm(
            adata_test,
            formula=formula,
            contrast_column="_pymast_group",
            layer=layer,
            method=method,
            use_highly_variable=False,  # already filtered
            use_ebayes=ebayes,
            n_jobs=n_jobs,
            verbose=True,
            cdr_key=cdr_key,
        )

        # Run LRT
        lr_df = fit.lr_test(contrast=None)

        # Compute log fold changes
        # contrast = column for _pymast_group in coef_names
        try:
            grp_idx = fit.coef_names.index(
                next(c for c in fit.coef_names if "_pymast_group" in c)
            )
        except (StopIteration, ValueError):
            grp_idx = 1  # fallback

        contrast_vec = np.zeros(fit.n_params)
        contrast_vec[grp_idx] = 1.0
        logfc_df = fit.get_log_fc(contrast_vec)

        # Compute detection fraction per group
        X_expr = get_expression_matrix(adata_test, layer=layer)
        group_bool = adata_test.obs["_pymast_group"].values.astype(bool)
        pts_group = (X_expr[group_bool, :] > 0).mean(axis=0)
        pts_rest  = (X_expr[~group_bool, :] > 0).mean(axis=0)

        # Map results back to gene_names (fit may have different gene order)
        fit_gene_idx = {g: i for i, g in enumerate(fit.gene_names)}
        for j, gene in enumerate(gene_names):
            if gene not in fit_gene_idx:
                continue
            gi = fit_gene_idx[gene]
            p_hurdle = lr_df["p_hurdle"].iloc[gi]
            names_arr[group_str][j] = gene
            # Score = -log10(p_hurdle) for ranking (matching scanpy convention)
            scores_arr[group_str][j] = (
                -np.log10(max(p_hurdle, 1e-300)) if np.isfinite(p_hurdle) else 0.0
            )
            pvals_arr[group_str][j]    = float(p_hurdle) if np.isfinite(p_hurdle) else 1.0
            logfc_arr[group_str][j]    = float(logfc_df["log2FC"].iloc[gi])

        # BH correction across genes for this group
        p_raw = np.array([float(pvals_arr[group_str][j]) for j in range(n_genes)])
        finite_mask = np.isfinite(p_raw) & (p_raw > 0)
        p_adj = p_raw.copy()
        if finite_mask.sum() > 1:
            _, p_adj_finite, _, _ = multipletests(p_raw[finite_mask], method="fdr_bh")
            p_adj[finite_mask] = p_adj_finite
        for j in range(n_genes):
            pvals_adj_arr[group_str][j] = float(p_adj[j])

        # Sort genes by score (descending) within this group
        sort_idx = np.argsort(-scores_arr[group_str])
        for field in [names_arr, scores_arr, pvals_arr, pvals_adj_arr, logfc_arr]:
            field[group_str] = field[group_str][sort_idx]

        # Store MAST-specific results
        mast_results[group_str] = {
            "hurdle_pvals": lr_df["p_hurdle"].values,
            "disc_pvals":   lr_df["p_D"].values,
            "cont_pvals":   lr_df["p_C"].values,
            "disc_logfc":   logfc_df["logFC_D"].values,
            "cont_logfc":   logfc_df["logFC_C"].values,
            "coef_C":       fit.coef_C,
            "coef_D":       fit.coef_D,
            "coef_names":   fit.coef_names,
            "gene_names":   fit.gene_names,
            "pts_group":    pts_group,
            "pts_rest":     pts_rest,
        }

    # ---- Store Scanpy-compatible results ----
    # R equivalent: adata.uns['rank_genes_groups'] = ...
    pts_df = pd.DataFrame(
        {g: mast_results[str(g)]["pts_group"] for g in test_groups},
        index=gene_names,
    )

    adata.uns[key_added] = {
        "params": {
            "groupby": groupby,
            "reference": reference,
            "groups": test_groups,
            "method": "mast",
            "use_raw": False,
            "layer": layer,
            "corr_method": "benjamini-hochberg",
        },
        "names":         names_arr,
        "scores":        scores_arr,
        "pvals":         pvals_arr,
        "pvals_adj":     pvals_adj_arr,
        "logfoldchanges": logfc_arr,
        "pts":           pts_df,
    }

    # ---- Store MAST-specific results ----
    cdr_vals = adata.obs[cdr_key].values if cdr_key in adata.obs.columns else None
    adata.uns["mast"] = {
        "groups": mast_results,
        "cdr":    cdr_vals,
        "params": {
            "formula_extra": formula_extra,
            "ebayes": ebayes,
            "method": method,
            "cdr_key": cdr_key,
        },
    }

    # Clean up temporary column
    if "_pymast_group" in adata.obs.columns:
        del adata.obs["_pymast_group"]

    if copy:
        return adata
    return None
