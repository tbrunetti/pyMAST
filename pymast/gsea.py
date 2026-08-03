"""
pymast/gsea.py
==============
Bootstrap-based Gene Set Enrichment Analysis (GSEA) after ZLM fitting.

Reimplements R MAST's ``gseaAfterBoot()`` from ``GSEA-by-boot.R``.  Uses the
bootstrap ZLM log fold change distribution from :func:`pymast.bootstrap.boot_zlm`
to compute normalized enrichment scores (NES) for gene sets by comparing
observed statistics to the bootstrap null distribution.

The algorithm follows Subramanian et al. (2005) enrichment score computation
applied to the MAST bootstrap framework.

R reference: RGLab/MAST main branch, R/GSEA-by-boot.R
             https://github.com/RGLab/MAST/blob/main/R/GSEA-by-boot.R

Citation: Subramanian A, Tamayo P, Mootha VK, et al. (2005). Gene set
enrichment analysis: A knowledge-based approach for interpreting genome-wide
expression profiles. PNAS 102(43):15545–15550.
DOI: 10.1073/pnas.0506580102
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .bootstrap import BootZlmResult


# ---------------------------------------------------------------------------
# GSEAResult
# ---------------------------------------------------------------------------

@dataclass
class GSEAResult:
    """Result of GSEA after bootstrap ZLM.

    Attributes
    ----------
    gene_set_name : str
        Name of the gene set tested.
    es : float
        Observed enrichment score.
    nes : float
        Normalized enrichment score (ES / mean of bootstrap null ESs).
    p_value : float
        One-sided p-value from bootstrap null distribution.
    p_adj : float
        BH-adjusted p-value (set to NaN if only one gene set tested).
    n_genes_in_set : int
        Number of genes in the gene set that were also in the fitted model.
    direction : str
        ``"up"`` or ``"down"`` based on sign of NES.
    """
    gene_set_name: str
    es: float
    nes: float
    p_value: float
    p_adj: float
    n_genes_in_set: int
    direction: str


# ---------------------------------------------------------------------------
# _enrichment_score
# R: GSEA-by-boot.R — .enrichmentScore(stat, in.set)
# ---------------------------------------------------------------------------

def _enrichment_score(stat: np.ndarray, in_set: np.ndarray) -> float:
    """Compute the Kolmogorov-Smirnov-style enrichment score.

    R: GSEA-by-boot.R ``.enrichmentScore(stat, in.set)``
       Ranks genes by statistic, computes running sum enrichment score.

    Parameters
    ----------
    stat : np.ndarray
        Per-gene statistics (e.g. log fold changes), length n_genes.
    in_set : np.ndarray
        Boolean array of length n_genes; True for genes in the gene set.

    Returns
    -------
    float
        Enrichment score (maximum deviation of running sum from zero).
    """
    # R: GSEA-by-boot.R — rank genes by statistic (descending)
    order = np.argsort(-stat)
    in_set_ordered = in_set[order]
    stat_ordered = np.abs(stat[order])

    n_total = len(stat)
    n_in_set = int(in_set.sum())
    n_not_in_set = n_total - n_in_set

    if n_in_set == 0 or n_not_in_set == 0:
        return 0.0

    # Running enrichment sum
    # R: GSEA-by-boot.R — sum(in.set * |stat|) / sum(in.set * |stat|) = Phit
    phit_denom = float(np.sum(stat_ordered[in_set_ordered]))
    pmiss_denom = float(n_not_in_set)

    running_sum = 0.0
    max_dev = 0.0
    min_dev = 0.0

    for i in range(n_total):
        if in_set_ordered[i]:
            running_sum += stat_ordered[i] / phit_denom if phit_denom > 0 else 1.0 / n_in_set
        else:
            running_sum -= 1.0 / pmiss_denom
        max_dev = max(max_dev, running_sum)
        min_dev = min(min_dev, running_sum)

    # ES = max deviation with largest absolute value
    if abs(max_dev) >= abs(min_dev):
        return max_dev
    return min_dev


# ---------------------------------------------------------------------------
# gsea_after_boot
# R: GSEA-by-boot.R — gseaAfterBoot(bootResult, gene.sets, ...)
# ---------------------------------------------------------------------------

def gsea_after_boot(
    boot_result: BootZlmResult,
    gene_sets: dict[str, list[str]],
    *,
    adjust_method: str = "fdr_bh",
    min_genes: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run GSEA using the bootstrap ZLM log fold change distribution.

    For each gene set, computes the enrichment score (ES) using the observed
    log fold changes, then normalizes by the mean ES from bootstrap replicates
    to get the NES.  P-values are estimated from the bootstrap null distribution.

    R: GSEA-by-boot.R ``gseaAfterBoot(bootResult, gene.sets, ...)``

    Parameters
    ----------
    boot_result : BootZlmResult
        Result from :func:`~pymast.bootstrap.boot_zlm`.
    gene_sets : dict[str, list[str]]
        Dictionary mapping gene set names to lists of gene names.
        E.g. ``{"HALLMARK_GLYCOLYSIS": ["ALDOA", "ALDOB", ...]}``.
    adjust_method : str
        Multiple testing correction method (default: ``"BH"``).
    min_genes : int
        Minimum number of genes in the set that are in the model to test
        a gene set (default: 5).
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        One row per gene set tested. Columns:
        ``["gene_set", "es", "nes", "p_value", "p_adj", "n_genes_in_set",
           "direction"]``.
        Sorted by absolute NES descending.
    """
    from statsmodels.stats.multitest import multipletests

    gene_names = np.array(boot_result.gene_names)
    n_genes = len(gene_names)
    n_boot = boot_result.n_boot

    # Use the mean log2FC across bootstrap as the observed statistic
    # R: GSEA-by-boot.R — stat <- colMeans(bootResult$logFC)
    observed_logfc = np.nanmean(boot_result.logfc_boot, axis=0)  # (n_genes,)

    results: list[GSEAResult] = []

    gene_set_names = list(gene_sets.keys())
    if verbose:
        from tqdm import tqdm
        gene_set_iter = tqdm(gene_set_names, desc="GSEA gene sets", unit="set")
    else:
        gene_set_iter = gene_set_names

    for gs_name in gene_set_iter:
        gs_genes = gene_sets[gs_name]
        in_set = np.isin(gene_names, gs_genes)
        n_in_set = int(in_set.sum())

        if n_in_set < min_genes:
            continue

        # Observed ES
        # R: GSEA-by-boot.R — es.obs <- .enrichmentScore(stat.obs, in.set)
        es_obs = _enrichment_score(observed_logfc, in_set)

        # Bootstrap null distribution of ES
        # R: GSEA-by-boot.R — es.boot <- apply(bootResult$logFC, 1, .enrichmentScore, in.set)
        es_boot = np.array([
            _enrichment_score(boot_result.logfc_boot[b], in_set)
            for b in range(n_boot)
        ])

        # Normalize: NES = ES_obs / mean(|ES_boot| where same direction)
        # R: GSEA-by-boot.R — nes <- es.obs / mean(es.boot[same.sign])
        same_sign = es_boot * es_obs >= 0
        if same_sign.sum() > 0:
            nes_denom = float(np.mean(np.abs(es_boot[same_sign])))
        else:
            nes_denom = float(np.mean(np.abs(es_boot))) if len(es_boot) > 0 else 1.0

        nes = es_obs / nes_denom if nes_denom > 0 else es_obs

        # P-value: fraction of same-direction null ESs with |ES_null| >= |ES_obs|
        # R: GSEA-by-boot.R — p.val <- mean(|es.boot| >= |es.obs|, same sign)
        if same_sign.sum() > 0:
            p_val = float(np.mean(np.abs(es_boot[same_sign]) >= abs(es_obs)))
        else:
            p_val = 1.0

        direction = "up" if nes >= 0 else "down"

        results.append(GSEAResult(
            gene_set_name=gs_name,
            es=float(es_obs),
            nes=float(nes),
            p_value=p_val,
            p_adj=np.nan,  # filled in below
            n_genes_in_set=n_in_set,
            direction=direction,
        ))

    if not results:
        return pd.DataFrame(columns=[
            "gene_set", "es", "nes", "p_value", "p_adj", "n_genes_in_set", "direction"
        ])

    result_df = pd.DataFrame([
        {
            "gene_set": r.gene_set_name,
            "es": r.es,
            "nes": r.nes,
            "p_value": r.p_value,
            "p_adj": r.p_adj,
            "n_genes_in_set": r.n_genes_in_set,
            "direction": r.direction,
        }
        for r in results
    ])

    # BH correction
    # R: GSEA-by-boot.R — p.adjust(p.vals, method="BH")
    finite_p = np.isfinite(result_df["p_value"].values)
    p_adj = result_df["p_adj"].values.copy()
    if finite_p.sum() > 1:
        _, p_adj_finite, _, _ = multipletests(
            result_df["p_value"].values[finite_p], method=adjust_method
        )
        p_adj[finite_p] = p_adj_finite
    elif finite_p.sum() == 1:
        p_adj[finite_p] = result_df["p_value"].values[finite_p]

    result_df["p_adj"] = p_adj

    # Sort by |NES| descending
    result_df = result_df.sort_values("nes", key=np.abs, ascending=False).reset_index(drop=True)

    return result_df
