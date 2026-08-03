"""
pymast/lrtest.py
================
Two-group likelihood ratio test (LRT) for the MAST hurdle model.

Implements the simple two-group LRT from R MAST — used for the fast
``lrtest()`` path where two groups are compared directly without fitting
a full ZLM model.  This is separate from the full-model LRT in ``zlm_fit.py``
which uses the fitted ZlmFit object.

R reference: RGLab/MAST main branch, R/lrtest.R
             https://github.com/RGLab/MAST/blob/main/R/lrtest.R

Citation: Finak G et al. (2015). MAST: a flexible statistical framework.
Genome Biology 16:278. DOI: 10.1186/s13059-015-0844-5

Original LRT theory: Neyman J & Pearson ES (1933). On the problem of the most
efficient tests of statistical hypotheses. Phil Trans R Soc A 231:289–337.
DOI: 10.1098/rsta.1933.0009
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# log_prod
# R: lrtest.R — logProd(prod, logand)
# ---------------------------------------------------------------------------

def log_prod(n: float, ratio: float) -> float:
    """Compute n * log(ratio), returning 0 when n == 0.

    Avoids ``0 * log(0)`` which is undefined; in the likelihood context this
    equals 0 (by the convention 0 * -inf = 0).

    R: lrtest.R ``logProd(prod, logand)``
       ``if(prod == 0) 0 else prod * log(logand)``

    Parameters
    ----------
    n : float
        The multiplying factor (e.g. number of events).
    ratio : float
        The ratio inside the log (must be > 0 when n != 0).

    Returns
    -------
    float
    """
    # R: lrtest.R L~5: if(prod == 0) 0 else prod * log(logand)
    if n == 0.0:
        return 0.0
    return float(n) * np.log(float(ratio))


# ---------------------------------------------------------------------------
# lrt — core two-group hurdle LRT statistic
# R: lrtest.R — lrt(et, group) / LRT()
# ---------------------------------------------------------------------------

def lrt(
    n_x: float,
    e_x: float,
    mu_x: float,
    ss_x: float,
    n_y: float,
    e_y: float,
    mu_y: float,
    ss_y: float,
) -> tuple[float, float]:
    """Compute the two-group hurdle log-likelihood ratio statistic.

    Combines the discrete (binomial detection) and continuous (Gaussian
    expression) components into a single hurdle LR statistic.

    The discrete component LR uses the pooled proportion under H0 and the
    group-specific proportions under H1.  The continuous component LR is
    based on the ratio of pooled to within-group variances.

    R: lrtest.R ``lrt()`` and ``LRT()`` — computes both components and sums.

    Parameters
    ----------
    n_x : float
        Total cells in group X.
    e_x : float
        Cells expressing gene in group X (detection count).
    mu_x : float
        Mean expression (log-scale) among expressing cells in group X.
    ss_x : float
        Sum of squares of expression among expressing cells in group X.
    n_y : float
        Total cells in group Y.
    e_y : float
        Cells expressing gene in group Y.
    mu_y : float
        Mean expression among expressing cells in group Y.
    ss_y : float
        Sum of squares of expression among expressing cells in group Y.

    Returns
    -------
    tuple[float, float]
        ``(log_lr, n_eff)`` where ``log_lr`` is the combined log-likelihood
        ratio (NOT multiplied by -2) and ``n_eff`` is the effective sample
        size used for the continuous component.
    """
    # ---- Discrete (binomial) component ----
    # R: lrtest.R L~15-25
    # H0: p0 = (e_x + e_y) / (n_x + n_y)
    # H1: p_x = e_x / n_x, p_y = e_y / n_y
    p0 = (e_x + e_y) / (n_x + n_y)   # pooled proportion under H0

    # Guard against division by zero when p_x or p_y are 0 or 1
    p_x = e_x / n_x if n_x > 0 else 0.0
    p_y = e_y / n_y if n_y > 0 else 0.0

    # log LR discrete = sum of log_prod terms for observed/not-observed
    # R: lrtest.R logProd(e_x, p0/p_x) + logProd(n_x - e_x, (1-p0)/(1-p_x)) + ...
    binom = (
        log_prod(e_x, p0 / p_x if p_x > 0 else 1.0)
        + log_prod(n_x - e_x, (1 - p0) / (1 - p_x) if p_x < 1 else 1.0)
        + log_prod(e_y, p0 / p_y if p_y > 0 else 1.0)
        + log_prod(n_y - e_y, (1 - p0) / (1 - p_y) if p_y < 1 else 1.0)
    )

    # ---- Continuous (Gaussian) component ----
    # R: lrtest.R L~27-38
    # Only computed when there are expressing cells in both groups
    norm_lr = 0.0
    n_eff = e_x + e_y  # effective sample size for continuous

    if e_x > 0 and e_y > 0 and (ss_x + ss_y) > 0:
        # T* = 1 + (e_x * e_y) / (e_x + e_y) * (mu_x - mu_y)^2 / (SS_x + SS_y)
        # R: lrtest.R L~30: T.star <- 1 + ex * ey / (ex + ey) * (mx - my)^2 / (ssx + ssy)
        t_star = 1.0 + (e_x * e_y) / (e_x + e_y) * (mu_x - mu_y) ** 2 / (ss_x + ss_y)
        # R: lrtest.R L~31: -(ex + ey)/2 * log(T.star)
        norm_lr = -(e_x + e_y) / 2.0 * np.log(t_star)

    # Combined log LR (NOTE: not yet negated by -2 — caller does that)
    log_lr = binom + norm_lr
    return log_lr, n_eff


# ---------------------------------------------------------------------------
# LRT — vectorised across genes (used internally by lrtest)
# R: lrtest.R — LRT(et, group)
# ---------------------------------------------------------------------------

def LRT(  # noqa: N802
    et: np.ndarray,
    group: np.ndarray,
) -> pd.DataFrame:
    """Vectorised hurdle LRT across all genes for two groups.

    R: lrtest.R ``LRT(et, group)`` — loops over genes, calling lrt() per gene.

    Parameters
    ----------
    et : np.ndarray
        Expression matrix (cells × genes), typically log-transformed Et values.
    group : np.ndarray
        1D array of group labels (0 or 1, or boolean) for each cell.
        Must have exactly two unique values.

    Returns
    -------
    pd.DataFrame
        One row per gene with columns:
        ``["lambda", "df", "p_value", "lambda_D", "lambda_C"]``.
    """
    group = np.asarray(group)
    unique_groups = np.unique(group)
    if len(unique_groups) != 2:
        raise ValueError(
            f"LRT requires exactly 2 groups, got {len(unique_groups)}: {unique_groups}"
        )

    mask_x = group == unique_groups[0]
    mask_y = group == unique_groups[1]
    n_genes = et.shape[1]

    results = []
    for g in range(n_genes):
        expr_x = et[mask_x, g]
        expr_y = et[mask_y, g]

        n_x = float(len(expr_x))
        n_y = float(len(expr_y))
        e_x = float((expr_x > 0).sum())
        e_y = float((expr_y > 0).sum())

        # Mean and SS only among expressing cells
        pos_x = expr_x[expr_x > 0]
        pos_y = expr_y[expr_y > 0]
        mu_x = float(pos_x.mean()) if len(pos_x) > 0 else 0.0
        mu_y = float(pos_y.mean()) if len(pos_y) > 0 else 0.0
        ss_x = float(np.sum((pos_x - mu_x) ** 2)) if len(pos_x) > 1 else 0.0
        ss_y = float(np.sum((pos_y - mu_y) ** 2)) if len(pos_y) > 1 else 0.0

        log_lr, _ = lrt(n_x, e_x, mu_x, ss_x, n_y, e_y, mu_y, ss_y)

        # R: lrtest.R — lambda = -2 * logLR ~ chi2(df=2)
        lambda_ = -2.0 * log_lr
        df = 2  # 1 df discrete + 1 df continuous
        p_val = chi2.sf(lambda_, df=df) if np.isfinite(lambda_) else np.nan

        results.append({
            "lambda": lambda_,
            "df": df,
            "p_value": p_val,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# lrtest — public API: two-group LRT on AnnData
# R: lrtest.R — lrtest(et, group)
# ---------------------------------------------------------------------------

def lrtest(
    et: np.ndarray,
    group: np.ndarray,
    *,
    gene_names: list[str] | None = None,
    adjust_method: str = "BH",
) -> pd.DataFrame:
    """Run the MAST two-group hurdle likelihood ratio test.

    This is the **simple two-group** version that does not require a ZLM fit.
    It compares two groups directly on each gene using the hurdle LRT statistic.

    For the full covariate-adjusted ZLM-based LRT (supporting arbitrary
    formulas and covariates), use :func:`pymast.tl.rank_genes_groups` or
    :class:`pymast.ZlmFit.lr_test`.

    R: lrtest.R ``lrtest(et, group)``

    Parameters
    ----------
    et : np.ndarray
        Expression matrix (cells × genes).
    group : np.ndarray
        Group label per cell (must have exactly 2 unique values).
    gene_names : list[str] or None
        Names for the genes (rows of result). Uses integer indices if None.
    adjust_method : str
        Multiple testing correction method. ``"BH"`` (Benjamini-Hochberg) is
        the R MAST default. Passed to ``statsmodels.stats.multitest``.

    Returns
    -------
    pd.DataFrame
        Columns: ``["lambda", "df", "p_value", "p_adj"]``.
        Indexed by gene names (or integer index).
    """
    from statsmodels.stats.multitest import multipletests  # lazy import

    result_df = LRT(et, group)

    # Multiple testing correction
    # R: lrtest.R — p.adjust(pvals, method="BH")
    valid_p = result_df["p_value"].values
    finite_mask = np.isfinite(valid_p)
    p_adj = np.full_like(valid_p, np.nan)

    if finite_mask.sum() > 0:
        _, p_adj[finite_mask], _, _ = multipletests(
            valid_p[finite_mask], method=adjust_method
        )

    result_df["p_adj"] = p_adj

    if gene_names is not None:
        result_df.index = gene_names

    return result_df
