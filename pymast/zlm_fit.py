"""
pymast/zlm_fit.py
=================
ZlmFit result container and associated LRT / Wald test methods.

Reimplements the R MAST ``ZlmFit`` S4 class from ``ZlmFit.R``, providing
the same interface for accessing fitted model results and running
likelihood ratio and Wald tests on contrasts.

R reference: RGLab/MAST main branch, R/ZlmFit.R
             https://github.com/RGLab/MAST/blob/main/R/ZlmFit.R

Citation: Finak G et al. (2015). MAST: a flexible statistical framework.
Genome Biology 16:278. DOI: 10.1186/s13059-015-0844-5
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests

from .lm_wrapper import ChiSqTable, make_chisq_table, wald_test
from .log_fc import get_log_fc


# ---------------------------------------------------------------------------
# ZlmFit — result container
# R: ZlmFit.R — ZlmFit S4 class
# ---------------------------------------------------------------------------

@dataclass
class ZlmFit:
    """Container for ZLM (zero-inflated linear model) fit results.

    Stores per-gene fitted coefficients, variances, and log-likelihoods for
    both the continuous (C) and discrete (D) components of the hurdle model.

    R: ZlmFit.R ``ZlmFit`` S4 class — slots:
       ``coefC``, ``coefD``, ``vcovC``, ``vcovD``,
       ``LLik``, ``LLikN``, ``df``, ``convergeC``, ``convergeD``.

    Parameters
    ----------
    gene_names : list[str]
        Gene names (length n_genes).
    coef_names : list[str]
        Coefficient names from the model formula (length n_params).
    coef_C : np.ndarray
        Continuous component coefficients (n_genes × n_params).
    coef_D : np.ndarray
        Discrete component coefficients (n_genes × n_params).
    vcov_C : np.ndarray
        Continuous covariance matrices (n_genes × n_params × n_params).
    vcov_D : np.ndarray
        Discrete covariance matrices (n_genes × n_params × n_params).
    loglik_C : np.ndarray
        Full-model log-likelihood for continuous component (n_genes,).
    loglik_D : np.ndarray
        Full-model log-likelihood for discrete component (n_genes,).
    loglik_C_null : np.ndarray
        Null-model log-likelihood for continuous component (n_genes,).
    loglik_D_null : np.ndarray
        Null-model log-likelihood for discrete component (n_genes,).
    df_C : np.ndarray
        Continuous degrees of freedom (difference) per gene (n_genes,).
    df_D : np.ndarray
        Discrete degrees of freedom (difference) per gene (n_genes,).
    converge_C : np.ndarray
        Boolean convergence flag for continuous component (n_genes,).
    converge_D : np.ndarray
        Boolean convergence flag for discrete component (n_genes,).
    n_expressing : np.ndarray
        Number of expressing cells used for continuous fit per gene (n_genes,).
    formula : str
        The formula string used for fitting.
    """
    gene_names: list[str]
    coef_names: list[str]
    coef_C: np.ndarray          # (n_genes, n_params)
    coef_D: np.ndarray          # (n_genes, n_params)
    vcov_C: np.ndarray          # (n_genes, n_params, n_params)
    vcov_D: np.ndarray          # (n_genes, n_params, n_params)
    loglik_C: np.ndarray        # (n_genes,) full model
    loglik_D: np.ndarray        # (n_genes,) full model
    loglik_C_null: np.ndarray   # (n_genes,) null model
    loglik_D_null: np.ndarray   # (n_genes,) null model
    df_C: np.ndarray            # (n_genes,) df for LRT
    df_D: np.ndarray            # (n_genes,) df for LRT
    converge_C: np.ndarray      # (n_genes,) bool
    converge_D: np.ndarray      # (n_genes,) bool
    n_expressing: np.ndarray    # (n_genes,) int
    formula: str = ""

    @property
    def n_genes(self) -> int:
        return len(self.gene_names)

    @property
    def n_params(self) -> int:
        return len(self.coef_names)

    # -----------------------------------------------------------------------
    # lr_test
    # R: ZlmFit.R — lrTest(zlmfit, hypothesis)
    # Also: .lrtZlmFit() — lines ~30-51
    # -----------------------------------------------------------------------

    def lr_test(
        self,
        contrast: str | np.ndarray,
        *,
        adjust_method: str = "fdr_bh",
    ) -> pd.DataFrame:
        """Run a likelihood ratio test on this ZlmFit.

        Computes the hurdle LRT statistic for each gene using:
        ``lambda = -2 * (loglik_null - loglik_full)``

        The combined hurdle statistic is ``lambda_C + lambda_D ~ chi2(df_C + df_D)``.

        R: ZlmFit.R ``.lrtZlmFit(object, hypothesis)`` — lines ~30-51.
           ``lambda <- -2 * (LLikN - LLik)``
           ``p <- 1 - pchisq(lambda.hurdle, df.hurdle)``

        Parameters
        ----------
        contrast : str or np.ndarray
            Currently unused (LRT uses the pre-stored null vs full log-likelihoods).
            Pass ``None`` or any string to use stored results.
        adjust_method : str
            Multiple testing correction method (default: ``"BH"``).

        Returns
        -------
        pd.DataFrame
            Columns: ``["lambda_C", "lambda_D", "lambda_hurdle",
            "df_C", "df_D", "df_hurdle", "p_C", "p_D", "p_hurdle", "p_adj"]``.
            Indexed by gene names.
        """
        # R: ZlmFit.R L~30: lambda <- -2 * (LLikN - LLik)
        # Use errstate to silence the "invalid value in subtract" warning that
        # fires when both loglik values are NaN (continuous component not fit
        # because too few cells express the gene). Those genes get lambda=0,
        # which correctly contributes nothing to the hurdle statistic.
        with np.errstate(invalid="ignore"):
            lambda_C = -2.0 * (self.loglik_C_null - self.loglik_C)
            lambda_D = -2.0 * (self.loglik_D_null - self.loglik_D)

        # Replace NaN (unfit component) with 0 before clipping
        lambda_C = np.nan_to_num(lambda_C, nan=0.0)
        lambda_D = np.nan_to_num(lambda_D, nan=0.0)

        # Clip to zero (numerical noise can give tiny negatives)
        lambda_C = np.clip(lambda_C, 0.0, None)
        lambda_D = np.clip(lambda_D, 0.0, None)

        lambda_hurdle = lambda_C + lambda_D

        # R: ZlmFit.R L~45: df.hurdle <- df.C + df.D
        df_hurdle = (self.df_C + self.df_D).astype(int)

        # p-values from chi2 survival function
        # R: ZlmFit.R L~48: p.hurdle <- 1 - pchisq(lambda.hurdle, df.hurdle)
        p_C = np.where(
            self.df_C > 0,
            chi2.sf(lambda_C, df=np.where(self.df_C > 0, self.df_C, 1)),
            np.nan,
        )
        p_D = np.where(
            self.df_D > 0,
            chi2.sf(lambda_D, df=np.where(self.df_D > 0, self.df_D, 1)),
            np.nan,
        )
        p_hurdle = np.where(
            df_hurdle > 0,
            chi2.sf(lambda_hurdle, df=np.where(df_hurdle > 0, df_hurdle, 1)),
            np.nan,
        )

        # BH correction on hurdle p-values
        finite_mask = np.isfinite(p_hurdle)
        p_adj = np.full_like(p_hurdle, np.nan)
        if finite_mask.sum() > 0:
            _, p_adj[finite_mask], _, _ = multipletests(
                p_hurdle[finite_mask], method=adjust_method
            )

        return pd.DataFrame(
            {
                "lambda_C": lambda_C,
                "lambda_D": lambda_D,
                "lambda_hurdle": lambda_hurdle,
                "df_C": self.df_C.astype(int),
                "df_D": self.df_D.astype(int),
                "df_hurdle": df_hurdle,
                "p_C": p_C,
                "p_D": p_D,
                "p_hurdle": p_hurdle,
                "p_adj": p_adj,
            },
            index=self.gene_names,
        )

    # -----------------------------------------------------------------------
    # wald_test
    # R: ZlmFit.R — waldTest(zlmfit, hypothesis)
    # -----------------------------------------------------------------------

    def wald_test(
        self,
        contrast: np.ndarray,
        *,
        adjust_method: str = "fdr_bh",
    ) -> pd.DataFrame:
        """Run a Wald test on this ZlmFit for a given contrast.

        R: ZlmFit.R ``waldTest(object, hypothesis)`` — applies Wald test
           per gene using the per-gene vcov and contrast matrix.

        Parameters
        ----------
        contrast : np.ndarray
            Contrast matrix L of shape (n_params,) or (n_params, k).
        adjust_method : str
            Multiple testing correction.

        Returns
        -------
        pd.DataFrame
            Columns: ``["lambda_C", "lambda_D", "lambda_hurdle",
            "df_C", "df_D", "df_hurdle", "p_C", "p_D", "p_hurdle", "p_adj"]``.
        """
        contrast = np.asarray(contrast, dtype=np.float64)
        if contrast.ndim == 1:
            contrast = contrast[:, np.newaxis]

        df = contrast.shape[1]
        lambda_C_arr = np.zeros(self.n_genes)
        lambda_D_arr = np.zeros(self.n_genes)
        p_C_arr = np.full(self.n_genes, np.nan)
        p_D_arr = np.full(self.n_genes, np.nan)

        for g in range(self.n_genes):
            if self.n_expressing[g] > self.n_params:
                lam_C, _, p_C = wald_test(contrast, self.coef_C[g], self.vcov_C[g])
                lambda_C_arr[g] = lam_C
                p_C_arr[g] = p_C

            lam_D, _, p_D = wald_test(contrast, self.coef_D[g], self.vcov_D[g])
            lambda_D_arr[g] = lam_D
            p_D_arr[g] = p_D

        lambda_hurdle = lambda_C_arr + lambda_D_arr
        df_hurdle = 2 * df
        p_hurdle = np.where(
            np.isfinite(lambda_hurdle),
            chi2.sf(lambda_hurdle, df=df_hurdle),
            np.nan,
        )

        finite_mask = np.isfinite(p_hurdle)
        p_adj = np.full_like(p_hurdle, np.nan)
        if finite_mask.sum() > 0:
            _, p_adj[finite_mask], _, _ = multipletests(
                p_hurdle[finite_mask], method=adjust_method
            )

        return pd.DataFrame(
            {
                "lambda_C": lambda_C_arr,
                "lambda_D": lambda_D_arr,
                "lambda_hurdle": lambda_hurdle,
                "df_C": df,
                "df_D": df,
                "df_hurdle": df_hurdle,
                "p_C": p_C_arr,
                "p_D": p_D_arr,
                "p_hurdle": p_hurdle,
                "p_adj": p_adj,
            },
            index=self.gene_names,
        )

    # -----------------------------------------------------------------------
    # collect_summaries
    # R: ZlmFit.R — collectSummaries(zlmfit)
    # -----------------------------------------------------------------------

    def collect_summaries(self) -> pd.DataFrame:
        """Collect a flat summary DataFrame of all fitted parameters.

        R: ZlmFit.R ``collectSummaries()`` — aggregates per-gene statistics.

        Returns
        -------
        pd.DataFrame
            One row per gene with continuous and discrete coefficients,
            log-likelihoods, and convergence flags.
        """
        rows = []
        for g, gene in enumerate(self.gene_names):
            row = {"gene": gene}
            for j, name in enumerate(self.coef_names):
                row[f"coef_C_{name}"] = self.coef_C[g, j]
                row[f"coef_D_{name}"] = self.coef_D[g, j]
            row["loglik_C"] = self.loglik_C[g]
            row["loglik_D"] = self.loglik_D[g]
            row["converge_C"] = self.converge_C[g]
            row["converge_D"] = self.converge_D[g]
            row["n_expressing"] = self.n_expressing[g]
            rows.append(row)
        return pd.DataFrame(rows).set_index("gene")

    def get_log_fc(self, contrast: np.ndarray) -> pd.DataFrame:
        """Compute log fold changes from fitted coefficients.

        Wraps :func:`pymast.log_fc.get_log_fc`.

        Parameters
        ----------
        contrast : np.ndarray
            Contrast vector or matrix.

        Returns
        -------
        pd.DataFrame
        """
        return get_log_fc(
            coef_C=self.coef_C,
            coef_D=self.coef_D,
            contrast=contrast,
            coef_names=self.coef_names,
            gene_names=self.gene_names,
        )
