"""
tests/test_zlm.py
=================
Tests for pymast.zlm — end-to-end ZLM fitting.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymast.zlm import zlm


class TestZlm:
    """End-to-end ZLM fitting tests."""

    def test_returns_zlmfit(self, small_adata) -> None:
        from pymast.zlm_fit import ZlmFit
        fit = zlm(
            small_adata,
            formula="~ group",
            n_jobs=1,
            verbose=False,
            use_ebayes=False,
        )
        assert isinstance(fit, ZlmFit)

    def test_gene_names_match(self, small_adata) -> None:
        fit = zlm(
            small_adata,
            formula="~ group",
            n_jobs=1,
            verbose=False,
            use_ebayes=False,
        )
        assert fit.gene_names == list(small_adata.var_names)

    def test_coef_shapes(self, small_adata) -> None:
        fit = zlm(
            small_adata,
            formula="~ group",
            n_jobs=1,
            verbose=False,
            use_ebayes=False,
        )
        n_genes = small_adata.n_vars
        n_params = fit.n_params
        assert fit.coef_C.shape == (n_genes, n_params)
        assert fit.coef_D.shape == (n_genes, n_params)
        assert fit.vcov_C.shape == (n_genes, n_params, n_params)
        assert fit.vcov_D.shape == (n_genes, n_params, n_params)

    def test_lr_test_runs(self, small_adata) -> None:
        fit = zlm(
            small_adata,
            formula="~ group",
            n_jobs=1,
            verbose=False,
            use_ebayes=False,
        )
        lr_df = fit.lr_test(contrast=None)
        assert "p_hurdle" in lr_df.columns
        assert len(lr_df) == small_adata.n_vars

    def test_p_values_in_range(self, small_adata) -> None:
        fit = zlm(
            small_adata,
            formula="~ group",
            n_jobs=1,
            verbose=False,
            use_ebayes=False,
        )
        lr_df = fit.lr_test(contrast=None)
        finite_p = lr_df["p_hurdle"].dropna()
        assert (finite_p >= 0).all() and (finite_p <= 1.0 + 1e-10).all()

    def test_invalid_method_raises(self, small_adata) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            zlm(small_adata, formula="~ group", method="invalid", verbose=False)

    def test_invalid_formula_raises(self, small_adata) -> None:
        with pytest.raises(ValueError, match="Formula"):
            zlm(
                small_adata,
                formula="~ nonexistent_column",
                n_jobs=1,
                verbose=False,
                use_ebayes=False,
            )

    def test_with_ebayes(self, small_adata) -> None:
        """EBayes should not crash and should return finite vcov."""
        fit = zlm(
            small_adata,
            formula="~ group",
            n_jobs=1,
            verbose=False,
            use_ebayes=True,
        )
        assert fit is not None
        # vcov_C should have some finite values
        assert np.isfinite(fit.vcov_C).any()
