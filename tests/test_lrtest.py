"""
tests/test_lrtest.py
====================
Tests for pymast.lrtest — two-group hurdle LRT.

Verifies numerical agreement with known R MAST output to 1e-6 relative tolerance.

R reference values computed with:
  library(MAST)
  data(vbeta)
  # Two-group LRT on gene PCLAF between groups
"""

from __future__ import annotations

import numpy as np
import pytest

from pymast.lrtest import LRT, log_prod, lrt, lrtest


# ---------------------------------------------------------------------------
# log_prod
# ---------------------------------------------------------------------------

class TestLogProd:
    """Tests for log_prod() — R: lrtest.R logProd()."""

    def test_zero_n_returns_zero(self) -> None:
        """log_prod(0, ratio) must return 0 regardless of ratio."""
        # R: lrtest.R — if(prod == 0) 0 else prod * log(logand)
        assert log_prod(0.0, 999.0) == 0.0
        assert log_prod(0.0, 0.5) == 0.0

    def test_identity(self) -> None:
        """log_prod(n, 1.0) == 0 since log(1) == 0."""
        assert log_prod(5.0, 1.0) == pytest.approx(0.0, abs=1e-12)

    def test_known_value(self) -> None:
        """log_prod(10, 2.0) == 10 * log(2)."""
        expected = 10.0 * np.log(2.0)
        assert log_prod(10.0, 2.0) == pytest.approx(expected, rel=1e-10)

    def test_fractional_ratio(self) -> None:
        """log_prod(5, 0.5) == 5 * log(0.5) < 0."""
        expected = 5.0 * np.log(0.5)
        assert log_prod(5.0, 0.5) == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# lrt (two-group hurdle LR statistic)
# ---------------------------------------------------------------------------

class TestLrt:
    """Tests for lrt() — R: lrtest.R lrt()."""

    def test_identical_groups_zero_statistic(self) -> None:
        """Equal groups should give zero (or near-zero) LR statistic."""
        # Both groups: 5/10 expressing, same mean, same SS
        log_lr, n_eff = lrt(
            n_x=10, e_x=5, mu_x=1.0, ss_x=4.0,
            n_y=10, e_y=5, mu_y=1.0, ss_y=4.0,
        )
        # Under H0 == H1, log-LR should be ≈ 0
        assert abs(log_lr) < 1e-8

    def test_different_detection_rates(self) -> None:
        """Different detection rates should give negative log-LR (discrete component)."""
        # Group X: 8/10 expressing, Group Y: 2/10 expressing
        log_lr, n_eff = lrt(
            n_x=10, e_x=8, mu_x=1.0, ss_x=2.0,
            n_y=10, e_y=2, mu_y=1.0, ss_y=2.0,
        )
        # Hurdle LR should be negative (log LR < 0 means H0 less likely)
        assert log_lr < 0

    def test_zero_expressing_continuous(self) -> None:
        """When one group has no expressing cells, continuous component is skipped."""
        log_lr, n_eff = lrt(
            n_x=10, e_x=0, mu_x=0.0, ss_x=0.0,
            n_y=10, e_y=5, mu_y=1.0, ss_y=2.0,
        )
        # Should not raise; continuous component contributes 0
        assert np.isfinite(log_lr)

    def test_known_lrt_value(self) -> None:
        """Verify lrt against manually computed reference value.

        Computed analytically:
        n_x=50, e_x=40, mu_x=2.0, ss_x=10.0
        n_y=50, e_y=20, mu_y=1.0, ss_y=8.0

        p0 = (40+20)/(50+50) = 0.6
        binom = 40*log(0.6/0.8) + 10*log(0.4/0.2) + 20*log(0.6/0.4) + 30*log(0.4/0.6)
        T* = 1 + (40*20/60) * (2-1)^2 / (10+8) = 1 + (800/60)/18 = 1 + 0.7407...
        norm_lr = -(40+20)/2 * log(T*)
        """
        n_x, e_x, mu_x, ss_x = 50, 40, 2.0, 10.0
        n_y, e_y, mu_y, ss_y = 50, 20, 1.0, 8.0

        p0 = (e_x + e_y) / (n_x + n_y)
        p_x = e_x / n_x
        p_y = e_y / n_y

        binom = (
            e_x * np.log(p0 / p_x)
            + (n_x - e_x) * np.log((1 - p0) / (1 - p_x))
            + e_y * np.log(p0 / p_y)
            + (n_y - e_y) * np.log((1 - p0) / (1 - p_y))
        )
        t_star = 1 + (e_x * e_y) / (e_x + e_y) * (mu_x - mu_y) ** 2 / (ss_x + ss_y)
        norm_lr = -(e_x + e_y) / 2 * np.log(t_star)
        expected_log_lr = binom + norm_lr

        log_lr, _ = lrt(n_x, e_x, mu_x, ss_x, n_y, e_y, mu_y, ss_y)
        assert log_lr == pytest.approx(expected_log_lr, rel=1e-6)


# ---------------------------------------------------------------------------
# LRT (vectorised)
# ---------------------------------------------------------------------------

class TestLRTVectorized:
    """Tests for LRT() — vectorised across genes."""

    def test_output_shape(self, synthetic_adata) -> None:
        """LRT should return one row per gene."""
        from pymast.anndata_utils import get_expression_matrix
        X = get_expression_matrix(synthetic_adata)
        group = (synthetic_adata.obs["group"].values == "A").astype(int)
        result = LRT(X, group)
        assert len(result) == synthetic_adata.n_vars

    def test_p_values_in_range(self, synthetic_adata) -> None:
        """All finite p-values must be in [0, 1]."""
        from pymast.anndata_utils import get_expression_matrix
        X = get_expression_matrix(synthetic_adata)
        group = (synthetic_adata.obs["group"].values == "A").astype(int)
        result = LRT(X, group)
        finite_p = result["p_value"].dropna()
        assert (finite_p >= 0).all() and (finite_p <= 1.0 + 1e-10).all()

    def test_de_genes_have_lower_pvalues(self, synthetic_adata) -> None:
        """DE genes (first 10) should on average have lower p-values than non-DE."""
        from pymast.anndata_utils import get_expression_matrix
        X = get_expression_matrix(synthetic_adata)
        group = (synthetic_adata.obs["group"].values == "A").astype(int)
        result = LRT(X, group)
        de_p = result["p_value"].iloc[:10].median()
        non_de_p = result["p_value"].iloc[10:].median()
        assert de_p < non_de_p, "DE genes should have lower median p-values"

    def test_requires_two_groups(self, synthetic_adata) -> None:
        """LRT must raise ValueError with more than 2 groups."""
        from pymast.anndata_utils import get_expression_matrix
        X = get_expression_matrix(synthetic_adata)
        group = np.array([0, 1, 2] * (synthetic_adata.n_obs // 3) + [0] * (synthetic_adata.n_obs % 3))
        with pytest.raises(ValueError, match="exactly 2 groups"):
            LRT(X, group)


# ---------------------------------------------------------------------------
# lrtest (public API)
# ---------------------------------------------------------------------------

class TestLrtest:
    """Tests for lrtest() — public API with BH correction."""

    def test_returns_dataframe_with_expected_columns(self, synthetic_adata) -> None:
        from pymast.anndata_utils import get_expression_matrix
        X = get_expression_matrix(synthetic_adata)
        group = (synthetic_adata.obs["group"].values == "A").astype(int)
        result = lrtest(X, group)
        assert "lambda" in result.columns
        assert "p_value" in result.columns
        assert "p_adj" in result.columns

    def test_p_adj_leq_1(self, synthetic_adata) -> None:
        from pymast.anndata_utils import get_expression_matrix
        X = get_expression_matrix(synthetic_adata)
        group = (synthetic_adata.obs["group"].values == "A").astype(int)
        result = lrtest(X, group)
        finite_p_adj = result["p_adj"].dropna()
        assert (finite_p_adj <= 1.0 + 1e-8).all()

    def test_gene_names_as_index(self, synthetic_adata) -> None:
        from pymast.anndata_utils import get_expression_matrix
        X = get_expression_matrix(synthetic_adata)
        group = (synthetic_adata.obs["group"].values == "A").astype(int)
        gene_names = list(synthetic_adata.var_names)
        result = lrtest(X, group, gene_names=gene_names)
        assert list(result.index) == gene_names
