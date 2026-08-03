"""
tests/test_ebayes.py
====================
Tests for pymast.ebayes — empirical Bayes variance shrinkage.

Verifies that hyperparameter estimation converges and that posterior
variances are shrunk toward the prior.

R reference: R/ebayes-helpers.R
"""

from __future__ import annotations

import numpy as np
import pytest

from pymast.ebayes import (
    GeneSufficientStats,
    ebayes,
    get_marginal_hyperlikelihood,
    get_ss_g_r_ng,
    solve_mom,
)


@pytest.fixture
def synthetic_stats(rng) -> GeneSufficientStats:
    """Synthetic sufficient statistics for 50 genes."""
    n_genes = 50
    # Simulate gene-level sufficient stats
    sigma2_true = rng.gamma(shape=2.0, scale=0.5, size=n_genes)  # true variances
    n_cells_per_gene = rng.integers(10, 50, size=n_genes)
    ss_g = sigma2_true * (n_cells_per_gene - 3).astype(float)    # SS = sigma2 * df
    r_ng = (n_cells_per_gene - 3).astype(float)
    return GeneSufficientStats(ss_g=ss_g, r_ng=r_ng)


class TestGetSSgRNg:
    """Tests for get_ss_g_r_ng()."""

    def test_output_shape(self, rng) -> None:
        n_obs, n_genes = 30, 5
        resid = rng.standard_normal((n_obs, n_genes))
        resid[rng.random((n_obs, n_genes)) < 0.3] = np.nan
        df_resid = np.array([20.0] * n_genes)
        stats = get_ss_g_r_ng(resid, df_resid)
        assert stats.ss_g.shape == (n_genes,)
        assert stats.r_ng.shape == (n_genes,)

    def test_ss_g_is_sum_of_squares(self, rng) -> None:
        n_obs, n_genes = 20, 3
        resid = rng.standard_normal((n_obs, n_genes))
        # No NaNs
        df_resid = np.array([17.0] * n_genes)
        stats = get_ss_g_r_ng(resid, df_resid)
        expected_ss = np.sum(resid ** 2, axis=0)
        np.testing.assert_allclose(stats.ss_g, expected_ss, rtol=1e-10)


class TestGetMarginalHyperlikelihood:
    """Tests for get_marginal_hyperlikelihood()."""

    def test_returns_finite_for_valid_inputs(self, synthetic_stats) -> None:
        ll = get_marginal_hyperlikelihood(1.0, 1.0, synthetic_stats.ss_g, synthetic_stats.r_ng)
        assert np.isfinite(ll)

    def test_returns_zero_for_all_zero_r_ng(self) -> None:
        ss = np.array([1.0, 2.0, 3.0])
        r_ng = np.zeros(3)
        ll = get_marginal_hyperlikelihood(1.0, 1.0, ss, r_ng)
        assert ll == 0.0

    def test_higher_b0_lower_ll_for_small_ss(self) -> None:
        """With small SS, larger b0 should decrease log-likelihood."""
        ss = np.array([0.1, 0.1, 0.1])
        r_ng = np.array([10.0, 10.0, 10.0])
        ll1 = get_marginal_hyperlikelihood(1.0, 0.1, ss, r_ng)
        ll2 = get_marginal_hyperlikelihood(1.0, 10.0, ss, r_ng)
        assert ll1 > ll2  # smaller b0 fits small ss better


class TestSolveMom:
    """Tests for solve_mom() — method of moments initialization."""

    def test_returns_positive_values(self, synthetic_stats) -> None:
        a0, b0 = solve_mom(synthetic_stats.ss_g, synthetic_stats.r_ng)
        assert a0 > 0
        assert b0 > 0

    def test_fallback_for_single_gene(self) -> None:
        ss = np.array([1.0])
        r_ng = np.array([5.0])
        a0, b0 = solve_mom(ss, r_ng)
        assert a0 > 0 and b0 > 0

    def test_fallback_for_zero_r_ng(self) -> None:
        ss = np.array([1.0, 2.0])
        r_ng = np.zeros(2)
        a0, b0 = solve_mom(ss, r_ng)
        assert a0 > 0 and b0 > 0


class TestEbayes:
    """Tests for ebayes() — main empirical Bayes estimation."""

    def test_returns_result_with_correct_shape(self, synthetic_stats) -> None:
        result = ebayes(synthetic_stats)
        n_genes = len(synthetic_stats.ss_g)
        assert result.sigma2_post.shape == (n_genes,)
        assert result.df_total.shape == (n_genes,)

    def test_hyperparameters_are_positive(self, synthetic_stats) -> None:
        result = ebayes(synthetic_stats)
        assert result.a0 > 0
        assert result.b0 > 0
        assert result.v >= 0
        assert result.df_prior >= 0

    def test_posterior_variance_shrinkage(self, synthetic_stats) -> None:
        """Posterior variances should be less variable than raw variances."""
        raw_sigma2 = synthetic_stats.ss_g / np.where(
            synthetic_stats.r_ng > 0, synthetic_stats.r_ng, 1
        )
        result = ebayes(synthetic_stats)
        # Posterior variance should have smaller coefficient of variation
        raw_cv = np.std(raw_sigma2) / np.mean(raw_sigma2)
        post_cv = np.std(result.sigma2_post) / np.mean(result.sigma2_post)
        assert post_cv <= raw_cv, "Posterior variance should be shrunk (less variable)"

    def test_df_total_is_prior_plus_residual(self, synthetic_stats) -> None:
        result = ebayes(synthetic_stats)
        expected_df_total = result.df_prior + synthetic_stats.r_ng
        np.testing.assert_allclose(result.df_total, expected_df_total, rtol=1e-6)

    def test_v_equals_b0_over_a0(self, synthetic_stats) -> None:
        result = ebayes(synthetic_stats)
        assert result.v == pytest.approx(result.b0 / result.a0, rel=1e-6)
