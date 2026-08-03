"""Tests for pymast.filter."""
import numpy as np
import pytest
from pymast.filter import burden_of_filtering, mast_filter


class TestMastFilter:
    def test_gene_filter(self, synthetic_adata) -> None:
        n_genes_before = synthetic_adata.n_vars
        filtered = mast_filter(
            synthetic_adata,
            min_cells_expressing=5,
            inplace=False,
            verbose=False,
        )
        assert filtered.n_vars <= n_genes_before

    def test_cell_filter(self, synthetic_adata) -> None:
        n_cells_before = synthetic_adata.n_obs
        filtered = mast_filter(
            synthetic_adata,
            min_genes_per_cell=2,
            inplace=False,
            verbose=False,
        )
        assert filtered.n_obs <= n_cells_before

    def test_no_filter_returns_same(self, synthetic_adata) -> None:
        filtered = mast_filter(
            synthetic_adata,
            inplace=False,
            verbose=False,
        )
        assert filtered.n_obs == synthetic_adata.n_obs
        assert filtered.n_vars == synthetic_adata.n_vars


class TestBurdenOfFiltering:
    def test_returns_dataframe(self, synthetic_adata) -> None:
        df = burden_of_filtering(synthetic_adata)
        assert "min_cells_expressing" in df.columns
        assert "n_genes_retained" in df.columns
        assert "n_cells_retained" in df.columns

    def test_monotone_decreasing_genes(self, synthetic_adata) -> None:
        df = burden_of_filtering(synthetic_adata, min_cells_list=[1, 5, 10, 20])
        genes_retained = df["n_genes_retained"].values
        assert (np.diff(genes_retained) <= 0).all()
