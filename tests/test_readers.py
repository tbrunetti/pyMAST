"""Tests for pymast.readers."""
import numpy as np
import pytest
from pymast.readers import from_anndata, from_flat_df, from_matrix


class TestFromAnndata:
    def test_validates_and_adds_cdr(self, synthetic_adata) -> None:
        adata = from_anndata(synthetic_adata, inplace=False)
        assert "cdr" in adata.obs.columns
        assert (adata.obs["cdr"] >= 0).all()
        assert (adata.obs["cdr"] <= 1).all()


class TestFromMatrix:
    def test_creates_anndata(self, rng) -> None:
        from anndata import AnnData
        X = rng.random((20, 10)).astype(np.float32)
        adata = from_matrix(X, gene_names=[f"g{i}" for i in range(10)])
        assert isinstance(adata, AnnData)
        assert adata.n_obs == 20
        assert adata.n_vars == 10


class TestFromFlatDf:
    def test_ct_to_et_conversion(self, rng) -> None:
        import pandas as pd
        ct_vals = rng.uniform(20, 40, size=(10, 5))
        df = pd.DataFrame(ct_vals, columns=[f"gene_{i}" for i in range(5)])
        df["cell_id"] = [f"cell_{i}" for i in range(10)]
        adata = from_flat_df(df, obs_cols=["cell_id"], is_ct=True, ct_threshold=40.0)
        # Et = 40 - Ct, clipped at 0
        assert (adata.X >= 0).all()
        assert adata.n_obs == 10
        assert adata.n_vars == 5
