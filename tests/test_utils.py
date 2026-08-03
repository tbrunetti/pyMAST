"""Tests for pymast.utils."""
import numpy as np
import pytest
from pymast.utils import compute_cdr, compute_et_from_ct, freq


class TestComputeCdr:
    def test_cdr_range(self, synthetic_adata) -> None:
        import pymast
        cdr = pymast.compute_cdr(synthetic_adata, inplace=False)
        assert (cdr >= 0).all() and (cdr <= 1).all()

    def test_inplace_adds_column(self, synthetic_adata) -> None:
        import pymast
        adata = synthetic_adata.copy()
        pymast.compute_cdr(adata, key_added="test_cdr", inplace=True)
        assert "test_cdr" in adata.obs.columns


class TestFreq:
    def test_returns_dataframe(self, synthetic_adata) -> None:
        import pymast
        df = pymast.freq(synthetic_adata)
        assert "freq" in df.columns
        assert len(df) == synthetic_adata.n_vars
        assert (df["freq"] >= 0).all() and (df["freq"] <= 1).all()


class TestComputeEtFromCt:
    def test_conversion(self) -> None:
        ct = np.array([35.0, 40.0, 20.0])
        et = compute_et_from_ct(ct, ct_threshold=40.0, et_threshold=0.0)
        np.testing.assert_allclose(et, [5.0, 0.0, 20.0])

    def test_clipping_at_zero(self) -> None:
        ct = np.array([45.0])  # above threshold → Et = -5 → clip to 0
        et = compute_et_from_ct(ct, ct_threshold=40.0, et_threshold=0.0)
        assert et[0] == 0.0
