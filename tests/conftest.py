"""
tests/conftest.py
=================
Shared test fixtures for pyMAST.

Provides synthetic AnnData objects mimicking the vbeta dataset used in the
R MAST vignette, allowing deterministic numerical testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData


# ---------------------------------------------------------------------------
# Synthetic AnnData fixture (vbeta-like)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Fixed-seed RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_adata(rng: np.random.Generator) -> AnnData:
    """Synthetic single-cell AnnData with two groups (vbeta-like).

    Properties
    ----------
    - 200 cells × 50 genes
    - Group A: 100 cells, higher expression of first 10 genes
    - Group B: 100 cells, lower expression of first 10 genes
    - ~30% zero expression (hurdle structure)
    - Columns in obs: "group", "batch"
    """
    n_obs = 200
    n_vars = 50
    n_de_genes = 10

    group = np.array(["A"] * 100 + ["B"] * 100)
    batch = np.array(["batch1"] * 50 + ["batch2"] * 50 + ["batch1"] * 50 + ["batch2"] * 50)

    # Base expression
    expr = rng.standard_normal((n_obs, n_vars)).astype(np.float32)

    # Add group effect for first 10 genes
    group_effect = np.where(group == "A", 1.5, -1.5)
    expr[:, :n_de_genes] += group_effect[:, np.newaxis]

    # Zero-inflate (~30% zeros)
    zero_prob = rng.random((n_obs, n_vars))
    expr[zero_prob < 0.30] = 0.0
    expr = np.clip(expr, 0.0, None)  # no negative expression

    obs = pd.DataFrame({"group": group, "batch": batch}, index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(
        {"is_de": [i < n_de_genes for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )

    return AnnData(X=expr, obs=obs, var=var)


@pytest.fixture(scope="session")
def small_adata(rng: np.random.Generator) -> AnnData:
    """Very small AnnData (20 cells × 5 genes) for fast unit tests."""
    n_obs = 20
    n_vars = 5
    group = ["A"] * 10 + ["B"] * 10
    expr = rng.standard_normal((n_obs, n_vars)).astype(np.float32)
    expr[rng.random((n_obs, n_vars)) < 0.3] = 0.0
    expr = np.clip(expr, 0.0, None)
    obs = pd.DataFrame({"group": group}, index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    return AnnData(X=expr, obs=obs, var=var)
