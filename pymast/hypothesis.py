"""
pymast/hypothesis.py
====================
Contrast and hypothesis specification classes for MAST ZLM testing.

Replicates the R ``CoefficientHypothesis`` and ``Hypothesis`` S4 classes
from ``R/Hypothesis.R``.  These objects encode which model coefficients
to test (e.g. a treatment effect) and are passed to ``ZlmFit.lr_test()``
and ``ZlmFit.wald_test()``.

R reference: RGLab/MAST main branch, R/Hypothesis.R
             https://github.com/RGLab/MAST/blob/main/R/Hypothesis.R
"""

from __future__ import annotations

import re

import numpy as np


# ---------------------------------------------------------------------------
# CoefficientHypothesis
# R: Hypothesis.R — CoefficientHypothesis S4 class
# ---------------------------------------------------------------------------

class CoefficientHypothesis:
    """Hypothesis that tests one or more named model coefficients.

    A thin wrapper encoding a contrast as a character string referencing
    one or more predictor names (as they appear in the model matrix after
    patsy formula expansion).

    R: Hypothesis.R ``CoefficientHypothesis`` — sets ``contrastMatrix``
       to a column-selection identity contrast from the model coefficient names.

    Parameters
    ----------
    hypothesis : str
        Coefficient name or pattern.
        - A single coefficient name: ``"groupB"``
        - A regex pattern (prefixed with ``"~"``): ``"~^group"``

    Examples
    --------
    >>> hyp = CoefficientHypothesis("groupB")
    >>> contrast = hyp.build_contrast(["Intercept", "groupB", "cdr"])
    # contrast is a (3, 1) matrix selecting column "groupB"
    """

    def __init__(self, hypothesis: str) -> None:
        self.hypothesis = hypothesis
        # R: Hypothesis.R — store the contrast string
        self._is_regex = hypothesis.startswith("~")
        self._pattern = hypothesis[1:] if self._is_regex else re.escape(hypothesis)

    def __repr__(self) -> str:
        return f"CoefficientHypothesis('{self.hypothesis}')"

    def build_contrast(self, coef_names: list[str]) -> np.ndarray:
        """Build a contrast matrix for the given coefficient names.

        Returns a ``(len(coef_names), k)`` float64 matrix where each column
        selects one matched coefficient (identity contrast for that column).

        R: Hypothesis.R ``makeContrastMatrix()`` — column-subset identity.

        Parameters
        ----------
        coef_names : list[str]
            Coefficient names from the fitted model (e.g. from patsy design).

        Returns
        -------
        np.ndarray
            Shape ``(n_coefs, k)`` where ``k`` is the number of matched
            coefficients.

        Raises
        ------
        ValueError
            If no coefficients match the hypothesis.
        """
        # R: Hypothesis.R — match hypothesis string to coefficient names
        matched = [
            i for i, name in enumerate(coef_names)
            if re.search(self._pattern, name)
        ]
        if not matched:
            raise ValueError(
                f"Hypothesis '{self.hypothesis}' matched no coefficients. "
                f"Available: {coef_names}"
            )

        # Build identity contrast matrix: (n_coefs, k)
        n = len(coef_names)
        k = len(matched)
        contrast = np.zeros((n, k), dtype=np.float64)
        for col, row in enumerate(matched):
            contrast[row, col] = 1.0

        return contrast

    @property
    def matched_names(self) -> list[str]:
        """Names matched by this hypothesis (populated after ``build_contrast``)."""
        return getattr(self, "_matched_names", [])


# ---------------------------------------------------------------------------
# Hypothesis
# R: Hypothesis.R — Hypothesis S4 class
# ---------------------------------------------------------------------------

class Hypothesis:
    """General linear hypothesis for a MAST ZLM model.

    Encodes a hypothesis of the form ``L @ beta = 0`` where ``L`` is a user-
    supplied contrast matrix.  Use :class:`CoefficientHypothesis` for the
    common case of testing named coefficients.

    R: Hypothesis.R ``Hypothesis`` S4 class — stores ``contrastMatrix``
       (the L matrix) and a ``name`` string.

    Parameters
    ----------
    contrast_matrix : np.ndarray
        Contrast matrix ``L`` of shape ``(n_coefs, n_hypotheses)``.
        Each column encodes one linear combination to test = 0.
    name : str
        Human-readable name for the hypothesis (used in summary output).
    coef_names : list[str] or None
        Coefficient names corresponding to rows of ``contrast_matrix``.
        If provided, stored for display purposes.

    Examples
    --------
    >>> L = np.array([[0, 1, 0]]).T   # test second coefficient
    >>> hyp = Hypothesis(L, name="groupB vs rest", coef_names=["Intercept", "groupB", "cdr"])
    """

    def __init__(
        self,
        contrast_matrix: np.ndarray,
        name: str = "custom_hypothesis",
        coef_names: list[str] | None = None,
    ) -> None:
        self.contrast_matrix = np.asarray(contrast_matrix, dtype=np.float64)
        self.name = name
        self.coef_names = coef_names

        if self.contrast_matrix.ndim == 1:
            # Promote 1D to column vector
            self.contrast_matrix = self.contrast_matrix[:, np.newaxis]

    def __repr__(self) -> str:
        shape = self.contrast_matrix.shape
        return f"Hypothesis(name='{self.name}', contrast_shape={shape})"

    @property
    def n_hypotheses(self) -> int:
        """Number of linear combinations (columns) in the contrast matrix."""
        return self.contrast_matrix.shape[1]

    @property
    def n_coefs(self) -> int:
        """Number of model coefficients (rows) in the contrast matrix."""
        return self.contrast_matrix.shape[0]

    @classmethod
    def from_coefficient(cls, hypothesis_str: str, coef_names: list[str]) -> Hypothesis:
        """Construct a Hypothesis from a :class:`CoefficientHypothesis` string.

        Convenience factory that wraps ``CoefficientHypothesis.build_contrast()``.

        Parameters
        ----------
        hypothesis_str : str
            Coefficient name or regex pattern (prefix with ``"~"`` for regex).
        coef_names : list[str]
            Full list of model coefficient names.

        Returns
        -------
        Hypothesis
        """
        coeff_hyp = CoefficientHypothesis(hypothesis_str)
        contrast = coeff_hyp.build_contrast(coef_names)
        return cls(contrast, name=hypothesis_str, coef_names=coef_names)
