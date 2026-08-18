"""Principal-component extraction for daily implied-volatility-surface changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from spx_risk.config import AppConfig


@dataclass(frozen=True)
class PCAResult:
    surface_matrix: pd.DataFrame
    changes: pd.DataFrame
    scores: pd.DataFrame
    loadings: pd.DataFrame
    explained_variance: pd.DataFrame
    model: PCA
    scaler: StandardScaler | None

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        values = self.model.inverse_transform(scores)
        if self.scaler is not None:
            values = self.scaler.inverse_transform(values)
        return values


def run_pca(surface: pd.DataFrame, config: AppConfig) -> PCAResult:
    matrix = surface.pivot(
        index="quote_date", columns=["maturity_days", "moneyness"], values="predicted_iv"
    ).sort_index()
    matrix = matrix.interpolate(limit_direction="both").dropna(axis=0, how="any")
    changes = matrix.diff().dropna()
    if len(changes) < 2:
        raise ValueError("At least three fitted surface dates are required for PCA")

    scaler: StandardScaler | None = None
    values = changes.to_numpy(float)
    if config.pca.standardize:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)

    component_count = min(config.pca.components, values.shape[0], values.shape[1])
    model = PCA(n_components=component_count, svd_solver="full")
    score_values = model.fit_transform(values)
    component_names = [f"PC{index + 1}" for index in range(component_count)]
    scores = pd.DataFrame(score_values, index=changes.index, columns=component_names)

    loadings_values = model.components_.copy()
    if scaler is not None:
        loadings_values *= scaler.scale_[None, :]
    loadings = pd.DataFrame(loadings_values, index=component_names, columns=changes.columns)

    explained = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_explained_variance": np.cumsum(model.explained_variance_ratio_),
            "eigenvalue": model.explained_variance_,
        }
    )
    return PCAResult(matrix, changes, scores, loadings, explained, model, scaler)
