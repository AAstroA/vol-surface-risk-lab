"""Option cleaning and configurable daily implied-volatility reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from spx_risk.analysis.black_scholes import implied_volatility
from spx_risk.config import AppConfig


@dataclass(frozen=True)
class SurfaceFit:
    observations: pd.DataFrame
    surface: pd.DataFrame
    diagnostics: pd.DataFrame
    method_comparison: pd.DataFrame
    comparison_surfaces: pd.DataFrame
    coverage: pd.DataFrame


def implied_volatility_from_price(row: pd.Series) -> float:
    if not np.isfinite(row["mid_price"]) or row["mid_price"] <= 0:
        return np.nan
    try:
        return implied_volatility(
            market_price=float(row["mid_price"]),
            option_type=str(row["type"]),
            spot=float(row["close"]),
            strike=float(row["strike"]),
            maturity=float(row["years_to_expiration"]),
            rate=float(row["rate"]),
            dividend_yield=float(row.get("dividend_yield", 0.0)),
            lower=1e-4,
            upper=5.0,
        )
    except (ValueError, RuntimeError):
        return np.nan


def _interpolate_zero_curve(options: pd.DataFrame, zero_curve: pd.DataFrame) -> np.ndarray:
    rates = np.full(len(options), np.nan, dtype=float)
    curves = {
        date: group.sort_values("days")[["days", "rate"]].dropna().to_numpy(float)
        for date, group in zero_curve.groupby("quote_date")
    }
    for quote_date, indices in options.groupby("quote_date").groups.items():
        curve = curves.get(quote_date)
        if curve is None or not len(curve):
            continue
        rates[np.asarray(indices)] = np.interp(
            options.loc[indices, "days_to_expiration"].to_numpy(float),
            curve[:, 0],
            curve[:, 1],
        )
    return rates


def prepare_options(
    options: pd.DataFrame,
    underlying: pd.DataFrame,
    zero_curve: pd.DataFrame,
    config: AppConfig,
) -> pd.DataFrame:
    data = options.copy()
    prices = underlying.copy()
    curve = zero_curve.copy()
    for frame, columns in (
        (data, ("quote_date", "expiration")),
        (prices, ("quote_date",)),
        (curve, ("quote_date",)),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column]).dt.normalize()

    if "type" not in data and "cp_flag" in data:
        data["type"] = data["cp_flag"].astype(str).str.upper().map({"C": "call", "P": "put"})
    data["type"] = data["type"].astype(str).str.lower()
    data = data[data["type"].isin(config.data.option_types)].copy()

    prices = prices.sort_values("quote_date").drop_duplicates("quote_date", keep="last")
    price_columns = ["quote_date", "close"]
    if "dividend_yield" in prices:
        price_columns.append("dividend_yield")
    data = data.merge(prices[price_columns], on="quote_date", how="inner")
    data["days_to_expiration"] = (data["expiration"] - data["quote_date"]).dt.days
    data["years_to_expiration"] = data["days_to_expiration"] / 365.25

    curve["rate"] = pd.to_numeric(curve["rate"], errors="coerce")
    # IvyDB zero-curve exports are commonly expressed in percentage points.
    # A 25% cutoff distinguishes that representation from decimal rates without
    # misclassifying ordinary low-rate periods.
    if curve["rate"].abs().quantile(0.99) > 0.25:
        curve["rate"] /= 100.0
    data.reset_index(drop=True, inplace=True)
    data["rate"] = _interpolate_zero_curve(data, curve)
    data["rate"] = data["rate"].fillna(curve.groupby("quote_date")["rate"].mean().median())

    numeric = ["strike", "bid", "ask", "open_interest", "implied_volatility", "close"]
    if "forward_price" in data:
        numeric.append("forward_price")
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["open_interest"] = data["open_interest"].fillna(0)
    data["mid_price"] = (data["bid"] + data["ask"]) / 2.0
    data["relative_spread"] = (data["ask"] - data["bid"]) / data["mid_price"].replace(0, np.nan)
    dividend_yield = (
        pd.to_numeric(data["dividend_yield"], errors="coerce").fillna(0.0)
        if "dividend_yield" in data
        else 0.0
    )
    model_forward = data["close"] * np.exp(
        (data["rate"] - dividend_yield) * data["years_to_expiration"]
    )
    if "forward_price" in data:
        valid_vendor_forward = np.isfinite(data["forward_price"]) & (
            data["forward_price"] > 0
        )
        data["forward"] = data["forward_price"].where(valid_vendor_forward, model_forward)
    else:
        data["forward"] = model_forward
    data["forward_moneyness"] = data["forward"] / data["strike"]
    data["normalized_moneyness"] = np.log(data["forward_moneyness"]) / np.sqrt(
        data["years_to_expiration"]
    )

    iv_median = data["implied_volatility"].abs().median()
    if pd.notna(iv_median) and iv_median > 2:
        data["implied_volatility"] /= 100.0

    f = config.filters
    base_valid = (
        (data["mid_price"] >= f.min_option_price)
        & (data["days_to_expiration"] >= f.min_days_to_expiry)
        & (data["days_to_expiration"] <= f.max_days_to_expiry)
        & (data["forward_moneyness"] >= f.min_forward_moneyness)
        & (data["forward_moneyness"] <= f.max_forward_moneyness)
        & (data["relative_spread"] <= f.max_relative_spread)
        & (data["open_interest"] >= f.min_open_interest)
    )
    missing_iv = ~np.isfinite(data["implied_volatility"]) | (
        data["implied_volatility"] <= 0
    )
    if f.impute_missing_iv:
        impute = base_valid & missing_iv
        data.loc[impute, "implied_volatility"] = data.loc[impute].apply(
            implied_volatility_from_price,
            axis=1,
        )
    valid = (
        base_valid
        & np.isfinite(data["implied_volatility"])
        & (data["implied_volatility"] > 0)
    )
    data = data.loc[valid].sort_values(["quote_date", "expiration", "strike", "type"])
    daily_counts = data.groupby("quote_date").size()
    keep_dates = daily_counts[daily_counts >= f.min_daily_observations].index
    return data[data["quote_date"].isin(keep_dates)].reset_index(drop=True)


def _design_matrix(normalized_moneyness: np.ndarray, maturity: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(normalized_moneyness)),
            normalized_moneyness,
            maturity,
            normalized_moneyness**2,
            normalized_moneyness * maturity,
        ]
    )


def _grid_coordinates(config: AppConfig) -> tuple[np.ndarray, np.ndarray, list[tuple[int, float]]]:
    labels = [
        (maturity_days, moneyness)
        for maturity_days in config.surface.maturity_days
        for moneyness in config.surface.moneyness_grid
    ]
    maturity = np.asarray([days / 365.25 for days, _ in labels], dtype=float)
    normalized = np.asarray(
        [np.log(moneyness) / np.sqrt(years) for years, (_, moneyness) in zip(maturity, labels)],
        dtype=float,
    )
    return normalized, maturity, labels


def _tensor_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ik->ijk", left, right).reshape(len(left), -1)


def _fit_predict_log_iv(
    method: str,
    train_x: np.ndarray,
    train_t: np.ndarray,
    train_y: np.ndarray,
    evaluation_x: np.ndarray,
    evaluation_t: np.ndarray,
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    config: AppConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit one reconstruction family and predict log-IV at two coordinate sets."""
    if method == "polynomial":
        design = _design_matrix(train_x, train_t)
        coefficients, _, rank, _ = np.linalg.lstsq(design, train_y, rcond=None)
        if rank < design.shape[1]:
            raise ValueError("rank-deficient polynomial surface")
        details = {f"beta_{index}": float(value) for index, value in enumerate(coefficients)}
        return (
            _design_matrix(evaluation_x, evaluation_t) @ coefficients,
            _design_matrix(grid_x, grid_t) @ coefficients,
            details,
        )

    if method == "b_spline":
        knots = max(3, config.surface.spline_knots)
        x_transformer = SplineTransformer(
            n_knots=knots, degree=3, knots="quantile", include_bias=True, extrapolation="linear"
        )
        t_transformer = SplineTransformer(
            n_knots=knots, degree=3, knots="quantile", include_bias=True, extrapolation="linear"
        )
        basis_x = x_transformer.fit_transform(train_x.reshape(-1, 1))
        basis_t = t_transformer.fit_transform(train_t.reshape(-1, 1))
        model = Ridge(alpha=config.surface.ridge_alpha, fit_intercept=False, solver="cholesky")
        model.fit(_tensor_product(basis_x, basis_t), train_y)

        def predict(x: np.ndarray, t: np.ndarray) -> np.ndarray:
            return model.predict(
                _tensor_product(
                    x_transformer.transform(x.reshape(-1, 1)),
                    t_transformer.transform(t.reshape(-1, 1)),
                )
            )

        return predict(evaluation_x, evaluation_t), predict(grid_x, grid_t), {
            "basis_functions": float(len(model.coef_))
        }

    if method == "thin_plate":
        x_scale = max(float(np.std(train_x)), 1e-8)
        t_scale = max(float(np.std(train_t)), 1e-8)
        x_center = float(np.mean(train_x))
        t_center = float(np.mean(train_t))
        x_quantiles = np.quantile(train_x, np.linspace(0.05, 0.95, config.surface.spline_knots))
        t_quantiles = np.quantile(train_t, np.linspace(0.05, 0.95, config.surface.spline_knots))
        centers = np.asarray([(x, t) for x in x_quantiles for t in t_quantiles], dtype=float)

        def features(x: np.ndarray, t: np.ndarray) -> np.ndarray:
            scaled_x = (x - x_center) / x_scale
            scaled_t = (t - t_center) / t_scale
            center_x = (centers[:, 0] - x_center) / x_scale
            center_t = (centers[:, 1] - t_center) / t_scale
            radius_squared = np.square(scaled_x[:, None] - center_x) + np.square(
                scaled_t[:, None] - center_t
            )
            radial = radius_squared * np.log(np.sqrt(radius_squared) + 1e-12)
            return np.column_stack([np.ones(len(x)), scaled_x, scaled_t, radial])

        model = Ridge(alpha=config.surface.ridge_alpha, fit_intercept=False, solver="cholesky")
        model.fit(features(train_x, train_t), train_y)
        return (
            model.predict(features(evaluation_x, evaluation_t)),
            model.predict(features(grid_x, grid_t)),
            {"radial_centers": float(len(centers))},
        )

    if method == "linear":
        training = pd.DataFrame({"x": train_x, "t": train_t, "y": train_y}).groupby(
            ["x", "t"], as_index=False
        )["y"].mean()
        points = training[["x", "t"]].to_numpy(float)
        values = training["y"].to_numpy(float)
        nearest = NearestNDInterpolator(points, values)
        try:
            linear = LinearNDInterpolator(points, values, fill_value=np.nan)

            def predict(x: np.ndarray, t: np.ndarray) -> np.ndarray:
                coordinates = np.column_stack([x, t])
                result = np.asarray(linear(coordinates), dtype=float)
                missing = ~np.isfinite(result)
                if missing.any():
                    result[missing] = nearest(coordinates[missing])
                return result

        except Exception:  # Qhull may fail on a nearly collinear daily cross-section.
            def predict(x: np.ndarray, t: np.ndarray) -> np.ndarray:
                return np.asarray(nearest(np.column_stack([x, t])), dtype=float)

        return predict(evaluation_x, evaluation_t), predict(grid_x, grid_t), {
            "unique_nodes": float(len(training))
        }

    raise ValueError(f"Unknown surface method: {method}")


def _surface_rows(
    quote_date: pd.Timestamp,
    method: str,
    predictions: np.ndarray,
    labels: list[tuple[int, float]],
    config: AppConfig,
    extrapolated: np.ndarray | None = None,
) -> list[dict[str, object]]:
    iv = np.clip(np.exp(predictions), config.surface.iv_floor, config.surface.iv_cap)
    if extrapolated is None:
        extrapolated = np.zeros(len(labels), dtype=bool)
    return [
        {
            "quote_date": quote_date,
            "method": method,
            "maturity_days": maturity_days,
            "moneyness": moneyness,
            "predicted_iv": float(value),
            "extrapolated": bool(is_extrapolated),
        }
        for (maturity_days, moneyness), value, is_extrapolated in zip(
            labels, iv, extrapolated, strict=True
        )
    ]


def _guard_grid_extrapolation(
    train_x: np.ndarray,
    train_t: np.ndarray,
    fitted_log_iv: np.ndarray,
    grid_x: np.ndarray,
    grid_t: np.ndarray,
    grid_prediction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace unsupported grid extrapolation while preserving all in-hull predictions."""
    points = np.column_stack([train_x, train_t])
    grid_points = np.column_stack([grid_x, grid_t])
    try:
        hull_probe = LinearNDInterpolator(points, np.ones(len(points)), fill_value=np.nan)
        outside = ~np.isfinite(np.asarray(hull_probe(grid_points), dtype=float))
    except Exception:
        outside = np.zeros(len(grid_points), dtype=bool)
    stabilized = np.asarray(grid_prediction, dtype=float).copy()
    if outside.any():
        nearest = NearestNDInterpolator(points, fitted_log_iv)
        stabilized[outside] = nearest(grid_points[outside])
    return stabilized, outside


def _coverage(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for quote_date, group in observations.groupby("quote_date", sort=True):
        rows.append(
            {
                "quote_date": quote_date,
                "contracts": int(group["contract"].nunique()) if "contract" in group else len(group),
                "strikes": int(group["strike"].nunique()),
                "expirations": int(group["expiration"].nunique()),
                "open_interest": float(group["open_interest"].sum()),
                "eligible_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def fit_daily_surfaces(observations: pd.DataFrame, config: AppConfig) -> SurfaceFit:
    surface_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    comparison_surface_rows: list[dict[str, object]] = []
    grid_x, grid_t, grid_labels = _grid_coordinates(config)
    grouped = observations.groupby("quote_date", sort=True)
    date_count = grouped.ngroups
    comparison_frequency = max(1, config.surface.comparison_frequency)
    comparison_methods = tuple(dict.fromkeys((config.surface.method, *config.surface.comparison_methods)))

    for date_index, (quote_date, group) in enumerate(grouped):
        x = group["normalized_moneyness"].to_numpy(float)
        maturity = group["years_to_expiration"].to_numpy(float)
        y = np.log(group["implied_volatility"].to_numpy(float))
        try:
            fitted, grid_prediction, details = _fit_predict_log_iv(
                config.surface.method, x, maturity, y, x, maturity, grid_x, grid_t, config
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = y - fitted
        grid_prediction, grid_extrapolated = _guard_grid_extrapolation(
            x, maturity, fitted, grid_x, grid_t, grid_prediction
        )
        total = np.square(y - y.mean()).sum()
        r_squared = 1.0 - np.square(residual).sum() / total if total > 0 else np.nan

        diagnostic_rows.append(
            {
                "quote_date": quote_date,
                "method": config.surface.method,
                "observations": len(group),
                "r_squared": r_squared,
                "log_iv_rmse": float(np.sqrt(np.mean(np.square(residual)))),
                **details,
            }
        )
        surface_rows.extend(
            _surface_rows(
                quote_date,
                config.surface.method,
                grid_prediction,
                grid_labels,
                config,
                grid_extrapolated,
            )
        )

        compare_today = date_index % comparison_frequency == 0 or date_index == date_count - 1
        if compare_today and len(group) >= 12:
            validation = np.arange(len(group)) % 5 == 0
            if validation.all() or (~validation).sum() < 6:
                validation[-max(1, len(group) // 5) :] = True
            for method in comparison_methods:
                try:
                    predicted, method_grid, _ = _fit_predict_log_iv(
                        method,
                        x[~validation],
                        maturity[~validation],
                        y[~validation],
                        x[validation],
                        maturity[validation],
                        grid_x,
                        grid_t,
                        config,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    continue
                errors = y[validation] - predicted
                total_validation = np.square(y[validation] - y[validation].mean()).sum()
                comparison_rows.append(
                    {
                        "quote_date": quote_date,
                        "method": method,
                        "training_observations": int((~validation).sum()),
                        "validation_observations": int(validation.sum()),
                        "log_iv_rmse": float(np.sqrt(np.mean(np.square(errors)))),
                        "log_iv_mae": float(np.mean(np.abs(errors))),
                        "iv_rmse": float(
                            np.sqrt(np.mean(np.square(np.exp(y[validation]) - np.exp(predicted))))
                        ),
                        "r_squared": (
                            float(1 - np.square(errors).sum() / total_validation)
                            if total_validation > 0
                            else np.nan
                        ),
                    }
                )
                if date_index == date_count - 1:
                    # Refit to all observations for an apples-to-apples final-date 3D comparison.
                    full_fitted, full_grid, _ = _fit_predict_log_iv(
                        method, x, maturity, y, x, maturity, grid_x, grid_t, config
                    )
                    full_grid, full_extrapolated = _guard_grid_extrapolation(
                        x, maturity, full_fitted, grid_x, grid_t, full_grid
                    )
                    comparison_surface_rows.extend(
                        _surface_rows(
                            quote_date,
                            method,
                            full_grid,
                            grid_labels,
                            config,
                            full_extrapolated,
                        )
                    )

    surface = pd.DataFrame(surface_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    comparison = pd.DataFrame(comparison_rows)
    comparison_surfaces = pd.DataFrame(comparison_surface_rows)
    if not surface.empty:
        surface = surface.sort_values(["quote_date", "maturity_days", "moneyness"])
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values("quote_date")
    if not comparison.empty:
        comparison = comparison.sort_values(["quote_date", "method"])
    return SurfaceFit(
        observations=observations,
        surface=surface,
        diagnostics=diagnostics,
        method_comparison=comparison,
        comparison_surfaces=comparison_surfaces,
        coverage=_coverage(observations),
    )
