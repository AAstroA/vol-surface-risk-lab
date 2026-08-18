from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

from spx_risk.analysis.black_scholes import (
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)
from spx_risk.analysis.heston import (
    HestonParameters,
    heston_normalized_call_price_grid,
    heston_normalized_call_prices,
    normalized_black_call_prices,
    normalized_implied_volatilities,
)
from spx_risk.analysis.heston_mc import simulate_heston_states
from spx_risk.analysis.pca import run_pca
from spx_risk.analysis.risk import run_var_backtest
from spx_risk.analysis.surface import fit_daily_surfaces, prepare_options
from spx_risk.config import load_config
from spx_risk.data.demo import generate_demo_dataset
from spx_risk.data.wrds import _safe_identifier
from spx_risk.pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_date_overrides_are_applied(self) -> None:
        config = load_config(
            PROJECT_ROOT / "configs/demo.yaml",
            start_date="2013-02-01",
            end_date="2013-02-28",
        )
        self.assertEqual(config.data.start_date, date(2013, 2, 1))
        self.assertEqual(config.data.end_date, date(2013, 2, 28))

    def test_invalid_sql_identifier_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _safe_identifier("optionm_all; DROP TABLE x")


class BlackScholesTests(unittest.TestCase):
    def test_price_greeks_and_implied_volatility(self) -> None:
        call = black_scholes_price("call", 100.0, 100.0, 1.0, 0.05, 0.20)
        put = black_scholes_price("put", 100.0, 100.0, 1.0, 0.05, 0.20)
        greeks = black_scholes_greeks("call", 100.0, 100.0, 1.0, 0.05, 0.20)
        recovered = implied_volatility(call, "call", 100.0, 100.0, 1.0, 0.05)
        self.assertAlmostEqual(call, 10.4506, places=4)
        self.assertAlmostEqual(put, 5.5735, places=4)
        self.assertAlmostEqual(greeks.vega, 37.5240, places=3)
        self.assertAlmostEqual(recovered, 0.20, places=10)


class HestonTests(unittest.TestCase):
    def test_constant_variance_limit_matches_black_scholes(self) -> None:
        import numpy as np

        strikes = np.array([0.8, 1.0, 1.2])
        maturities = np.ones(3)
        parameters = HestonParameters(
            v0=0.04, kappa=2.0, theta=0.04, xi=1e-4, rho=-0.7
        )
        heston = heston_normalized_call_prices(
            strikes, maturities, parameters, integration_nodes=96
        )
        black = normalized_black_call_prices(
            strikes, maturities, np.full(3, 0.20)
        )
        np.testing.assert_allclose(heston, black, rtol=0, atol=8e-6)

    def test_normalized_implied_volatility_round_trip(self) -> None:
        import numpy as np

        strikes = np.array([0.85, 1.0, 1.15])
        maturities = np.array([30, 180, 360]) / 365.25
        volatility = np.array([0.24, 0.19, 0.22])
        prices = normalized_black_call_prices(strikes, maturities, volatility)
        recovered = normalized_implied_volatilities(prices, strikes, maturities)
        np.testing.assert_allclose(recovered, volatility, rtol=0, atol=2e-10)

    def test_rectangular_variance_price_grid_matches_scalar_pricer(self) -> None:
        import numpy as np

        strikes = np.array([0.82, 1.0, 1.18])
        variances = np.array([0.025, 0.04, 0.075])
        structural = HestonParameters(0.04, 2.5, 0.045, 0.65, -0.72)
        grid = heston_normalized_call_price_grid(
            strikes, 0.5, variances, structural, integration_nodes=96
        )
        scalar = np.vstack(
            [
                heston_normalized_call_prices(
                    strikes,
                    np.full(len(strikes), 0.5),
                    HestonParameters(
                        variance,
                        structural.kappa,
                        structural.theta,
                        structural.xi,
                        structural.rho,
                    ),
                    integration_nodes=96,
                )
                for variance in variances
            ]
        )
        np.testing.assert_allclose(grid, scalar, rtol=0, atol=2e-12)

    def test_projected_heston_states_remain_positive(self) -> None:
        import numpy as np

        rng = np.random.default_rng(17)
        spot, variance = simulate_heston_states(
            spot=100.0,
            variance=0.04,
            mu_total=0.03,
            dividend_yield=0.01,
            parameters=HestonParameters(0.04, 3.0, 0.04, 0.8, -0.75),
            standard_normals=rng.standard_normal((8, 5_000, 2)),
            steps=8,
        )
        self.assertTrue(np.all(spot > 0.0))
        self.assertTrue(np.all(variance >= 0.0))
        self.assertLess(abs(np.mean(spot / 100.0 - 1.0)), 0.002)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = load_config(
            PROJECT_ROOT / "configs/demo.yaml",
            start_date="2013-01-02",
            end_date="2013-03-29",
        )
        cls.config = replace(
            base,
            risk=replace(
                base.risk,
                scenarios=120,
                rolling_window=20,
                minimum_history=8,
            ),
        )
        cls.dataset = generate_demo_dataset(cls.config)

    def test_surface_pca_and_risk_contracts(self) -> None:
        observations = prepare_options(
            self.dataset.options,
            self.dataset.underlying,
            self.dataset.zero_curve,
            self.config,
        )
        fitted = fit_daily_surfaces(observations, self.config)
        pca = run_pca(fitted.surface, self.config)
        risk = run_var_backtest(pca, self.config)

        self.assertGreater(len(observations), 1_000)
        self.assertEqual(
            fitted.surface.groupby("quote_date").size().nunique(),
            1,
        )
        self.assertEqual(pca.loadings.shape[1], 36)
        self.assertIn("var_95_pca", risk.backtest)
        self.assertIn("var_95_pwg", risk.backtest)
        self.assertIn("var_95_psp", risk.backtest)
        self.assertEqual(set(risk.diagnostics["method"]), {"PSP", "PCA", "PWG"})
        self.assertEqual(set(risk.diagnostics["confidence"]), {0.90, 0.95, 0.99})
        self.assertEqual(len(risk.diagnostics), 9)
        self.assertIn("conditional_coverage_p_value", risk.diagnostics)
        self.assertIn("mean_quantile_loss", risk.diagnostics)
        self.assertEqual(
            set(fitted.method_comparison["method"]),
            {"polynomial", "b_spline", "thin_plate", "linear"},
        )

    def test_zero_curve_percentage_points_are_normalized(self) -> None:
        decimal = prepare_options(
            self.dataset.options,
            self.dataset.underlying,
            self.dataset.zero_curve,
            self.config,
        )
        percentage_curve = self.dataset.zero_curve.copy()
        percentage_curve["rate"] *= 100.0
        percentage = prepare_options(
            self.dataset.options,
            self.dataset.underlying,
            percentage_curve,
            self.config,
        )
        self.assertAlmostEqual(decimal["rate"].median(), percentage["rate"].median(), places=12)

    def test_pipeline_writes_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = replace(
                self.config.output,
                root=Path(directory) / "run",
            )
            config = replace(self.config, output=output)
            result = run_pipeline(config, dataset=self.dataset)
            self.assertTrue((result.output_root / "run_summary.json").is_file())
            self.assertTrue((result.output_root / "tables/var_backtest.csv").is_file())
            self.assertTrue((result.output_root / "tables/var_model_ranking.csv").is_file())
            self.assertEqual(len(result.figures), 17)


if __name__ == "__main__":
    unittest.main()
