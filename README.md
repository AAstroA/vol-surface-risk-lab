# Volatility Surface Risk Lab

**Public-safe research-software release.** This project began as **Alireza Moslemi Haghighi's master's-thesis work on Parametric Surface Projection (PSP)**, developed with Shiva Zamani and Hamid Arian and documented in arXiv:2311.14985. Alireza subsequently extended it into a reproducible option-surface risk platform with rolling PCA, Heston models, nonlinear repricing, and formal validation.

## Research objective

The project studies option-portfolio market risk while allowing the complete implied-volatility surface to move jointly across forward moneyness and maturity. It separates four tasks that must not be confused:

1. reconstruct the current surface;
2. specify a dated physical transition law;
3. price or reprice the portfolio under the projected state; and
4. validate VaR and Expected Shortfall forecasts out of sample.

PSP forms complete historical surface shocks and applies them to the current market state. The controlled experiment maps shocks through a disclosed vega vector. The full-revaluation experiment fixes strikes, rolls maturities, preserves paired spot/surface dependence, and reprices the portfolio.

## Public-release boundary

This public branch is intended to contain only:

- project-authored source code and configuration templates;
- deterministic synthetic demo data produced by `src/spx_risk/data/demo.py`;
- tests and non-empirical documentation; and
- public teaching interfaces driven only by project-authored synthetic inputs.

It is **not** intended to contain credentials, raw or cleaned WRDS/OptionMetrics observations, licensed caches, dated empirical panels, or detailed vendor-data-derived reports while publication rights are under institutional review. Repository access does not grant database access or redistribution rights.

## Required WRDS acknowledgment

> Wharton Research Data Services (WRDS) was used in preparing the SPX Volatility-Surface Risk master's-thesis research and subsequent research-extension report. This service and the data available thereon constitute valuable intellectual property and trade secrets of WRDS and/or its third-party suppliers.

This acknowledgment is attribution, not a license. Use of WRDS services is governed by Bocconi University's subscription agreement, and OptionMetrics requires a separate institutional license.

## Synthetic public demo

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
spx-risk demo --config configs/demo.yaml
```

The checked demo under `sample_outputs/demo/` is deterministic and synthetic. It is not a sample of OptionMetrics, Cboe, exchange, or other vendor observations.

## Authorized licensed-data use

Only independently authorized users may run the WRDS pipeline, using their own credentials under their institution's agreements:

```bash
cp .env.example .env
chmod 600 .env
spx-risk probe-wrds --config configs/default.yaml
```

Credentials and licensed outputs must remain local and uncommitted. Raw, interim, and processed data paths are Git-ignored.

## Software and data licensing

The MIT license covers project-authored software unless a file says otherwise. It does **not** license WRDS, OptionMetrics, exchange data, third-party databases, documentation, names, marks, or contract-restricted outputs. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md), [THIRD_PARTY_DATA_AND_LICENSE_NOTICE.md](THIRD_PARTY_DATA_AND_LICENSE_NOTICE.md), and [PUBLIC_RELEASE_POLICY.md](PUBLIC_RELEASE_POLICY.md).

## Empirical report status

The detailed 2005–2021 WRDS/OptionMetrics-derived report and generated empirical fragments are withheld from the public branch pending written confirmation from Bocconi's electronic-resource licensing staff. See [reports/README.md](reports/README.md).

## Public-release audit

Run before every public push:

```bash
python scripts/audit_public_release.py .
```

## Attribution and non-endorsement

References to WRDS, Wharton, OptionMetrics, Bocconi, Cboe, SPX, or other names identify sources, institutions, instruments, or methodology only. No affiliation, sponsorship, certification, or endorsement is implied.
