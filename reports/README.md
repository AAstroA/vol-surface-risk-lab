# MSc thesis report and reproducible artifacts

`SPX_WRDS_Thesis_Report_2005_2021.pdf` is the public empirical report for Alireza Moslemi Haghighi's MSc thesis project and subsequent research extensions.

`SPX_WRDS_Thesis_Report_2005_2021.md` is the clean text source for the public report. Machine-generated empirical table fragments and the table-generation utility remain under `latex/generated/` and `latex/generate_tables.py` for auditability.

The default live WRDS run creates `default_run/figures/`, `default_run/tables/`, `default_run/processed/`, and `default_run/run_summary.json`. These paths are Git-ignored because they derive from licensed WRDS data. Recreate them with:

```bash
spx-risk run --config configs/default.yaml
```

Shareable deterministic equivalents are committed under `sample_outputs/demo/`. The browser teaching interface uses model-generated values and does not expose WRDS/OptionMetrics observations.
