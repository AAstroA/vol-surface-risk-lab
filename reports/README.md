# Generated and source reports

Original reports and proposals are retained only in the private local project and are Git-ignored.

The default live WRDS run creates `default_run/figures/`, `default_run/tables/`, `default_run/processed/`, and `default_run/run_summary.json`. These paths are Git-ignored because they derive from licensed WRDS data. Recreate them with:

```bash
spx-risk run --config configs/default.yaml
```

Shareable deterministic equivalents are committed under `sample_outputs/demo/`.

`SPX_WRDS_Thesis_Report_2005_2021.pdf` is the publication-ready project report. Its reproducible source, generated result-table fragments, and table-generation script are in `latex/`.
