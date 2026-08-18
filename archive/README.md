# Private legacy boundary

Legacy material is retained locally for provenance but is not part of the public repository or the
executable production path.

- Private notebooks, benchmarks, figures, MATLAB code, and the migration manifest are Git-ignored.
- The obsolete PwG interest-rate workbooks are excluded from both the maintained analysis and the
  public release.
- The maintained implementation under `src/spx_risk/` does not import private archive content.

This boundary prevents disclosure of private paths, document metadata, licensed data, and obsolete
credentials while preserving a clean, reproducible public project.
