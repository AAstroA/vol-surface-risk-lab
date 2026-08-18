# Public release policy

## Allowed on the public branch

- Project-authored source code and configuration templates.
- Deterministic synthetic demonstrations and their generated outputs.
- Tests, methodological explanations, and non-empirical diagrams.
- Aggregate empirical material only after written institutional clearance identifies it as permitted.

## Not allowed without written clearance

- Credentials or authentication artifacts.
- Raw, cleaned, transformed, or reconstructed vendor observations.
- Dated panels, extracts, caches, or APIs that substitute for licensed access.
- Detailed WRDS/OptionMetrics-derived reports, tables, and figures whose publication status has not been confirmed under Bocconi's agreements.

## Release procedure

1. Run `python scripts/audit_public_release.py .`.
2. Confirm every bundled dataset has a provenance statement.
3. Confirm the WRDS acknowledgment appears in communications prepared using WRDS.
4. Retain written clearance for any empirical outputs restored to the public branch.
5. Re-run a complete history audit after any accidental-data incident.
