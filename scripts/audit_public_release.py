#!/usr/bin/env python3
"""Conservative public-release gate for accidental licensed-data or secret disclosure."""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
FORBIDDEN_PARTS = {"data/raw", "data/interim", "data/processed", "reports/latex", "reports/v4_1"}
FORBIDDEN_NAMES = {".env", "SPX_WRDS_Thesis_Report_2005_2021.pdf", "SPX_WRDS_Thesis_Report_2005_2021_v4_1.pdf"}
BINARY_DATA = {".parquet", ".feather", ".arrow", ".sas7bdat", ".dta", ".rdata", ".rds", ".xlsx"}
SECRET_PATTERNS = [
    re.compile(r"WRDS_PASSWORD\s*=\s*[^\s#]+", re.I),
    re.compile(r"WRDS_USERNAME\s*=\s*[^\s#]+", re.I),
    re.compile(r"(?:api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]+", re.I),
]
ALLOW_SYNTHETIC = {"sample_outputs/demo/processed/clean_options.parquet", "sample_outputs/demo/processed/fitted_surface.parquet"}
errors=[]
for path in ROOT.rglob("*"):
    if not path.is_file() or ".git" in path.parts:
        continue
    rel=path.relative_to(ROOT).as_posix()
    if path.name in FORBIDDEN_NAMES:
        errors.append(f"forbidden public file: {rel}")
    if any(rel == p or rel.startswith(p + "/") for p in FORBIDDEN_PARTS):
        if not rel.endswith(".gitkeep"):
            errors.append(f"licensed/generated data path: {rel}")
    if path.suffix.lower() in BINARY_DATA and rel not in ALLOW_SYNTHETIC:
        errors.append(f"unapproved binary data file: {rel}")
    if path.stat().st_size > 20_000_000:
        errors.append(f"oversized file requires review: {rel}")
    if path.suffix.lower() in {".md", ".txt", ".py", ".yml", ".yaml", ".json", ".toml", ".env", ".js", ".html"}:
        text=path.read_text(encoding="utf-8", errors="ignore")
        if path.name != ".env.example":
            for pat in SECRET_PATTERNS:
                if pat.search(text): errors.append(f"possible secret in {rel}")
if errors:
    print("PUBLIC RELEASE AUDIT FAILED", file=sys.stderr)
    for item in sorted(set(errors)): print(f"- {item}", file=sys.stderr)
    raise SystemExit(1)
print("Public release audit passed.")
