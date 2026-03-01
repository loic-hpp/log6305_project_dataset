"""
01_build_dataset_fixed.py
--------------------------
Fixed version that works with the actual scraped TravisTorrent CSV.

Since log_snippet is truncated (Travis API v3 limitation), we cannot
extract individual test names. Instead we label at the build level:
  - Failed builds in projects that HAVE flaky tests in iDFlakies
    → could be FA or TP → we use failure rate heuristic
  - Failed builds in projects with NO flaky tests in iDFlakies → TP
  - Passed builds → used as negative context (not labelled)

Labelling strategy:
  - If a project has known flaky tests (in iDFlakies) AND the build
    failed repeatedly for the same reason → FA (flaky)
  - If a project has known flaky tests AND build failed once → TP
  - If a project has NO known flaky tests → TP
"""

import pandas as pd
import numpy as np
from pathlib import Path

TRAVIS_CSV      = Path("data/raw/travistorrent_8_17_2017.csv")
IDOFT_MAVEN_CSV = Path("data/raw/pr-data.csv")   # IDoFT Maven (Java)
OUT_CSV         = Path("data/processed/labelled_builds.csv")

# ── Load ─────────────────────────────────────────────────────────────────
print("Loading data...")
tt    = pd.read_csv(TRAVIS_CSV, low_memory=False)
idoft = pd.read_csv(IDOFT_MAVEN_CSV)

print(f"  TravisTorrent: {tt.shape}")
print(f"  IDoFT:         {idoft.shape}")

# ── Prepare iDFlakies ─────────────────────────────────────────────────────
# Extract project name from URL
idoft["project_name"] = (
    idoft["Project URL"]
    .str.replace("https://github.com/", "", regex=False)
    .str.lower()
)
# Extract test class name (drop method suffix after last dot)
idoft["test_class"] = (
    idoft["Test Name"]
    .str.rsplit(".", n=1).str[0]   # e.g. org.apache.Foo.testBar → org.apache.Foo
    .str.rsplit(".", n=1).str[-1]  # → Foo (simple class name)
)

# Projects with known flaky tests
flaky_projects = set(idoft["project_name"].unique())

# Flaky test classes per project
flaky_by_project = (
    idoft.groupby("project_name")["test_class"]
    .apply(set)
    .to_dict()
)
print(f"\n  Projects with flaky tests in IDoFT: {len(flaky_projects)}")

# ── Filter TravisTorrent to failed Java builds ────────────────────────────
tt["gh_project_name_lower"] = tt["gh_project_name"].str.lower()
failed = tt[tt["tr_status"].isin(["failed", "errored"])].copy()
print(f"  Failed/errored builds: {len(failed)}")

# ── Compute per-build failure rate within project ─────────────────────────
# Sort by build_id (proxy for time)
tt_sorted = tt.sort_values("build_id")

# Rolling failure rate: for each build, what fraction of last 10 builds failed?
def rolling_fail_rate(group, k=10):
    is_fail = (group["tr_status"].isin(["failed", "errored"])).astype(int)
    return is_fail.rolling(window=k, min_periods=1).mean().shift(1).fillna(0.5)

tt_sorted["hist_fail_rate"] = (
    tt_sorted.groupby("gh_project_name", group_keys=False)
    .apply(rolling_fail_rate)
)

failed = failed.merge(
    tt_sorted[["build_id", "hist_fail_rate"]],
    on="build_id", how="left"
)
failed["hist_fail_rate"] = failed["hist_fail_rate"].fillna(0.5)

# ── Label construction ────────────────────────────────────────────────────
#
# Logic:
#   1. Project NOT in iDFlakies → TP (no known flaky tests, failure is real)
#   2. Project IN iDFlakies AND hist_fail_rate > 0.5 → FA (high recurrence = flaky)
#   3. Project IN iDFlakies AND hist_fail_rate <= 0.5 → TP (isolated failure = real)

def assign_label(row):
    proj = row["gh_project_name_lower"]
    if proj not in flaky_projects:
        return 1   # TP — no known flaky tests
    elif row["hist_fail_rate"] > 0.5:
        return 0   # FA — high recurring failure rate → likely flaky
    else:
        return 1   # TP — isolated failure even in flaky project

failed["label"] = failed.apply(assign_label, axis=1)

# ── Feature columns to keep ───────────────────────────────────────────────
keep_cols = [
    "build_id", "gh_project_name", "gh_lang", "gh_is_pr",
    "tr_status", "duration", "event_type",
    "tr_tests_run", "tr_tests_failed", "tr_tests_passed",
    "git_diff_src_churn", "git_diff_test_churn",
    "gh_team_size", "gh_repo_age",
    "hist_fail_rate", "log_snippet", "label",
]

# keep only columns that exist
keep_cols = [c for c in keep_cols if c in failed.columns]
out = failed[keep_cols].copy()

# ── Save ──────────────────────────────────────────────────────────────────
out.to_csv(OUT_CSV, index=False)

print(f"\n{'='*50}")
print(f"Output: {OUT_CSV}")
print(f"Total labelled builds : {len(out):,}")
print(f"  TP (label=1)        : {(out.label==1).sum():,}")
print(f"  FA (label=0)        : {(out.label==0).sum():,}")
print(f"\nLabel distribution by project:")
print(out.groupby(["gh_project_name", "label"]).size().unstack(fill_value=0))
