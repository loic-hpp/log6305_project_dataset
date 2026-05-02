"""
02_feature_engineering.py
--------------------------
Phase 2: Extract the three feature families from the labelled CSV
and produce a ready-to-train feature matrix (features.npz + metadata.csv).

Requirements:
    pip install pandas numpy scikit-learn scipy tqdm
"""

import re
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

IN_CSV       = Path("data/processed/labelled_builds.csv")
OUT_FEATURES = Path("data/processed/features.npz")
OUT_META     = Path("data/processed/metadata.csv")

# ── column name mapping (scraped CSV uses different names) ────────────────
LOG_COL      = "log_snippet"   # scraped CSV has log_snippet not log_text
DURATION_COL = "duration"      # scraped CSV has duration not tr_duration

# ── keyword lists ────────────────────────────────────────────────────────────
FA_KEYWORDS  = [          # strongly associated with flaky / infrastructure failures
    "timeout", "connection refused", "socket", "network",
    "SocketException", "ConnectException", "read timed out",
    "flaky", "intermittent", "retry", "race condition",
]
TP_KEYWORDS  = [          # strongly associated with genuine code defects
    "AssertionError", "NullPointerException", "ClassCastException",
    "IndexOutOfBoundsException", "StackOverflowError",
    "expected but was", "expected:<", "but was:<",
    "comparison failure",
]


def keyword_features(text: str) -> dict:
    t = text.lower()
    feats = {}
    for kw in FA_KEYWORDS:
        feats[f"kw_fa_{kw.replace(' ', '_')}"] = int(kw.lower() in t)
    for kw in TP_KEYWORDS:
        feats[f"kw_tp_{kw.replace(' ', '_')}"] = int(kw.lower() in t)
    feats["n_stacktrace_lines"] = text.count("\tat ")
    feats["n_assertion_lines"]  = sum(1 for l in text.splitlines()
                                      if "assert" in l.lower())
    return feats


def compute_historical_instability(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    For each (project) compute rolling failure rate over the last k builds.
    Uses build-level label since we don't have individual test names.
    """
    df = df.sort_values("build_id").copy()

    history: defaultdict[str, list] = defaultdict(list)
    fail_rate = {}

    for _, r in df.iterrows():
        key  = str(r["gh_project_name"])
        hist = history[key][-k:]
        fail_rate[r["build_id"]] = np.mean(hist) if hist else 0.5
        history[key].append(int(r["label"]))

    df["hist_fail_rate_mean"] = df["build_id"].map(fail_rate).fillna(0.5)
    return df[["build_id", "hist_fail_rate_mean"]]


def main() -> None:
    log.info("Loading labelled builds …")
    df = pd.read_csv(IN_CSV)

    # ── deduplicate: keep one row per build_id (first occurrence) ────────
    before = len(df)
    df = df.drop_duplicates(subset="build_id", keep="first").reset_index(drop=True)
    log.info(f"  Deduplicated: {before:,} → {len(df):,} unique builds")
    log.info(f"  TP={( df.label==1).sum():,}  FA={(df.label==0).sum():,}")

    # ── 1. keyword features ──────────────────────────────────────────────────
    log.info("Extracting keyword features …")
    kw_feats = df[LOG_COL].fillna("").apply(keyword_features)
    kw_df    = pd.DataFrame(list(kw_feats))

    # ── 2. contextual / metadata features ───────────────────────────────────
    log.info("Extracting contextual features …")
    ctx_cols = [
        "gh_is_pr", "git_diff_src_churn", "git_diff_test_churn",
        "tr_tests_run", "tr_tests_failed",
        "gh_team_size", "gh_repo_age",
    ]
    ctx_df = df[[c for c in ctx_cols if c in df.columns]].fillna(0).copy()
    # add duration — column may be called "duration" or "tr_duration"
    ctx_df["duration"] = df.get(DURATION_COL, pd.Series(0, index=df.index)).fillna(0)
    ctx_df["fail_ratio"] = (
        ctx_df.get("tr_tests_failed", pd.Series(0, index=ctx_df.index))
        / ctx_df.get("tr_tests_run", pd.Series(1, index=ctx_df.index)).clip(lower=1)
    )

    # ── 3. historical instability ────────────────────────────────────────────
    log.info("Computing historical instability (k=5) …")
    hist_df = compute_historical_instability(df, k=5)
    df = df.merge(hist_df, on="build_id", how="left")
    df["hist_fail_rate_mean"] = df["hist_fail_rate_mean"].fillna(0.5)

    # ── 4. TF-IDF on log error section ──────────────────────────────────────
    log.info("Fitting TF-IDF (200 terms) on log text …")
    # trim log to lines after "BUILD FAILURE" or last 100 lines
    def error_section(text: str) -> str:
        text = text or ""
        idx = text.rfind("BUILD FAILURE")
        return text[idx:] if idx != -1 else text[-3000:]

    error_texts = df[LOG_COL].fillna("").apply(error_section)
    tfidf = TfidfVectorizer(
        max_features=200,
        token_pattern=r"[A-Za-z][A-Za-z0-9_\.]{2,}",
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(error_texts).toarray()
    tfidf_cols    = [f"tfidf_{t}" for t in tfidf.get_feature_names_out()]
    tfidf_df      = pd.DataFrame(tfidf_matrix, columns=tfidf_cols)

    # ── 5. Assemble + scale ──────────────────────────────────────────────────
    log.info("Assembling feature matrix …")
    tabular = pd.concat(
        [kw_df.reset_index(drop=True),
         ctx_df.reset_index(drop=True),
         df[["hist_fail_rate_mean"]].reset_index(drop=True),
         tfidf_df.reset_index(drop=True)],
        axis=1,
    ).astype(float)

    scaler = MinMaxScaler()
    X      = scaler.fit_transform(tabular)
    y      = df["label"].values
    groups = df["gh_project_name"].values   # for group-aware splitting

    log.info(f"Feature matrix shape: {X.shape}")
    np.savez(OUT_FEATURES, X=X, y=y, feature_names=tabular.columns.tolist())
    meta = df[["build_id", "gh_project_name", "label"]].copy()
    meta["group"] = groups
    meta.to_csv(OUT_META, index=False)
    log.info(f"Saved → {OUT_FEATURES}  and  {OUT_META}")


if __name__ == "__main__":
    main()
