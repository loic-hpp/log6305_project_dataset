"""
99_sensitivity_k.py
-------------------
Sensitivity analysis on the rolling-window size k used by the historical
instability feature `hist_fail_rate_mean`.

For each k in {5, 10, 20}:
  1. Recompute hist_fail_rate over the deduplicated build series.
  2. Substitute it into the feature matrix produced by
     02_feature_engineering.py.
  3. Train a Random Forest under leave-one-project-out cross-validation
     (the same protocol described in Section IV.G of the report) and
     report mean F1, precision and recall on held-out projects.

Output:
  data/processed/sensitivity_k.csv   — one row per k, holdout-averaged
                                       precision / recall / F1.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut

LABELLED_CSV = Path("data/processed/labelled_builds.csv")
FEATURES_NPZ = Path("data/processed/features.npz")
METADATA_CSV = Path("data/processed/metadata.csv")
OUT_CSV      = Path("data/processed/sensitivity_k.csv")

K_VALUES = [5, 10, 20]
SEED     = 42


def compute_historical_instability(df: pd.DataFrame, k: int) -> pd.Series:
    """Mirrors 02_feature_engineering.compute_historical_instability."""
    df = df.sort_values("build_id").copy()
    history: defaultdict[str, list] = defaultdict(list)
    fail_rate: dict[int, float] = {}
    for _, r in df.iterrows():
        key = str(r["gh_project_name"])
        hist = history[key][-k:]
        fail_rate[r["build_id"]] = float(np.mean(hist)) if hist else 0.5
        history[key].append(int(r["label"]))
    return pd.Series(fail_rate, name=f"hist_fail_rate_k{k}")


def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    logo = LeaveOneGroupOut()
    p_list, r_list, f_list = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            # Some projects are 100% TP; skip degenerate folds for the macro avg.
            continue
        clf = RandomForestClassifier(
            n_estimators=200, random_state=SEED, n_jobs=-1
        )
        clf.fit(X[train_idx], y[train_idx])
        y_hat = clf.predict(X[test_idx])
        p, r, f, _ = precision_recall_fscore_support(
            y[test_idx], y_hat, average="binary", pos_label=0, zero_division=0
        )
        p_list.append(p); r_list.append(r); f_list.append(f)
    return {
        "n_folds_used": len(f_list),
        "precision_FA_mean": float(np.mean(p_list)) if p_list else float("nan"),
        "recall_FA_mean":    float(np.mean(r_list)) if r_list else float("nan"),
        "f1_FA_mean":        float(np.mean(f_list)) if f_list else float("nan"),
    }


def main() -> None:
    print("Loading data...")
    bld  = pd.read_csv(LABELLED_CSV).drop_duplicates("build_id").reset_index(drop=True)
    meta = pd.read_csv(METADATA_CSV)
    npz  = np.load(FEATURES_NPZ, allow_pickle=True)
    X    = npz["X"].copy()
    y    = npz["y"]
    feat_names = list(npz["feature_names"])

    if "hist_fail_rate_mean" not in feat_names:
        sys.exit("hist_fail_rate_mean column not found in features.npz")
    hist_idx = feat_names.index("hist_fail_rate_mean")

    # Align bld to the metadata order used to build features.npz.
    bld = bld.set_index("build_id").loc[meta["build_id"]].reset_index()
    assert (bld["label"].values == y).all(), "label/order mismatch"

    groups = meta["gh_project_name"].values

    rows = []
    for k in K_VALUES:
        print(f"  Recomputing hist_fail_rate with k={k} ...")
        s = compute_historical_instability(bld, k)
        new_col = bld["build_id"].map(s).fillna(0.5).values

        # MinMax-scale the new column to [0, 1] like 02_feature_engineering.py
        # does for the whole matrix; hist_fail_rate is already in [0, 1] so
        # this is a no-op, but we clip for safety.
        new_col = np.clip(new_col, 0.0, 1.0)

        X_k = X.copy()
        X_k[:, hist_idx] = new_col

        print(f"    Training RandomForest under leave-one-project-out CV ...")
        m = evaluate(X_k, y, groups)
        m["k"] = k
        rows.append(m)
        print(f"      F1(FA)={m['f1_FA_mean']:.4f}  "
              f"P={m['precision_FA_mean']:.4f}  R={m['recall_FA_mean']:.4f}  "
              f"folds={m['n_folds_used']}")

    out = pd.DataFrame(rows)[["k", "n_folds_used",
                               "precision_FA_mean",
                               "recall_FA_mean",
                               "f1_FA_mean"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
