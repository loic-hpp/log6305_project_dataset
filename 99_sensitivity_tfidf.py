"""
99_sensitivity_tfidf.py
-----------------------
Honest empirical justification for the TF-IDF max_features cap.

The report originally claimed that 200 was chosen by inspecting the
``cumulative explained variance'' of the TF-IDF terms. In practice
02_feature_engineering.py simply passes max_features=200 to
TfidfVectorizer, which selects the top-N terms by overall corpus
frequency. This script:

  1. Measures the *actual* full vocabulary size on the truncated
     500-character log snippets;
  2. Sweeps max_features in {25, 50, 100, 200, 500, full} and reports
     the F1 score (FA class) of a Logistic Regression trained on the
     TF-IDF matrix alone, under leave-one-project-out CV;
  3. Saves the table so the report can cite real numbers instead of
     the variance-curve claim.

Output:
  data/processed/sensitivity_tfidf.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut

LABELLED_CSV = Path("data/processed/labelled_builds.csv")
OUT_CSV      = Path("data/processed/sensitivity_tfidf.csv")

K_SWEEP = [25, 50, 100, 200, 500, None]   # None = no cap (full vocab)
SEED    = 42


def error_section(text: str) -> str:
    text = text or ""
    idx = text.rfind("BUILD FAILURE")
    return text[idx:] if idx != -1 else text[-3000:]


def evaluate(X, y, groups) -> dict:
    logo = LeaveOneGroupOut()
    p_list, r_list, f_list = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        if len(np.unique(y[test_idx])) < 2:
            continue
        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
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
    print("Loading data ...")
    df = (pd.read_csv(LABELLED_CSV)
            .drop_duplicates("build_id")
            .reset_index(drop=True))
    texts  = df["log_snippet"].fillna("").apply(error_section)
    y      = df["label"].values
    groups = df["gh_project_name"].values

    # Discover the real upper bound first.
    full = TfidfVectorizer(
        token_pattern=r"[A-Za-z][A-Za-z0-9_\.]{2,}", sublinear_tf=True
    )
    full.fit(texts)
    full_vocab = len(full.vocabulary_)
    print(f"  Full TF-IDF vocabulary size on truncated logs: {full_vocab}")

    rows = []
    for cap in K_SWEEP:
        eff_cap = full_vocab if cap is None else min(cap, full_vocab)
        cap_label = "full" if cap is None else str(cap)
        print(f"  max_features={cap_label} (effective={eff_cap}) ...")
        v = TfidfVectorizer(
            max_features=eff_cap if eff_cap else None,
            token_pattern=r"[A-Za-z][A-Za-z0-9_\.]{2,}",
            sublinear_tf=True,
        )
        X = v.fit_transform(texts).toarray()
        m = evaluate(X, y, groups)
        m["max_features_requested"] = cap_label
        m["effective_vocab_size"]   = X.shape[1]
        rows.append(m)
        print(f"    F1(FA)={m['f1_FA_mean']:.4f}  "
              f"P={m['precision_FA_mean']:.4f}  R={m['recall_FA_mean']:.4f}  "
              f"folds={m['n_folds_used']}  dim={X.shape[1]}")

    out = pd.DataFrame(rows)[["max_features_requested",
                               "effective_vocab_size",
                               "n_folds_used",
                               "precision_FA_mean",
                               "recall_FA_mean",
                               "f1_FA_mean"]]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
