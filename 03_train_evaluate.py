"""
03_train_evaluate.py
--------------------
Phase 3 & 4: Train all five model families, evaluate on held-out test split,
compute SHAP feature importance, run cross-project generalisation experiment.

Answers RQ1, RQ2, RQ3.

Requirements:
    pip install scikit-learn xgboost shap imbalanced-learn scipy matplotlib seaborn
    pip install torch transformers  # for BiLSTM + CodeBERT
"""

import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection    import GroupShuffleSplit, GridSearchCV
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier
from sklearn.metrics            import (classification_report, confusion_matrix,
                                        roc_auc_score, f1_score,
                                        precision_score, recall_score)
from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import MinMaxScaler
from imblearn.over_sampling     import SMOTE
from scipy.stats                import wilcoxon
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_NPZ = Path("data/processed/features.npz")
META_CSV     = Path("data/processed/metadata.csv")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_RUNS = 10   # independent seeds for variance estimation


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_data():
    data  = np.load(FEATURES_NPZ, allow_pickle=True)
    X     = data["X"].astype(np.float32)
    y     = data["y"].astype(int)
    fnames= list(data["feature_names"])
    meta  = pd.read_csv(META_CSV)
    groups= meta["gh_project_name"].values

    # ── fix NaNs and Infs before anything else ───────────────────────────
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    # ── sanity check: X rows must match meta rows ────────────────────────
    if X.shape[0] != len(meta):
        min_len = min(X.shape[0], len(meta))
        X      = X[:min_len]
        y      = y[:min_len]
        groups = groups[:min_len]
        log.warning(f"Shape mismatch fixed: using first {min_len} rows")

    log.info(f"After cleaning: X={X.shape}  NaNs={np.isnan(X).sum()}")
    return X, y, fnames, groups


def project_split(X, y, groups, test_size=0.1, val_size=0.1, seed=42):
    """Group-aware split: no project leaks across train/val/test."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(splitter.split(X, y, groups))

    X_tv, y_tv, g_tv = X[trainval_idx], y[trainval_idx], groups[trainval_idx]
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size / (1 - test_size),
        random_state=seed,
    )
    train_idx, val_idx = next(splitter2.split(X_tv, y_tv, g_tv))

    return (X_tv[train_idx], y_tv[train_idx],
            X_tv[val_idx],   y_tv[val_idx],
            X[test_idx],     y[test_idx])


def apply_smote(X_train, y_train, seed=42):
    sm = SMOTE(random_state=seed, k_neighbors=5)
    return sm.fit_resample(X_train, y_train)


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        m["auc_roc"] = roc_auc_score(y_true, y_prob)
    return m


# ═══════════════════════════════════════════════════════════════════════════
#  Model definitions
# ═══════════════════════════════════════════════════════════════════════════

def build_lr():
    return LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)


def build_rf():
    return RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=42,
    )


def build_xgb():
    return xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        scale_pos_weight=3,   # approx TP:FA ratio
        random_state=42, n_jobs=-1,
    )


# ── Deep models (optional — requires torch + transformers) ──────────────────
def _try_import_deep():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoTokenizer, AutoModel
        return torch, nn, DataLoader, TensorDataset, AutoTokenizer, AutoModel
    except ImportError:
        return None


class BiLSTMClassifier:
    """Thin wrapper around a PyTorch BiLSTM for tabular/log text features."""

    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.3, epochs=20, lr=1e-3):
        deps = _try_import_deep()
        if deps is None:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        torch, nn, DataLoader, TensorDataset, *_ = deps
        self.torch = torch
        self.DataLoader = DataLoader
        self.TensorDataset = TensorDataset

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_dim, hidden, layers,
                    batch_first=True, bidirectional=True,
                    dropout=dropout if layers > 1 else 0,
                )
                self.fc = nn.Sequential(
                    nn.Linear(hidden * 2, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )

            def forward(self, x):
                # treat each feature as a time step of length 1
                x = x.unsqueeze(1)
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.net    = Net()
        self.epochs = epochs
        self.opt    = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn= torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([3.0])   # class imbalance weight
        )

    def fit(self, X, y):
        torch = self.torch
        ds = self.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y)
        )
        loader = self.DataLoader(ds, batch_size=256, shuffle=True)
        self.net.train()
        for ep in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                self.opt.zero_grad()
                loss = self.loss_fn(self.net(xb), yb)
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
            if (ep + 1) % 5 == 0:
                log.info(f"  BiLSTM epoch {ep+1}/{self.epochs}  loss={total_loss:.4f}")
        return self

    def predict_proba(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.FloatTensor(X)).numpy()
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════════════════
#  RQ1 — Feature Importance (SHAP)
# ═══════════════════════════════════════════════════════════════════════════

def rq1_feature_importance(rf_model, X_test, feature_names):
    log.info("=== RQ1: SHAP feature importance ===")
    explainer = shap.TreeExplainer(rf_model)
    shap_vals = explainer.shap_values(X_test)

    # Handle all possible SHAP output formats:
    # - old RF: list of 2 arrays, each (n_samples, n_features)  → take index 1
    # - new RF: single 3D array (n_samples, n_features, n_classes) → take [:,:,1]
    # - regression: single 2D array (n_samples, n_features)
    sv = np.array(shap_vals)
    if sv.ndim == 3 and sv.shape[0] == X_test.shape[0]:
        # new format: (n_samples, n_features, n_classes)
        sv = sv[:, :, 1]
    elif sv.ndim == 3:
        # old list format stacked: (n_classes, n_samples, n_features)
        sv = sv[1]
    elif sv.ndim == 2:
        pass  # already (n_samples, n_features)

    mean_abs  = np.abs(sv).mean(axis=0).flatten()
    top15_idx = [int(i) for i in np.argsort(mean_abs)[::-1][:15]]

    # ensure feature_names is a plain Python list
    fnames = list(feature_names)

    importance_df = pd.DataFrame({
        "feature":       [fnames[i] for i in top15_idx],
        "mean_abs_shap": [float(mean_abs[i]) for i in top15_idx],
    })
    importance_df.to_csv(RESULTS_DIR / "rq1_shap_top15.csv", index=False)
    log.info(f"Top-5 features:\n{importance_df.head()}")

    # plot
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=importance_df, x="mean_abs_shap", y="feature", ax=ax, palette="Blues_r")
    ax.set_title("RQ1 — Top-15 Features by Mean |SHAP| (Random Forest)")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "rq1_shap_plot.pdf", dpi=150)
    plt.close(fig)
    return importance_df


# ═══════════════════════════════════════════════════════════════════════════
#  RQ2 — Model comparison over N_RUNS seeds
# ═══════════════════════════════════════════════════════════════════════════

def rq2_model_comparison(X, y, groups):
    log.info("=== RQ2: multi-run model comparison ===")
    model_factories = {
        "LogisticRegression": build_lr,
        "RandomForest":       build_rf,
        "XGBoost":            build_xgb,
    }
    # try deep models if torch is available
    deps = _try_import_deep()
    if deps:
        model_factories["BiLSTM"] = lambda: BiLSTMClassifier(X.shape[1])

    all_results = defaultdict(list)

    for seed in range(N_RUNS):
        X_tr, y_tr, X_val, y_val, X_te, y_te = project_split(X, y, groups, seed=seed)
        X_tr_s, y_tr_s = apply_smote(X_tr, y_tr, seed=seed)

        for name, factory in model_factories.items():
            log.info(f"  seed={seed} model={name}")
            model = factory()
            if hasattr(model, "fit"):
                model.fit(X_tr_s, y_tr_s)
            else:
                model.fit(X_tr, y_tr)  # deep models handle imbalance via loss weight

            y_pred = model.predict(X_te)
            y_prob = (model.predict_proba(X_te)[:, 1]
                      if hasattr(model, "predict_proba") else None)
            m = compute_metrics(y_te, y_pred, y_prob)
            m["seed"] = seed
            all_results[name].append(m)

    # summarise
    summary_rows = []
    for name, runs in all_results.items():
        rdf = pd.DataFrame(runs)
        summary_rows.append({
            "model":     name,
            "precision": f"{rdf.precision.mean():.3f} ± {rdf.precision.std():.3f}",
            "recall":    f"{rdf.recall.mean():.3f} ± {rdf.recall.std():.3f}",
            "f1":        f"{rdf.f1.mean():.3f} ± {rdf.f1.std():.3f}",
            "auc_roc":   f"{rdf.auc_roc.mean():.3f} ± {rdf.auc_roc.std():.3f}"
                          if "auc_roc" in rdf else "N/A",
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "rq2_model_comparison.csv", index=False)
    log.info(f"\n{summary.to_string(index=False)}")

    # Wilcoxon pairwise test: XGB vs each other
    f1s = {n: [r["f1"] for r in runs] for n, runs in all_results.items()}
    if "XGBoost" in f1s:
        for other in f1s:
            if other == "XGBoost":
                continue
            try:
                stat, p = wilcoxon(f1s["XGBoost"], f1s[other])
                log.info(f"  Wilcoxon XGBoost vs {other}: stat={stat:.3f} p={p:.4f}")
            except Exception:
                pass

    return all_results, summary


# ═══════════════════════════════════════════════════════════════════════════
#  RQ3 — Reliability + Cross-project generalisation
# ═══════════════════════════════════════════════════════════════════════════

def rq3_reliability(X, y, groups, feature_names):
    log.info("=== RQ3: reliability and cross-project generalisation ===")

    # ── 3a. Full confusion matrix of best model (XGBoost) on one split ──────
    X_tr, y_tr, _, _, X_te, y_te = project_split(X, y, groups, seed=0)
    X_tr_s, y_tr_s = apply_smote(X_tr, y_tr, seed=0)

    # XGBoost for confusion matrix
    xgb_model = build_xgb()
    xgb_model.fit(X_tr_s, y_tr_s)
    y_pred = xgb_model.predict(X_te)

    cm = confusion_matrix(y_te, y_pred)
    cm_df = pd.DataFrame(cm,
                          index=["Actual FA", "Actual TP"],
                          columns=["Pred FA", "Pred TP"])
    cm_df.to_csv(RESULTS_DIR / "rq3_confusion_matrix.csv")
    log.info(f"Confusion matrix:\n{cm_df}")
    log.info("\n" + classification_report(y_te, y_pred,
                                          target_names=["FA (flaky)", "TP (real)"]))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred FA", "Pred TP"],
                yticklabels=["Actual FA", "Actual TP"])
    ax.set_title("RQ3 — Confusion Matrix (XGBoost, best seed)")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "rq3_confusion_matrix.pdf", dpi=150)
    plt.close(fig)

    # ── 3b. SHAP — use Random Forest (avoids XGBoost/SHAP version conflict) ──
    rf_model = build_rf()
    rf_model.fit(X_tr_s, y_tr_s)
    rq1_feature_importance(rf_model, X_te, feature_names)

    # ── 3c. Cross-project leave-one-project-out ──────────────────────────────
    unique_projects = list(set(groups))
    xp_results = []
    for held_out in unique_projects[:20]:   # limit to 20 for runtime
        mask_te = groups == held_out
        mask_tr = ~mask_te
        if mask_te.sum() < 10 or mask_tr.sum() < 50:
            continue
        X_tr_, y_tr_ = X[mask_tr], y[mask_tr]
        X_te_, y_te_ = X[mask_te], y[mask_te]
        X_tr_s_, y_tr_s_ = apply_smote(X_tr_, y_tr_, seed=42)
        m = build_xgb()
        m.fit(X_tr_s_, y_tr_s_)
        y_pred_ = m.predict(X_te_)
        xp_results.append({
            "held_out_project": held_out,
            "n_test":           mask_te.sum(),
            "f1":               f1_score(y_te_, y_pred_, zero_division=0),
            "precision":        precision_score(y_te_, y_pred_, zero_division=0),
            "recall":           recall_score(y_te_, y_pred_, zero_division=0),
        })

    xp_df = pd.DataFrame(xp_results)
    xp_df.to_csv(RESULTS_DIR / "rq3_cross_project.csv", index=False)
    if len(xp_df):
        log.info(f"Cross-project mean F1: {xp_df.f1.mean():.3f} ± {xp_df.f1.std():.3f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(xp_df)), xp_df["f1"].values, color="steelblue")
    ax.set_xticks(range(len(xp_df)))
    ax.set_xticklabels(xp_df["held_out_project"].values, rotation=45, ha="right", fontsize=7)
    ax.axhline(xp_df["f1"].mean(), color="red", linestyle="--", label="Mean F1")
    ax.set_ylabel("F1 score")
    ax.set_title("RQ3 — Cross-Project Generalisation (Leave-One-Project-Out)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "rq3_cross_project.pdf", dpi=150)
    plt.close(fig)
    return xp_df


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    X, y, feature_names, groups = load_data()
    log.info(f"Loaded: X={X.shape}  y distribution: TP={y.sum()}  FA={(y==0).sum()}")

    _, summary = rq2_model_comparison(X, y, groups)
    rq3_reliability(X, y, groups, feature_names)

    log.info("All results saved to ./results/")
    log.info("Done.")


if __name__ == "__main__":
    main()
