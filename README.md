# Flaky Test Triage — Full Pipeline

Classifies CI/CD test failure events as **True Positive (TP)** (real regression)
or **False Alarm (FA)** (flaky noise). Implements RQ1, RQ2, RQ3 of the final
report (Submission D).

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data Preparation

### 1. TravisTorrent (custom rebuild)

The canonical TravisTorrent dataset is no longer available for download —
the official page (https://travistorrent.testroots.org/) stopped serving
its links in early 2022. We therefore rebuild a TravisTorrent-style CSV
ourselves by scraping the Travis CI v3 REST API for the eight target Java
projects, following the four-stage methodology documented in the
[`TestRoots/travistorrent-tools`](https://github.com/TestRoots/travistorrent-tools)
README.

```bash
docker run -it --rm \
  -e TRAVIS_TOKEN="YOUR_TOKEN_HERE" \
  -v $(pwd)/data:/data \
  python:3.11-slim bash
  # Then run fetch_travis_data.sh inside the container
```

→ produces `data/raw/travistorrent_8_17_2017.csv` (25-column schema,
~15 830 builds across 8 Java projects).

> **Important caveat.** `fetch_travis_data.sh` deliberately retains only
> the **last 500 characters** of each build log as the `log_snippet`
> field, to keep the resulting CSV compact. This is a design choice, not
> an API limitation, and it propagates downstream: it prevents per-test
> name extraction, collapses the TF-IDF vocabulary to 13 tokens, and
> rules out CodeBERT fine-tuning. Removing this truncation is logged as
> the highest-priority future-work item in the report.

### 2. IDoFT (International Dataset of Flaky Tests)

```bash
wget https://raw.githubusercontent.com/TestingResearchIllinois/idoft/main/pr-data.csv
mv pr-data.csv data/raw/pr-data.csv   # Maven/Java
```

---

## Running the Pipeline

```bash
# Step 1 — build labelled dataset (build-level join on project slug)
python 01_build_dataset.py

# Step 2 — extract features (k=5 rolling window by default)
python 02_feature_engineering.py

# Step 3 — train all models, evaluate, produce all RQ outputs
python 03_train_evaluate.py
```

### Sensitivity analyses (optional, ≤ 1 minute each)

```bash
# Sweep over k ∈ {5, 10, 20} for the rolling failure rate
python 99_sensitivity_k.py
# → data/processed/sensitivity_k.csv

# Sweep over TF-IDF max_features ∈ {25, 50, 100, 200, 500, full}
python 99_sensitivity_tfidf.py
# → data/processed/sensitivity_tfidf.csv
```

---

## Output Files

| File | Answers |
|------|---------|
| `results/rq1_shap_top15.csv` + `rq1_shap_plot.pdf` | RQ1 — feature importance |
| `results/rq2_model_comparison.csv` | RQ2 — model comparison (precision/recall/F1/AUC) |
| `results/rq3_confusion_matrix.csv` + `.pdf` | RQ3 — best model reliability |
| `results/rq3_cross_project.csv` + `.pdf` | RQ3 — cross-project generalisation (LOPO) |
| `results_k10/` | Frozen baseline run with the previous k=10 default (kept for the discrepancies table in the report) |
| `data/processed/sensitivity_k.csv` | Sensitivity sweep on `hist_fail_rate` window size |
| `data/processed/sensitivity_tfidf.csv` | Sensitivity sweep on `TfidfVectorizer.max_features` |

---

## Project Structure

```
.
├── 01_build_dataset.py        # Phase 1: build-level label construction
├── 02_feature_engineering.py  # Phase 2: keyword, contextual, TF-IDF, instability features
├── 03_train_evaluate.py       # Phases 3-4: LR/RF/XGBoost/BiLSTM training + SHAP + LOPO
├── 99_sensitivity_k.py        # Sensitivity sweep on rolling-window k
├── 99_sensitivity_tfidf.py    # Sensitivity sweep on TF-IDF cap
├── fetch_travis_data.sh       # Travis CI v3 API scraper (rebuilds the TravisTorrent CSV)
├── requirements.txt
├── data/
│   ├── raw/                   # travistorrent_8_17_2017.csv, pr-data.csv
│   ├── logs/                  # cached Travis logs (optional)
│   └── processed/             # labelled_builds.csv, features.npz, metadata.csv, sensitivity_*.csv
├── results/                   # current results (k=5 default)
└── results_k10/               # frozen k=10 baseline
```

---

## Label Construction Logic

The labelling join is performed at the **build level** on the project
slug — not at the (project, commit, test_name) granularity originally
planned in Submission B. Per-test names are not recoverable from the
truncated 500-character log snippets.

```
                TravisTorrent CSV (failed/errored builds)
                                 │
                                 ▼
                    Compute hist_fail_rate (k=5 default)
                                 │
                                 ▼
                  ┌────────────────────────────┐
                  │ project ∈ flaky_projects ? │
                  └────────────────────────────┘
                       │ No                       │ Yes
                       ▼                          ▼
                    label = TP            ┌─────────────────────┐
              (no flaky test in           │ hist_fail_rate>0.5 ?│
               IDoFT for project)         └─────────────────────┘
                                              │ Yes        │ No
                                              ▼            ▼
                                       label = FA     label = TP
                                       (flaky)        (isolated
                                                       failure)
                                 │
                                 ▼
              Step 2 deduplicates by build_id → 1 376 labelled builds
                       (1 019 TP / 357 FA)
```

`flaky_projects` is the set of project slugs that appear at least once in
IDoFT (`pr-data.csv`). Mixed-build and source-modified-test exclusions
that were planned in Submission B are **not** performed in this pipeline:
the `git_diff_src_churn` field is propagated as a feature instead.

---

## Feature Families (45 dimensions total)

| Family | Examples | Count |
|--------|----------|-------|
| Log keywords | `kw_fa_timeout`, `kw_tp_AssertionError`, … | 20 |
| Contextual | `gh_is_pr`, `git_diff_src_churn`, `tr_duration`, `gh_team_size`, … | 9 |
| Historical instability | `hist_fail_rate_mean` (rolling failure rate, k=5) | 1 |
| TF-IDF (error section) | `tfidf_*` tokens (`max_features=200`) | 15 |

> The TF-IDF cap is set to 200 for forward compatibility but is **not
> binding** in practice: the truncated logs only yield 13 distinct
> alphabetic tokens, so the effective dimensionality is at most 13. The
> remaining 2 spots reflect cumulative-document-frequency rounding when
> the cap is requested. See `99_sensitivity_tfidf.py` for the proof
> (F1(FA) is identical for max_features ∈ {25, …, full}).

> The k=5 default for `hist_fail_rate_mean` was selected by an explicit
> sensitivity sweep over k ∈ {5, 10, 20} — F1(FA) ranges from 0.367
> (k=20) to 0.599 (k=5). See `99_sensitivity_k.py`.

---

## Models

| Model | Type | Imbalance handling |
|-------|------|--------------------|
| Logistic Regression | Linear baseline | class_weight="balanced" |
| Random Forest | Tree ensemble | class_weight="balanced" + SMOTE on training split |
| XGBoost | Gradient boosting | scale_pos_weight=3 + SMOTE on training split |
| BiLSTM | Deep model | weighted BCE loss (`pos_weight=3`), no SMOTE |

> CodeBERT was originally planned in Submission B but was dropped: the
> 500-character log snippets average <200 tokens, too short for
> meaningful transformer fine-tuning.

SMOTE is applied **only on the training partition**, after the
project-disjoint 80/10/10 split, and re-sampled independently for each
of the 10 evaluation seeds. See `03_train_evaluate.py:284-294` and the
helper `apply_smote` (lines 94-96) for the exact ordering.

---

## Headline Results (k=5 default)

| RQ | Metric | Value |
|----|--------|-------|
| RQ1 | Top SHAP feature | `hist_fail_rate_mean` (mean \|SHAP\| = 0.291) |
| RQ1 | Zero-SHAP features | All 20 keyword + 15 TF-IDF features |
| RQ2 | Best F1 | **BiLSTM** = 0.930 ± 0.104 |
| RQ2 | Best traditional F1 | XGBoost = 0.911 ± 0.118 |
| RQ3 | Mean LOPO F1 (XGBoost) | 0.891 ± 0.117 |
| RQ3 | Worst LOPO F1 | `apache/commons-math` = 0.702 |
