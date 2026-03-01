# Flaky Test Triage — Full Pipeline

Classifies CI/CD test failure events as **True Positive (TP)** (real regression)
or **False Alarm (FA)** (flaky noise). Answers RQ1, RQ2, RQ3 of the registered report.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Data Preparation

### 1. TravisTorrent
```bash
docker run -it --rm \
  -e TRAVIS_TOKEN="YOUR_TOKEN_HERE" \
  -v $(pwd)/data:/data \
  python:3.11-slim bash
  # Then run file fetch_travis_data.sh inside the container
``` 
→ Place as `data/raw/travistorrent_8_17_2017.csv`

### 2. IDoFT (International Dataset of Flaky Tests)
```bash
wget https://raw.githubusercontent.com/TestingResearchIllinois/idoft/main/pr-data.csv
mv pr-data.csv data/raw/idoft/pr-data.csv   # Maven/Java
```

---

## Running the Pipeline

```bash
# Step 1 — build labelled dataset (fetches Travis logs, ~hours depending on size)
python 01_build_dataset.py

# Step 2 — extract features
python 02_feature_engineering.py

# Step 3 — train all models, evaluate, produce all RQ outputs
python 03_train_evaluate.py
```

---

## Output Files (in `results/`)

| File | Answers |
|------|---------|
| `rq1_shap_top15.csv` + `rq1_shap_plot.pdf` | RQ1 — feature importance |
| `rq2_model_comparison.csv` | RQ2 — model comparison (precision/recall/F1/AUC) |
| `rq3_confusion_matrix.csv` + `.pdf` | RQ3 — best model reliability |
| `rq3_cross_project.csv` + `.pdf` | RQ3 — cross-project generalisation |

---

## Project Structure

```
.
├── 01_build_dataset.py       # Phase 1: label construction (TravisTorrent + IDoFT join)
├── 02_feature_engineering.py # Phase 2: keyword, contextual, TF-IDF, instability features
├── 03_train_evaluate.py      # Phase 3-4: LR, RF, XGB, BiLSTM training + SHAP + evaluation
├── requirements.txt
├── data/
│   ├── raw/                  # place TravisTorrent CSV and IDoFT here
│   ├── logs/                 # auto-populated Travis log cache
│   └── processed/            # labelled_builds.csv, features.npz, metadata.csv
└── results/                  # all output CSVs and PDF figures
```

---

## Label Construction Logic

```
TravisTorrent (failed Java builds)
        │
        ▼  extract failing test class names from Maven Surefire logs
        │
        ▼  join with IDoFT on (project, test_class)
        │
        ├─ ALL failing tests in IDoFT  → label = FA (0)
        │   (skip if src_churn > 0 — confounded)
        │
        └─ ANY failing test NOT in IDoFT → label = TP (1)
           (mixed → excluded)
```

---

## Feature Families (~220 dimensions total)

| Family | Examples | Count |
|--------|----------|-------|
| Log keywords | timeout, NullPointerException, AssertionError, … | ~20 |
| Contextual | gh_is_pr, src_churn, test_churn, tr_duration, … | ~9 |
| Historical instability | rolling failure rate over last 10 builds | 1 |
| TF-IDF (error section) | top 200 log tokens | 200 |

---

## Models

| Model | Type | Imbalance handling |
|-------|------|--------------------|
| Logistic Regression | Linear baseline | class_weight="balanced" |
| Random Forest | Tree ensemble | class_weight="balanced" + SMOTE |
| XGBoost | Gradient boosting | scale_pos_weight=3 + SMOTE |
| BiLSTM | Deep (optional) | weighted BCE loss |
| CodeBERT | Transformer (optional) | weighted BCE loss |
