# Career Path Prediction — Stacking Ensemble Evaluation

Evaluation and reporting script for a trained stacking-ensemble model that predicts a student's most suitable career path / job role from academic performance and skill data. It loads a previously trained pipeline, scores it against a held-out test set, and produces a full set of evaluation charts plus a text summary report.

**Model architecture:** Random Forest + XGBoost + LightGBM base learners → Logistic Regression meta-learner (stacking ensemble).

## What it does

Running `main.py`:

1. Loads the trained pipeline, label encoder, held-out test set, and the source dataset.
2. Predicts on the unseen test set and computes accuracy plus a per-class classification report.
3. Runs a quick 3-fold cross-validation check on a sample of the full dataset.
4. Renders 11 evaluation assets (10 charts + 1 text report) to `results/stacking/`.

## Outputs

| File | What it shows |
|---|---|
| `Final_Confusion_Matrix.png` | Predicted vs. actual career path, as a heatmap |
| `Final_ROC_Curve.png` | Multi-class ROC curves with per-class AUC |
| `Final_Precision_Recall_Curve.png` | Precision-recall curve per class |
| `Final_CV_Boxplot.png` | Spread of 3-fold cross-validation scores |
| `Final_Learning_Curve.png` | Training vs. validation score across sample sizes |
| `Final_Performance_Summary.png` | F1-score per class, as a bar chart |
| `Final_Model_Competency_Radar.png` | Per-class F1-score, as a radar chart |
| `Final_Skill_Gap_Radar.png` | Model output vs. reference benchmark, as a radar chart |
| `Final_Confidence_Density.png` | Prediction-confidence distribution, correct vs. incorrect |
| `Final_Accuracy_Benchmark.png` | Test accuracy vs. mean CV accuracy |
| `Stacking_Master_Report.txt` | Text summary: accuracy, CV score, architecture, full classification report |

All files are saved to `results/stacking/`, which is created automatically if it doesn't exist.

## Expected project structure

Paths are resolved relative to the script's own location, one level below the project root:

```
project-root/
├── datasets/
│   └── student_dataset_final.xlsx
├── models/
│   └── stacking/
│       ├── final_career_pipeline_stacking.pkl
│       ├── career_label_mapper_stacking.pkl
│       └── stacking_test_set.pkl
├── results/
│   └── stacking/              # created automatically
└── <scripts folder>/
    ├── train_model_stacking.py
    └── main.py                # this file
```

If your layout differs, adjust `CURRENT_DIR` / `BASE_DIR` at the top of the script.

## Prerequisites

- **`train_model_stacking.py` must be run first.** This script exits immediately if the trained pipeline, label encoder, or test set aren't found in `models/stacking/`.
- The dataset (`datasets/student_dataset_final.xlsx`) should include:
  - a target column: `Target_Job_Role` or `Career_Path`
  - a skills text column: `Known_Skills` or `Skills`
  - numeric columns: `CGPA`, `Logic_Score`, `Technical_Skill_Count`, `Soft_Skill_Count`, `Has_Technical_Skills`
  - a `Degree` column

## Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm openpyxl
```

Requires Python 3.8+. `xgboost` and `lightgbm` aren't imported directly in this file, but the saved pipeline contains fitted models from both, so they're needed to unpickle it. `openpyxl` lets pandas read the `.xlsx` dataset.

## Usage

```bash
python main.py
```

No arguments needed — paths are resolved automatically. Expect console output like:

```
🚀 Initializing Supreme Evaluation Suite for Stacking Ensemble...
✅ Master Assets and Secure Test Set Loaded Successfully.
🧠 Meta-Model is processing unseen data...
📊 Crafting 11 Elite Academic Assets (Unique Stacking Design)...
🖼️ Running Cross-Validation Analysis (This takes a moment)...
🖼️ Generating Learning Convergence...
🎉 MASTERPIECE COMPLETED! All 11 Elite Assets saved in 'results/stacking'.
```

## Notes

- The CV Boxplot uses only the first 1,500 rows of the full dataset, not the entire dataset, to keep runtime reasonable.
- The Learning Curve is computed over the held-out test set (`X_test`, `y_test`) rather than the training set.
- In the Skill Gap Radar, only the "Technical" dimension is derived from actual model accuracy — the other four dimensions and the entire "Industry Standard" series are fixed placeholder values in the script.
- All warnings are suppressed globally via `warnings.filterwarnings('ignore')`.

## License

Add a license of your choice (e.g., MIT) here.
