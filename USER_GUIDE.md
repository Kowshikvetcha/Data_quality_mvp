# AI Data Cleaning Pro — User Guide

This guide walks through the full workflow: uploading data, cleaning it, engineering features, training ML models, evaluating results, and generating predictions. The app runs in your browser via Streamlit.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Sidebar & Data Upload](#2-sidebar--data-upload)
3. [Data Inspector](#3-data-inspector)
4. [Chat & Transform (AI Cleaning)](#4-chat--transform-ai-cleaning)
5. [Manual Transform](#5-manual-transform)
6. [Join Datasets](#6-join-datasets)
7. [AI Suggestions](#7-ai-suggestions)
8. [Export Cleaned Data](#8-export-cleaned-data)
9. [History & Code](#9-history--code)
10. [ML Pipeline — Feature Engineering](#10-ml-pipeline--feature-engineering)
11. [ML Pipeline — Model Training (AutoML)](#11-ml-pipeline--model-training-automl)
12. [ML Pipeline — Evaluation](#12-ml-pipeline--evaluation)
13. [ML Pipeline — Predictions](#13-ml-pipeline--predictions)
14. [Sample Datasets](#14-sample-datasets)
15. [Tips & Best Practices](#15-tips--best-practices)

---

## 1. Getting Started

```bash
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt

# Set your OpenAI API key
cp .env.example .env
# Edit .env → OPENAI_API_KEY=sk-proj-...

# Launch the app
streamlit run app_enhanced.py
```

The app opens at `http://localhost:8501`.

---

## 2. Sidebar & Data Upload

The left sidebar is your control panel throughout the session.

**Uploading data:**
- Click **Upload File** and select a CSV, Excel (.xlsx), or Parquet file (up to 200 MB).
- The app automatically infers column types (numeric, string, datetime) using an 80% threshold heuristic.
- A quality report and health score are computed immediately.

**Sidebar info panel (after upload):**
- Original vs. cleaned dataset shape (rows/columns).
- Number of AI suggestions available.
- Undo stack depth.
- OpenAI API status indicator.

**Column type override:** Expand the type override section in the sidebar to manually correct any misdetected column types. This affects which operations are available for each column.

**Reset App:** Clears all data, history, and ML pipeline state — starts a fresh session.

---

## 3. Data Inspector

A side-by-side view of your original and cleaned data.

| Panel | What it shows |
|-------|---------------|
| Left | Original uploaded data (first 50 rows), health score, issues found |
| Right | Current cleaned data, health score with delta vs. original |

**Advanced Visualizations** (tabs below the comparison):

- **Column Profile** — Select any column to see its distribution chart and `describe()` statistics.
- **Correlation Matrix** — Heatmap of pairwise numeric correlations. Use this to spot multicollinearity before ML.
- **Box Plots** — Outlier visualization per numeric column. Useful for deciding outlier treatment strategy.
- **Scatter Plot** — Pick two numeric columns to check relationships visually.

---

## 4. Chat & Transform (AI Cleaning)

Describe what you want in plain English. The AI translates your request into one or more data operations.

**How it works:**

1. Type a request in the chat input, e.g.:
   - *"Fill missing values in the price column with the median"*
   - *"Trim whitespace and convert all string columns to lowercase"*
   - *"Remove outliers from revenue using the IQR method"*
   - *"Split the full_name column by space into first_name and last_name"*
2. The AI proposes a set of transformations with descriptions.
3. Review the proposed changes — a diff preview shows how many rows/columns/cells will change.
4. Click **Apply Transformations** to execute, or **Cancel** to reject.
5. Every applied batch is added to the undo stack.

**Supported operations via chat (35+ tools):**

| Category | Operations |
|----------|-----------|
| Null handling | `fill_nulls` (mean, median, mode, zero, ffill, bfill, custom), `drop_rows_with_nulls` |
| String ops | `trim_spaces`, `standardize_case`, `replace_text`, `replace_text_regex`, `remove_special_chars`, `pad_string`, `slice_string`, `add_prefix_suffix` |
| Numeric ops | `round_numeric`, `clip_numeric`, `scale_numeric`, `apply_math` (sqrt, log, abs, square, negate), `bin_numeric`, `replace_negative_values` |
| Outlier handling | `remove_outliers` (IQR/Z-score; actions: null, drop, clip, mean, median) |
| Date ops | `convert_to_datetime`, `extract_date_part` (year, month, day, weekday, quarter), `offset_date`, `date_difference` |
| Type conversion | `convert_column_type` (numeric, string, datetime, boolean, categorical) |
| Dataset-level | `deduplicate_rows`, `drop_column`, `rename_column`, `split_column`, `merge_columns` |
| Batch ops | `fill_nulls_batch`, `trim_spaces_batch`, `standardize_case_batch`, `drop_columns_batch` |
| Advanced | `create_calculated_column` (pandas eval formulas, e.g. `Price * Quantity`) |

**Undo:** Every batch of transformations can be undone. The undo stack holds up to 20 states by default (configurable via `MAX_UNDO_HISTORY`).

---

## 5. Manual Transform

Point-and-click interface for when you know exactly what operation you want. Organized into collapsible sections:

### Dataset Operations
- **Deduplicate Rows** — Select columns to check for duplicates, choose to keep first or last occurrence.
- **Rename Column** — Pick a column, enter the new name.
- **Drop Column(s)** — Multi-select columns to remove.

### Calculated Columns
- Enter a new column name and a pandas eval formula (e.g., `Revenue - Cost`).
- Formulas are validated to prevent code injection — `import`, `exec`, `eval`, etc. are blocked.

### Text Operations
Select a string column, then choose: Trim Whitespace, Convert to Lowercase/Uppercase/Title Case, Remove Special Characters, Replace Text, or Regex Replace.

### Missing Values
Select a column and a fill method: Drop Rows, Mean, Median, Mode, Zero, Forward Fill, Backward Fill, or Custom Value.

### Numeric Operations
Select a numeric column, then choose:
- **Round** — Decimal places (0–10).
- **Clip** — Lower and upper bounds.
- **Scale** — MinMax or Z-score normalization.
- **Bin** — Number of bins (2–100), optional new column name.
- **Remove Outliers** — IQR or Z-score method; action: null, drop, clip, mean, or median.
- **Replace Negatives** — Replacement value (default 0).

### Date & Type Operations
- **Convert Column Type** — Target: numeric, string, datetime, boolean, categorical.
- **Convert to Datetime** — Handles ISO, US (MM/DD/YYYY), EU (DD-MM-YYYY) formats.
- **Extract Date Part** — Year, month, day, weekday, or quarter into a new column.
- **Offset Date** — Add/subtract days, weeks, months, or years.
- **Date Difference** — Days/weeks/hours/years from a reference date (use `today` or a specific `YYYY-MM-DD`).

---

## 6. Join Datasets

Merge your main dataset with additional files.

1. Upload extra datasets via the sidebar's additional file uploader.
2. Navigate to **Join Datasets**.
3. Select the right dataset from the dropdown.
4. Choose a join type:
   - **inner** — only matching rows from both sides.
   - **left** — all rows from main, matching from right.
   - **right** — all rows from right, matching from main.
   - **outer** — all rows from both sides.
5. Select key column(s) for each side.
6. Click **Preview Join** to inspect the first 10 result rows.
7. Click **Apply & Set as Main Dataset** to commit. This replaces both `original_df` and `cleaned_df`.

---

## 7. AI Suggestions

The app automatically analyzes your data quality report and generates prioritized cleaning suggestions.

- **High priority** (red) — Issues affecting >20% of data (e.g., heavy nulls, many duplicates).
- **Medium priority** (yellow) — Issues affecting 5–20% (e.g., outliers, mixed casing).
- **Low priority** (green) — Minor issues <5% (e.g., leading/trailing spaces).

Each suggestion shows the tool it will use, a description, the impact score (cells/rows affected), and a category (missing, duplicates, outliers, formatting, type).

- **Apply All Suggestions** — Runs all suggestions as a batch with rollback on error.
- **Individual Apply** — Apply one suggestion at a time.
- **Refresh Suggestions** — Re-analyze after cleaning to see what's left.

---

## 8. Export Cleaned Data

Available once at least one transformation has been applied.

Three download formats in a single row:
- **CSV** — Standard comma-separated values.
- **Excel** (.xlsx) — Single sheet, limited to 1,048,576 rows.
- **JSON** — Records orientation.

Below the download buttons, a **Transformation Summary** lists every operation that was applied.

---

## 9. History & Code

**Audit trail** of every transformation applied during the session.

**Auto-generated Python script** — A reproducible cleaning pipeline that replays your transformations:

```python
import pandas as pd
from core.cleaning import *

def clean(df):
    df = fill_nulls(df, column='price', method='median')
    df = trim_spaces(df, column='name')
    df = remove_outliers(df, column='revenue', method='iqr', action='clip')
    return df
```

Click **Download Script** to save as a `.txt` file.

---

## 10. ML Pipeline — Feature Engineering

Navigate to **Feature Engineering** to start the ML pipeline. The cleaned dataset is the starting point.

### Step 1: Task Setup

| Setting | Options | Notes |
|---------|---------|-------|
| Task Type | classification, regression, clustering | Determines which algorithms and metrics are used |
| Target Column | Any column (dropdown) | Disabled for clustering |
| Feature Columns | Multi-select (default: all except target) | Remove irrelevant columns like IDs |

Click **Initialize ML Pipeline**. The app copies your cleaned data into a separate ML workspace — cleaning and ML pipelines don't interfere with each other.

A **data leakage check** runs automatically: if any feature has >99% correlation with the target, you'll see a warning.

### Step 2: Feature Engineering Operations

Apply transformations in any order. Each operation updates the working dataset immediately.

| Operation | When to use | Key options |
|-----------|-------------|-------------|
| **Encoding** | Non-numeric columns that models can't consume directly | Label encoding (ordinal), One-hot encoding (nominal, max 20 categories) |
| **Scaling** | Numeric features on different scales (e.g., age vs. salary) | StandardScaler (mean=0, std=1), MinMaxScaler (0–1 range) |
| **Polynomial Features** | Suspected non-linear relationships | Degree 2–4, interaction-only option |
| **Interaction Terms** | Known feature interactions (e.g., price × quantity) | Select two numeric columns |
| **Binning** | Convert continuous to categorical | 2–10 bins, strategy: quantile/uniform/kmeans |
| **PCA** | Reduce dimensionality, remove multicollinearity | Select columns, choose number of components; variance explained shown after applying |
| **Feature Selection** | Too many features, want the most predictive subset | Uses Random Forest importance; select top K |
| **Handle Remaining Nulls** | ML algorithms can't handle NaN | Drop rows, fill with mean/median/zero |

**Reset Feature Engineering** reverts to the clean dataset snapshot — useful if you want to try a different approach.

### Step 3: Review & Split

- **Before/After comparison** — Shape, columns added/removed.
- **Operations Log** — Expandable list of everything you applied.
- **Preview** — First 20 rows of the engineered dataset.
- **Test Size slider** — Proportion held out for evaluation (default 20%). For clustering, no split is performed.
- Click **Split Data & Proceed to Training**.

---

## 11. ML Pipeline — Model Training (AutoML)

The AutoML engine trains multiple algorithms, cross-validates each, and ranks them.

### Algorithms by task type

| Classification | Regression | Clustering |
|---------------|-----------|-----------|
| Logistic Regression | Linear Regression | K-Means |
| Random Forest | Ridge Regression | Agglomerative |
| Gradient Boosting | Lasso Regression | Gaussian Mixture |
| SVM | Random Forest | |
| K-Nearest Neighbors | Gradient Boosting | |
| Decision Tree | SVR | |
| | K-Nearest Neighbors | |

### Controls

- **Algorithm multi-select** — All selected by default. Deselect any you want to skip.
- **Cross-validation folds** — 2–10, default 5 (not applicable for clustering).
- **Cluster range** (clustering only) — Min and max number of clusters to try (default 2–10).

Click **Run AutoML**. A spinner shows while models train.

### Leaderboard

Results appear in a ranked table:

| Column | Description |
|--------|-------------|
| Rank | Position by primary metric |
| Algorithm | Model name (for clustering, includes k value) |
| Primary Metric | Accuracy (classification), R-squared (regression), Silhouette Score (clustering) |
| CV Mean ± Std | Cross-validation stability (supervised only) |
| Training Time | Seconds to fit |

The top-ranked model is auto-selected. Use the dropdown to choose a different model if you prefer (e.g., a simpler model with similar performance).

---

## 12. ML Pipeline — Evaluation

Four tabs for thorough model assessment.

### Tab 1: Metrics

**Classification:** Accuracy, Precision, Recall, F1 Score, ROC AUC (binary only) — displayed as metric cards plus a full table.

**Regression:** R-squared, MAE, RMSE, MSE.

**Clustering:** Silhouette Score (-1 to 1, higher is better), Calinski-Harabasz (higher is better), Davies-Bouldin (lower is better), plus a cluster sizes table.

### Tab 2: Visualizations

| Task | Charts |
|------|--------|
| Classification | Confusion matrix (table), ROC curve (binary), prediction distribution (bar chart) |
| Regression | Actual vs. Predicted scatter, Residual plot (predicted vs. residual), Residual distribution histogram |
| Clustering | 2D scatter plot colored by cluster (select axes), Cluster sizes bar chart |

### Tab 3: Feature Importance

- Horizontal bar chart of feature importances (sorted descending).
- Table with exact importance values.
- Available for tree-based models (Random Forest, Gradient Boosting, Decision Tree) and linear models (Logistic Regression, Ridge, Lasso). Not available for KNN or SVM with non-linear kernels.

### Tab 4: Model Comparison

- Side-by-side table of all trained models with all their metrics.
- Bar chart comparing the primary metric across algorithms — quick visual for picking the best model.

---

## 13. ML Pipeline — Predictions

Three tabs for using your trained model.

### Tab 1: Test Set Predictions

- Table showing features, actual values, and predictions for the held-out test set.
- For classification: probability columns for each class are included.
- Prediction summary: value counts (classification/clustering) or descriptive stats (regression).
- **Download Test Predictions (CSV)**.

### Tab 2: Predict on New Data

1. Upload a new CSV, Excel, or Parquet file.
2. The app validates that all required feature columns are present (missing columns trigger an error with a list of what's expected).
3. Extra columns in the new file are preserved but ignored by the model.
4. Click **Run Predictions**.
5. Results table shows all original columns plus predictions (and probabilities for classification).
6. **Download Predictions (CSV)**.

### Tab 3: Export

| Download | Format | Contents |
|----------|--------|----------|
| Trained Model | `.joblib` | Serialized scikit-learn model — load with `joblib.load("trained_model.joblib")` |
| Pipeline Config | `.json` | Feature columns, target, task type, preprocessing steps (encoding mappings, scaler params, PCA components) |
| Full Report | `.txt` | Text summary: task type, best model, all metrics, leaderboard |

**Loading a saved model in your own code:**

```python
import joblib
import pandas as pd

model = joblib.load("trained_model.joblib")
new_data = pd.read_csv("new_data.csv")
predictions = model.predict(new_data[["feature1", "feature2", "feature3"]])
```

---

## 14. Sample Datasets

The `sample_data/` folder includes test files you can upload to explore the app:

| File | Issues demonstrated |
|------|-------------------|
| `sales_bad.csv` | Mixed date formats, negative amounts, leading/trailing spaces, null values, duplicates |
| `strings_bad.csv` | Whitespace problems, mixed casing, empty strings, special characters |
| `numeric_bad.csv` | Negative values, outliers, constant columns |
| `dates_bad.csv` | Multiple date formats (ISO, US, EU), invalid date strings |
| `test_csv.csv` | General-purpose test file |

---

## 15. Tips & Best Practices

**Data Cleaning:**
- Start with the **Data Inspector** to understand the scope of issues before cleaning.
- Use **AI Suggestions** as a first pass — they catch the most impactful issues automatically.
- Use the **Chat** for multi-step operations described in one sentence (e.g., "trim all string columns and fill missing numeric values with the median").
- Use **Manual Transform** when you need precise control over parameters.
- Check the **Health Score delta** after each operation to verify improvement.

**Feature Engineering:**
- Always handle nulls before splitting — most scikit-learn models can't handle NaN.
- Encode all non-numeric columns before training. One-hot is better for low-cardinality nominals (< 20 unique values). Label encoding is better for ordinals or high cardinality.
- Scale features when using distance-based algorithms (KNN, SVM, PCA). Tree-based models (Random Forest, Gradient Boosting) don't require scaling.
- Use PCA when you have many correlated features and want to reduce dimensionality.
- Use **Feature Selection** when you have 20+ features and want to focus on the most predictive ones.

**Model Training:**
- Start with all algorithms selected — AutoML is fast and the leaderboard tells you what works best.
- Look at **CV Mean ± Std**, not just test accuracy. High std means the model is unstable across folds.
- For imbalanced classification, prefer F1 over accuracy — a model predicting the majority class always gets high accuracy.

**Clustering:**
- Look at the **Silhouette Score**: values above 0.5 indicate good separation. Below 0.25 is weak.
- Try the full range of k values (2–10) first, then narrow down based on the elbow in the silhouette plot.
- Scale your features before clustering — K-Means is distance-based and sensitive to feature magnitude.

**General:**
- The cleaning and ML pipelines use separate state. Going back to clean more data doesn't erase your ML work, but you'll need to re-initialize the ML pipeline to pick up the new changes.
- Use **History & Code** to get a reproducible Python script of your cleaning steps.
- Export the trained model as `.joblib` to use it in production pipelines outside this app.
