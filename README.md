# AI Data Cleaning Pro

A production-ready, AI-assisted data cleaning and ML pipeline tool built with Streamlit and OpenAI. Upload datasets, identify quality issues, clean data through a conversational chat interface, then build and evaluate ML models — all in one app.

---

## Features

- **AI Chat Interface** — Describe what you want to clean in plain English; the AI translates it into data operations
- **30+ Cleaning Operations** — Fill nulls, trim spaces, standardize case, round/clip/scale numerics, regex replace, date parsing, column split/merge, and more
- **Batch Operations** — Apply transformations across multiple columns at once
- **Data Quality Dashboard** — Automated quality checks with health scores and column-level summaries
- **Undo/Redo** — Full history stack (configurable depth) so every change is reversible
- **Multi-Dataset Join** — Load and join multiple datasets on shared keys
- **Export** — Download cleaned data as CSV, Excel, or JSON
- **Manual Transformations** — Point-and-click cleaning alongside the AI chat
- **Formula Sanitization** — Calculated column formulas are validated to prevent code injection

### ML Pipeline

- **Feature Engineering** — Label/one-hot encoding, standard/minmax scaling, polynomial features, interaction terms, binning, PCA dimensionality reduction, importance-based feature selection
- **AutoML** — Automatically trains and ranks multiple algorithms for classification (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Decision Tree), regression (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, SVR, KNN), and clustering (K-Means, Agglomerative, Gaussian Mixture)
- **Model Evaluation** — Accuracy, precision, recall, F1, ROC AUC, R-squared, MAE, RMSE, silhouette score; confusion matrix, ROC curves, residual plots, cluster scatter plots, feature importance charts
- **Predictions** — Predict on test set or upload new data; download trained model (.joblib), pipeline config (.json), and full report

---

## Prerequisites

- **Python 3.9+**
- **An OpenAI API key**
- Internet connection (for AI calls)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Kowshikvetcha/Data_quality_mvp.git
cd Data_quality_mvp
```

### 2. Create and activate a virtual environment

```bash
python -m venv env
source env/bin/activate        # Linux / macOS
env\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and add your API key:

```bash
cp .env.example .env           # Linux / macOS
copy .env.example .env         # Windows
```

Edit `.env` and set your key:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

See `.env.example` for all optional overrides (model, timeout, retries, upload limit, log level, etc.).

### 5. Run the application

```bash
streamlit run app_enhanced.py
```

The app opens at `http://localhost:8501`.

---

## Configuration

All settings live in `config.py` and can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model used for AI chat |
| `OPENAI_TIMEOUT_SECONDS` | `30` | API call timeout |
| `OPENAI_MAX_RETRIES` | `3` | Retry attempts with exponential backoff |
| `MAX_UPLOAD_MB` | `200` | Maximum upload file size (MB) |
| `MAX_UNDO_HISTORY` | `20` | Undo stack depth |
| `OUTPUT_DIR` | `outputs` | Directory for exported files |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FILE` | `app.log` | Log file path (rotating, 5 MB, 3 backups) |
| `ML_DEFAULT_TEST_SIZE` | `0.2` | Default train/test split ratio |
| `ML_DEFAULT_CV_FOLDS` | `5` | Cross-validation folds for AutoML |
| `ML_MAX_ONEHOT_CATEGORIES` | `20` | Max unique values for one-hot encoding |
| `ML_DEFAULT_RANDOM_STATE` | `42` | Random seed for reproducibility |

---

## Project Structure

```
Data_quality_mvp/
├── app_enhanced.py          # Streamlit application (main entry point)
├── config.py                # Centralized configuration
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container build
├── .env.example             # Environment variable template
├── .streamlit/
│   └── config.toml          # Streamlit server & theme settings
├── core/                    # Data cleaning logic
│   ├── ai_router.py         # OpenAI integration with retry logic
│   ├── ai_tools.py          # Tool definitions for function calling
│   ├── checks.py            # Data quality checks
│   ├── cleaning.py          # 30+ cleaning/transformation functions
│   ├── cleaning_executor.py # Tool dispatch with validation
│   ├── confirm.py           # User confirmation logic
│   ├── diff_preview.py      # Before/after change preview
│   ├── export.py            # CSV, Excel, JSON export
│   ├── logger.py            # Rotating file + console logging
│   ├── suggestions.py       # AI-generated cleaning suggestions
│   ├── summary.py           # Dataset health scoring
│   └── validators.py        # Input validation utilities
├── ml/                      # ML pipeline
│   ├── config.py            # ML-specific configuration
│   ├── validators.py        # ML input validation & leakage detection
│   ├── feature_engineering.py # Encoding, scaling, PCA, polynomial features, etc.
│   ├── training.py          # AutoML engine with algorithm registries
│   ├── evaluation.py        # Metrics, feature importance, visualization data
│   ├── predictions.py       # Predict on new data, model export
│   └── pages/               # Streamlit page renderers
│       ├── feature_engineering_page.py
│       ├── training_page.py
│       ├── evaluation_page.py
│       └── predictions_page.py
├── tests/                   # 382 pytest tests
│   ├── test_checks.py
│   ├── test_cleaning.py
│   ├── test_cleaning_executor.py
│   ├── test_confirm.py
│   ├── test_diff_preview.py
│   ├── test_export.py
│   ├── test_join.py
│   ├── test_suggestions.py
│   ├── test_summary.py
│   ├── test_ml_validators.py
│   ├── test_ml_feature_engineering.py
│   ├── test_ml_training.py
│   ├── test_ml_evaluation.py
│   └── test_ml_predictions.py
├── sample_data/             # Example CSV files for testing
└── archive/                 # Legacy app versions and scripts
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker

Build and run in a container:

```bash
docker build -t dq-app .
docker run -p 8501:8501 --env-file .env dq-app
```

The container includes a health check at `/_stcore/health`.

---

## License

This project is proprietary. All rights reserved.



