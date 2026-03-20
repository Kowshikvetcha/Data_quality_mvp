# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Data Cleaning Pro — a Streamlit-based data cleaning and ML pipeline tool. The cleaning side uses OpenAI's function-calling API for natural-language transformations. The ML side provides an end-to-end AutoML pipeline: feature engineering, model training (classification, regression, clustering), evaluation, and predictions.

## Commands

```bash
# Run the app
streamlit run app_enhanced.py

# Run all tests (382 tests across 16 modules)
pytest tests/ -v

# Run a single test file
pytest tests/test_cleaning.py -v

# Run tests matching a pattern
pytest tests/ -k "fill_nulls"

# Docker
docker build -t dq-app .
docker run -p 8501:8501 --env-file .env dq-app
```

No linter or formatter is configured. No Makefile or build step exists.

## Architecture

```
User (Streamlit UI in app_enhanced.py)
  → ai_router.py        sends user prompt + tool definitions to OpenAI
  → ai_tools.py          defines 30+ tools in function-calling format
  → cleaning_executor.py validates & dispatches tool calls
  → cleaning.py          performs the actual pandas transformations
  → diff_preview.py      computes before/after diff for user confirmation
  → confirm.py           generates human-readable change descriptions
```

**Key flow:** upload → type inference (`checks.py`) → quality report (`summary.py`) → user describes cleaning → AI returns tool calls → executor validates → diff preview shown → user confirms → transformation applied → undo stack updated.

### Core modules (`core/`)

| Module | Role |
|---|---|
| `ai_router.py` | OpenAI API calls with exponential-backoff retry (tenacity) |
| `ai_tools.py` | Tool schema definitions for function calling |
| `cleaning.py` | 30+ transformation functions (null fill, trim, regex, dates, outliers, splits, merges, batch ops) |
| `cleaning_executor.py` | Routes tool names to cleaning functions; validates columns/types before execution; rolls back on failure |
| `checks.py` | Column type inference, null/duplicate/outlier detection |
| `suggestions.py` | AI-generated proactive cleaning recommendations |
| `summary.py` | Dataset health scoring algorithm |
| `diff_preview.py` | Row/column/cell-level change counts |
| `confirm.py` | Human-readable pending-change descriptions |
| `export.py` | CSV, Excel, JSON export |
| `validators.py` | Input validation utilities |
| `logger.py` | Rotating file logger (5 MB, 3 backups) via `get_logger(name)` |

### ML pipeline modules (`ml/`)

| Module | Role |
|---|---|
| `validators.py` | ML-specific validation (target column, feature columns, leakage detection) |
| `feature_engineering.py` | Encoding, scaling, polynomial features, interaction terms, binning, PCA, feature selection, train/test split |
| `training.py` | AutoML engine — algorithm registries for classification/regression/clustering, trains all, ranks by primary metric |
| `evaluation.py` | Metrics computation, feature importance, confusion matrix/ROC/residual/cluster visualization data |
| `predictions.py` | Predict on new data, model serialization (joblib), pipeline config export |
| `config.py` | ML-specific configuration (test size, CV folds, random state, etc.) |
| `pages/` | Streamlit page renderers: feature_engineering_page, training_page, evaluation_page, predictions_page |

**ML flow:** cleaned_df → Feature Engineering (encode, scale, PCA, etc.) → train/test split → AutoML (try all algorithms) → Evaluation (metrics + visualizations) → Predictions (test set or new data upload) → Export (model, predictions, report).

### UI pages in `app_enhanced.py`

**Cleaning:** Data Inspector, Chat & Transform, Manual Transform, Join Datasets, AI Suggestions, Export, History & Code.
**ML Pipeline:** Feature Engineering, Model Training, Evaluation, Predictions.

Session state tracks `cleaned_df`, `df_history` (undo stack, max 20), `pending_tool_calls`, `executed_actions`, `chat_history`, plus ML state: `ml_engineered_df`, `ml_split_data`, `ml_automl_results`, `ml_best_model`, `ml_preprocessing_pipeline`.

## Configuration

All config lives in `config.py`, read from environment variables (loaded via `python-dotenv` from `.env`). Key settings: `OPENAI_API_KEY` (required), `OPENAI_MODEL` (default `gpt-4o-mini`), `MAX_UPLOAD_MB` (200), `MAX_UNDO_HISTORY` (20). See `.env.example` for the full list.

## Testing

Tests are in `tests/` with shared fixtures in `conftest.py`. Fixtures provide DataFrames for various scenarios (nulls, duplicates, mixed types, dates, large 1000-row sets). Tests mock OpenAI calls — no API key needed to run the suite.

## Dependencies

Core: `streamlit`, `pandas`, `openai`, `tenacity`, `pydantic`, `altair`, `openpyxl`, `pyarrow`. ML: `scikit-learn`, `matplotlib`, `seaborn`, `joblib`. All pinned in `requirements.txt`.
