# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Data Cleaning Pro — a Streamlit-based data cleaning tool that uses OpenAI's function-calling API to let users describe transformations in natural language. The AI translates requests into tool calls, which are validated and executed against pandas DataFrames.

## Commands

```bash
# Run the app
streamlit run app_enhanced.py

# Run all tests (247 tests across 11 modules)
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

### UI pages in `app_enhanced.py`

Data Inspector, Chat & Transform, Manual Transform, Join Datasets, AI Suggestions, Export, History & Code. Session state tracks `cleaned_df`, `df_history` (undo stack, max 20), `pending_tool_calls`, `executed_actions`, and `chat_history`.

## Configuration

All config lives in `config.py`, read from environment variables (loaded via `python-dotenv` from `.env`). Key settings: `OPENAI_API_KEY` (required), `OPENAI_MODEL` (default `gpt-4o-mini`), `MAX_UPLOAD_MB` (200), `MAX_UNDO_HISTORY` (20). See `.env.example` for the full list.

## Testing

Tests are in `tests/` with shared fixtures in `conftest.py`. Fixtures provide DataFrames for various scenarios (nulls, duplicates, mixed types, dates, large 1000-row sets). Tests mock OpenAI calls — no API key needed to run the suite.

## Dependencies

Core: `streamlit`, `pandas`, `openai`, `tenacity`, `pydantic`, `altair`, `openpyxl`, `pyarrow`. All pinned in `requirements.txt`.
