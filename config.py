"""
Centralized configuration for Data Quality MVP.
All values can be overridden via environment variables.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- AI / OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "30"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

# --- Application ---
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
MAX_UNDO_HISTORY = int(os.getenv("MAX_UNDO_HISTORY", "20"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
DATAFRAME_PREVIEW_ROWS = int(os.getenv("DATAFRAME_PREVIEW_ROWS", "50"))
MAX_CHART_CATEGORIES = int(os.getenv("MAX_CHART_CATEGORIES", "20"))

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

# --- App Metadata ---
APP_TITLE = "AI Data Cleaning Pro"
APP_VERSION = "1.0.0"
