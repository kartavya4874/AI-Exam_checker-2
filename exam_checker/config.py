"""
Exam Checker Configuration Module.

Loads environment variables from .env file, validates required settings,
parses complex config values (grade boundaries, paths), and creates
necessary directories. All configuration is centralized here.
"""

import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Locate .env — walk upward from this file until we find one
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR  # exam_checker/ directory

_env_path = _PROJECT_ROOT / ".env"
if not _env_path.exists():
    _env_path = _PROJECT_ROOT.parent / ".env"

load_dotenv(dotenv_path=str(_env_path), override=True)

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

def _validate_api_key() -> bool:
    """Return True if the API key looks valid."""
    if not OPENAI_API_KEY:
        return False
    if not OPENAI_API_KEY.startswith("sk-"):
        return False
    return True

API_KEY_VALID: bool = _validate_api_key()

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: str = os.getenv("DB_PATH", str(_PROJECT_ROOT / "exam_checker.db"))
DATABASE_URL: str = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------
MAX_THREADS: int = int(os.getenv("MAX_THREADS", "8"))

# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------
TOTAL_MARKS_DEFAULT: int = int(os.getenv("TOTAL_MARKS_DEFAULT", "100"))

_raw_boundaries = os.getenv(
    "GRADE_BOUNDARIES",
    '{"A+":95,"A":85,"B+":75,"B":65,"C":55,"D":45,"F":0}',
)
try:
    GRADE_BOUNDARIES: dict = json.loads(_raw_boundaries)
except json.JSONDecodeError:
    GRADE_BOUNDARIES = {"A+": 95, "A": 85, "B+": 75, "B": 65, "C": 55, "D": 45, "F": 0}

NEGATIVE_MARKING_FACTOR: float = float(os.getenv("NEGATIVE_MARKING_FACTOR", "0.0"))

# ---------------------------------------------------------------------------
# Portal
# ---------------------------------------------------------------------------
PORTAL_PORT: int = int(os.getenv("PORTAL_PORT", "8000"))

# ---------------------------------------------------------------------------
# Detection / Model weights
# ---------------------------------------------------------------------------
BLANK_DETECTION_THRESHOLD: float = float(os.getenv("BLANK_DETECTION_THRESHOLD", "0.02"))
TROCR_BATCH_SIZE: int = int(os.getenv("TROCR_BATCH_SIZE", "8"))
CLIP_SIMILARITY_WEIGHT: float = float(os.getenv("CLIP_SIMILARITY_WEIGHT", "0.3"))
GPT4O_WEIGHT: float = float(os.getenv("GPT4O_WEIGHT", "0.7"))

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
REPROCESS_EXISTING: bool = os.getenv("REPROCESS_EXISTING", "false").lower() in ("true", "1", "yes")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", str(_PROJECT_ROOT / "temp_processing")))
MODELS_CACHE_DIR: Path = Path(os.getenv("MODELS_CACHE_DIR", str(_PROJECT_ROOT / "model_cache")))

TEMP_DIR.mkdir(parents=True, exist_ok=True)
MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Point HuggingFace cache to our models directory
os.environ["HF_HOME"] = str(MODELS_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_CACHE_DIR)
os.environ["TORCH_HOME"] = str(MODELS_CACHE_DIR / "torch")

# ---------------------------------------------------------------------------
# Model identifiers (centralised so every module uses the same strings)
# ---------------------------------------------------------------------------
TROCR_MODEL: str = "microsoft/trocr-large-handwritten"
CLIP_MODEL: str = "openai/clip-vit-base-patch32"
SAM_MODEL: str = "facebook/sam-vit-base"
SENTENCE_MODEL: str = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Startup summary
# ---------------------------------------------------------------------------

def print_startup_summary() -> None:
    """Print a human-readable configuration summary to stdout."""
    border = "═" * 60
    print(f"\n{border}")
    print("  HYBRID AI EXAM CHECKER — Configuration Summary")
    print(border)
    print(f"  API Key valid     : {'✅ Yes' if API_KEY_VALID else '❌ No / Missing'}")
    print(f"  Database          : {DB_PATH}")
    print(f"  Max threads       : {MAX_THREADS}")
    print(f"  Default marks     : {TOTAL_MARKS_DEFAULT}")
    print(f"  Grade boundaries  : {GRADE_BOUNDARIES}")
    print(f"  Negative marking  : {NEGATIVE_MARKING_FACTOR}")
    print(f"  Blank threshold   : {BLANK_DETECTION_THRESHOLD}")
    print(f"  TrOCR batch size  : {TROCR_BATCH_SIZE}")
    print(f"  CLIP weight       : {CLIP_SIMILARITY_WEIGHT}")
    print(f"  GPT-4o weight     : {GPT4O_WEIGHT}")
    print(f"  Reprocess         : {REPROCESS_EXISTING}")
    print(f"  Log level         : {LOG_LEVEL}")
    print(f"  Temp dir          : {TEMP_DIR}")
    print(f"  Models cache      : {MODELS_CACHE_DIR}")
    print(f"  Portal port       : {PORTAL_PORT}")
    print()
    print("  Models that will be used:")
    print(f"    • TrOCR       — {TROCR_MODEL}")
    print(f"    • CLIP        — {CLIP_MODEL}")
    print(f"    • SAM         — {SAM_MODEL}")
    print(f"    • Sentence-TF — {SENTENCE_MODEL}")
    print(f"    • Pix2Text    — bundled default")
    print(f"    • DECIMER     — bundled default")
    print(f"    • Tesseract   — system install")
    print(f"    • EasyOCR     — en, ur, hi, ar")
    print(border + "\n")


if __name__ == "__main__":
    print_startup_summary()
