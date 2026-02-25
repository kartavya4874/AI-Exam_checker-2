"""
Configuration module for the Hybrid AI Exam Checking System.

Loads environment variables from .env, validates required settings,
creates necessary directories, and exposes a singleton Config object.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv


class Config:
    """Central configuration loaded from environment variables."""

    def __init__(self, env_path: str = None):
        """
        Initialize configuration from .env file.

        Args:
            env_path: Optional explicit path to .env file.
                      Defaults to .env in the project root.
        """
        self.project_root: Path = Path(__file__).resolve().parent
        env_file = Path(env_path) if env_path else self.project_root / ".env"

        if env_file.exists():
            load_dotenv(dotenv_path=str(env_file))
        else:
            alt = self.project_root / ".env.example"
            if alt.exists():
                load_dotenv(dotenv_path=str(alt))
                logging.warning(
                    ".env not found — loaded .env.example (API key will be invalid)"
                )

        # ── OpenAI ──
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

        # ── Database ──
        self.DB_PATH: str = os.getenv(
            "DB_PATH", str(self.project_root / "exam_checker.db")
        )

        # ── Threading ──
        self.MAX_THREADS: int = int(os.getenv("MAX_THREADS", "8"))

        # ── Grading ──
        self.TOTAL_MARKS_DEFAULT: int = int(
            os.getenv("TOTAL_MARKS_DEFAULT", "100")
        )
        self.GRADE_BOUNDARIES: Dict[str, int] = self._parse_grade_boundaries(
            os.getenv(
                "GRADE_BOUNDARIES",
                '{"A+":95,"A":85,"B+":75,"B":65,"C":55,"D":45,"F":0}',
            )
        )
        self.NEGATIVE_MARKING_FACTOR: float = float(
            os.getenv("NEGATIVE_MARKING_FACTOR", "0.0")
        )

        # ── Web Portal ──
        self.PORTAL_HOST: str = os.getenv("PORTAL_HOST", "0.0.0.0")
        self.PORTAL_PORT: int = int(os.getenv("PORTAL_PORT", "8000"))

        # ── Preprocessing ──
        self.BLANK_DETECTION_THRESHOLD: float = float(
            os.getenv("BLANK_DETECTION_THRESHOLD", "0.02")
        )

        # ── OCR ──
        self.TROCR_BATCH_SIZE: int = int(os.getenv("TROCR_BATCH_SIZE", "8"))

        # ── Evaluation Weights ──
        self.CLIP_SIMILARITY_WEIGHT: float = float(
            os.getenv("CLIP_SIMILARITY_WEIGHT", "0.3")
        )
        self.GPT4O_WEIGHT: float = float(os.getenv("GPT4O_WEIGHT", "0.7"))

        # ── Processing ──
        self.REPROCESS_EXISTING: bool = (
            os.getenv("REPROCESS_EXISTING", "false").lower() == "true"
        )
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.TEMP_DIR: Path = Path(
            os.getenv("TEMP_DIR", str(self.project_root / "temp_processing"))
        )
        self.MODELS_CACHE_DIR: Path = Path(
            os.getenv(
                "MODELS_CACHE_DIR", str(self.project_root / "model_cache")
            )
        )

        # ── Create directories ──
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # ── Set HuggingFace cache ──
        os.environ["HF_HOME"] = str(self.MODELS_CACHE_DIR)
        os.environ["TRANSFORMERS_CACHE"] = str(self.MODELS_CACHE_DIR)

    # ─── helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse_grade_boundaries(raw: str) -> Dict[str, int]:
        """Parse grade boundaries JSON string into a dict sorted by value descending."""
        try:
            boundaries = json.loads(raw)
            return dict(
                sorted(boundaries.items(), key=lambda x: x[1], reverse=True)
            )
        except json.JSONDecodeError:
            return {"A+": 95, "A": 85, "B+": 75, "B": 65, "C": 55, "D": 45, "F": 0}

    def validate(self) -> bool:
        """
        Validate that all critical configuration is present.

        Returns:
            True if valid, False otherwise. Prints issues to stderr.
        """
        valid = True

        if not self.OPENAI_API_KEY or not self.OPENAI_API_KEY.startswith("sk-"):
            print(
                "⚠️  OPENAI_API_KEY is missing or invalid. "
                "Set it in .env (must start with 'sk-').",
                file=sys.stderr,
            )
            valid = False

        if self.MAX_THREADS < 1:
            print("⚠️  MAX_THREADS must be >= 1.", file=sys.stderr)
            valid = False

        if self.CLIP_SIMILARITY_WEIGHT + self.GPT4O_WEIGHT == 0:
            print(
                "⚠️  Evaluation weights sum to zero — at least one must be > 0.",
                file=sys.stderr,
            )
            valid = False

        return valid

    def print_summary(self) -> None:
        """Print startup configuration summary."""
        border = "═" * 56
        print(f"\n{border}")
        print("  Hybrid AI Exam Checker — Configuration Summary")
        print(border)
        print(f"  Project root       : {self.project_root}")
        print(f"  Database           : {self.DB_PATH}")
        print(f"  Temp directory     : {self.TEMP_DIR}")
        print(f"  Models cache       : {self.MODELS_CACHE_DIR}")
        print(f"  Max threads        : {self.MAX_THREADS}")
        print(f"  Portal             : http://{self.PORTAL_HOST}:{self.PORTAL_PORT}")
        print(f"  Log level          : {self.LOG_LEVEL}")
        print(f"  Reprocess existing : {self.REPROCESS_EXISTING}")
        print(f"  Blank threshold    : {self.BLANK_DETECTION_THRESHOLD}")
        print(f"  Negative marking   : {self.NEGATIVE_MARKING_FACTOR}")
        api_display = (
            self.OPENAI_API_KEY[:8] + "..." if self.OPENAI_API_KEY else "NOT SET"
        )
        print(f"  OpenAI API key     : {api_display}")
        print(f"  Grade boundaries   : {json.dumps(self.GRADE_BOUNDARIES)}")
        print(border)
        print("  Models that will be used:")
        print("    • microsoft/trocr-large-handwritten  (handwriting OCR)")
        print("    • openai/clip-vit-base-patch32       (diagram similarity)")
        print("    • facebook/sam-vit-base              (region segmentation)")
        print("    • all-MiniLM-L6-v2                   (text similarity)")
        print("    • Pix2Text                           (math OCR)")
        print("    • DECIMER                            (chemistry structures)")
        print("    • EasyOCR                            (multilingual)")
        print("    • Tesseract                          (printed text)")
        print("    • GPT-4o                             (final evaluation)")
        print(f"{border}\n")


# ── Singleton ──
_config_instance: Config = None


def get_config(env_path: str = None) -> Config:
    """
    Return the singleton Config instance.

    Args:
        env_path: Optional path to .env file (used only on first call).

    Returns:
        Config singleton.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(env_path=env_path)
    return _config_instance
