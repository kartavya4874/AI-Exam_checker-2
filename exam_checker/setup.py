"""
One-command setup script for the Hybrid AI Exam Checking System.

Run:  python setup.py

Steps performed:
  1. Install all pip requirements
  2. Download and cache HuggingFace models
  3. Verify Tesseract installation
  4. Verify OPENAI_API_KEY in .env
  5. Initialize SQLite database
  6. Run a self-test with a synthetic blank image
  7. Print final status summary
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

MODELS_TO_DOWNLOAD = [
    ("microsoft/trocr-large-handwritten", "TrOCR handwriting OCR"),
    ("openai/clip-vit-base-patch32", "CLIP diagram similarity"),
    ("sentence-transformers/all-MiniLM-L6-v2", "Text similarity"),
]

SAM_MODEL = ("facebook/sam-vit-base", "SAM region segmentation")


def print_step(step_num: int, description: str) -> None:
    """Print a numbered setup step."""
    print(f"\n{'─' * 50}")
    print(f"  Step {step_num}: {description}")
    print(f"{'─' * 50}")


def step1_install_requirements() -> bool:
    """Install pip requirements."""
    print_step(1, "Installing pip requirements")
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        print("  ❌ requirements.txt not found!")
        return False
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
            stdout=subprocess.DEVNULL if "--quiet" in sys.argv else None,
        )
        print("  ✅ All requirements installed")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  ❌ pip install failed: {exc}")
        return False


def step2_download_models() -> bool:
    """Download and cache HuggingFace models."""
    print_step(2, "Downloading AI models (this may take a while)")

    # Set cache dir
    cache_dir = PROJECT_ROOT / "model_cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

    all_ok = True

    for model_name, description in MODELS_TO_DOWNLOAD:
        print(f"\n  📦 {description}  ({model_name})")
        try:
            if "sentence-transformers" in model_name:
                from sentence_transformers import SentenceTransformer

                SentenceTransformer(
                    model_name.replace("sentence-transformers/", ""),
                    cache_folder=str(cache_dir),
                )
            elif "trocr" in model_name:
                from transformers import VisionEncoderDecoderModel, AutoProcessor

                AutoProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
                VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=str(cache_dir))
            else:
                from transformers import AutoModel, AutoProcessor

                AutoProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
                AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir))
            print(f"     ✅ Downloaded")
        except Exception as exc:
            print(f"     ⚠️  Failed: {exc}")
            print(f"     (Will be downloaded on first use)")
            all_ok = False

    # SAM model — optional
    print(f"\n  📦 {SAM_MODEL[1]}  ({SAM_MODEL[0]})")
    try:
        from transformers import SamModel, SamProcessor

        SamProcessor.from_pretrained(SAM_MODEL[0], cache_dir=str(cache_dir))
        SamModel.from_pretrained(SAM_MODEL[0], cache_dir=str(cache_dir))
        print(f"     ✅ Downloaded")
    except Exception as exc:
        print(f"     ⚠️  Optional model failed: {exc}")
        print(f"     (Projection-profile fallback will be used)")
        all_ok = False

    return all_ok


def step3_verify_tesseract() -> bool:
    """Check that Tesseract OCR is installed."""
    print_step(3, "Verifying Tesseract OCR installation")
    try:
        import pytesseract

        version = pytesseract.get_tesseract_version()
        print(f"  ✅ Tesseract version: {version}")
        return True
    except Exception:
        print("  ❌ Tesseract not found!")
        print("     Install Tesseract OCR:")
        print("     • Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("     • macOS:   brew install tesseract")
        print("     • Linux:   sudo apt install tesseract-ocr")
        print("     Then ensure 'tesseract' is in your system PATH.")
        return False


def step4_verify_env() -> bool:
    """Check that .env file exists and has an API key."""
    print_step(4, "Verifying .env configuration")
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        example = PROJECT_ROOT / ".env.example"
        if example.exists():
            import shutil

            shutil.copy(str(example), str(env_file))
            print("  📝 Created .env from .env.example")
            print("  ⚠️  Edit .env and set your OPENAI_API_KEY!")
            return False
        else:
            print("  ❌ No .env or .env.example found!")
            return False

    from dotenv import load_dotenv

    load_dotenv(str(env_file))
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and api_key.startswith("sk-") and len(api_key) > 10:
        print(f"  ✅ OPENAI_API_KEY found: {api_key[:8]}...")
        return True
    else:
        print("  ⚠️  OPENAI_API_KEY is missing or invalid in .env")
        print("     Set: OPENAI_API_KEY=sk-your-actual-key")
        return False


def step5_init_database() -> bool:
    """Initialize the SQLite database."""
    print_step(5, "Initializing database")
    try:
        sys.path.insert(0, str(PROJECT_ROOT.parent))
        from exam_checker.database.db_manager import DatabaseManager

        db = DatabaseManager()                    # singleton; reads DB_PATH from config
        print(f"  ✅ Database initialized at: {db.engine.url}")
        return True
    except Exception as exc:
        print(f"  ❌ Database init failed: {exc}")
        return False


def step6_self_test() -> bool:
    """Run a basic self-test with a synthetic blank image."""
    print_step(6, "Running self-test")
    try:
        from PIL import Image
        import numpy as np

        # Create a synthetic blank white image
        blank = Image.fromarray(
            np.ones((200, 400, 3), dtype=np.uint8) * 255
        )

        # Test blank detector
        sys.path.insert(0, str(PROJECT_ROOT.parent))
        from exam_checker.preprocessing.blank_detector import is_region_blank

        assert is_region_blank(blank) is True, "Blank detector failed on white image"
        print("  ✅ Blank detector: working")

        # Test image utils
        from exam_checker.utils.image_utils import (
            image_to_base64,
            base64_to_image,
            pil_to_numpy,
            numpy_to_pil,
        )

        b64 = image_to_base64(blank)
        restored = base64_to_image(b64)
        assert restored.size == blank.size, "Base64 roundtrip failed"
        print("  ✅ Image utils: working")

        arr = pil_to_numpy(blank)
        back = numpy_to_pil(arr)
        assert back.size == blank.size, "Numpy conversion failed"
        print("  ✅ Numpy conversion: working")

        # Test config
        from exam_checker.config import get_config

        cfg = get_config()
        assert cfg.TOTAL_MARKS_DEFAULT > 0, "Config default marks invalid"
        print("  ✅ Configuration: working")

        print("\n  ✅ All self-tests passed!")
        return True
    except Exception as exc:
        print(f"  ❌ Self-test failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


def main() -> None:
    """Execute all setup steps and print final summary."""
    print("=" * 56)
    print("  Hybrid AI Exam Checker — Setup")
    print("=" * 56)

    results = {}
    results["Requirements"] = step1_install_requirements()
    results["Models"] = step2_download_models()
    results["Tesseract"] = step3_verify_tesseract()
    results["Environment"] = step4_verify_env()
    results["Database"] = step5_init_database()
    results["Self-test"] = step6_self_test()

    print("\n" + "=" * 56)
    print("  Setup Summary")
    print("=" * 56)

    all_passed = True
    for name, passed in results.items():
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 System ready! Run: python main.py --gui")
    else:
        print("  ⚠️  Some steps failed — check above for details.")
        print("  The system may still work with reduced functionality.")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
