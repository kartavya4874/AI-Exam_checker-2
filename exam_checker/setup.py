"""
Setup Utility — Hybrid AI University Exam Checker

Handles model downloads, directory creation, and system dependency checks.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    print("Checking system dependencies...")
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"[OK] Tesseract found: {version}")
    except Exception:
        print("[WARN] Tesseract not found. Printed OCR will fail.")
        print("      Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")

def prepare_directories():
    dirs = ["data", "logs", "temp", "models_cache", "output_reports"]
    print("Preparing directories...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  [Created/Exists] {d}/")

def download_models():
    """
    Trigger initial downloads for Hugging Face models if they aren't cached.
    """
    print("Starting model pre-downloads (this may take several minutes)...")
    try:
        from transformers import CLIPProcessor, CLIPModel, TrOCRProcessor, VisionEncoderDecoderModel
        from sentence_transformers import SentenceTransformer
        
        # We don't download everything here to save time, 
        # but we check if the basic ones can be initialized.
        print("  - Checking sentence-transformers...")
        SentenceTransformer('all-MiniLM-L6-v2')
        
        print("  - Checking TrOCR (Microsoft/trocr-large-handwritten)...")
        # Just loading the processor is a good check
        TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
        
        print("[OK] Models checked/downloaded.")
    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")
        print("Please check your internet connection and 'requirements.txt' installation.")

def main():
    print("=== Hybrid AI Exam Checker Setup ===")
    prepare_directories()
    check_dependencies()
    
    # Check if .env exists
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("[OK] Created .env from .env.example. PLEASE ADD YOUR OPENAI_API_KEY!")
        else:
            print("[WARN] .env.example missing. Create a .env file with OPENAI_API_KEY.")

    print("\nSetup complete. You can now run the system using:")
    print("python -m exam_checker.main --mode gui")

if __name__ == "__main__":
    main()
