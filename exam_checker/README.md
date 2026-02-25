# Hybrid AI University Exam Checking System

An AI-powered exam evaluation system that processes scanned answer sheets using OCR, specialized content analyzers, and GPT-4o for automated grading.

## Architecture

```
PDF/Image Input → Ingestion → Preprocessing → OCR Router → Content Analyzers → GPT-4o Evaluator → Grading Engine → Database → GUI / Web Portal
```

### Modules

| Module | Purpose | Key Technologies |
|--------|---------|-----------------|
| **Ingestion** | Folder scanning, PDF conversion, file organization | PyMuPDF |
| **Preprocessing** | Scan enhancement, blank detection, region segmentation | OpenCV, SAM |
| **OCR** | Handwriting, printed, multilingual text extraction | TrOCR, Tesseract, EasyOCR |
| **Content Analyzers** | Math, chemistry, diagram, code, text analysis | Pix2Text, SymPy, DECIMER, RDKit, CLIP |
| **Evaluation** | Content classification, GPT-4o grading, few-shot learning | GPT-4o, sentence-transformers |
| **Processing** | Student & course orchestration with threading | ThreadPoolExecutor |
| **Database** | SQLAlchemy ORM with SQLite | SQLAlchemy |
| **GUI** | Desktop interface | Tkinter |
| **Web Portal** | REST API + HTML dashboard | FastAPI, Jinja2 |

## Prerequisites

- Python 3.9+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and on PATH
- OpenAI API key (GPT-4o access)

## Setup

```bash
cd exam_checker

# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY

# 3. Run setup (downloads AI models, verifies Tesseract)
python setup.py

# 4. Launch
python main.py --gui       # Desktop GUI
python main.py --portal    # Web portal (http://localhost:8000)
python main.py --both      # Both simultaneously
```

## Input File Naming Convention

```
COURSECODE_ROLLNUMBER.pdf       → Student answer sheet
COURSECODE_questionpaper.pdf    → Question paper
COURSECODE_answerkey.pdf        → Answer key
COURSECODE_sample_01.pdf        → Teacher-marked sample (for few-shot learning)
```

## License

MIT
