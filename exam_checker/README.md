# Hybrid AI University Exam Checker

A production-ready system for automated grading of university exam sheets using a hybrid AI pipeline.

## Features

- **Hybrid OCR Pipeline**: Combines TrOCR (Handwriting), Tesseract (Printed), and EasyOCR (Multilingual) for robust text extraction.
- **Multimodal Evaluation**: Specialized analyzers for:
  - **Math**: Symbolic verification via SymPy and Pix2Text.
  - **Chemistry**: Structure recognition with DECIMER and RDKit similarity.
  - **Code**: Static analysis and RestrictedPython sandbox execution.
  - **Diagrams**: CLIP-based visual similarity + GPT-4o structural review.
  - **Theory**: Semantic similarity and few-shot LLM evaluation.
- **Teacher Pattern Loading**: Learns from 3-5 pre-graded sample sheets to match a specific teacher's grading style.
- **Automated Reporting**: Generates multi-sheet Excel reports with statistics and feedback.
- **Flexible UI**: Includes a Tkinter control panel and a FastAPI results portal.

## Installation

1. **System Dependencies**:
   - Python 3.9+
   - Tesseract OCR engine (installed on your OS)

2. **Python Environment**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup**:
   ```bash
   python -m exam_checker.setup
   ```
   *This will create directories and a template `.env` file.*

4. **Configuration**:
   Add your `OPENAI_API_KEY` to the `.env` file created in the root directory.

## Usage

### GUI Mode (Recommended)
```bash
python -m exam_checker.main --mode gui
```

### CLI Processing
```bash
python -m exam_checker.main --mode process --root path/to/exams
```

### Results Portal
```bash
python -m exam_checker.main --mode portal
```

## Input Folder Structure
The system expects a folder per exam session containing:
- `COURSECODE_answerkey.json`: The mapping of question numbers to types, marks, and correct answers.
- `COURSECODE_sample_01.pdf`: Pre-graded sample files (PDF + same-named JSON).
- `COURSECODE_ROLLNUMBER.pdf`: Student answer sheets.

## Technical Details
- **Database**: SQLite with SQLAlchemy ORM.
- **Async & Threading**: ThreadPoolExecutor for parallel processing of students.
- **Retry Mechanism**: Exponential backoff for OpenAI API calls.
- **Preprocessing**: OpenCV-based deskewing, denoising, and SAM/Projection-based segmentation.
