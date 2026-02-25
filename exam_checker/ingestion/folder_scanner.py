"""
Folder scanner for grouping exam files by course code.

Scans a root directory for answer sheets, question papers, answer keys,
and teacher samples. Groups them by course code extracted from filenames.

Naming convention:
  COURSECODE_ROLLNUMBER.pdf/.jpg/.png   → student answer sheet
  COURSECODE_questionpaper.pdf          → question paper
  COURSECODE_answerkey.pdf              → answer key
  COURSECODE_sample_01.pdf              → teacher sample
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

SPECIAL_SUFFIXES = {
    "questionpaper": "question_paper",
    "answerkey": "answer_key",
    "sample": "teacher_sample",
}


def scan_folder(root_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan a root folder and group files by course code.

    Args:
        root_path: Path to the root directory containing exam files.

    Returns:
        Dictionary keyed by course_code with structure:
        {
            "CS101": {
                "student_files": [
                    {"path": "...", "roll_number": "2024-CS-001", "filename": "..."},
                    ...
                ],
                "question_paper": "path/to/CS101_questionpaper.pdf" or None,
                "answer_key": "path/to/CS101_answerkey.pdf" or None,
                "teacher_samples": ["path/to/CS101_sample_01.pdf", ...],
                "file_count": 45,
            },
            ...
        }
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root_path}")

    courses: Dict[str, Dict[str, Any]] = {}

    # Walk through all files (including subdirectories)
    for dirpath, _dirnames, filenames in os.walk(str(root)):
        for fname in filenames:
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                continue

            # Skip zero-byte files
            if fpath.stat().st_size == 0:
                logger.warning("Skipping zero-byte file: %s", fpath)
                continue

            stem = fpath.stem  # filename without extension
            parts = stem.split("_", 1)


            if len(parts) < 2:
                logger.warning(
                    "File '%s' does not match COURSECODE_XXX pattern — skipping.",
                    fname,
                )
                continue

            course_code = parts[0].upper().strip()
            remainder = parts[1].strip()

            # Initialize course entry
            if course_code not in courses:
                courses[course_code] = {
                    "student_files": [],
                    "question_paper": None,
                    "answer_key": None,
                    "teacher_samples": [],
                    "file_count": 0,
                }

            # Classify file type
            remainder_lower = remainder.lower().replace("-", "").replace(" ", "")

            if remainder_lower == "questionpaper":
                courses[course_code]["question_paper"] = str(fpath)
                logger.info("Found question paper for %s: %s", course_code, fname)
            elif remainder_lower == "answerkey":
                courses[course_code]["answer_key"] = str(fpath)
                logger.info("Found answer key for %s: %s", course_code, fname)
            elif remainder_lower.startswith("sample"):
                courses[course_code]["teacher_samples"].append(str(fpath))
                logger.info("Found teacher sample for %s: %s", course_code, fname)
            else:
                # It's a student answer sheet
                roll_number = remainder  # e.g. "2024-CS-001"
                courses[course_code]["student_files"].append(
                    {
                        "path": str(fpath),
                        "roll_number": roll_number,
                        "filename": fname,
                    }
                )

    # Update file counts
    for code, info in courses.items():
        info["file_count"] = len(info["student_files"])
        # Sort student files by roll number for consistent ordering
        info["student_files"].sort(key=lambda x: x["roll_number"])
        # Sort teacher samples
        info["teacher_samples"].sort()

    # Log summary
    total_students = sum(c["file_count"] for c in courses.values())
    logger.info(
        "Scan complete: %d courses, %d student files",
        len(courses),
        total_students,
    )
    for code, info in sorted(courses.items()):
        qp = "✓" if info["question_paper"] else "✗"
        ak = "✓" if info["answer_key"] else "✗"
        ts = len(info["teacher_samples"])
        logger.info(
            "  %s: %d students, QP=%s, AK=%s, Samples=%d",
            code,
            info["file_count"],
            qp,
            ak,
            ts,
        )

    return courses


def get_course_codes(root_path: str) -> List[str]:
    """
    Return a sorted list of all course codes found in a folder.

    Args:
        root_path: Path to the root directory.

    Returns:
        Sorted list of course code strings.
    """
    courses = scan_folder(root_path)
    return sorted(courses.keys())


def detect_duplicates(courses: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Detect duplicate roll numbers within each course.

    Args:
        courses: Output from scan_folder().

    Returns:
        Dict mapping course_code → list of duplicate roll numbers.
    """
    duplicates: Dict[str, List[str]] = {}

    for code, info in courses.items():
        rolls = [s["roll_number"] for s in info["student_files"]]
        seen = set()
        dupes = set()
        for r in rolls:
            if r in seen:
                dupes.add(r)
            seen.add(r)
        if dupes:
            duplicates[code] = sorted(dupes)
            logger.warning(
                "Duplicate roll numbers in %s: %s", code, ", ".join(sorted(dupes))
            )

    return duplicates
