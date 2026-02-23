"""
Folder scanner — groups files by course code.

Naming convention:
  Student sheet  :  COURSECODE_ROLLNUMBER.pdf/.jpg/.png
  Question paper :  COURSECODE_questionpaper.pdf
  Answer key     :  COURSECODE_answerkey.pdf
  Teacher sample :  COURSECODE_sample_01.pdf
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

_VALID_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
_SPECIAL_SUFFIXES = ("questionpaper", "answerkey", "sample")


def scan_folder(root_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan *root_dir* recursively and group files by course code.

    Returns:
        ``{course_code: {
            "students":        [path, ...],
            "question_paper":  path | None,
            "answer_key":      path | None,
            "samples":         [path, ...],
         }}``
    """
    root = Path(root_dir)
    if not root.is_dir():
        log.error("Root directory does not exist: %s", root_dir)
        return {}

    courses: Dict[str, Dict[str, Any]] = {}

    for dirpath, _dirs, filenames in os.walk(root):
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext not in _VALID_EXTENSIONS:
                continue

            full_path = str(Path(dirpath) / fname)
            stem = Path(fname).stem  # without extension

            parts = stem.split("_", 1)
            if len(parts) < 2:
                log.warning("Skipping file with unexpected name format: %s", fname)
                continue

            course_code = parts[0].upper()
            remainder = parts[1].lower()

            if course_code not in courses:
                courses[course_code] = {
                    "students": [],
                    "question_paper": None,
                    "answer_key": None,
                    "samples": [],
                }

            bucket = courses[course_code]

            if remainder == "questionpaper":
                bucket["question_paper"] = full_path
                log.debug("Found question paper: %s", full_path)
            elif remainder == "answerkey":
                bucket["answer_key"] = full_path
                log.debug("Found answer key: %s", full_path)
            elif remainder.startswith("sample"):
                bucket["samples"].append(full_path)
                log.debug("Found teacher sample: %s", full_path)
            else:
                # Treat as student answer sheet
                bucket["students"].append(full_path)

    # Log summary
    for cc, data in courses.items():
        log.info(
            "Course %s: %d students, QP=%s, AK=%s, %d samples",
            cc,
            len(data["students"]),
            "Yes" if data["question_paper"] else "No",
            "Yes" if data["answer_key"] else "No",
            len(data["samples"]),
        )

    return courses


def extract_roll_number(filepath: str) -> str:
    """
    Extract the roll number from a student answer sheet filename.

    ``CS101_2024-CS-001.pdf`` → ``2024-CS-001``
    """
    stem = Path(filepath).stem
    parts = stem.split("_", 1)
    if len(parts) >= 2:
        return parts[1]
    return stem


def extract_course_code(filepath: str) -> str:
    """
    Extract the course code from any filename following the convention.

    ``CS101_2024-CS-001.pdf`` → ``CS101``
    """
    stem = Path(filepath).stem
    return stem.split("_")[0].upper()
