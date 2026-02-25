"""
File organizer that physically moves files into course-code subfolders.

Takes scan results from folder_scanner and reorganizes the directory
structure for cleaner processing.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def organize_files(
    root_path: str,
    courses: Dict[str, Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, List[str]]:
    """
    Physically move files into course-code subfolders.

    Creates subdirectories per course code and moves student files,
    question papers, answer keys, and teacher samples into them.

    Args:
        root_path: Root directory containing the files.
        courses: Course dictionary from folder_scanner.scan_folder().
        dry_run: If True, only log what would happen without moving files.

    Returns:
        Dict mapping course_code → list of moved file paths (new locations).

    Directory structure after organizing:
        root/
        ├── CS101/
        │   ├── students/
        │   │   ├── CS101_2024-CS-001.pdf
        │   │   └── ...
        │   ├── CS101_questionpaper.pdf
        │   ├── CS101_answerkey.pdf
        │   └── samples/
        │       ├── CS101_sample_01.pdf
        │       └── ...
        └── MATH201/
            └── ...
    """
    root = Path(root_path)
    moved: Dict[str, List[str]] = {}

    for course_code, info in courses.items():
        course_dir = root / course_code
        students_dir = course_dir / "students"
        samples_dir = course_dir / "samples"

        moved[course_code] = []

        if not dry_run:
            course_dir.mkdir(parents=True, exist_ok=True)
            students_dir.mkdir(parents=True, exist_ok=True)
            samples_dir.mkdir(parents=True, exist_ok=True)

        # Move student files
        for student in info["student_files"]:
            src = Path(student["path"])
            dst = students_dir / src.name

            if src == dst:
                continue  # Already in the right place
            if dst.exists():
                logger.warning(
                    "Destination already exists, skipping: %s", dst
                )
                continue

            if dry_run:
                logger.info("[DRY RUN] Would move: %s → %s", src, dst)
            else:
                try:
                    shutil.move(str(src), str(dst))
                    moved[course_code].append(str(dst))
                    logger.debug("Moved: %s → %s", src, dst)
                except Exception as exc:
                    logger.error("Failed to move %s: %s", src, exc)

        # Move question paper
        if info["question_paper"]:
            src = Path(info["question_paper"])
            dst = course_dir / src.name
            if src != dst and not dst.exists():
                if dry_run:
                    logger.info("[DRY RUN] Would move QP: %s → %s", src, dst)
                else:
                    try:
                        shutil.move(str(src), str(dst))
                        moved[course_code].append(str(dst))
                        logger.debug("Moved QP: %s → %s", src, dst)
                    except Exception as exc:
                        logger.error("Failed to move QP %s: %s", src, exc)

        # Move answer key
        if info["answer_key"]:
            src = Path(info["answer_key"])
            dst = course_dir / src.name
            if src != dst and not dst.exists():
                if dry_run:
                    logger.info("[DRY RUN] Would move AK: %s → %s", src, dst)
                else:
                    try:
                        shutil.move(str(src), str(dst))
                        moved[course_code].append(str(dst))
                        logger.debug("Moved AK: %s → %s", src, dst)
                    except Exception as exc:
                        logger.error("Failed to move AK %s: %s", src, exc)

        # Move teacher samples
        for sample_path in info["teacher_samples"]:
            src = Path(sample_path)
            dst = samples_dir / src.name
            if src != dst and not dst.exists():
                if dry_run:
                    logger.info("[DRY RUN] Would move sample: %s → %s", src, dst)
                else:
                    try:
                        shutil.move(str(src), str(dst))
                        moved[course_code].append(str(dst))
                        logger.debug("Moved sample: %s → %s", src, dst)
                    except Exception as exc:
                        logger.error("Failed to move sample %s: %s", src, exc)

        total_moved = len(moved[course_code])
        if total_moved > 0:
            logger.info(
                "%s: organized %d files into %s",
                course_code,
                total_moved,
                course_dir,
            )

    return moved


def is_already_organized(root_path: str, course_code: str) -> bool:
    """
    Check if files for a course are already organized into subfolders.

    Args:
        root_path: Root directory.
        course_code: Course code to check.

    Returns:
        True if the course subfolder with students/ exists and has files.
    """
    students_dir = Path(root_path) / course_code / "students"
    if not students_dir.exists():
        return False
    return any(students_dir.iterdir())
