"""
File organizer — physically moves scanned files into course-code subfolders.

Given a root directory full of flat files, this module creates per-course-code
subfolders and moves each file into the correct one.

    root/
      CS101_2024-CS-001.pdf
      CS101_2024-CS-002.pdf
      MATH201_2024-EE-045.jpg

    becomes:

    root/
      CS101/
        CS101_2024-CS-001.pdf
        CS101_2024-CS-002.pdf
      MATH201/
        MATH201_2024-EE-045.jpg
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

_VALID_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def organize_files(root_dir: str, dry_run: bool = False) -> Dict[str, List[str]]:
    """
    Organize files in *root_dir* into course-code subfolders.

    Args:
        root_dir: Flat directory containing answer sheets.
        dry_run: If True, compute moves but don't execute them.

    Returns:
        ``{course_code: [destination_path, …]}``
    """
    root = Path(root_dir)
    if not root.is_dir():
        log.error("Directory does not exist: %s", root_dir)
        return {}

    moves: Dict[str, List[str]] = {}

    for item in root.iterdir():
        if item.is_dir():
            continue
        if item.suffix.lower() not in _VALID_EXTENSIONS:
            continue

        parts = item.stem.split("_", 1)
        if len(parts) < 2:
            log.warning("Skipping file with no course code: %s", item.name)
            continue

        course_code = parts[0].upper()
        target_dir = root / course_code
        target_path = target_dir / item.name

        if course_code not in moves:
            moves[course_code] = []

        if not dry_run:
            target_dir.mkdir(exist_ok=True)
            if target_path.exists():
                log.warning("Destination already exists, skipping: %s", target_path)
            else:
                shutil.move(str(item), str(target_path))
                log.debug("Moved %s → %s", item, target_path)

        moves[course_code].append(str(target_path))

    for cc, files in moves.items():
        log.info("Organized %d files into %s/", len(files), cc)

    return moves
