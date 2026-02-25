"""
Course processor — orchestrates processing of all students in a course.

Uses threading for concurrent processing with progress tracking.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional

from exam_checker.config import get_config
from exam_checker.processing.student_processor import process_student
from exam_checker.evaluation.grading_engine import calculate_course_statistics
from exam_checker.evaluation.few_shot_builder import (
    load_teacher_samples,
    build_text_patterns,
)
from exam_checker.utils.report_generator import generate_excel_report

logger = logging.getLogger(__name__)


def process_course(
    course_data: Dict[str, Any],
    openai_client,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Process all students in a course.

    Args:
        course_data: Dict with:
            - course_code: str
            - student_files: list of (roll_number, file_path) tuples
            - question_paper: str (path)
            - answer_key: str (path)
            - sample_files: list of file paths
            - answer_key_data: dict mapping qnum to answer data
            - marks_map: dict mapping qnum to marks
            - temp_dir: str
        openai_client: OpenAI client.
        progress_callback: Optional callable(roll_number, status, current, total).

    Returns:
        Course results dict.
    """
    config = get_config()
    course_code = course_data["course_code"]
    student_files = course_data.get("student_files", [])
    answer_key = course_data.get("answer_key_data", {})
    marks_map = course_data.get("marks_map", {})
    sample_files = course_data.get("sample_files", [])
    temp_dir = course_data.get("temp_dir", str(config.TEMP_DIR))

    logger.info(
        "Processing course %s: %d students",
        course_code, len(student_files),
    )

    result = {
        "course_code": course_code,
        "total_students": len(student_files),
        "processed": 0,
        "errors": 0,
        "student_results": [],
        "statistics": {},
        "status": "processing",
    }

    # Load few-shot examples
    few_shot_examples = []
    teacher_patterns = []
    if sample_files:
        try:
            answer_key_text = " ".join(
                v.get("text", "") for v in answer_key.values()
            )
            few_shot_examples = load_teacher_samples(sample_files, answer_key_text)
            teacher_patterns = build_text_patterns(sample_files)
            logger.info(
                "Loaded %d few-shot examples, %d text patterns",
                len(few_shot_examples), len(teacher_patterns),
            )
        except Exception as exc:
            logger.warning("Failed to load teacher samples: %s", exc)

    max_threads = config.MAX_THREADS

    if max_threads > 1:
        result = _process_concurrent(
            student_files, course_code, answer_key, marks_map,
            openai_client, few_shot_examples, teacher_patterns,
            temp_dir, max_threads, progress_callback, result,
        )
    else:
        result = _process_sequential(
            student_files, course_code, answer_key, marks_map,
            openai_client, few_shot_examples, teacher_patterns,
            temp_dir, progress_callback, result,
        )

    # Calculate course statistics
    if result["student_results"]:
        result["statistics"] = calculate_course_statistics(result["student_results"])

    # Generate report
    try:
        import os
        report_buf = generate_excel_report(
            course_code, result["student_results"], result["statistics"]
        )
        report_path = os.path.join(temp_dir, f"{course_code}_report.xlsx")
        with open(report_path, "wb") as f:
            f.write(report_buf.getvalue())
        result["report_path"] = report_path
        logger.info("Report saved: %s", report_path)
    except Exception as exc:
        logger.warning("Report generation failed: %s", exc)

    result["status"] = "completed"
    logger.info(
        "Course %s complete: %d/%d processed, %d errors",
        course_code, result["processed"], result["total_students"], result["errors"],
    )

    return result


def _process_concurrent(
    student_files, course_code, answer_key, marks_map,
    openai_client, few_shot, patterns, temp_dir, max_threads,
    callback, result,
):
    """Process students concurrently."""
    total = len(student_files)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {}
        for roll_number, file_path in student_files:
            future = executor.submit(
                process_student,
                file_path, course_code, roll_number,
                answer_key, marks_map, openai_client,
                temp_dir, few_shot, patterns,
            )
            futures[future] = roll_number

        for future in as_completed(futures):
            roll = futures[future]
            try:
                student_result = future.result(timeout=300)
                result["student_results"].append(student_result)
                result["processed"] += 1
                if student_result.get("status") == "error":
                    result["errors"] += 1

                if callback:
                    callback(roll, student_result.get("status"), result["processed"], total)

            except Exception as exc:
                logger.error("Student %s failed: %s", roll, exc)
                result["errors"] += 1
                result["processed"] += 1
                result["student_results"].append({
                    "roll_number": roll, "status": "error", "error": str(exc),
                })
                if callback:
                    callback(roll, "error", result["processed"], total)

    return result


def _process_sequential(
    student_files, course_code, answer_key, marks_map,
    openai_client, few_shot, patterns, temp_dir,
    callback, result,
):
    """Process students sequentially."""
    total = len(student_files)

    for i, (roll_number, file_path) in enumerate(student_files, 1):
        try:
            student_result = process_student(
                file_path, course_code, roll_number,
                answer_key, marks_map, openai_client,
                temp_dir, few_shot, patterns,
            )
            result["student_results"].append(student_result)
            result["processed"] += 1
            if student_result.get("status") == "error":
                result["errors"] += 1

            if callback:
                callback(roll_number, student_result.get("status"), i, total)

        except Exception as exc:
            logger.error("Student %s failed: %s", roll_number, exc)
            result["errors"] += 1
            result["processed"] += 1
            result["student_results"].append({
                "roll_number": roll_number, "status": "error", "error": str(exc),
            })
            if callback:
                callback(roll_number, "error", i, total)

    return result
