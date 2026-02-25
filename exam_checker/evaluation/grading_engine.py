"""
Grading engine — calculates grades from marks using configurable boundaries.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional

from exam_checker.config import get_config

logger = logging.getLogger(__name__)


def calculate_grade(
    percentage: float,
    grade_boundaries: Optional[Dict[str, int]] = None,
) -> str:
    """
    Calculate letter grade from percentage.

    Args:
        percentage: Student's percentage score (0-100).
        grade_boundaries: Optional custom boundaries {grade: min_percent}.
            Defaults to config-loaded boundaries.

    Returns:
        Letter grade string (e.g. "A+", "B", "F").
    """
    if grade_boundaries is None:
        try:
            config = get_config()
            grade_boundaries = config.GRADE_BOUNDARIES
        except Exception:
            grade_boundaries = {
                "A+": 90, "A": 80, "B+": 70, "B": 60,
                "C+": 50, "C": 40, "D": 33, "F": 0,
            }

    # Sort boundaries descending by threshold
    sorted_grades = sorted(grade_boundaries.items(), key=lambda x: x[1], reverse=True)

    for grade, threshold in sorted_grades:
        if percentage >= threshold:
            return grade

    return "F"


def calculate_student_results(
    evaluation_results: Dict[str, Any],
    grade_boundaries: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Calculate final results for a student.

    Args:
        evaluation_results: Output from evaluator_orchestrator.evaluate_student().
        grade_boundaries: Optional custom grade boundaries.

    Returns:
        Dict with total_marks, percentage, grade, question_results, statistics.
    """
    total_obtained = evaluation_results.get("total_marks_obtained", 0)
    total_allocated = evaluation_results.get("total_marks_allocated", 1)
    percentage = round(total_obtained / max(total_allocated, 1) * 100, 1)
    grade = calculate_grade(percentage, grade_boundaries)

    question_results = evaluation_results.get("question_results", {})

    attempted = sum(
        1 for q in question_results.values()
        if q.get("status") not in ("unattempted", "blank")
    )
    total_questions = len(question_results)
    unattempted = total_questions - attempted

    return {
        "total_marks_obtained": total_obtained,
        "total_marks_allocated": total_allocated,
        "percentage": percentage,
        "grade": grade,
        "questions_attempted": attempted,
        "questions_unattempted": unattempted,
        "total_questions": total_questions,
        "question_results": question_results,
    }


def calculate_course_statistics(
    student_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate aggregate statistics for a course.

    Args:
        student_results: List of per-student result dicts.

    Returns:
        Course statistics dict.
    """
    if not student_results:
        return {"error": "No results to analyze"}

    import numpy as np

    percentages = [r.get("percentage", 0) for r in student_results]
    grades = [r.get("grade", "F") for r in student_results]

    grade_distribution = {}
    for g in grades:
        grade_distribution[g] = grade_distribution.get(g, 0) + 1

    return {
        "total_students": len(student_results),
        "mean_percentage": round(float(np.mean(percentages)), 1),
        "median_percentage": round(float(np.median(percentages)), 1),
        "std_deviation": round(float(np.std(percentages)), 1),
        "min_percentage": round(float(np.min(percentages)), 1),
        "max_percentage": round(float(np.max(percentages)), 1),
        "grade_distribution": grade_distribution,
        "pass_rate": round(
            sum(1 for p in percentages if p >= 33) / len(percentages) * 100, 1
        ),
        "distinction_rate": round(
            sum(1 for p in percentages if p >= 75) / len(percentages) * 100, 1
        ),
    }


def question_wise_analysis(
    student_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze performance per question across all students.

    Args:
        student_results: List of per-student result dicts.

    Returns:
        Dict mapping question_number to analysis stats.
    """
    import numpy as np

    question_data: Dict[str, List[float]] = {}

    for result in student_results:
        qr = result.get("question_results", {})
        for qnum, qdata in qr.items():
            if qnum not in question_data:
                question_data[qnum] = []
            allocated = qdata.get("marks_allocated", 1)
            obtained = qdata.get("marks_obtained", 0)
            question_data[qnum].append(obtained / max(allocated, 1) * 100)

    analysis = {}
    for qnum, scores in sorted(question_data.items()):
        arr = np.array(scores)
        analysis[qnum] = {
            "mean_score_pct": round(float(np.mean(arr)), 1),
            "median_score_pct": round(float(np.median(arr)), 1),
            "std_deviation": round(float(np.std(arr)), 1),
            "attempted_count": sum(1 for s in scores if s > 0),
            "total_count": len(scores),
            "difficulty": _classify_difficulty(float(np.mean(arr))),
        }

    return analysis


def _classify_difficulty(mean_pct: float) -> str:
    """Classify question difficulty based on average score."""
    if mean_pct >= 80:
        return "easy"
    elif mean_pct >= 50:
        return "moderate"
    elif mean_pct >= 25:
        return "difficult"
    else:
        return "very_difficult"
