"""
Database manager — all CRUD operations, statistics, and Excel export.

Uses SQLAlchemy sessions scoped to each public function so the module
is safe to call from multiple threads.
"""

from __future__ import annotations

import statistics as _stats
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

from exam_checker.config import DATABASE_URL
from exam_checker.database.models import (
    Base,
    Course,
    EvaluationResult,
    ProcessingLog,
    QuestionResult,
    Student,
)
from exam_checker.utils.logger import get_logger
from exam_checker.utils.report_generator import generate_excel_report

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine & session factory (module-level singletons)
# ---------------------------------------------------------------------------
_engine = create_engine(DATABASE_URL, echo=False, future=True)
_SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False)


def _session() -> Session:
    return _SessionFactory()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(_engine)
    log.info("Database initialised at %s", DATABASE_URL)


# ---------------------------------------------------------------------------
# Course helpers
# ---------------------------------------------------------------------------

def get_or_create_course(
    course_code: str,
    total_marks: int = 100,
    question_paper_path: Optional[str] = None,
    answer_key_path: Optional[str] = None,
) -> Course:
    """Return existing course or insert a new one."""
    with _session() as s:
        course = s.query(Course).filter_by(course_code=course_code).first()
        if course is None:
            course = Course(
                course_code=course_code,
                total_marks=total_marks,
                question_paper_path=question_paper_path,
                answer_key_path=answer_key_path,
            )
            s.add(course)
            s.commit()
            log.info("Created course %s", course_code)
        else:
            # Update paths if provided
            if question_paper_path:
                course.question_paper_path = question_paper_path
            if answer_key_path:
                course.answer_key_path = answer_key_path
            course.updated_at = datetime.utcnow()
            s.commit()
        return course


# ---------------------------------------------------------------------------
# Save evaluation results
# ---------------------------------------------------------------------------

def save_result(eval_dict: Dict[str, Any]) -> int:
    """
    Upsert an evaluation result (keyed on roll_number + course_code).

    Args:
        eval_dict: Must contain at minimum ``roll_number``, ``course_code``,
            ``obtained_marks``, ``total_marks``, ``percentage``, ``grade``.
            Optionally ``question_results`` as a list of dicts and
            ``overall_feedback``, ``processing_time_seconds``.

    Returns:
        The ``EvaluationResult.id``.
    """
    rn = eval_dict["roll_number"]
    cc = eval_dict["course_code"]

    with _session() as s:
        # Ensure student exists
        student = s.query(Student).filter_by(roll_number=rn, course_code=cc).first()
        if student is None:
            student = Student(roll_number=rn, course_code=cc)
            s.add(student)
            s.flush()

        # Upsert evaluation
        ev = s.query(EvaluationResult).filter_by(roll_number=rn, course_code=cc).first()
        if ev is None:
            ev = EvaluationResult(
                roll_number=rn,
                course_code=cc,
                obtained_marks=eval_dict.get("obtained_marks", 0),
                total_marks=eval_dict.get("total_marks", 100),
                percentage=eval_dict.get("percentage", 0),
                grade=eval_dict.get("grade", ""),
                overall_feedback=eval_dict.get("overall_feedback", ""),
                processing_time_seconds=eval_dict.get("processing_time_seconds"),
            )
            s.add(ev)
        else:
            ev.obtained_marks = eval_dict.get("obtained_marks", ev.obtained_marks)
            ev.total_marks = eval_dict.get("total_marks", ev.total_marks)
            ev.percentage = eval_dict.get("percentage", ev.percentage)
            ev.grade = eval_dict.get("grade", ev.grade)
            ev.overall_feedback = eval_dict.get("overall_feedback", ev.overall_feedback)
            ev.processing_time_seconds = eval_dict.get("processing_time_seconds", ev.processing_time_seconds)
            ev.reprocessed_count = (ev.reprocessed_count or 0) + 1
            ev.evaluated_at = datetime.utcnow()
            # Remove old question results
            s.query(QuestionResult).filter_by(result_id=ev.id).delete()

        s.flush()  # ensure ev.id is populated

        # Question-level results
        for qr_dict in eval_dict.get("question_results", []):
            qr = QuestionResult(
                result_id=ev.id,
                question_number=qr_dict.get("question_number", ""),
                marks_allocated=qr_dict.get("marks_allocated", 0),
                marks_obtained=qr_dict.get("marks_obtained", 0),
                status=qr_dict.get("status", "attempted"),
                content_type=qr_dict.get("content_type", ""),
                feedback=qr_dict.get("feedback", ""),
                error_analysis=qr_dict.get("error_analysis", ""),
                analyzer_used=qr_dict.get("analyzer_used", ""),
                confidence_score=qr_dict.get("confidence_score"),
            )
            s.add(qr)

        s.commit()
        log.info("Saved result for %s / %s → %.1f%%", rn, cc, ev.percentage)
        return ev.id


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_results_by_course(course_code: str) -> List[Dict[str, Any]]:
    """Return all evaluation results for *course_code* as dicts."""
    with _session() as s:
        evs = s.query(EvaluationResult).filter_by(course_code=course_code).all()
        out: List[Dict[str, Any]] = []
        for ev in evs:
            qrs = s.query(QuestionResult).filter_by(result_id=ev.id).all()
            out.append(_ev_to_dict(ev, qrs))
        return out


def get_student_result(roll_number: str, course_code: str) -> Optional[Dict[str, Any]]:
    """Return a single student's evaluation with question breakdown."""
    with _session() as s:
        ev = s.query(EvaluationResult).filter_by(
            roll_number=roll_number, course_code=course_code
        ).first()
        if ev is None:
            return None
        qrs = s.query(QuestionResult).filter_by(result_id=ev.id).all()
        return _ev_to_dict(ev, qrs)


def get_all_course_codes() -> List[str]:
    """Return list of distinct course codes that have results."""
    with _session() as s:
        rows = s.query(EvaluationResult.course_code).distinct().all()
        return [r[0] for r in rows]


def get_course_stats(course_code: str) -> Dict[str, Any]:
    """Aggregate statistics for a course."""
    with _session() as s:
        evs = s.query(EvaluationResult).filter_by(course_code=course_code).all()
        if not evs:
            return {
                "total_students": 0, "average": 0, "max": 0, "min": 0,
                "pass_rate": 0, "std_dev": 0, "grade_distribution": {},
            }

        percentages = [e.percentage for e in evs]
        grades = [e.grade or "" for e in evs]
        grade_dist: Dict[str, int] = {}
        for g in grades:
            grade_dist[g] = grade_dist.get(g, 0) + 1

        passing = sum(1 for p in percentages if p >= 45)

        return {
            "total_students": len(evs),
            "average": _stats.mean(percentages),
            "max": max(percentages),
            "min": min(percentages),
            "median": _stats.median(percentages),
            "std_dev": _stats.stdev(percentages) if len(percentages) > 1 else 0,
            "pass_rate": (passing / len(evs)) * 100,
            "grade_distribution": grade_dist,
        }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_excel(course_code: str):
    """Generate Excel report for *course_code*, returned as ``BytesIO``."""
    results = get_results_by_course(course_code)
    stats = get_course_stats(course_code)
    return generate_excel_report(course_code, results, stats)


# ---------------------------------------------------------------------------
# Processing log
# ---------------------------------------------------------------------------

def log_processing_event(
    roll_number: Optional[str],
    course_code: Optional[str],
    status: str,
    error: Optional[str] = None,
    stage: Optional[str] = None,
) -> None:
    """Insert a processing-log entry for audit / debugging."""
    with _session() as s:
        entry = ProcessingLog(
            roll_number=roll_number,
            course_code=course_code,
            status=status,
            error_message=error,
            stage_failed=stage,
        )
        s.add(entry)
        s.commit()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ev_to_dict(ev: EvaluationResult, qrs: List[QuestionResult]) -> Dict[str, Any]:
    return {
        "id": ev.id,
        "roll_number": ev.roll_number,
        "course_code": ev.course_code,
        "obtained_marks": ev.obtained_marks,
        "total_marks": ev.total_marks,
        "percentage": ev.percentage,
        "grade": ev.grade,
        "overall_feedback": ev.overall_feedback,
        "processing_time_seconds": ev.processing_time_seconds,
        "evaluated_at": str(ev.evaluated_at) if ev.evaluated_at else None,
        "reprocessed_count": ev.reprocessed_count,
        "question_results": [
            {
                "question_number": qr.question_number,
                "marks_allocated": qr.marks_allocated,
                "marks_obtained": qr.marks_obtained,
                "status": qr.status,
                "content_type": qr.content_type,
                "feedback": qr.feedback,
                "error_analysis": qr.error_analysis,
                "analyzer_used": qr.analyzer_used,
                "confidence_score": qr.confidence_score,
            }
            for qr in qrs
        ],
    }
