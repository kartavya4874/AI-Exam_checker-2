"""
Database manager — CRUD operations and session management.

Provides a singleton DatabaseManager for all DB interactions.
"""

import logging
from typing import Dict, Any, List, Optional
from threading import Lock

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from exam_checker.config import get_config
from exam_checker.database.models import (
    Base, Course, Student, QuestionResult, EvaluationResult, ProcessingLog,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe singleton database manager."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        config = get_config()
        db_path = config.DB_PATH
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self._initialized = True
        logger.info("Database initialized: %s", db_path)

    def get_session(self) -> Session:
        return self.SessionLocal()

    # ── Course CRUD ──

    def create_course(self, course_code: str, course_name: str = "") -> Course:
        with self.get_session() as session:
            course = session.query(Course).filter_by(course_code=course_code).first()
            if course:
                return course
            course = Course(course_code=course_code, course_name=course_name)
            session.add(course)
            session.commit()
            session.refresh(course)
            return course

    def get_course(self, course_code: str) -> Optional[Course]:
        with self.get_session() as session:
            return session.query(Course).filter_by(course_code=course_code).first()

    def get_all_courses(self) -> List[Dict]:
        with self.get_session() as session:
            courses = session.query(Course).all()
            return [
                {
                    "id": c.id, "course_code": c.course_code,
                    "course_name": c.course_name, "total_students": c.total_students,
                    "processed_students": c.processed_students, "status": c.status,
                    "created_at": str(c.created_at), "statistics": c.statistics,
                }
                for c in courses
            ]

    def update_course(self, course_code: str, **kwargs) -> None:
        with self.get_session() as session:
            course = session.query(Course).filter_by(course_code=course_code).first()
            if course:
                for k, v in kwargs.items():
                    if hasattr(course, k):
                        setattr(course, k, v)
                session.commit()

    # ── Student CRUD ──

    def save_student_result(self, course_code: str, result: Dict[str, Any]) -> None:
        with self.get_session() as session:
            course = session.query(Course).filter_by(course_code=course_code).first()
            if not course:
                course = Course(course_code=course_code)
                session.add(course)
                session.flush()

            roll = result.get("roll_number", "")
            student = session.query(Student).filter_by(
                roll_number=roll, course_id=course.id
            ).first()
            if not student:
                student = Student(roll_number=roll, course_id=course.id)
                session.add(student)

            evaluation = result.get("evaluation", {})
            student.file_path = result.get("file_path", "")
            student.total_marks_obtained = evaluation.get("total_marks_obtained", 0)
            student.total_marks_allocated = evaluation.get("total_marks_allocated", 0)
            student.percentage = evaluation.get("percentage", 0)
            student.grade = evaluation.get("grade", "")
            student.status = result.get("status", "error")
            student.questions_attempted = evaluation.get("questions_attempted", 0)
            student.questions_total = evaluation.get("total_questions", 0)
            student.error_message = result.get("error", "") or ""

            session.flush()

            # Save question results
            for qnum, qdata in evaluation.get("question_results", {}).items():
                qr = session.query(QuestionResult).filter_by(
                    student_id=student.id, question_number=qnum
                ).first()
                if not qr:
                    qr = QuestionResult(student_id=student.id, question_number=qnum)
                    session.add(qr)
                qr.marks_allocated = qdata.get("marks_allocated", 0)
                qr.marks_obtained = qdata.get("marks_obtained", 0)
                qr.content_type = qdata.get("content_type", "text")
                qr.status = qdata.get("status", "attempted")
                qr.feedback = qdata.get("feedback", "")
                qr.error_analysis = qdata.get("error_analysis", "")
                qr.ocr_text = qdata.get("ocr_text", "")
                qr.ocr_engine = qdata.get("ocr_engine", "")

            # Save evaluation result summary
            er = EvaluationResult(
                course_code=course_code, roll_number=roll,
                total_marks=evaluation.get("total_marks_obtained", 0),
                max_marks=evaluation.get("total_marks_allocated", 0),
                percentage=evaluation.get("percentage", 0),
                grade=evaluation.get("grade", ""),
                evaluation_data=evaluation,
            )
            session.add(er)
            session.commit()

    def get_student_results(self, course_code: str) -> List[Dict]:
        with self.get_session() as session:
            course = session.query(Course).filter_by(course_code=course_code).first()
            if not course:
                return []
            students = session.query(Student).filter_by(course_id=course.id).all()
            results = []
            for s in students:
                qrs = [
                    {
                        "question_number": qr.question_number,
                        "marks_allocated": qr.marks_allocated,
                        "marks_obtained": qr.marks_obtained,
                        "content_type": qr.content_type,
                        "status": qr.status,
                        "feedback": qr.feedback,
                    }
                    for qr in s.question_results
                ]
                results.append({
                    "roll_number": s.roll_number, "percentage": s.percentage,
                    "grade": s.grade, "status": s.status,
                    "total_marks_obtained": s.total_marks_obtained,
                    "total_marks_allocated": s.total_marks_allocated,
                    "questions": qrs,
                })
            return results

    def get_student_detail(self, course_code: str, roll_number: str) -> Optional[Dict]:
        with self.get_session() as session:
            course = session.query(Course).filter_by(course_code=course_code).first()
            if not course:
                return None
            student = session.query(Student).filter_by(
                course_id=course.id, roll_number=roll_number
            ).first()
            if not student:
                return None
            return {
                "roll_number": student.roll_number,
                "percentage": student.percentage,
                "grade": student.grade,
                "total_marks_obtained": student.total_marks_obtained,
                "total_marks_allocated": student.total_marks_allocated,
                "status": student.status,
                "questions": [
                    {
                        "question_number": qr.question_number,
                        "marks_allocated": qr.marks_allocated,
                        "marks_obtained": qr.marks_obtained,
                        "content_type": qr.content_type,
                        "status": qr.status,
                        "feedback": qr.feedback,
                        "error_analysis": qr.error_analysis,
                        "ocr_text": qr.ocr_text,
                    }
                    for qr in student.question_results
                ],
            }

    # ── Processing Logs ──

    def log_action(self, course_code: str, action: str,
                   status: str = "info", message: str = "",
                   roll_number: str = "", details: dict = None) -> None:
        with self.get_session() as session:
            log = ProcessingLog(
                course_code=course_code, roll_number=roll_number,
                action=action, status=status, message=message,
                details=details or {},
            )
            session.add(log)
            session.commit()

    def get_processing_logs(self, course_code: str, limit: int = 100) -> List[Dict]:
        with self.get_session() as session:
            logs = (
                session.query(ProcessingLog)
                .filter_by(course_code=course_code)
                .order_by(ProcessingLog.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "action": l.action, "status": l.status,
                    "message": l.message, "roll_number": l.roll_number,
                    "created_at": str(l.created_at),
                }
                for l in logs
            ]


def init_database() -> DatabaseManager:
    """Initialize and return the database manager."""
    return DatabaseManager()
