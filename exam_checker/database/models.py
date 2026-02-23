"""
SQLAlchemy ORM models for the Exam Checker database.

Tables
------
- courses           — one row per unique course code
- students          — one row per enrolled student
- evaluation_results — one row per student×course evaluation
- question_results  — one row per individual question evaluation
- processing_logs   — audit trail for debugging
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all models."""
    pass


class Course(Base):
    """Represents a university course."""

    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_code = Column(String(32), unique=True, nullable=False, index=True)
    course_name = Column(String(256), nullable=True)
    total_marks = Column(Integer, nullable=False, default=100)
    question_paper_path = Column(Text, nullable=True)
    answer_key_path = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # relationships
    students = relationship("Student", back_populates="course", cascade="all, delete-orphan")
    evaluations = relationship("EvaluationResult", back_populates="course", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Course {self.course_code}>"


class Student(Base):
    """Represents a student enrolled in a course."""

    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    roll_number = Column(String(64), nullable=False, index=True)
    course_code = Column(String(32), ForeignKey("courses.course_code"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("roll_number", "course_code", name="uq_student_course"),)

    course = relationship("Course", back_populates="students")

    def __repr__(self) -> str:
        return f"<Student {self.roll_number} / {self.course_code}>"


class EvaluationResult(Base):
    """Stores overall evaluation for one student in one course."""

    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    roll_number = Column(String(64), nullable=False, index=True)
    course_code = Column(String(32), ForeignKey("courses.course_code"), nullable=False)
    obtained_marks = Column(Float, nullable=False, default=0)
    total_marks = Column(Float, nullable=False, default=100)
    percentage = Column(Float, nullable=False, default=0)
    grade = Column(String(8), nullable=True)
    overall_feedback = Column(Text, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)
    evaluated_at = Column(DateTime, default=datetime.utcnow)
    reprocessed_count = Column(Integer, default=0)

    __table_args__ = (UniqueConstraint("roll_number", "course_code", name="uq_eval_student_course"),)

    course = relationship("Course", back_populates="evaluations")
    question_results = relationship(
        "QuestionResult", back_populates="evaluation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<EvaluationResult {self.roll_number} {self.course_code} {self.percentage:.1f}%>"


class QuestionResult(Base):
    """Stores per-question evaluation detail."""

    __tablename__ = "question_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(Integer, ForeignKey("evaluation_results.id"), nullable=False)
    question_number = Column(String(16), nullable=False)
    marks_allocated = Column(Float, nullable=False, default=0)
    marks_obtained = Column(Float, nullable=False, default=0)
    status = Column(String(16), nullable=False, default="attempted")  # attempted / unattempted / partial
    content_type = Column(String(32), nullable=True)
    feedback = Column(Text, nullable=True)
    error_analysis = Column(Text, nullable=True)
    analyzer_used = Column(String(64), nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    evaluation = relationship("EvaluationResult", back_populates="question_results")

    def __repr__(self) -> str:
        return f"<QuestionResult {self.question_number} {self.marks_obtained}/{self.marks_allocated}>"


class ProcessingLog(Base):
    """Audit trail for processing events and errors."""

    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    roll_number = Column(String(64), nullable=True)
    course_code = Column(String(32), nullable=True)
    status = Column(String(32), nullable=False, default="info")  # info / warning / error / success
    error_message = Column(Text, nullable=True)
    stage_failed = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<ProcessingLog {self.status} {self.roll_number}>"
