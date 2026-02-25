"""
SQLAlchemy ORM models for the exam checker database.

Tables: Course, Student, EvaluationResult, QuestionResult, ProcessingLog.
"""

import datetime
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime,
    ForeignKey, Boolean, JSON, create_engine,
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Course(Base):
    """Course table — one row per course being evaluated."""
    __tablename__ = "courses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_code = Column(String(50), unique=True, nullable=False, index=True)
    course_name = Column(String(200), default="")
    total_students = Column(Integer, default=0)
    processed_students = Column(Integer, default=0)
    status = Column(String(30), default="pending")  # pending, processing, completed, error
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    statistics = Column(JSON, default=dict)
    report_path = Column(Text, default="")

    students = relationship("Student", back_populates="course", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Course {self.course_code}>"


class Student(Base):
    """Student table — one row per student-course evaluation."""
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, autoincrement=True)
    roll_number = Column(String(50), nullable=False, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"), nullable=False)
    file_path = Column(Text, default="")
    total_marks_obtained = Column(Float, default=0.0)
    total_marks_allocated = Column(Float, default=0.0)
    percentage = Column(Float, default=0.0)
    grade = Column(String(5), default="")
    status = Column(String(30), default="pending")
    questions_attempted = Column(Integer, default=0)
    questions_total = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    error_message = Column(Text, default="")

    course = relationship("Course", back_populates="students")
    question_results = relationship("QuestionResult", back_populates="student", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Student {self.roll_number}>"


class QuestionResult(Base):
    """Per-question evaluation results."""
    __tablename__ = "question_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    question_number = Column(String(20), nullable=False)
    marks_allocated = Column(Float, default=0.0)
    marks_obtained = Column(Float, default=0.0)
    content_type = Column(String(30), default="text")
    status = Column(String(30), default="attempted")
    feedback = Column(Text, default="")
    error_analysis = Column(Text, default="")
    ocr_text = Column(Text, default="")
    ocr_engine = Column(String(30), default="")
    pre_analysis = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    student = relationship("Student", back_populates="question_results")

    def __repr__(self):
        return f"<QuestionResult {self.question_number}: {self.marks_obtained}/{self.marks_allocated}>"


class EvaluationResult(Base):
    """Summary evaluation record for audit and tracking."""
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_code = Column(String(50), index=True)
    roll_number = Column(String(50), index=True)
    total_marks = Column(Float, default=0.0)
    max_marks = Column(Float, default=0.0)
    percentage = Column(Float, default=0.0)
    grade = Column(String(5), default="")
    evaluation_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<EvaluationResult {self.course_code}/{self.roll_number}>"


class ProcessingLog(Base):
    """Processing log for debugging and audit."""
    __tablename__ = "processing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    course_code = Column(String(50), index=True)
    roll_number = Column(String(50), default="")
    action = Column(String(100), default="")
    status = Column(String(30), default="info")  # info, warning, error
    message = Column(Text, default="")
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return f"<ProcessingLog {self.action}: {self.status}>"
