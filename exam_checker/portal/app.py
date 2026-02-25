"""
FastAPI web portal for viewing exam results.

Provides REST API endpoints and Jinja2-rendered HTML pages:
  - Dashboard with course overview
  - Course detail with student list
  - Student detail with per-question results
  - API endpoints for AJAX access
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from exam_checker.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

app = FastAPI(title="Exam Checker Portal", version="1.0.0")

# Templates directory
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Create dirs if they don't exist
os.makedirs(TEMPLATE_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATE_DIR)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard showing all courses."""
    db = DatabaseManager()
    courses = db.get_all_courses()
    return templates.TemplateResponse("dashboard.html", {
        "request": request, "courses": courses,
    })


@app.get("/course/{course_code}", response_class=HTMLResponse)
async def course_detail(request: Request, course_code: str):
    """Course detail page with student results."""
    db = DatabaseManager()
    course = db.get_course(course_code)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    students = db.get_student_results(course_code)
    return templates.TemplateResponse("course_detail.html", {
        "request": request, "course_code": course_code,
        "course": course, "students": students,
    })


@app.get("/course/{course_code}/student/{roll_number}", response_class=HTMLResponse)
async def student_detail(request: Request, course_code: str, roll_number: str):
    """Student detail page with per-question evaluation."""
    db = DatabaseManager()
    student = db.get_student_detail(course_code, roll_number)
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return templates.TemplateResponse("student_detail.html", {
        "request": request, "course_code": course_code,
        "student": student,
    })


# ── API Endpoints ──

@app.get("/api/courses")
async def api_courses():
    """API: List all courses."""
    db = DatabaseManager()
    return JSONResponse(db.get_all_courses())


@app.get("/api/course/{course_code}/students")
async def api_students(course_code: str):
    """API: List students for a course."""
    db = DatabaseManager()
    return JSONResponse(db.get_student_results(course_code))


@app.get("/api/course/{course_code}/student/{roll_number}")
async def api_student_detail(course_code: str, roll_number: str):
    """API: Get student detail."""
    db = DatabaseManager()
    result = db.get_student_detail(course_code, roll_number)
    if not result:
        raise HTTPException(status_code=404)
    return JSONResponse(result)


@app.get("/api/course/{course_code}/logs")
async def api_logs(course_code: str, limit: int = 100):
    """API: Get processing logs for a course."""
    db = DatabaseManager()
    return JSONResponse(db.get_processing_logs(course_code, limit))


def create_app(config=None):
    """Factory function to create and configure the FastAPI app.

    Args:
        config: Optional Config instance. If None, uses singleton.

    Returns:
        Configured FastAPI app instance.
    """
    if config is not None:
        app.state.config = config
    return app


def launch_portal(host: str = "0.0.0.0", port: int = 8000):
    """Launch the FastAPI portal using uvicorn."""
    import uvicorn
    logger.info("Starting web portal on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
