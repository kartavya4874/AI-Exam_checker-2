"""
Portal Routes — Defines web endpoints for the results portal.
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

from exam_checker.database.db_manager import (
    get_all_courses, 
    get_evaluation_results, 
    get_student_result,
    get_course_stats
)

router = APIRouter()
base_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home page showing all processed courses."""
    courses = get_all_courses()
    return templates.TemplateResponse("index.html", {"request": request, "courses": courses})

@router.get("/course/{course_code}", response_class=HTMLResponse)
async def course_results(request: Request, course_code: str):
    """Results page for a specific course."""
    results = get_evaluation_results(course_code)
    stats = get_course_stats(course_code)
    if not results:
        raise HTTPException(status_code=404, detail="Course not found or no results yet.")
        
    return templates.TemplateResponse("results.html", {
        "request": request, 
        "course_code": course_code, 
        "results": results,
        "stats": stats
    })

@router.get("/student/{course_code}/{roll_number}", response_class=HTMLResponse)
async def student_detail(request: Request, course_code: str, roll_number: str):
    """Detailed result for a single student."""
    result = get_student_result(course_code, roll_number)
    if not result:
        raise HTTPException(status_code=404, detail="Student result not found.")
        
    return templates.TemplateResponse("student.html", {
        "request": request, 
        "result": result
    })
