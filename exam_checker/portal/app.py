"""
Web Portal App — FastAPI application for viewing results.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os

from exam_checker.config import PORTAL_PORT
from exam_checker.portal.routes import router

app = FastAPI(title="Exam Checker Results Portal")

# Static and Templates setup
# Assuming templates are in the same directory as app.py
base_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# Include routes
app.include_router(router)

def start_portal():
    """Run the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=PORTAL_PORT)

if __name__ == "__main__":
    start_portal()
