"""
Grade Calculator — Computes final grades and handles negative marking.
"""

from typing import Dict, Optional
from exam_checker.config import GRADE_BOUNDARIES, NEGATIVE_MARKING_FACTOR
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

def calculate_grade(percentage: float) -> str:
    """
    Calculate grade based on percentage using boundaries from config.
    """
    # Sort boundaries descending
    sorted_boundaries = sorted(GRADE_BOUNDARIES.items(), key=lambda x: x[1], reverse=True)
    
    for grade, min_pct in sorted_boundaries:
        if percentage >= min_pct:
            return grade
            
    return "F"

def apply_negative_marking(marks_obtained: float, marks_allocated: float, is_correct: bool) -> float:
    """
    Apply negative marking if configured.
    """
    if is_correct or marks_obtained > 0:
        return marks_obtained
        
    penalty = marks_allocated * NEGATIVE_MARKING_FACTOR
    return -penalty

def compute_final_score(question_results: list) -> Dict[str, float]:
    """
    Sum marks and calculate overall percentage.
    """
    total_obtained = sum(q['marks_obtained'] for q in question_results)
    total_allocated = sum(q['marks_allocated'] for q in question_results)
    
    if total_allocated == 0:
        return {"obtained": 0.0, "total": 0.0, "percentage": 0.0}
        
    percentage = (total_obtained / total_allocated) * 100
    return {
        "obtained": round(total_obtained, 2),
        "total": round(total_allocated, 2),
        "percentage": round(percentage, 2)
    }
