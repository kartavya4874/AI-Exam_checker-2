"""
Evaluator Orchestrator — Full per-student pipeline coordinator.

Coordinates:
  1. Preprocessing (deskew, enhance)
  2. Segmentation (crop regions)
  3. Question Mapping (match regions to Q numbers)
  4. Per-question Analysis (OCR + specialized solvers)
  5. Final Evaluation (GPT-4o)
"""

import time
from typing import Any, Dict, List, Optional
from PIL import Image

from exam_checker.utils.logger import get_logger
from exam_checker.utils.image_utils import save_temp_image
from exam_checker.preprocessing.scan_enhancer import enhance_pages
from exam_checker.preprocessing.blank_detector import is_region_blank
from exam_checker.preprocessing.region_segmenter import segment_page
from exam_checker.preprocessing.question_mapper import map_questions_to_regions
from exam_checker.ocr.ocr_router import route_ocr
from exam_checker.evaluation.content_classifier import classify_content_type
from exam_checker.evaluation.gpt4o_evaluator import GPT4oEvaluator
from exam_checker.evaluation.grade_calculator import compute_final_score, calculate_grade

# Analyzers
from exam_checker.content_analyzers.math_analyzer import analyze_math_answer
from exam_checker.content_analyzers.chemistry_analyzer import analyze_structure, analyze_equation
from exam_checker.content_analyzers.diagram_analyzer import analyze_diagram
from exam_checker.content_analyzers.code_analyzer import analyze_code
from exam_checker.content_analyzers.text_analyzer import analyze_text

log = get_logger(__name__)

class EvaluatorOrchestrator:
    """
    Coordinates the full pipeline for a student answer sheet.
    """

    def __init__(self):
        self.gpt_evaluator = GPT4oEvaluator()

    def process_student_sheet(
        self,
        student_id: str,
        course_code: str,
        page_images: List[Image.Image],
        answer_key: Dict[str, Any],
        teacher_patterns: List[Dict[str, Any]],
        course_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run the full pipeline for one student.
        """
        start_time = time.time()
        log.info("Processing sheet: %s for %s", student_id, course_code)
        
        # 1. PREPROCESSING
        log.info("Step 1: Enhancing pages…")
        enhanced_pages = enhance_pages(page_images)
        
        # 2. BLANK PAGE FILTERING
        non_blank_pages = []
        for p in enhanced_pages:
            if not is_region_blank(p, threshold=0.01):
                non_blank_pages.append(p)
        
        if not non_blank_pages:
            log.warning("All pages are blank for %s", student_id)
            return self._all_unattempted_result(student_id, course_code, answer_key)

        # 3 & 4. SEGMENTATION & QUESTION MAPPING
        all_questions_map: Dict[str, Image.Image] = {}
        for page_idx, page in enumerate(non_blank_pages):
            log.info("Processing Page %d…", page_idx + 1)
            
            # Get OCR for mapping
            ocr_text = route_ocr(page, context="printed")
            
            # Segment into regions
            regions = segment_page(page)
            
            # Map questions to images
            q_map = map_questions_to_regions(ocr_text, regions)
            all_questions_map.update(q_map)

        # 5. PER-QUESTION PIPELINE
        question_results = []
        for q_num, q_image in all_questions_map.items():
            # Get key info for this question
            q_key = answer_key.get(q_num)
            if not q_key:
                log.warning("No answer key entry for Q%s. Using theory fallback.", q_num)
                q_key = {"type": "text", "marks": 5, "answer": "Consult expert."}

            # a. Skip if blank
            if is_region_blank(q_image):
                question_results.append({
                    "question_number": q_num,
                    "marks_allocated": q_key["marks"],
                    "marks_obtained": 0,
                    "status": "unattempted",
                    "feedback": "Not attempted."
                })
                continue

            # b. OCR
            student_text = route_ocr(q_image)
            
            # c. Classify
            content_type = classify_content_type(q_image, student_text)
            
            # d. Specialized Analyzer
            analyzer_result = {}
            try:
                if content_type == 'math':
                    analyzer_result = analyze_math_answer(q_image, q_key.get('answer', ''), q_key['marks'])
                elif content_type == 'chemistry_structure':
                    analyzer_result = analyze_structure(q_image, q_key.get('answer', ''), q_key['marks'])
                elif content_type == 'chemistry_equation':
                    analyzer_result = analyze_equation(student_text, q_key.get('answer', ''), q_key['marks'])
                elif content_type == 'diagram':
                    analyzer_result = analyze_diagram(q_image, None, q_key.get('answer', ''), q_key['marks'])
                elif content_type == 'code':
                    analyzer_result = analyze_code(student_text, q_key.get('answer', ''), marks_allocated=q_key['marks'])
                else: # text, mixed
                    analyzer_result = analyze_text(student_text, q_key.get('answer', ''), teacher_patterns, q_key['marks'])
            except Exception as e:
                log.error("Analyzer failed for Q%s: %s", q_num, e)
                analyzer_result = {"error": str(e), "suggested_marks": 0}

            # e. Final GPT-4o Evaluation
            gpt_result = self.gpt_evaluator.evaluate_with_context(
                question_number=q_num,
                student_image=q_image,
                student_text=student_text,
                content_type=content_type,
                analyzer_results=analyzer_result,
                answer_key_text=q_key.get('answer', ''),
                marks_allocated=q_key['marks'],
                few_shot_examples=teacher_patterns
            )
            question_results.append(gpt_result)

        # 6. COMPILE FULL RESULT
        stats = compute_final_score(question_results)
        grade = calculate_grade(stats['percentage'])
        
        processing_time = time.time() - start_time
        
        return {
            "roll_number": student_id,
            "course_code": course_code,
            "obtained_marks": stats['obtained'],
            "total_marks": stats['total'],
            "percentage": stats['percentage'],
            "grade": grade,
            "overall_feedback": f"Evaluated {len(question_results)} questions in {processing_time:.1f}s.",
            "processing_time_seconds": processing_time,
            "question_results": question_results
        }

    def _all_unattempted_result(self, student_id: str, course_code: str, answer_key: Dict[str, Any]) -> Dict[str, Any]:
        """Helper for completely blank submissions."""
        q_results = []
        for q_num, q_key in answer_key.items():
            q_results.append({
                "question_number": q_num,
                "marks_allocated": q_key.get("marks", 0),
                "marks_obtained": 0,
                "status": "unattempted",
                "feedback": "Not attempted."
            })
        
        stats = compute_final_score(q_results)
        return {
            "roll_number": student_id,
            "course_code": course_code,
            "obtained_marks": 0.0,
            "total_marks": stats['total'],
            "percentage": 0.0,
            "grade": "F",
            "overall_feedback": "Answer sheet appeared to be blank.",
            "processing_time_seconds": 0.1,
            "question_results": q_results
        }
