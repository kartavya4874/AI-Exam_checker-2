"""
Pipeline Runner — The main execution engine that ties everything together.

Coordinates the high-level flow:
1. Scan root folder for courses
2. For each course:
   a. Load answer key and teacher samples
   b. Identify all student PDF/Image files
   c. Instantiate ThreadManager and run EvaluatorOrchestrator
   d. Save results to DB
   e. Generate reports
"""

import os
from typing import Dict, Any, List, Optional

from exam_checker.utils.logger import get_logger
from exam_checker.utils.report_generator import generate_excel_report
from exam_checker.ingestion.folder_scanner import scan_folder
from exam_checker.ingestion.pdf_converter import load_file_as_images
from exam_checker.database.db_manager import save_result, init_db, get_course_stats, export_course_to_excel
from exam_checker.evaluation.evaluator_orchestrator import EvaluatorOrchestrator
from exam_checker.evaluation.pattern_loader import PatternLoader
from exam_checker.processing.thread_manager import ThreadManager

log = get_logger(__name__)

class PipelineRunner:
    """
    Master controller for the exam checking process.
    """

    def __init__(self):
        init_db()
        self.orchestrator = EvaluatorOrchestrator()
        self.thread_manager = ThreadManager()
        self.pattern_loader = PatternLoader()

    def run_full_process(self, root_dir: str) -> Dict[str, Any]:
        """
        Scan and process all courses in the root directory.
        """
        log.info("Starting full evaluation process on: %s", root_dir)
        
        # 1. SCAN FOLDER
        courses_data = scan_folder(root_dir)
        processed_stats = {}

        for course_code, data in courses_data.items():
            log.info("========================================")
            log.info("PROCESSING COURSE: %s", course_code)
            log.info("========================================")
            
            # 2. LOAD KEY & PATTERNS
            answer_key = data.get('answer_key', {}) # Loaded in scanner
            samples = data.get('samples', [])
            few_shot = self.pattern_loader.load_samples(samples)
            
            # 3. PROCESS STUDENTS
            students = data.get('students', {})
            course_results = []
            
            def process_one_student(student_item):
                roll_num, file_path = student_item
                try:
                    images = load_file_as_images(file_path)
                    result = self.orchestrator.process_student_sheet(
                        student_id=roll_num,
                        course_code=course_code,
                        page_images=images,
                        answer_key=answer_key,
                        teacher_patterns=few_shot,
                        course_config={} # Placeholder for future specific settings
                    )
                    
                    # Save to DB
                    save_result(result)
                    return result
                except Exception as e:
                    log.error("Failed to process student %s: %s", roll_num, e)
                    return None

            # Run in parallel
            results = self.thread_manager.run_parallel(process_one_student, list(students.items()))
            course_results = [r for r in results if r is not None]
            
            # 4. GENERATE COURSE REPORT
            if course_results:
                stats = get_course_stats(course_code)
                report_path = export_course_to_excel(course_code, f"{course_code}_Final_Report.xlsx")
                log.info("Generated report for %s at: %s", course_code, report_path)
                processed_stats[course_code] = stats
            
        log.info("Full process complete.")
        return processed_stats

    def cleanup(self):
        """Shutdown resources."""
        self.thread_manager.shutdown()
