"""
Pattern Loader — Loads teacher sample sheets for few-shot evaluation.

Accepts list of pre-graded answer sheet paths and their JSON sidecars.
Converts to conversation-format few-shot examples for GPT-4o.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image

from exam_checker.utils.logger import get_logger
from exam_checker.utils.image_utils import pil_to_base64
from exam_checker.ingestion.pdf_converter import load_file_as_images

log = get_logger(__name__)

class PatternLoader:
    """
    Loads and prepares teacher grading patterns for the evaluator.
    """

    @staticmethod
    def load_samples(sample_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Load teacher samples and their matching JSON feedback files.
        Example filename: CS101_sample_01.pdf should have CS101_sample_01.json
        """
        few_shot_examples = []
        
        for path_str in sample_paths:
            path = Path(path_str)
            json_path = path.with_suffix('.json')
            
            if not json_path.exists():
                log.warning("Sample JSON not found for %s. Skipping.", path_str)
                continue
                
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
                    
                # Load images (could be multiple pages)
                images = load_file_as_images(path_str)
                if not images:
                    continue
                
                # Assume the feedback data is a list of question evaluations
                # We'll pick a few key ones to avoid hitting context limits
                for idx, q_eval in enumerate(feedback_data[:3]):
                    # If the sample is a single page/region, use the image directly
                    # If it's a PDF, we'd ideally crop but for patterns we often use
                    # the specific cropped region image if provided.
                    # For now, we'll assume the JSON contains the necessary reference.
                    
                    # Construct a few-shot turn
                    # In a real system, we'd match the question region. 
                    # Here we'll use the first page as a representative for the pattern.
                    img_b64 = pil_to_base64(images[0])
                    
                    few_shot_examples.append({
                        "prompt": f"Below is a teacher-graded example for Question {q_eval.get('question_number')}.\n"
                                  f"Context: {q_eval.get('context', 'Theory answer')}\n"
                                  f"Student Answer Text: {q_eval.get('text', '')}",
                        "response": {
                            "marks_obtained": q_eval.get("marks_obtained"),
                            "feedback": q_eval.get("feedback"),
                            "partial_credit_breakdown": q_eval.get("partial_credit_breakdown", {})
                        }
                    })
                    
                log.info("Loaded %d few-shot turns from %s", len(feedback_data), path.name)
                
            except Exception as e:
                log.error("Failed to load sample %s: %s", path_str, e)
                
        return few_shot_examples[:5] # Hard limit of 5 diverse examples
