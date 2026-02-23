"""
GPT-4o Evaluator — Final evaluation coordinator using LLM vision.

Uses GPT-4o to make the final mark decision based on pre-analyzed signals
from specialized models and the raw student answer image.
"""

import json
from typing import Any, Dict, List, Optional
from PIL import Image
from openai import OpenAI

from exam_checker.config import OPENAI_API_KEY
from exam_checker.utils.image_utils import pil_to_base64
from exam_checker.utils.logger import get_logger
from exam_checker.utils.retry_utils import retry

log = get_logger(__name__)

class GPT4oEvaluator:
    """
    Expert university examiner using GPT-4o Vision for final grading.
    """

    SYSTEM_PROMPT = """
You are an expert university examiner with 20 years of experience.
You evaluate answers across all academic disciplines.

You receive pre-analyzed signals from specialized AI models:
- Math: SymPy verified LaTeX equivalence result
- Chemistry: RDKit Tanimoto similarity score  
- Diagrams: CLIP visual similarity + structural analysis
- Code: Static analysis + execution test results
- Text: Semantic similarity score from sentence-transformers

Your job is to make the FINAL evaluation decision using all these signals
combined with your own understanding. Do not ignore the pre-computed signals.

STRICT RULES:
1. Match answers to questions by question number written by student, NEVER by position
2. 'Not Attempted' questions get 0 marks with note "Not Attempted"
3. Alternative correct approaches get FULL marks (especially in code and math)
4. Partial credit must be granular — don't just give half marks, justify each mark
5. Feedback must be constructive, specific, and educationally valuable
6. For mixed-language answers: evaluate in English regardless of answer language
7. NEVER penalize for messy handwriting if the content is correct
8. If a student makes a conceptual error but correct calculation on wrong premise:
   award calculation marks, deduct conceptual marks separately
9. Always return valid JSON. No text outside JSON.
"""

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    @retry(max_retries=3)
    def evaluate_with_context(
        self,
        question_number: str,
        student_image: Image.Image,
        student_text: str,
        content_type: str,
        analyzer_results: Dict[str, Any],
        answer_key_text: str,
        marks_allocated: int,
        few_shot_examples: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Final evaluation of a single question.
        """
        log.info("Starting GPT-4o evaluation for Q%s (Type: %s)", question_number, content_type)
        
        student_b64 = pil_to_base64(student_image)
        
        user_content = f"""
Question {question_number} Evaluation

Content Type: {content_type}
Marks Allocated: {marks_allocated}

Pre-computed Analysis Signals:
{json.dumps(analyzer_results, indent=2)}

Student's Answer Text (OCR extracted):
{student_text}

Answer Key:
{answer_key_text}

Using the pre-computed signals and your own analysis of the attached
student answer image, provide final evaluation as JSON:
{{
  "question_number": "{question_number}",
  "marks_allocated": {marks_allocated},
  "marks_obtained": number,
  "status": "attempted|unattempted|partial",
  "content_type": "{content_type}",
  "pre_analysis_used": true/false,
  "feedback": "string",
  "error_analysis": "string",
  "partial_credit_breakdown": {{"step_or_component": marks, ...}}
}}
"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # Add few-shot examples if provided
        if few_shot_examples:
            for ex in few_shot_examples:
                messages.append({"role": "user", "content": ex["prompt"]})
                messages.append({"role": "assistant", "content": json.dumps(ex["response"])})
                
        # Current question
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_content},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{student_b64}"}
                }
            ]
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=2048,
            temperature=0.1
        )
        
        raw_result = response.choices[0].message.content
        log.debug("GPT-4o Raw Response for Q%s: %s", question_number, raw_result)
        
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            log.error("Failed to parse GPT-4o response as JSON for Q%s", question_number)
            return {
                "question_number": question_number,
                "marks_allocated": marks_allocated,
                "marks_obtained": 0,
                "status": "error",
                "content_type": content_type,
                "feedback": "Internal error: GPT-4o returned malformed response.",
                "error_analysis": "JSON decoding failed."
            }
