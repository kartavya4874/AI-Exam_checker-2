"""
GPT-4o evaluation engine with few-shot prompting.

The final evaluator that combines pre-computed analyzer signals
with GPT-4o vision for the ultimate grading decision.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional

from PIL import Image

from exam_checker.utils.image_utils import image_to_base64
from exam_checker.utils.retry_utils import retry_with_backoff, TokenBucket

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert university examiner with 20 years of experience.
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
9. Always return valid JSON. No text outside JSON."""

# Rate limiter
_token_bucket = TokenBucket(tokens_per_minute=50)


class GPT4OEvaluator:
    """GPT-4o based final evaluator."""

    def __init__(self, openai_client):
        """
        Initialize with an OpenAI client.

        Args:
            openai_client: Configured openai.OpenAI instance.
        """
        self.client = openai_client

    @retry_with_backoff(max_retries=3, base_delay=2.0)
    def evaluate_with_context(
        self,
        question_number: str,
        student_image: Optional[Image.Image],
        student_text: str,
        content_type: str,
        analyzer_results: Dict[str, Any],
        answer_key_text: str,
        marks_allocated: int,
        few_shot_examples: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Final evaluation combining analyzer signals with GPT-4o.

        Args:
            question_number: Question identifier (e.g. "Q1").
            student_image: PIL Image of student's answer (optional).
            student_text: OCR-extracted text.
            content_type: Classified content type.
            analyzer_results: Pre-computed signals from specialized analyzers.
            answer_key_text: Correct answer text.
            marks_allocated: Maximum marks.
            few_shot_examples: Teacher sample evaluations for few-shot.

        Returns:
            Evaluation result dict.
        """
        _token_bucket.acquire()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add few-shot examples
        if few_shot_examples:
            for ex in few_shot_examples[:5]:
                messages.append({"role": "user", "content": ex.get("user_content", "")})
                messages.append({"role": "assistant", "content": ex.get("assistant_content", "")})

        # Build user message
        user_content = self._build_user_message(
            question_number, student_text, content_type,
            analyzer_results, answer_key_text, marks_allocated,
        )

        # Build content list (text + optional image)
        content_parts = [{"type": "text", "text": user_content}]

        if student_image is not None:
            try:
                b64 = image_to_base64(student_image)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
            except Exception as exc:
                logger.warning("Could not encode image: %s", exc)

        messages.append({"role": "user", "content": content_parts})

        # Call GPT-4o
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        result = self._parse_response(response_text, question_number, marks_allocated)

        # Validate and retry if needed
        if result is None:
            result = self._retry_json_parse(messages, question_number, marks_allocated)

        return result

    def _build_user_message(self, qnum, text, ctype, analyzer, key, marks):
        """Build the evaluation prompt."""
        analyzer_str = json.dumps(analyzer, indent=2, default=str)
        return f"""Question {qnum} Evaluation

Content Type: {ctype}
Marks Allocated: {marks}

Pre-computed Analysis Signals:
{analyzer_str}

Student's Answer Text (OCR extracted):
{text if text else '[No text extracted — check attached image]'}

Answer Key:
{key if key else '[No answer key provided]'}

Provide final evaluation as JSON:
{{
  "question_number": "{qnum}",
  "marks_allocated": {marks},
  "marks_obtained": <number>,
  "status": "attempted|unattempted|partial",
  "content_type": "{ctype}",
  "pre_analysis_used": true,
  "feedback": "<specific feedback>",
  "error_analysis": "<what went wrong>",
  "partial_credit_breakdown": {{"step_or_component": <marks>}}
}}"""

    def _parse_response(self, text, qnum, marks):
        """Parse GPT-4o JSON response."""
        try:
            if "```" in text:
                m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                if m:
                    text = m.group(1)
            result = json.loads(text)
            # Ensure marks don't exceed allocated
            result["marks_obtained"] = min(
                result.get("marks_obtained", 0), marks
            )
            return result
        except (json.JSONDecodeError, KeyError):
            logger.warning("Invalid JSON from GPT-4o for %s", qnum)
            return None

    def _retry_json_parse(self, messages, qnum, marks):
        """Retry with explicit JSON instruction."""
        try:
            messages.append({"role": "user", "content": (
                "Your previous response was not valid JSON. "
                "Return ONLY the JSON object, nothing else."
            )})
            resp = self.client.chat.completions.create(
                model="gpt-4o", messages=messages,
                max_tokens=800, temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            return self._parse_response(text, qnum, marks) or self._default_result(qnum, marks)
        except Exception:
            return self._default_result(qnum, marks)

    @staticmethod
    def _default_result(qnum, marks):
        """Return a safe default when GPT-4o fails."""
        return {
            "question_number": qnum, "marks_allocated": marks,
            "marks_obtained": 0, "status": "attempted",
            "content_type": "unknown", "pre_analysis_used": False,
            "feedback": "Automated evaluation could not be completed. Manual review needed.",
            "error_analysis": "GPT-4o response parsing failed",
            "partial_credit_breakdown": {},
        }
