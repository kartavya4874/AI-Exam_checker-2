"""
Text answer analyzer using sentence-transformers and GPT-4o.

Pipeline: semantic similarity pre-check → GPT-4o few-shot evaluation.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
from threading import Lock

logger = logging.getLogger(__name__)

_st_model = None
_st_lock = Lock()


def _get_st_model():
    """Lazy-load sentence-transformers model."""
    global _st_model
    with _st_lock:
        if _st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SentenceTransformer loaded")
            except Exception as exc:
                logger.warning("sentence-transformers not available: %s", exc)
    return _st_model


def analyze_text(
    student_text: str,
    answer_key_text: str,
    teacher_patterns: Optional[List[Dict]] = None,
    marks_allocated: int = 10,
    openai_client=None,
) -> Dict[str, Any]:
    """
    Analyze a text/theory answer.

    Args:
        student_text: Student's answer text.
        answer_key_text: Correct answer text.
        teacher_patterns: Few-shot examples from teacher samples.
        marks_allocated: Maximum marks.
        openai_client: OpenAI client for GPT-4o evaluation.

    Returns:
        Analysis result dict.
    """
    result = {
        "similarity_score": 0.0, "suggested_marks": 0.0,
        "feedback": "", "method": "none", "confidence": 0.0,
    }

    if not student_text.strip():
        result["feedback"] = "No text content found."
        return result

    # Step 1: Semantic similarity
    similarity = _compute_similarity(student_text, answer_key_text)
    result["similarity_score"] = similarity

    # Step 2: GPT-4o evaluation with few-shot
    if openai_client is not None:
        gpt_result = _gpt4o_evaluate(
            student_text, answer_key_text, similarity,
            teacher_patterns or [], marks_allocated, openai_client
        )
        result.update(gpt_result)
        result["method"] = "similarity+gpt4o"
    else:
        # Fallback: similarity-based scoring
        result["suggested_marks"] = round(marks_allocated * similarity, 1)
        result["feedback"] = f"Semantic similarity: {similarity:.2f}"
        result["method"] = "similarity_only"
        result["confidence"] = 0.5

    return result


def _compute_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts."""
    model = _get_st_model()
    if model is None:
        return 0.0
    try:
        import numpy as np
        emb1 = model.encode(text1, normalize_embeddings=True)
        emb2 = model.encode(text2, normalize_embeddings=True)
        sim = float(np.dot(emb1, emb2))
        return max(0.0, min(1.0, sim))
    except Exception as exc:
        logger.error("Similarity computation failed: %s", exc)
        return 0.0


def _gpt4o_evaluate(
    student_text: str, answer_key: str, similarity: float,
    teacher_patterns: List[Dict], marks_allocated: int,
    openai_client,
) -> Dict[str, Any]:
    """Evaluate text answer using GPT-4o with few-shot patterns."""
    try:
        messages = [{"role": "system", "content": (
            "You are an expert university examiner. Evaluate student answers "
            "against the answer key. Award partial credit granularly. "
            "Return ONLY valid JSON."
        )}]

        # Add few-shot examples (max 5)
        for ex in teacher_patterns[:5]:
            messages.append({"role": "user", "content": (
                f"Question: {ex.get('question', '')}\n"
                f"Student Answer: {ex.get('answer', '')}\n"
                f"Marks Available: {ex.get('marks', marks_allocated)}"
            )})
            messages.append({"role": "assistant", "content": json.dumps({
                "marks_obtained": ex.get("marks_awarded", 0),
                "feedback": ex.get("feedback", ""),
            })})

        hint = ""
        if similarity > 0.92:
            hint = f" Pre-computed semantic similarity is HIGH ({similarity:.2f})."
        elif similarity < 0.30:
            hint = f" Pre-computed semantic similarity is LOW ({similarity:.2f})."

        messages.append({"role": "user", "content": (
            f"Answer Key: {answer_key}\n\nStudent Answer: {student_text}\n\n"
            f"Marks Available: {marks_allocated}{hint}\n\n"
            f"Return JSON: {{\"marks_obtained\": number, \"feedback\": str, "
            f"\"key_points_covered\": [str], \"key_points_missing\": [str]}}"
        )})

        resp = openai_client.chat.completions.create(
            model="gpt-4o", messages=messages,
            max_tokens=600, temperature=0.1,
        )
        text = resp.choices[0].message.content.strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)
        parsed = json.loads(text)
        return {
            "suggested_marks": min(parsed.get("marks_obtained", 0), marks_allocated),
            "feedback": parsed.get("feedback", ""),
            "key_points_covered": parsed.get("key_points_covered", []),
            "key_points_missing": parsed.get("key_points_missing", []),
            "confidence": 0.85,
        }
    except Exception as exc:
        logger.error("GPT-4o text evaluation failed: %s", exc)
        return {"suggested_marks": 0, "feedback": f"Evaluation error: {exc}", "confidence": 0.0}
