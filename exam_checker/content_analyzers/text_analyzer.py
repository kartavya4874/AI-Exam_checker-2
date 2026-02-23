"""
Text analyzer — sentence-transformers semantic similarity + GPT-4o
few-shot evaluation for theory answers.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

from exam_checker.utils.logger import get_logger
from exam_checker.utils.retry_utils import retry

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton sentence-transformer model
# ---------------------------------------------------------------------------
_st_model = None


def _load_st():
    """Lazy-load the sentence-transformers model."""
    global _st_model
    if _st_model is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer
        from exam_checker.config import SENTENCE_MODEL

        log.info("Loading sentence-transformer model: %s…", SENTENCE_MODEL)
        _st_model = SentenceTransformer(SENTENCE_MODEL)
        log.info("Sentence-transformer loaded")
    except Exception as exc:
        log.warning("sentence-transformers not available: %s", exc)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

def _compute_similarity(student_text: str, answer_text: str) -> float:
    """Return cosine similarity between student and key text embeddings."""
    _load_st()
    if _st_model is None:
        return 0.0

    emb_student = _st_model.encode(student_text)
    emb_key = _st_model.encode(answer_text)
    sim = _cosine_similarity(emb_student, emb_key)
    log.debug("Semantic similarity: %.4f", sim)
    return sim


# ---------------------------------------------------------------------------
# GPT-4o evaluation
# ---------------------------------------------------------------------------

def _build_few_shot_messages(
    teacher_patterns: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Build conversation-format few-shot examples from teacher patterns.

    Each pattern is a dict with ``question``, ``answer``, ``marks``, ``feedback``.
    """
    messages: List[Dict[str, str]] = []
    for pat in teacher_patterns[:5]:  # max 5
        messages.append({
            "role": "user",
            "content": f"Question: {pat.get('question', '')}\n"
                       f"Student answer: {pat.get('answer', '')}\n"
                       f"Marks allocated: {pat.get('marks', '10')}",
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "marks_obtained": pat.get("marks_awarded", 0),
                "feedback": pat.get("feedback", ""),
            }),
        })
    return messages


def _gpt4o_evaluate_text(
    student_text: str,
    answer_key_text: str,
    similarity_score: float,
    marks_allocated: int,
    teacher_patterns: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Send text answer to GPT-4o with few-shot teacher patterns."""
    from openai import OpenAI
    from exam_checker.config import OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are an expert university examiner evaluating a student's theory answer. "
        "A semantic similarity model has pre-computed a score between the student answer "
        "and the answer key. Consider this as one signal but make your own thorough judgment.\n"
        "STRICT RULES:\n"
        "1. Alternative correct approaches get FULL marks.\n"
        "2. Partial credit must be granular and justified.\n"
        "3. Feedback must be constructive and educationally valuable.\n"
        "4. Never penalize for language/grammar if content is correct.\n"
        "5. Always return valid JSON only."
    )

    few_shot = _build_few_shot_messages(teacher_patterns)

    # Simplify prompt if similarity is very high
    if similarity_score > 0.92:
        hint = (
            f"Pre-computed semantic similarity is very high ({similarity_score:.2f}), "
            f"suggesting a strong match. Verify content and award marks accordingly."
        )
    elif similarity_score < 0.30:
        hint = (
            f"Pre-computed semantic similarity is low ({similarity_score:.2f}), "
            f"suggesting a potential mismatch. Evaluate carefully for alternative phrasings."
        )
    else:
        hint = f"Pre-computed semantic similarity: {similarity_score:.2f}."

    user_msg = (
        f"{hint}\n\n"
        f"Answer Key:\n{answer_key_text}\n\n"
        f"Student Answer:\n{student_text}\n\n"
        f"Marks Allocated: {marks_allocated}\n\n"
        f"Return JSON:\n"
        f'{{"marks_obtained": number, "feedback": "string", "key_points_covered": '
        f'["list"], "key_points_missed": ["list"]}}'
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(few_shot)
    messages.append({"role": "user", "content": user_msg})

    @retry(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def _call():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content

    try:
        raw = _call()
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "marks_obtained": 0,
            "feedback": raw if isinstance(raw, str) else "Evaluation unavailable.",
            "key_points_covered": [],
            "key_points_missed": [],
        }
    except Exception as exc:
        log.error("GPT-4o text evaluation failed: %s", exc)
        return {
            "marks_obtained": 0,
            "feedback": f"Evaluation failed: {exc}",
            "key_points_covered": [],
            "key_points_missed": [],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_text(
    student_text: str,
    answer_key_text: str,
    teacher_patterns: Optional[List[Dict[str, str]]] = None,
    marks_allocated: int = 10,
) -> Dict[str, Any]:
    """
    Evaluate a theory/text answer.

    Pipeline:
      1. Sentence-transformers cosine similarity.
      2. GPT-4o few-shot evaluation with teacher patterns.
      3. Merge results.

    Args:
        student_text: OCR-extracted student answer.
        answer_key_text: Reference answer from the key.
        teacher_patterns: Pre-graded examples for few-shot context.
        marks_allocated: Total marks.

    Returns:
        Evaluation dict.
    """
    if teacher_patterns is None:
        teacher_patterns = []

    # Semantic similarity
    similarity = _compute_similarity(student_text, answer_key_text)

    # GPT-4o evaluation
    gpt_result = _gpt4o_evaluate_text(
        student_text, answer_key_text, similarity, marks_allocated, teacher_patterns,
    )

    suggested_marks = float(gpt_result.get("marks_obtained", 0))

    return {
        "similarity_score": similarity,
        "gpt4o_marks": suggested_marks,
        "suggested_marks": suggested_marks,
        "feedback": gpt_result.get("feedback", ""),
        "key_points_covered": gpt_result.get("key_points_covered", []),
        "key_points_missed": gpt_result.get("key_points_missed", []),
        "method": "sentence_transformers+gpt4o",
    }
