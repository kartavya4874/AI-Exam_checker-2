"""
Diagram analyzer — CLIP visual similarity + GPT-4o description evaluation.

Computes cosine similarity between student and answer key diagram embeddings
using CLIP, then sends to GPT-4o for detailed structural evaluation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from exam_checker.config import CLIP_MODEL, CLIP_SIMILARITY_WEIGHT, GPT4O_WEIGHT
from exam_checker.utils.image_utils import pil_to_base64
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Singleton CLIP model
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None


def _load_clip():
    """Lazy-load CLIP model and processor."""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return

    try:
        from transformers import CLIPModel, CLIPProcessor

        log.info("Loading CLIP model: %s…", CLIP_MODEL)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
        _clip_model.eval()
        log.info("CLIP model loaded")
    except Exception as exc:
        log.warning("CLIP loading failed: %s", exc)


# ---------------------------------------------------------------------------
# CLIP similarity
# ---------------------------------------------------------------------------

def _clip_similarity(img_a: Image.Image, img_b: Image.Image) -> float:
    """Compute cosine similarity between two images using CLIP embeddings."""
    _load_clip()
    if _clip_model is None:
        return 0.0

    import torch

    try:
        inputs_a = _clip_processor(images=img_a.convert("RGB"), return_tensors="pt")
        inputs_b = _clip_processor(images=img_b.convert("RGB"), return_tensors="pt")

        with torch.no_grad():
            emb_a = _clip_model.get_image_features(**inputs_a)
            emb_b = _clip_model.get_image_features(**inputs_b)

        # Normalise
        emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
        emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)

        sim = (emb_a @ emb_b.T).item()
        log.debug("CLIP cosine similarity: %.4f", sim)
        return float(sim)
    except Exception as exc:
        log.warning("CLIP similarity computation failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# GPT-4o diagram evaluation
# ---------------------------------------------------------------------------

def _gpt4o_evaluate_diagram(
    student_image: Image.Image,
    answer_description: str,
) -> Dict[str, Any]:
    """
    Send the student diagram to GPT-4o with the correct description
    and get a structured evaluation.
    """
    from openai import OpenAI
    from exam_checker.config import OPENAI_API_KEY
    from exam_checker.utils.retry_utils import retry

    client = OpenAI(api_key=OPENAI_API_KEY)
    b64 = pil_to_base64(student_image)

    prompt = f"""This is a student's hand-drawn diagram in a university exam.
The correct diagram should show: {answer_description}

Evaluate this diagram and return JSON:
{{
  "elements_present": ["list of correctly drawn elements"],
  "elements_missing": ["list of missing or wrong elements"],
  "labels_correct": true/false,
  "overall_accuracy": 0.0 to 1.0,
  "feedback": "specific feedback for student"
}}"""

    @retry(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def _call():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content

    try:
        raw = _call()
        # Try to parse JSON from the response
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        result = json.loads(cleaned)
        return result
    except json.JSONDecodeError:
        log.warning("GPT-4o returned invalid JSON for diagram evaluation")
        return {
            "elements_present": [],
            "elements_missing": [],
            "labels_correct": False,
            "overall_accuracy": 0.0,
            "feedback": raw if isinstance(raw, str) else "Evaluation unavailable.",
        }
    except Exception as exc:
        log.error("GPT-4o diagram evaluation failed: %s", exc)
        return {
            "elements_present": [],
            "elements_missing": [],
            "labels_correct": False,
            "overall_accuracy": 0.0,
            "feedback": f"Evaluation failed: {exc}",
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_diagram(
    student_image: Image.Image,
    answer_key_image: Optional[Image.Image],
    answer_description: str,
    marks_allocated: int,
) -> Dict[str, Any]:
    """
    Evaluate a student's diagram using CLIP similarity and GPT-4o.

    Args:
        student_image: Student's diagram as PIL Image.
        answer_key_image: Correct diagram (optional).
        answer_description: Textual description of the correct diagram.
        marks_allocated: Total marks.

    Returns:
        Evaluation dict.
    """
    # CLIP similarity (if answer key image exists)
    clip_score = 0.0
    if answer_key_image is not None:
        clip_score = _clip_similarity(student_image, answer_key_image)

    # GPT-4o evaluation (always)
    gpt_result = _gpt4o_evaluate_diagram(student_image, answer_description)
    gpt4o_accuracy = gpt_result.get("overall_accuracy", 0.0)

    # Combine scores
    if answer_key_image is not None:
        final_score = (clip_score * CLIP_SIMILARITY_WEIGHT + gpt4o_accuracy * GPT4O_WEIGHT)
    else:
        final_score = gpt4o_accuracy  # No answer key image → GPT-4o only

    suggested_marks = round(marks_allocated * final_score, 1)

    # Interpret CLIP score
    clip_interpretation = "N/A"
    if answer_key_image is not None:
        if clip_score >= 0.90:
            clip_interpretation = "Strong visual match"
        elif clip_score >= 0.75:
            clip_interpretation = "Partial visual match"
        else:
            clip_interpretation = "Poor visual match"

    return {
        "clip_score": clip_score,
        "clip_interpretation": clip_interpretation,
        "gpt4o_accuracy": gpt4o_accuracy,
        "elements_present": gpt_result.get("elements_present", []),
        "elements_missing": gpt_result.get("elements_missing", []),
        "labels_correct": gpt_result.get("labels_correct", False),
        "final_score": final_score,
        "suggested_marks": suggested_marks,
        "feedback": gpt_result.get("feedback", ""),
        "method": "clip+gpt4o" if answer_key_image else "gpt4o_only",
    }
