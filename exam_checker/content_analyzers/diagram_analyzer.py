"""
Diagram analyzer using CLIP similarity and GPT-4o visual evaluation.

Combines:
  - CLIP visual embedding similarity (if reference image available)
  - GPT-4o structural diagram evaluation (always)
"""

import json
import logging
from typing import Dict, Any, Optional
from threading import Lock

import numpy as np
from PIL import Image

from exam_checker.utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_clip_model = None
_clip_processor = None
_clip_lock = Lock()


def _load_clip():
    """Lazy-load CLIP model and processor."""
    global _clip_model, _clip_processor
    with _clip_lock:
        if _clip_model is None:
            try:
                from transformers import CLIPModel, CLIPProcessor

                _clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                _clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                _clip_model.eval()
                logger.info("CLIP model loaded")
            except Exception as exc:
                logger.warning("CLIP not available: %s", exc)
    return _clip_model, _clip_processor


def analyze_diagram(
    student_image: Image.Image,
    answer_key_image: Optional[Image.Image],
    answer_description: str,
    marks_allocated: int,
    openai_client=None,
) -> Dict[str, Any]:
    """
    Analyze a student's diagram answer.

    Combines CLIP visual similarity (30% weight) with
    GPT-4o structural evaluation (70% weight).

    Args:
        student_image: PIL Image of student's diagram.
        answer_key_image: Optional PIL Image of correct diagram.
        answer_description: Text description of what the correct diagram shows.
        marks_allocated: Maximum marks.
        openai_client: OpenAI client instance (optional, for GPT-4o eval).

    Returns:
        Analysis result dict.
    """
    result = {
        "clip_score": 0.0,
        "gpt4o_accuracy": 0.0,
        "final_score": 0.0,
        "suggested_marks": 0.0,
        "elements_present": [],
        "elements_missing": [],
        "labels_correct": False,
        "feedback": "",
        "method": "none",
    }

    # Step 1: CLIP similarity (if answer key image exists)
    clip_score = 0.0
    has_clip = False

    if answer_key_image is not None:
        clip_score = _compute_clip_similarity(student_image, answer_key_image)
        result["clip_score"] = clip_score
        has_clip = True
        logger.info("CLIP similarity: %.3f", clip_score)

    # Step 2: GPT-4o evaluation (if client available)
    gpt4o_result = {}
    if openai_client is not None:
        gpt4o_result = _evaluate_with_gpt4o(
            student_image, answer_description, openai_client
        )
        result["gpt4o_accuracy"] = gpt4o_result.get("overall_accuracy", 0.0)
        result["elements_present"] = gpt4o_result.get("elements_present", [])
        result["elements_missing"] = gpt4o_result.get("elements_missing", [])
        result["labels_correct"] = gpt4o_result.get("labels_correct", False)
    else:
        # Without GPT-4o, estimate from CLIP alone
        result["gpt4o_accuracy"] = clip_score  # Use CLIP as proxy

    # Step 3: Combine scores
    if has_clip and gpt4o_result:
        final_score = clip_score * 0.3 + result["gpt4o_accuracy"] * 0.7
        result["method"] = "clip+gpt4o"
    elif has_clip:
        final_score = clip_score
        result["method"] = "clip_only"
    elif gpt4o_result:
        final_score = result["gpt4o_accuracy"]
        result["method"] = "gpt4o_only"
    else:
        final_score = 0.0
        result["method"] = "none"

    result["final_score"] = round(final_score, 3)
    result["suggested_marks"] = round(marks_allocated * final_score, 1)

    # Build feedback
    feedback_parts = []
    if has_clip:
        if clip_score >= 0.90:
            feedback_parts.append("Strong visual match with reference diagram.")
        elif clip_score >= 0.75:
            feedback_parts.append("Partial visual match with reference diagram.")
        else:
            feedback_parts.append("Poor visual match with reference diagram.")

    if gpt4o_result.get("feedback"):
        feedback_parts.append(gpt4o_result["feedback"])

    result["feedback"] = " ".join(feedback_parts) if feedback_parts else "Diagram analysis incomplete."

    return result


def _compute_clip_similarity(
    image1: Image.Image,
    image2: Image.Image,
) -> float:
    """
    Compute CLIP cosine similarity between two images.

    Args:
        image1: First PIL Image.
        image2: Second PIL Image.

    Returns:
        Cosine similarity score (0.0 to 1.0).
    """
    model, processor = _load_clip()
    if model is None or processor is None:
        return 0.0

    try:
        import torch

        # Ensure RGB
        if image1.mode != "RGB":
            image1 = image1.convert("RGB")
        if image2.mode != "RGB":
            image2 = image2.convert("RGB")

        inputs1 = processor(images=image1, return_tensors="pt")
        inputs2 = processor(images=image2, return_tensors="pt")

        with torch.no_grad():
            emb1 = model.get_image_features(**inputs1)
            emb2 = model.get_image_features(**inputs2)

        # Normalize
        emb1 = emb1 / emb1.norm(dim=-1, keepdim=True)
        emb2 = emb2 / emb2.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()

        # Clamp to [0, 1]
        return max(0.0, min(1.0, float(similarity)))

    except Exception as exc:
        logger.error("CLIP similarity computation failed: %s", exc)
        return 0.0


def _evaluate_with_gpt4o(
    student_image: Image.Image,
    answer_description: str,
    openai_client,
) -> Dict[str, Any]:
    """
    Evaluate a diagram using GPT-4o vision.

    Args:
        student_image: PIL Image of student's diagram.
        answer_description: What the correct diagram should show.
        openai_client: OpenAI client instance.

    Returns:
        Dict with elements_present, elements_missing, labels_correct,
        overall_accuracy, feedback.
    """
    try:
        b64_image = image_to_base64(student_image)

        prompt = f"""This is a student's hand-drawn diagram in a university exam.
The correct diagram should show: {answer_description}

Evaluate this diagram and return ONLY a JSON object:
{{
    "elements_present": ["list of correctly drawn elements"],
    "elements_missing": ["list of missing or wrong elements"],
    "labels_correct": true/false,
    "overall_accuracy": 0.0 to 1.0,
    "feedback": "specific feedback for student"
}}"""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=800,
            temperature=0.1,
        )

        response_text = response.choices[0].message.content.strip()

        # Parse JSON — try to extract from markdown code blocks if present
        if "```" in response_text:
            import re
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)

        parsed = json.loads(response_text)
        return parsed

    except json.JSONDecodeError:
        logger.warning("GPT-4o returned invalid JSON for diagram evaluation")
        return {"overall_accuracy": 0.5, "feedback": "Could not parse evaluation."}
    except Exception as exc:
        logger.error("GPT-4o diagram evaluation failed: %s", exc)
        return {"overall_accuracy": 0.0, "feedback": f"Evaluation error: {exc}"}
