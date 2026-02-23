"""
Chemistry analyzer — DECIMER for structure recognition, RDKit for
comparison, and rule-based equation balancing.

Two main functions:
  - analyze_structure: DECIMER → SMILES → RDKit Tanimoto similarity
  - analyze_equation:  parse chemical equation, check balance, compare
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from exam_checker.utils.image_utils import save_temp_image
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structure analysis (DECIMER + RDKit)
# ---------------------------------------------------------------------------

def _predict_smiles(image: Image.Image) -> str:
    """Run DECIMER to predict SMILES from a chemical structure image."""
    try:
        from DECIMER import predict_SMILES
        tmp_path = save_temp_image(image, prefix="chem_struct_", subdir="chemistry")
        smiles = predict_SMILES(tmp_path)
        log.info("DECIMER predicted SMILES: %s", smiles)
        return smiles
    except Exception as exc:
        log.warning("DECIMER prediction failed: %s", exc)
        return ""


def _tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """Compute Tanimoto similarity between two SMILES using RDKit Morgan fingerprints."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        if mol_a is None or mol_b is None:
            log.warning("Invalid SMILES — cannot compute similarity")
            return 0.0

        fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
        log.debug("Tanimoto similarity: %.4f", sim)
        return float(sim)

    except Exception as exc:
        log.warning("RDKit Tanimoto failed: %s", exc)
        return 0.0


def analyze_structure(
    student_image: Image.Image,
    answer_smiles: str,
    marks_allocated: int,
) -> Dict[str, Any]:
    """
    Evaluate a student's drawn chemical structure against the answer key.

    Pipeline:
      1. DECIMER → SMILES from student image.
      2. RDKit Tanimoto similarity.
      3. Grade thresholds: ≥0.95 → full, ≥0.80 → 75%, ≥0.60 → 50%.
      4. If DECIMER fails: leave blank for GPT-4o fallback.

    Args:
        student_image: Cropped image of the drawn structure.
        answer_smiles: SMILES string of the correct structure.
        marks_allocated: Total marks for this question.

    Returns:
        Evaluation dict.
    """
    student_smiles = _predict_smiles(student_image)

    if not student_smiles:
        return {
            "student_smiles": "",
            "answer_smiles": answer_smiles,
            "similarity": 0.0,
            "suggested_marks": 0.0,
            "method": "decimer_failed",
            "feedback": "Could not recognise the chemical structure. Needs GPT-4o review.",
            "needs_gpt4o_fallback": True,
        }

    similarity = _tanimoto_similarity(student_smiles, answer_smiles)

    if similarity >= 0.95:
        suggested = float(marks_allocated)
        feedback = f"Structure is a near-exact match (similarity={similarity:.2f}). Full marks."
    elif similarity >= 0.80:
        suggested = round(marks_allocated * 0.75, 1)
        feedback = f"Minor structural difference (similarity={similarity:.2f}). 75% marks."
    elif similarity >= 0.60:
        suggested = round(marks_allocated * 0.50, 1)
        feedback = f"Related structure with significant errors (similarity={similarity:.2f}). 50% marks."
    else:
        suggested = 0.0
        feedback = f"Structure does not match (similarity={similarity:.2f}). 0 marks."

    return {
        "student_smiles": student_smiles,
        "answer_smiles": answer_smiles,
        "similarity": similarity,
        "suggested_marks": suggested,
        "method": "decimer_rdkit",
        "feedback": feedback,
        "needs_gpt4o_fallback": False,
    }


# ---------------------------------------------------------------------------
# Chemical equation analysis (rule-based)
# ---------------------------------------------------------------------------

_ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")
_COMPOUND_PATTERN = re.compile(r"(\d*)\s*([A-Z][a-z0-9()]*)")


def _parse_formula(formula: str) -> Counter:
    """
    Parse a chemical formula into element counts.

    ``H2O`` → ``Counter({'H': 2, 'O': 1})``
    ``Ca(OH)2`` → ``Counter({'Ca': 1, 'O': 2, 'H': 2})``
    """
    counts: Counter = Counter()
    # Handle parenthetical groups
    formula = formula.strip()

    # Expand groups like (OH)2
    paren_pat = re.compile(r"\(([^)]+)\)(\d*)")
    while paren_pat.search(formula):
        match = paren_pat.search(formula)
        group_content = match.group(1)
        multiplier = int(match.group(2)) if match.group(2) else 1
        expanded = ""
        for elem_match in _ELEMENT_PATTERN.finditer(group_content):
            elem = elem_match.group(1)
            num = int(elem_match.group(2)) if elem_match.group(2) else 1
            expanded += f"{elem}{num * multiplier}"
        formula = formula[: match.start()] + expanded + formula[match.end() :]

    for match in _ELEMENT_PATTERN.finditer(formula):
        elem = match.group(1)
        if not elem:
            continue
        num = int(match.group(2)) if match.group(2) else 1
        counts[elem] += num

    return counts


def _parse_equation_side(side: str) -> Tuple[List[str], Counter]:
    """
    Parse one side of a chemical equation.

    ``2H2O + O2`` → (["2H2O", "O2"], Counter of total atoms)
    """
    compounds = [c.strip() for c in re.split(r"\+", side) if c.strip()]
    total_counts: Counter = Counter()

    for compound in compounds:
        # Extract coefficient
        coeff_match = re.match(r"(\d+)\s*(.*)", compound)
        if coeff_match:
            coeff = int(coeff_match.group(1))
            formula = coeff_match.group(2)
        else:
            coeff = 1
            formula = compound

        elem_counts = _parse_formula(formula)
        for elem, count in elem_counts.items():
            total_counts[elem] += count * coeff

    return compounds, total_counts


def _split_equation(text: str) -> Tuple[str, str]:
    """Split a chemical equation at the arrow / equals sign."""
    for separator in ["→", "⟶", "->", "=", "⇌", "⟹"]:
        if separator in text:
            parts = text.split(separator, 1)
            return parts[0].strip(), parts[1].strip()
    return text.strip(), ""


def analyze_equation(
    student_text: str,
    answer_text: str,
    marks_allocated: int,
) -> Dict[str, Any]:
    """
    Evaluate a student's chemical equation against the answer key.

    Checks:
      1. Correct products / reactants.
      2. Whether the equation is balanced.
      3. Comparison to answer key.

    Args:
        student_text: Student's equation (OCR text).
        answer_text: Correct equation from the answer key.
        marks_allocated: Total marks.

    Returns:
        Evaluation dict.
    """
    student_left, student_right = _split_equation(student_text)
    key_left, key_right = _split_equation(answer_text)

    if not student_right:
        return {
            "student_equation": student_text,
            "answer_equation": answer_text,
            "is_balanced": False,
            "reactants_correct": False,
            "products_correct": False,
            "suggested_marks": 0.0,
            "feedback": "Could not parse student equation (no arrow/separator found).",
        }

    # Parse both sides
    _, student_l_counts = _parse_equation_side(student_left)
    _, student_r_counts = _parse_equation_side(student_right)
    _, key_l_counts = _parse_equation_side(key_left)
    _, key_r_counts = _parse_equation_side(key_right)

    # Check balance
    is_balanced = student_l_counts == student_r_counts

    # Compare to key
    reactants_correct = set(student_l_counts.keys()) == set(key_l_counts.keys())
    products_correct = set(student_r_counts.keys()) == set(key_r_counts.keys())
    coefficients_match = student_l_counts == key_l_counts and student_r_counts == key_r_counts

    # Scoring
    score = 0.0
    feedback_parts: List[str] = []

    if coefficients_match and is_balanced:
        score = float(marks_allocated)
        feedback_parts.append("Equation fully correct and balanced.")
    else:
        if reactants_correct:
            score += marks_allocated * 0.25
            feedback_parts.append("Correct reactant elements.")
        else:
            feedback_parts.append(f"Reactant elements mismatch: expected {dict(key_l_counts)}.")

        if products_correct:
            score += marks_allocated * 0.25
            feedback_parts.append("Correct product elements.")
        else:
            feedback_parts.append(f"Product elements mismatch: expected {dict(key_r_counts)}.")

        if is_balanced:
            score += marks_allocated * 0.25
            feedback_parts.append("Equation is balanced.")
        else:
            diff_elems = set(student_l_counts.keys()) | set(student_r_counts.keys())
            unbalanced = [
                e for e in diff_elems
                if student_l_counts.get(e, 0) != student_r_counts.get(e, 0)
            ]
            feedback_parts.append(f"Equation is NOT balanced. Unbalanced elements: {unbalanced}.")

        if coefficients_match:
            score += marks_allocated * 0.25
            feedback_parts.append("Coefficients match the key.")

    return {
        "student_equation": student_text,
        "answer_equation": answer_text,
        "is_balanced": is_balanced,
        "reactants_correct": reactants_correct,
        "products_correct": products_correct,
        "coefficients_match": coefficients_match,
        "suggested_marks": round(score, 1),
        "feedback": " ".join(feedback_parts),
    }
