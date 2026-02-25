"""
Chemistry analyzer using DECIMER and RDKit.

Handles:
  - Chemical structure recognition (DECIMER → SMILES → RDKit comparison)
  - Chemical equation balancing and comparison
"""

import re
import logging
import tempfile
from typing import Dict, Any, List, Tuple
from collections import Counter

from PIL import Image

logger = logging.getLogger(__name__)


def analyze_structure(
    student_image: Image.Image,
    answer_smiles: str,
    marks_allocated: int,
) -> Dict[str, Any]:
    """
    Analyze a chemical structure answer using DECIMER and RDKit.

    Pipeline:
      1. DECIMER prediction → student SMILES
      2. RDKit fingerprint comparison → Tanimoto similarity
      3. Score based on similarity thresholds

    Args:
        student_image: PIL Image of student's drawn structure.
        answer_smiles: SMILES string of the correct structure.
        marks_allocated: Maximum marks for this question.

    Returns:
        Analysis result dict.
    """
    result = {
        "student_smiles": "",
        "answer_smiles": answer_smiles,
        "similarity": 0.0,
        "suggested_marks": 0.0,
        "feedback": "",
        "method": "none",
        "is_valid_student": False,
        "is_valid_answer": False,
    }

    # Step 1: DECIMER SMILES prediction
    student_smiles = _predict_smiles_decimer(student_image)
    result["student_smiles"] = student_smiles

    if not student_smiles:
        # Fallback: will be handled by GPT-4o in the orchestrator
        result["feedback"] = (
            "Could not recognize chemical structure. "
            "Falling back to GPT-4o visual evaluation."
        )
        result["method"] = "decimer_failed"
        return result

    # Step 2: RDKit comparison
    similarity = _compare_structures_rdkit(student_smiles, answer_smiles)
    result["similarity"] = similarity
    result["method"] = "rdkit_tanimoto"

    # Validate molecules
    result["is_valid_student"] = _is_valid_smiles(student_smiles)
    result["is_valid_answer"] = _is_valid_smiles(answer_smiles)

    # Step 3: Score based on similarity
    if similarity >= 0.95:
        result["suggested_marks"] = float(marks_allocated)
        result["feedback"] = "Chemical structure is correct (exact or near-exact match)."
    elif similarity >= 0.80:
        result["suggested_marks"] = round(marks_allocated * 0.75, 1)
        result["feedback"] = (
            f"Structure is mostly correct (similarity: {similarity:.2f}). "
            "Minor structural differences detected."
        )
    elif similarity >= 0.60:
        result["suggested_marks"] = round(marks_allocated * 0.50, 1)
        result["feedback"] = (
            f"Related structure with significant differences (similarity: {similarity:.2f})."
        )
    else:
        result["suggested_marks"] = 0.0
        result["feedback"] = (
            f"Structure does not match expected answer (similarity: {similarity:.2f})."
        )

    return result


def analyze_equation(
    student_text: str,
    answer_text: str,
    marks_allocated: int,
) -> Dict[str, Any]:
    """
    Analyze a chemical equation answer.

    Compares reactants, products, and coefficients, and checks balancing.

    Args:
        student_text: Student's equation text (from OCR).
        answer_text: Correct equation text.
        marks_allocated: Maximum marks.

    Returns:
        Analysis result dict.
    """
    result = {
        "student_equation": student_text,
        "answer_equation": answer_text,
        "student_balanced": False,
        "answer_balanced": False,
        "equations_match": False,
        "suggested_marks": 0.0,
        "feedback": "",
        "student_elements": {},
        "answer_elements": {},
    }

    # Parse both equations
    student_parsed = _parse_equation(student_text)
    answer_parsed = _parse_equation(answer_text)

    if not student_parsed:
        result["feedback"] = "Could not parse student's chemical equation."
        return result

    # Check if student equation is balanced
    student_balanced = _is_balanced(student_parsed)
    answer_balanced = _is_balanced(answer_parsed) if answer_parsed else False
    result["student_balanced"] = student_balanced
    result["answer_balanced"] = answer_balanced

    # Compare equations
    equations_match = _equations_equivalent(student_parsed, answer_parsed)
    result["equations_match"] = equations_match

    # Element analysis
    if student_parsed:
        result["student_elements"] = {
            "reactants": student_parsed.get("reactant_elements", {}),
            "products": student_parsed.get("product_elements", {}),
        }
    if answer_parsed:
        result["answer_elements"] = {
            "reactants": answer_parsed.get("reactant_elements", {}),
            "products": answer_parsed.get("product_elements", {}),
        }

    # Scoring
    marks = 0.0
    feedback_parts = []

    if equations_match and student_balanced:
        marks = float(marks_allocated)
        feedback_parts.append("Equation is correct and balanced.")
    elif equations_match and not student_balanced:
        marks = marks_allocated * 0.6
        feedback_parts.append("Correct reactants and products, but equation is not balanced.")
    elif student_balanced and not equations_match:
        marks = marks_allocated * 0.3
        feedback_parts.append(
            "Equation is balanced but does not match the expected reaction."
        )
    else:
        # Partial credit for correct elements
        if student_parsed and answer_parsed:
            s_elements = set(student_parsed.get("all_elements", []))
            a_elements = set(answer_parsed.get("all_elements", []))
            common = s_elements & a_elements
            if common:
                ratio = len(common) / max(len(a_elements), 1)
                marks = marks_allocated * ratio * 0.3
                feedback_parts.append(
                    f"Partially correct: {len(common)}/{len(a_elements)} elements match."
                )
            else:
                feedback_parts.append("Equation does not match expected answer.")

    result["suggested_marks"] = round(marks, 1)
    result["feedback"] = " ".join(feedback_parts)

    return result


def _predict_smiles_decimer(image: Image.Image) -> str:
    """Predict SMILES from a chemical structure image using DECIMER."""
    try:
        from DECIMER import predict_SMILES

        # DECIMER requires a file path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            smiles = predict_SMILES(f.name)

        if smiles:
            logger.info("DECIMER predicted: %s", smiles)
            return smiles.strip()
        return ""

    except ImportError:
        logger.warning("DECIMER not installed. Run: pip install DECIMER")
        return ""
    except Exception as exc:
        logger.warning("DECIMER prediction failed: %s", exc)
        return ""


def _compare_structures_rdkit(smiles1: str, smiles2: str) -> float:
    """Compare two SMILES strings using RDKit Tanimoto similarity."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            logger.warning(
                "Invalid SMILES: mol1=%s, mol2=%s",
                "valid" if mol1 else "invalid",
                "valid" if mol2 else "invalid",
            )
            return 0.0

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        logger.debug("Tanimoto similarity: %.4f", similarity)
        return float(similarity)

    except ImportError:
        logger.warning("RDKit not installed")
        return 0.0
    except Exception as exc:
        logger.warning("RDKit comparison failed: %s", exc)
        return 0.0


def _is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is chemically valid."""
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(smiles) is not None
    except Exception:
        return False


def _parse_equation(text: str) -> Dict[str, Any]:
    """
    Parse a chemical equation into structured components.

    Handles formats like:
      2H2 + O2 → 2H2O
      2H2 + O2 = 2H2O
      2H2 + O2 -> 2H2O
    """
    if not text or not text.strip():
        return {}

    # Normalize arrow/equals
    text = text.replace("⟶", "→").replace("->", "→").replace("=>", "→")
    text = text.replace("—>", "→").replace("-->", "→")

    # Split by arrow or equals (that's not part of →)
    if "→" in text:
        parts = text.split("→", 1)
    elif "=" in text:
        parts = text.split("=", 1)
    else:
        return {}

    if len(parts) != 2:
        return {}

    reactant_str = parts[0].strip()
    product_str = parts[1].strip()

    reactants = _parse_side(reactant_str)
    products = _parse_side(product_str)

    reactant_elements = _count_elements(reactants)
    product_elements = _count_elements(products)

    all_elements = sorted(set(list(reactant_elements.keys()) + list(product_elements.keys())))

    return {
        "reactants": reactants,
        "products": products,
        "reactant_elements": reactant_elements,
        "product_elements": product_elements,
        "all_elements": all_elements,
    }


def _parse_side(side_str: str) -> List[Tuple[int, str]]:
    """Parse one side of a chemical equation into (coefficient, formula) tuples."""
    compounds = []
    parts = re.split(r"\s*\+\s*", side_str.strip())

    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract coefficient
        match = re.match(r"^(\d*)\s*(.+)$", part)
        if match:
            coeff_str = match.group(1)
            formula = match.group(2).strip()
            coeff = int(coeff_str) if coeff_str else 1
            compounds.append((coeff, formula))

    return compounds


def _count_elements(compounds: List[Tuple[int, str]]) -> Dict[str, int]:
    """Count total atoms of each element in a list of compounds."""
    total: Counter = Counter()

    for coeff, formula in compounds:
        elements = _parse_formula(formula)
        for elem, count in elements.items():
            total[elem] += coeff * count

    return dict(total)


def _parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a chemical formula into element counts.

    Handles: H2O, Ca(OH)2, Fe2(SO4)3, etc.
    """
    elements: Counter = Counter()
    i = 0
    n = len(formula)
    stack = [elements]

    while i < n:
        if formula[i] == "(":
            new_group: Counter = Counter()
            stack.append(new_group)
            i += 1
        elif formula[i] == ")":
            i += 1
            num_str = ""
            while i < n and formula[i].isdigit():
                num_str += formula[i]
                i += 1
            multiplier = int(num_str) if num_str else 1
            group = stack.pop()
            for elem, count in group.items():
                stack[-1][elem] += count * multiplier
        elif formula[i].isupper():
            elem = formula[i]
            i += 1
            while i < n and formula[i].islower():
                elem += formula[i]
                i += 1
            num_str = ""
            while i < n and formula[i].isdigit():
                num_str += formula[i]
                i += 1
            count = int(num_str) if num_str else 1
            stack[-1][elem] += count
        else:
            i += 1

    return dict(stack[0])


def _is_balanced(parsed: Dict[str, Any]) -> bool:
    """Check if a parsed equation is balanced."""
    if not parsed:
        return False
    r_elem = parsed.get("reactant_elements", {})
    p_elem = parsed.get("product_elements", {})
    return r_elem == p_elem


def _equations_equivalent(
    parsed1: Dict[str, Any],
    parsed2: Dict[str, Any],
) -> bool:
    """Check if two parsed equations represent the same reaction."""
    if not parsed1 or not parsed2:
        return False

    # Compare element counts on both sides
    r1 = parsed1.get("reactant_elements", {})
    p1 = parsed1.get("product_elements", {})
    r2 = parsed2.get("reactant_elements", {})
    p2 = parsed2.get("product_elements", {})

    return r1 == r2 and p1 == p2
