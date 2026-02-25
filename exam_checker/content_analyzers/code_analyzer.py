"""
Code answer analyzer.

Pipeline: language detection → static analysis → sandbox execution → GPT-4o review.
"""

import re
import io
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def analyze_code(
    student_text: str,
    answer_code: str,
    language: str = "auto",
    marks_allocated: int = 10,
    test_cases: Optional[List[Dict]] = None,
    openai_client=None,
) -> Dict[str, Any]:
    """
    Analyze a student's code answer.

    Args:
        student_text: Student's code as extracted text.
        answer_code: Reference correct code.
        language: Programming language ("auto" for detection).
        marks_allocated: Maximum marks.
        test_cases: Optional list of test case dicts.
        openai_client: OpenAI client for GPT-4o review.

    Returns:
        Analysis result dict.
    """
    result = {
        "language": language, "syntax_errors": [],
        "static_analysis_score": 0.0, "execution_passed": False,
        "execution_output": "", "test_results": [],
        "gpt4o_evaluation": {}, "suggested_marks": 0.0,
        "feedback": "", "method": "none",
    }
    if not student_text.strip():
        result["feedback"] = "No code found in the answer."
        return result

    if language == "auto":
        language = _detect_language(student_text)
    result["language"] = language

    static_result = _run_static_analysis(student_text, language)
    result["syntax_errors"] = static_result.get("errors", [])
    result["static_analysis_score"] = static_result.get("score", 0.0)

    execution_marks = 0.0
    if language == "python":
        exec_result = _sandbox_execute(student_text, test_cases)
        result["execution_passed"] = exec_result.get("passed", False)
        result["execution_output"] = exec_result.get("output", "")
        result["test_results"] = exec_result.get("test_results", [])
        if exec_result["passed"]:
            execution_marks = float(marks_allocated)
        elif exec_result.get("partial_pass", 0) > 0:
            total = max(exec_result.get("total_tests", 1), 1)
            execution_marks = marks_allocated * (exec_result["partial_pass"] / total)

    gpt4o_marks = 0.0
    if openai_client is not None:
        gpt4o_result = _gpt4o_code_review(student_text, answer_code, language, openai_client)
        result["gpt4o_evaluation"] = gpt4o_result
        gpt4o_marks = min(gpt4o_result.get("suggested_marks", 0.0), marks_allocated)

    result["suggested_marks"] = round(max(execution_marks, gpt4o_marks), 1)
    fb = []
    if result["syntax_errors"]:
        fb.append(f"Syntax issues: {'; '.join(result['syntax_errors'][:3])}")
    if result["execution_passed"]:
        fb.append("Code executes correctly.")
    if result["gpt4o_evaluation"].get("feedback"):
        fb.append(result["gpt4o_evaluation"]["feedback"])
    result["feedback"] = " ".join(fb) if fb else "Code analysis complete."
    result["method"] = "static+execution+gpt4o" if openai_client else "static+execution"
    return result


def _detect_language(code: str) -> str:
    """Detect programming language from code content."""
    cl = code.lower()
    if any(kw in cl for kw in ["def ", "import ", "print(", "elif "]):
        return "python"
    if any(kw in cl for kw in ["#include", "printf(", "int main"]):
        return "cpp" if ("cout" in cl or "std::" in cl) else "c"
    if any(kw in cl for kw in ["public class", "system.out.println"]):
        return "java"
    if any(kw in cl for kw in ["select ", "insert into", "create table"]):
        return "sql"
    if any(kw in cl for kw in ["function ", "console.log", "=>"]):
        return "javascript"
    return "unknown"


def _run_static_analysis(code: str, language: str) -> Dict[str, Any]:
    """Run static analysis on code."""
    errors = []
    score = 1.0
    if language == "python":
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as exc:
            errors.append(f"SyntaxError line {exc.lineno}: {exc.msg}")
            score -= 0.3
    elif language == "sql":
        try:
            import sqlparse
            parsed = sqlparse.parse(code)
            if not parsed:
                errors.append("Could not parse SQL")
                score = 0.3
        except ImportError:
            pass
    elif language in ("c", "cpp", "java"):
        if code.count("{") != code.count("}"):
            errors.append("Unbalanced braces")
            score -= 0.2
        if code.count("(") != code.count(")"):
            errors.append("Unbalanced parentheses")
            score -= 0.1
    return {"errors": errors, "score": max(0.0, min(1.0, score))}


def _sandbox_execute(code: str, test_cases: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Execute Python code in a restricted sandbox."""
    result = {"passed": False, "output": "", "test_results": [],
              "partial_pass": 0, "total_tests": 0, "error": None}
    try:
        from RestrictedPython import compile_restricted, safe_globals
        from RestrictedPython.Guards import safe_builtins, guarded_unpack_sequence
        from RestrictedPython.Eval import default_guarded_getiter, default_guarded_getitem
    except ImportError:
        logger.warning("RestrictedPython not available")
        return result

    rg = safe_globals.copy()
    rg["__builtins__"] = safe_builtins.copy()
    rg["_getiter_"] = default_guarded_getiter
    rg["_getitem_"] = default_guarded_getitem
    rg["_unpack_sequence_"] = guarded_unpack_sequence
    rg["_write_"] = lambda x: x
    rg["_inplacevar_"] = lambda op, x, y: op(x, y)

    try:
        byte_code = compile_restricted(code, "<student>", "exec")
    except SyntaxError as exc:
        result["error"] = f"SyntaxError: {exc}"
        return result

    if not test_cases:
        try:
            captured = io.StringIO()
            rg["__builtins__"]["print"] = lambda *a, **k: captured.write(
                " ".join(str(x) for x in a) + k.get("end", "\n"))
            exec(byte_code, rg)
            result["output"] = captured.getvalue()
            result["passed"] = True
        except Exception as exc:
            result["error"] = str(exc)
        return result

    result["total_tests"] = len(test_cases)
    passed = 0
    for i, tc in enumerate(test_cases):
        tr = {"test_num": i + 1, "passed": False, "output": "", "expected": ""}
        try:
            cap = io.StringIO()
            tg = rg.copy()
            tg["__builtins__"] = rg["__builtins__"].copy()
            tg["__builtins__"]["print"] = lambda *a, **k: cap.write(
                " ".join(str(x) for x in a) + k.get("end", "\n"))
            if "input" in tc:
                vals = tc["input"].split("\n") if isinstance(tc["input"], str) else tc["input"]
                it = iter(vals)
                tg["__builtins__"]["input"] = lambda p="": next(it, "")
            exec(byte_code, tg)
            out = cap.getvalue().strip()
            exp = str(tc.get("expected_output", "")).strip()
            tr["output"], tr["expected"] = out, exp
            tr["passed"] = out == exp
            if tr["passed"]:
                passed += 1
        except Exception as exc:
            tr["error"] = str(exc)
        result["test_results"].append(tr)
    result["partial_pass"] = passed
    result["passed"] = passed == len(test_cases)
    return result


def _gpt4o_code_review(student_code: str, answer_code: str,
                       language: str, openai_client) -> Dict[str, Any]:
    """Review code using GPT-4o."""
    try:
        prompt = (f"Evaluate this student's code.\nStudent:\n```{language}\n{student_code}\n```\n"
                  f"Reference:\n```{language}\n{answer_code}\n```\nLanguage: {language}\n"
                  f"Return ONLY JSON: {{\"logic_correct\":bool,\"syntax_errors\":[],\"time_complexity_correct\":bool,"
                  f"\"alternative_valid_approach\":bool,\"suggested_marks\":number(0-10),\"feedback\":str}}")
        resp = openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}],
            max_tokens=800, temperature=0.1)
        text = resp.choices[0].message.content.strip()
        if "```" in text:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if m:
                text = m.group(1)
        return json.loads(text)
    except Exception as exc:
        logger.error("GPT-4o code review failed: %s", exc)
        return {"suggested_marks": 0, "feedback": f"Review error: {exc}"}
