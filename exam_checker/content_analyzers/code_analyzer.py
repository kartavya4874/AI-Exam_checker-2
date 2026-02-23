"""
Code analyzer — static analysis + RestrictedPython sandbox + GPT-4o review.

Evaluates code answers by:
  1. Detecting the programming language
  2. Running static analysis (pylint for Python, regex for others)
  3. Sandbox execution (Python only, via RestrictedPython)
  4. GPT-4o code review (always)
"""

from __future__ import annotations

import io
import json
import re
import sys
from typing import Any, Dict, List, Optional

from exam_checker.utils.logger import get_logger
from exam_checker.utils.retry_utils import retry

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_LANGUAGE_HINTS = {
    "python": [r"\bdef\b", r"\bimport\b", r"\bprint\s*\(", r"\bclass\b.*:"],
    "c": [r"#include\b", r"\bint\s+main\b", r"\bprintf\b"],
    "cpp": [r"#include\b", r"\bcout\b", r"std::", r"using namespace"],
    "java": [r"\bpublic\s+class\b", r"\bSystem\.out", r"\bvoid\s+main\b"],
    "sql": [r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b", r"\bINSERT\b"],
    "html": [r"<html", r"<div", r"<body", r"<script"],
    "javascript": [r"\bfunction\b", r"\bconsole\.log\b", r"\bconst\b", r"\blet\b"],
}


def detect_language(code: str) -> str:
    """
    Auto-detect programming language from code text.

    Returns:
        Language name (``python``, ``c``, ``cpp``, ``java``, ``sql``,
        ``html``, ``javascript``). Defaults to ``python``.
    """
    scores: Dict[str, int] = {}
    for lang, patterns in _LANGUAGE_HINTS.items():
        score = sum(1 for p in patterns if re.search(p, code, re.IGNORECASE))
        if score > 0:
            scores[lang] = score

    if not scores:
        return "python"

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Static analysis
# ---------------------------------------------------------------------------

def _pylint_check(code: str) -> Dict[str, Any]:
    """Run pylint on a code string and capture results."""
    try:
        from pylint.lint import Run
        from pylint.reporters.text import TextReporter

        output = io.StringIO()
        reporter = TextReporter(output)

        import tempfile
        import os

        fd, tmp_path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(code)
            results = Run(
                [tmp_path, "--disable=C,R", "--errors-only"],
                reporter=reporter,
                exit=False,
            )
            score = results.linter.stats.global_note if hasattr(results.linter.stats, "global_note") else 5.0
        finally:
            os.unlink(tmp_path)

        return {
            "pylint_output": output.getvalue(),
            "pylint_score": score,
            "errors": [line for line in output.getvalue().split("\n") if line.strip()],
        }
    except Exception as exc:
        log.warning("Pylint check failed: %s", exc)
        return {"pylint_output": "", "pylint_score": 5.0, "errors": []}


def _sql_check(code: str) -> Dict[str, Any]:
    """Basic SQL syntax check using sqlparse."""
    try:
        import sqlparse
        parsed = sqlparse.parse(code)
        if parsed:
            formatted = sqlparse.format(code, reindent=True)
            return {"valid": True, "formatted": formatted, "errors": []}
        return {"valid": False, "formatted": "", "errors": ["Empty SQL statement"]}
    except Exception as exc:
        return {"valid": False, "formatted": "", "errors": [str(exc)]}


def _basic_syntax_check(code: str, language: str) -> Dict[str, Any]:
    """Regex-based syntax check for C/C++/Java."""
    errors: List[str] = []

    # Check brace balance
    open_braces = code.count("{")
    close_braces = code.count("}")
    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} {{ vs {close_braces} }}")

    # Check parenthesis balance
    open_parens = code.count("(")
    close_parens = code.count(")")
    if open_parens != close_parens:
        errors.append(f"Unbalanced parentheses: {open_parens} ( vs {close_parens} )")

    # Check semicolons at end of statements (very basic)
    if language in ("c", "cpp", "java"):
        lines = code.strip().split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and not stripped.endswith((";", "{", "}", "//", "*/", "*", "#")):
                if not any(stripped.startswith(k) for k in ("if", "else", "for", "while", "do", "//")):
                    if not stripped.startswith("#"):
                        pass  # Don't flag — too many false positives

    return {"errors": errors, "valid": len(errors) == 0}


def run_static_analysis(code: str, language: str) -> Dict[str, Any]:
    """Run language-appropriate static analysis."""
    if language == "python":
        return _pylint_check(code)
    elif language == "sql":
        return _sql_check(code)
    else:
        return _basic_syntax_check(code, language)


# ---------------------------------------------------------------------------
# Sandbox execution (Python only)
# ---------------------------------------------------------------------------

def _sandbox_execute(
    code: str,
    test_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Execute Python code in RestrictedPython sandbox.

    Args:
        code: Student's Python code.
        test_cases: Optional list of ``{'input': str, 'expected_output': str}``.

    Returns:
        ``{'executed': bool, 'output': str, 'error': str, 'test_results': list}``
    """
    try:
        from RestrictedPython import compile_restricted, safe_globals

        byte_code = compile_restricted(code, "<student_code>", "exec")
        if byte_code is None:
            return {"executed": False, "output": "", "error": "Compilation failed", "test_results": []}

        # Safe globals with limited builtins
        globs = safe_globals.copy()
        globs["_print_"] = lambda *args, **kwargs: print(*args, **kwargs)
        globs["_getattr_"] = getattr
        globs["_getitem_"] = lambda obj, key: obj[key]
        globs["_getiter_"] = iter
        globs["_write_"] = lambda obj: obj

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        try:
            exec(byte_code, globs)
            output = captured.getvalue()
        except Exception as exec_err:
            output = captured.getvalue()
            return {
                "executed": False,
                "output": output,
                "error": str(exec_err),
                "test_results": [],
            }
        finally:
            sys.stdout = old_stdout

        # Run test cases
        test_results: List[Dict[str, Any]] = []
        if test_cases:
            for tc in test_cases:
                expected = tc.get("expected_output", "").strip()
                actual = output.strip()
                passed = actual == expected
                test_results.append({
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                })

        return {
            "executed": True,
            "output": output,
            "error": "",
            "test_results": test_results,
        }

    except Exception as exc:
        log.warning("Sandbox execution failed: %s", exc)
        return {"executed": False, "output": "", "error": str(exc), "test_results": []}


# ---------------------------------------------------------------------------
# GPT-4o code review
# ---------------------------------------------------------------------------

def _gpt4o_review(
    student_code: str,
    answer_code: str,
    language: str,
    marks_allocated: int,
) -> Dict[str, Any]:
    """Send code to GPT-4o for review."""
    from openai import OpenAI
    from exam_checker.config import OPENAI_API_KEY

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""Evaluate this student's code answer.
Student code:
```{language}
{student_code}
```

Reference answer:
```{language}
{answer_code}
```

Language: {language}
Marks allocated: {marks_allocated}

Evaluate: logic correctness, time complexity, edge case handling,
code style. Award marks even if syntax differs but logic is correct.
A working alternative algorithm gets full marks.

Return JSON only:
{{
  "logic_correct": true/false,
  "syntax_errors": ["list of syntax errors"],
  "time_complexity_correct": true/false,
  "alternative_valid_approach": true/false,
  "suggested_marks": number,
  "feedback": "string"
}}"""

    @retry(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def _call():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
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
            "logic_correct": False,
            "syntax_errors": [],
            "time_complexity_correct": False,
            "alternative_valid_approach": False,
            "suggested_marks": 0,
            "feedback": raw if isinstance(raw, str) else "Review unavailable.",
        }
    except Exception as exc:
        log.error("GPT-4o code review failed: %s", exc)
        return {
            "logic_correct": False,
            "syntax_errors": [],
            "time_complexity_correct": False,
            "alternative_valid_approach": False,
            "suggested_marks": 0,
            "feedback": f"Review failed: {exc}",
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_code(
    student_text: str,
    answer_code: str,
    language: str = "",
    marks_allocated: int = 10,
    test_cases: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate a student's code answer.

    Pipeline:
      1. Auto-detect language if not specified.
      2. Static analysis.
      3. Sandbox execution (Python only).
      4. GPT-4o review.
      5. Final mark = max(execution marks, GPT-4o marks).

    Args:
        student_text: Student's code (OCR-extracted text).
        answer_code: Reference answer code.
        language: Programming language (auto-detected if empty).
        marks_allocated: Total marks.
        test_cases: Optional test cases for execution.

    Returns:
        Evaluation dict.
    """
    if not language:
        language = detect_language(student_text)
        log.info("Auto-detected language: %s", language)

    # Static analysis
    static = run_static_analysis(student_text, language)

    # Sandbox execution (Python only)
    execution_result: Dict[str, Any] = {"executed": False}
    execution_marks = 0.0
    if language == "python":
        execution_result = _sandbox_execute(student_text, test_cases)
        if execution_result["executed"]:
            if execution_result.get("test_results"):
                passed = sum(1 for t in execution_result["test_results"] if t["passed"])
                total = len(execution_result["test_results"])
                execution_marks = marks_allocated * (passed / total)
            else:
                # No test cases, but code executed without error
                execution_marks = marks_allocated * 0.5

    # GPT-4o review
    gpt_result = _gpt4o_review(student_text, answer_code, language, marks_allocated)
    gpt_marks = float(gpt_result.get("suggested_marks", 0))

    # Final: take the higher score (benefit of the doubt)
    final_marks = max(execution_marks, gpt_marks)

    feedback_parts: List[str] = []
    if static.get("errors"):
        feedback_parts.append(f"Static analysis issues: {'; '.join(static['errors'][:3])}")
    if execution_result.get("executed"):
        feedback_parts.append(f"Code executed successfully. Output: {execution_result.get('output', '')[:200]}")
    elif execution_result.get("error"):
        feedback_parts.append(f"Execution error: {execution_result['error'][:200]}")
    feedback_parts.append(gpt_result.get("feedback", ""))

    return {
        "language": language,
        "static_analysis": static,
        "execution_result": execution_result,
        "execution_marks": execution_marks,
        "gpt4o_result": gpt_result,
        "gpt4o_marks": gpt_marks,
        "suggested_marks": final_marks,
        "feedback": " | ".join(feedback_parts),
        "method": "static+sandbox+gpt4o" if language == "python" else "static+gpt4o",
    }
