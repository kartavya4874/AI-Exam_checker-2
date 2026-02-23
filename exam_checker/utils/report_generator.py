"""
Excel report builder.

Generates multi-sheet XLSX files:
  Sheet 1 — Summary (Roll No, Marks, %, Grade)
  Sheet 2 — Question-wise breakdown
  Sheet 3 — Individual feedback
  Sheet 4 — Statistics with charts
"""

import io
from typing import Any, Dict, List

from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
_HEADER_FONT = Font(bold=True, color="FFFFFF", size=12)
_HEADER_FILL = PatternFill("solid", fgColor="2F5496")
_CENTER = Alignment(horizontal="center", vertical="center")
_WRAP = Alignment(wrap_text=True, vertical="top")

_GRADE_COLOURS = {
    "A+": "27AE60",
    "A": "2ECC71",
    "B+": "F39C12",
    "B": "E67E22",
    "C": "E74C3C",
    "D": "C0392B",
    "F": "7F8C8D",
}


def _style_header_row(ws, col_count: int) -> None:
    """Apply header styling to the first row of *ws*."""
    for col in range(1, col_count + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = _CENTER


def _auto_width(ws) -> None:
    """Set each column width to fit the widest cell (max 50 chars)."""
    for col_cells in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col_cells[0].column)
        for cell in col_cells:
            try:
                length = len(str(cell.value or ""))
                if length > max_len:
                    max_len = length
            except Exception:
                pass
        ws.column_dimensions[col_letter].width = min(max_len + 4, 50)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_excel_report(
    course_code: str,
    results: List[Dict[str, Any]],
    stats: Dict[str, Any],
) -> io.BytesIO:
    """
    Build a complete Excel report for a course.

    Args:
        course_code: Course identifier (e.g. ``CS101``).
        results: List of evaluation result dicts (one per student).
        stats: Aggregate statistics dict (mean, max, min, etc.).

    Returns:
        :class:`io.BytesIO` containing the XLSX data.
    """
    wb = Workbook()

    # ------ Sheet 1: Summary -------------------------------------------------
    ws_summary = wb.active
    ws_summary.title = "Summary"
    headers = ["Roll Number", "Obtained Marks", "Total Marks", "Percentage", "Grade"]
    ws_summary.append(headers)
    _style_header_row(ws_summary, len(headers))

    for r in results:
        grade = r.get("grade", "")
        ws_summary.append([
            r.get("roll_number", ""),
            r.get("obtained_marks", 0),
            r.get("total_marks", 0),
            round(r.get("percentage", 0), 2),
            grade,
        ])
        # Colour-code grade cell
        row_idx = ws_summary.max_row
        grade_cell = ws_summary.cell(row=row_idx, column=5)
        colour = _GRADE_COLOURS.get(grade, "FFFFFF")
        grade_cell.fill = PatternFill("solid", fgColor=colour)
        grade_cell.font = Font(bold=True, color="FFFFFF")
        grade_cell.alignment = _CENTER

    _auto_width(ws_summary)

    # ------ Sheet 2: Question-wise breakdown ---------------------------------
    ws_qwise = wb.create_sheet("Question Breakdown")
    # Determine all question numbers across results
    all_qs: list[str] = []
    for r in results:
        for qr in r.get("question_results", []):
            qn = qr.get("question_number", "")
            if qn and qn not in all_qs:
                all_qs.append(qn)
    all_qs.sort(key=lambda q: (q.replace("Q", "").replace("q", ""), q))

    q_headers = ["Roll Number"] + all_qs + ["Total"]
    ws_qwise.append(q_headers)
    _style_header_row(ws_qwise, len(q_headers))

    for r in results:
        row: list[Any] = [r.get("roll_number", "")]
        q_map = {qr["question_number"]: qr.get("marks_obtained", 0) for qr in r.get("question_results", [])}
        total = 0
        for qn in all_qs:
            marks = q_map.get(qn, 0)
            row.append(marks)
            total += marks
        row.append(total)
        ws_qwise.append(row)

    _auto_width(ws_qwise)

    # ------ Sheet 3: Feedback ------------------------------------------------
    ws_fb = wb.create_sheet("Feedback")
    fb_headers = ["Roll Number", "Question", "Status", "Marks", "Feedback", "Error Analysis"]
    ws_fb.append(fb_headers)
    _style_header_row(ws_fb, len(fb_headers))

    for r in results:
        for qr in r.get("question_results", []):
            ws_fb.append([
                r.get("roll_number", ""),
                qr.get("question_number", ""),
                qr.get("status", ""),
                qr.get("marks_obtained", 0),
                qr.get("feedback", ""),
                qr.get("error_analysis", ""),
            ])

    # Wrap text for feedback columns
    for row in ws_fb.iter_rows(min_row=2, min_col=5, max_col=6):
        for cell in row:
            cell.alignment = _WRAP

    _auto_width(ws_fb)

    # ------ Sheet 4: Statistics + Chart --------------------------------------
    ws_stats = wb.create_sheet("Statistics")
    stat_rows = [
        ("Course Code", course_code),
        ("Total Students", stats.get("total_students", len(results))),
        ("Average Marks", round(stats.get("average", 0), 2)),
        ("Highest Marks", stats.get("max", 0)),
        ("Lowest Marks", stats.get("min", 0)),
        ("Pass Rate (%)", round(stats.get("pass_rate", 0), 2)),
        ("Standard Deviation", round(stats.get("std_dev", 0), 2)),
    ]
    for label, value in stat_rows:
        ws_stats.append([label, value])

    # Grade distribution sub-table
    ws_stats.append([])
    ws_stats.append(["Grade", "Count"])
    grade_dist: Dict[str, int] = stats.get("grade_distribution", {})
    chart_start_row = ws_stats.max_row + 1
    for grade, count in sorted(grade_dist.items()):
        ws_stats.append([grade, count])

    # Bar chart
    if grade_dist:
        chart_end_row = ws_stats.max_row
        chart = BarChart()
        chart.type = "col"
        chart.title = f"{course_code} — Grade Distribution"
        chart.y_axis.title = "Students"
        chart.x_axis.title = "Grade"
        cats = Reference(ws_stats, min_col=1, min_row=chart_start_row, max_row=chart_end_row)
        data = Reference(ws_stats, min_col=2, min_row=chart_start_row - 1, max_row=chart_end_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.shape = 4
        ws_stats.add_chart(chart, "D2")

    _auto_width(ws_stats)

    # ------ Write to BytesIO -------------------------------------------------
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    log.info("Generated Excel report for course %s (%d students)", course_code, len(results))
    return output
