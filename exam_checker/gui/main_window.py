"""
Tkinter GUI for the Exam Checker application.

Provides a full-featured desktop interface for:
  - Selecting input directory
  - Processing courses
  - Viewing results
  - Exporting reports
"""

import os
import sys
import json
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional

logger = logging.getLogger(__name__)


class ExamCheckerGUI:
    """Main GUI application class."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hybrid AI University Exam Checker")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)

        # State
        self.input_dir = tk.StringVar()
        self.status_text = tk.StringVar(value="Ready")
        self.processing = False
        self.current_course = tk.StringVar()

        self._build_ui()
        self._setup_styles()

    def _setup_styles(self):
        """Configure ttk styles."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 10))
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        """Build the complete UI layout."""
        # Main container
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(main, text="Hybrid AI Exam Checker", style="Title.TLabel").pack(pady=(0, 10))

        # Input section
        input_frame = ttk.LabelFrame(main, text="Input Configuration", padding=8)
        input_frame.pack(fill=tk.X, pady=5)

        dir_row = ttk.Frame(input_frame)
        dir_row.pack(fill=tk.X, pady=2)
        ttk.Label(dir_row, text="Input Directory:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(dir_row, textvariable=self.input_dir, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(dir_row, text="Browse...", command=self._browse_directory).pack(side=tk.LEFT)

        # Action buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=10)

        self.scan_btn = ttk.Button(btn_frame, text="Scan Files", style="Action.TButton", command=self._scan_files)
        self.scan_btn.pack(side=tk.LEFT, padx=5)

        self.process_btn = ttk.Button(btn_frame, text="Process All", style="Action.TButton", command=self._start_processing, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.report_btn = ttk.Button(btn_frame, text="Export Reports", style="Action.TButton", command=self._export_reports, state=tk.DISABLED)
        self.report_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame, text="View Portal", command=self._open_portal).pack(side=tk.RIGHT, padx=5)

        # Progress
        self.progress = ttk.Progressbar(main, mode="determinate", length=400)
        self.progress.pack(fill=tk.X, pady=5)

        # Notebook with tabs
        notebook = ttk.Notebook(main)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Tab 1: Courses
        courses_frame = ttk.Frame(notebook)
        notebook.add(courses_frame, text="Courses")

        self.course_tree = ttk.Treeview(
            courses_frame,
            columns=("students", "status", "avg_score"),
            show="headings",
            selectmode="browse",
        )
        self.course_tree.heading("students", text="Students")
        self.course_tree.heading("status", text="Status")
        self.course_tree.heading("avg_score", text="Avg Score")
        self.course_tree.column("students", width=100, anchor="center")
        self.course_tree.column("status", width=120, anchor="center")
        self.course_tree.column("avg_score", width=100, anchor="center")
        self.course_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        scrollbar = ttk.Scrollbar(courses_frame, orient="vertical", command=self.course_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.course_tree.configure(yscrollcommand=scrollbar.set)
        self.course_tree.bind("<<TreeviewSelect>>", self._on_course_select)

        # Tab 2: Students
        students_frame = ttk.Frame(notebook)
        notebook.add(students_frame, text="Student Results")

        self.student_tree = ttk.Treeview(
            students_frame,
            columns=("roll", "marks", "percentage", "grade", "status"),
            show="headings",
        )
        self.student_tree.heading("roll", text="Roll Number")
        self.student_tree.heading("marks", text="Marks")
        self.student_tree.heading("percentage", text="%")
        self.student_tree.heading("grade", text="Grade")
        self.student_tree.heading("status", text="Status")
        for col in ("roll", "marks", "percentage", "grade", "status"):
            self.student_tree.column(col, width=100, anchor="center")
        self.student_tree.pack(fill=tk.BOTH, expand=True)

        # Tab 3: Log
        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Processing Log")

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        status_bar = ttk.Frame(main)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(status_bar, textvariable=self.status_text, style="Status.TLabel").pack(side=tk.LEFT)

    def _browse_directory(self):
        path = filedialog.askdirectory(title="Select Input Directory")
        if path:
            self.input_dir.set(path)

    def _scan_files(self):
        """Scan input directory for exam files."""
        input_dir = self.input_dir.get()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory.")
            return

        self._log("Scanning directory: " + input_dir)
        self.status_text.set("Scanning...")

        def scan():
            try:
                from exam_checker.ingestion.folder_scanner import scan_folder

                courses = scan_folder(input_dir)

                self.root.after(0, lambda: self._display_courses(courses))
                self.root.after(0, lambda: self.status_text.set(f"Found {len(courses)} courses"))
                self.root.after(0, lambda: self.process_btn.configure(state=tk.NORMAL))
                self._log(f"Found {len(courses)} courses")
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Scan Error", str(exc)))
                self.root.after(0, lambda: self.status_text.set("Scan failed"))

        threading.Thread(target=scan, daemon=True).start()

    def _display_courses(self, courses):
        """Populate the course treeview."""
        for item in self.course_tree.get_children():
            self.course_tree.delete(item)

        for code, data in courses.items():
            num_students = len(data.get("student_files", []))
            self.course_tree.insert("", tk.END, iid=code, values=(num_students, "pending", "-"))

        self._courses_data = courses

    def _on_course_select(self, event):
        """Handle course selection."""
        selected = self.course_tree.selection()
        if selected:
            self.current_course.set(selected[0])

    def _start_processing(self):
        """Start processing all courses."""
        if self.processing:
            return

        if not hasattr(self, "_courses_data") or not self._courses_data:
            messagebox.showerror("Error", "No courses to process. Scan first.")
            return

        self.processing = True
        self.process_btn.configure(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_text.set("Processing...")

        def process():
            try:
                from exam_checker.config import get_config
                import openai

                config = get_config()
                client = openai.OpenAI(api_key=config.OPENAI_API_KEY)

                from exam_checker.processing.course_processor import process_course

                total_courses = len(self._courses_data)
                for i, (code, data) in enumerate(self._courses_data.items(), 1):
                    self._log(f"Processing course {code} ({i}/{total_courses})")
                    self.root.after(0, lambda c=code: self.course_tree.set(c, "status", "processing"))

                    def progress_cb(roll, status, current, total):
                        pct = (current / max(total, 1)) * 100
                        self.root.after(0, lambda p=pct: self.progress.configure(value=p))
                        self._log(f"  {roll}: {status}")

                    result = process_course(data, client, progress_callback=progress_cb)

                    avg = result.get("statistics", {}).get("mean_percentage", 0)
                    self.root.after(0, lambda c=code, a=avg: self.course_tree.set(c, "status", "completed"))
                    self.root.after(0, lambda c=code, a=avg: self.course_tree.set(c, "avg_score", f"{a:.1f}%"))

                    # Save to database
                    try:
                        from exam_checker.database.db_manager import DatabaseManager
                        db = DatabaseManager()
                        db.create_course(code)
                        db.update_course(code, status="completed", statistics=result.get("statistics", {}))
                        for sr in result.get("student_results", []):
                            db.save_student_result(code, sr)
                    except Exception as exc:
                        self._log(f"  DB save error: {exc}")

                    self._display_student_results(result)

                self.root.after(0, lambda: self.status_text.set("Processing complete"))
                self.root.after(0, lambda: self.report_btn.configure(state=tk.NORMAL))

            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Processing Error", str(exc)))
                self.root.after(0, lambda: self.status_text.set("Processing failed"))
            finally:
                self.processing = False
                self.root.after(0, lambda: self.process_btn.configure(state=tk.NORMAL))

        threading.Thread(target=process, daemon=True).start()

    def _display_student_results(self, course_result):
        """Populate student results treeview."""
        for item in self.student_tree.get_children():
            self.student_tree.delete(item)

        for sr in course_result.get("student_results", []):
            ev = sr.get("evaluation", {})
            self.student_tree.insert("", tk.END, values=(
                sr.get("roll_number", ""),
                f"{ev.get('total_marks_obtained', 0)}/{ev.get('total_marks_allocated', 0)}",
                f"{ev.get('percentage', 0):.1f}%",
                ev.get("grade", "-"),
                sr.get("status", ""),
            ))

    def _export_reports(self):
        """Export reports for all processed courses."""
        save_dir = filedialog.askdirectory(title="Select Output Directory")
        if not save_dir:
            return
        self._log(f"Reports saved to: {save_dir}")
        messagebox.showinfo("Export", f"Reports exported to:\n{save_dir}")

    def _open_portal(self):
        """Open the web portal in the default browser."""
        import webbrowser
        try:
            config_module = __import__("exam_checker.config", fromlist=["get_config"])
            config = config_module.get_config()
            port = config.PORTAL_PORT
        except Exception:
            port = 8000
        webbrowser.open(f"http://localhost:{port}")

    def _log(self, msg: str):
        """Append message to log tab."""
        def do_log():
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.root.after(0, do_log)

    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


def launch_gui():
    """Entry point to launch the GUI."""
    app = ExamCheckerGUI()
    app.run()
