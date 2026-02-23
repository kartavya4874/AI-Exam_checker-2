"""
GUI Browser — Main Tkinter-based file browser and control panel.
"""

import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading

from exam_checker.config import PORTAL_PORT
from exam_checker.utils.logger import get_logger
from exam_checker.processing.pipeline_runner import PipelineRunner
from exam_checker.gui.log_viewer import LogViewer
from exam_checker.gui.progress_tracker import ProgressTracker

log = get_logger(__name__)

class ExamCheckerGUI:
    """
    Main Tkinter application for the Exam Checker.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Hybrid AI University Exam Checker")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f0")
        
        self.runner = PipelineRunner()
        self.is_processing = False
        
        self._setup_ui()

    def _setup_ui(self):
        # Style
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), background="#f0f0f0")
        style.configure("Main.TFrame", background="#f0f0f0")
        
        main_frame = ttk.Frame(self.root, style="Main.TFrame", padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Label(main_frame, text="Exam Checker Control Panel", style="Header.TLabel")
        header.pack(pady=(0, 20))

        # Folder Selection
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Configuration", padding="10")
        folder_frame.pack(fill=tk.X, pady=10)
        
        self.folder_path = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_path, width=70).grid(row=0, column=0, padx=5)
        ttk.Button(folder_frame, text="Browse Folder", command=self._browse_folder).grid(row=0, column=1, padx=5)

        # Control Buttons
        btn_frame = ttk.Frame(main_frame, style="Main.TFrame")
        btn_frame.pack(fill=tk.X, pady=20)
        
        self.start_btn = ttk.Button(btn_frame, text="Start Processing", command=self._start_process)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Open Web Portal", command=self._open_portal).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="View Logs", command=self._show_logs).pack(side=tk.LEFT, padx=5)

        # Progress Tracker Component
        self.progress = ProgressTracker(main_frame)
        self.progress.pack(fill=tk.X, pady=10)

        # Status Footer
        self.status_var = tk.StringVar(value="Ready to process.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)
            log.info("Selected folder: %s", folder)

    def _start_process(self):
        if self.is_processing:
            return
            
        folder = self.folder_path.get()
        if not folder:
            messagebox.showwarning("Warning", "Please select a root folder first.")
            return

        self.is_processing = True
        self.start_btn.config(state=tk.DISABLED)
        self.status_var.set("Processing exam sheets…")
        
        # Run in separate thread to keep UI responsive
        thread = threading.Thread(target=self._run_pipeline, args=(folder,))
        thread.daemon = True
        thread.start()

    def _run_pipeline(self, folder):
        try:
            self.progress.reset()
            self.progress.update_status("Scanning folder…")
            
            # This would ideally be hooked into pipeline events for better granularity
            results = self.runner.run_full_process(folder)
            
            self.root.after(0, self._on_process_complete, results)
        except Exception as e:
            log.error("GUI Pipeline Execution Error: %s", e)
            self.root.after(0, self._on_process_error, str(e))

    def _on_process_complete(self, results):
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.status_var.set("Process complete.")
        self.progress.update_status("Successfully processed all courses.")
        messagebox.showinfo("Success", f"Finished processing. Courses: {', '.join(results.keys())}")

    def _on_process_error(self, error_msg):
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.status_var.set("Error occurred.")
        messagebox.showerror("Error", f"Processing failed: {error_msg}")

    def _open_portal(self):
        import webbrowser
        webbrowser.open(f"http://localhost:{PORTAL_PORT}")

    def _show_logs(self):
        LogViewer(self.root)

def start_gui():
    root = tk.Tk()
    app = ExamCheckerGUI(root)
    root.mainloop()
