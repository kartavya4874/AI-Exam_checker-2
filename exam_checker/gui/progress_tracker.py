"""
Progress Tracker — Tkinter component for visual feedback.
"""

import tkinter as tk
from tkinter import ttk

class ProgressTracker(ttk.Frame):
    """
    Shows progress bars and status text.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        # Status Label
        self.status_label = ttk.Label(self, text="Waiting...", font=("Helvetica", 10, "italic"))
        self.status_label.pack(anchor=tk.W, pady=(0, 5))

        # Overall Progress
        self.progress_bar = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress_bar.pack(fill=tk.X, expand=True)
        
        # Sub-status
        self.sub_status = ttk.Label(self, text="", font=("Helvetica", 9))
        self.sub_status.pack(anchor=tk.W, pady=5)

    def update_status(self, main_text: str, sub_text: str = ""):
        self.status_label.config(text=main_text)
        self.sub_status.config(text=sub_text)

    def set_progress(self, value: float):
        """Value from 0 to 100."""
        self.progress_bar['value'] = value

    def reset(self):
        self.progress_bar['value'] = 0
        self.status_label.config(text="Ready.")
        self.sub_status.config(text="")
