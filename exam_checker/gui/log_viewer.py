"""
Log Viewer — Real-time Tkinter log window.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from exam_checker.utils.logger import register_gui_handler
import logging

class LogViewer(tk.Toplevel):
    """
    A separate window to view log messages as they are generated.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.title("System Logs")
        self.geometry("800x400")
        self._setup_ui()
        
        # Register this viewer with the logging system
        register_gui_handler(self)

    def _setup_ui(self):
        self.log_area = scrolledtext.ScrolledText(self, state=tk.DISABLED, bg="black", fg="white", font=("Consolas", 10))
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure tags for colors
        self.log_area.tag_config("INFO", foreground="white")
        self.log_area.tag_config("WARNING", foreground="yellow")
        self.log_area.tag_config("ERROR", foreground="red")
        self.log_area.tag_config("DEBUG", foreground="gray")

    def write_log(self, message, level="INFO"):
        """
        Write a log message to the text area. Thread-safe using .after().
        """
        self.after(0, self._append_text, message, level)

    def _append_text(self, message, level):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n", level)
        self.log_area.see(tk.END)
        self.log_area.config(state=tk.DISABLED)

    def handle_record(self, record):
        """
        Called by the custom logging handler.
        """
        msg = f"[{record.levelname}] {record.name} - {record.getMessage()}"
        self.write_log(msg, record.levelname)
