"""
Main Entry Point — Hybrid AI University Exam Checker

This script launches the system. It can start the GUI, the Web Portal, 
or run a CLI-based processing session.
"""

import argparse
import sys
import threading
import os

from exam_checker.config import print_startup_summary, PORTAL_PORT
from exam_checker.utils.logger import get_logger

log = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Hybrid AI Exam Checker")
    parser.add_argument("--mode", choices=["gui", "portal", "process", "all"], default="gui",
                        help="Mode to run: gui (default), portal (web view), process (cli run), or all")
    parser.add_argument("--root", type=str, help="Root folder to process (for --mode process)")
    
    args = parser.parse_args()
    
    print_startup_summary()

    if args.mode == "gui":
        from exam_checker.gui.browser import start_gui
        start_gui()
        
    elif args.mode == "portal":
        from exam_checker.portal.app import start_portal
        start_portal()
        
    elif args.mode == "process":
        if not args.root:
            print("Error: --root directory required for processing mode.")
            sys.exit(1)
        from exam_checker.processing.pipeline_runner import PipelineRunner
        runner = PipelineRunner()
        try:
            runner.run_full_process(args.root)
        finally:
            runner.cleanup()
            
    elif args.mode == "all":
        # Run portal in background thread
        from exam_checker.portal.app import start_portal
        from exam_checker.gui.browser import start_gui
        
        log.info("Starting Web Portal on http://localhost:%d", PORTAL_PORT)
        portal_thread = threading.Thread(target=start_portal, daemon=True)
        portal_thread.start()
        
        log.info("Starting GUI...")
        start_gui()

if __name__ == "__main__":
    main()
