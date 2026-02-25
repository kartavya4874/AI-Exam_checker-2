"""
Main entry point for the Hybrid AI University Exam Checking System.

Supports three modes:
  python main.py --gui       Launch the Tkinter desktop GUI
  python main.py --portal    Launch the FastAPI web portal
  python main.py --both      Launch both GUI and portal
"""

import argparse
import sys
import threading
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exam_checker.config import get_config
from exam_checker.utils.logger import setup_logging, get_logger
from exam_checker.database.db_manager import DatabaseManager


def launch_portal(config) -> None:
    """Start the FastAPI web portal in a background thread (or foreground)."""
    import uvicorn
    from exam_checker.portal.app import create_app

    app = create_app(config)
    uvicorn.run(
        app,
        host=config.PORTAL_HOST,
        port=config.PORTAL_PORT,
        log_level=config.LOG_LEVEL.lower(),
    )


def launch_gui(config) -> None:
    """Start the Tkinter desktop GUI (blocks until window closes)."""
    from exam_checker.gui.main_window import ExamCheckerGUI

    gui = ExamCheckerGUI()
    gui.run()


def main() -> None:
    """Parse CLI arguments and launch the requested mode."""
    parser = argparse.ArgumentParser(
        description="Hybrid AI University Exam Checking System"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Launch the desktop GUI"
    )
    parser.add_argument(
        "--portal", action="store_true", help="Launch the web portal"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Launch GUI and web portal simultaneously",
    )
    parser.add_argument(
        "--env", type=str, default=None, help="Path to .env file"
    )
    args = parser.parse_args()

    # Load configuration
    config = get_config(env_path=args.env)
    setup_logging(config.LOG_LEVEL)
    logger = get_logger(__name__)

    config.print_summary()

    if not config.validate():
        logger.warning(
            "Configuration has warnings — some features may not work."
        )

    # Initialize database
    db_manager = DatabaseManager()
    logger.info("Database initialized at %s", config.DB_PATH)

    # Determine launch mode
    if args.both:
        logger.info("Starting web portal in background thread...")
        portal_thread = threading.Thread(
            target=launch_portal, args=(config,), daemon=True
        )
        portal_thread.start()
        logger.info(
            "Portal running at http://%s:%d",
            config.PORTAL_HOST,
            config.PORTAL_PORT,
        )
        launch_gui(config)
    elif args.portal:
        logger.info(
            "Starting web portal at http://%s:%d",
            config.PORTAL_HOST,
            config.PORTAL_PORT,
        )
        launch_portal(config)
    elif args.gui:
        launch_gui(config)
    else:
        # Default to GUI
        logger.info("No mode specified — defaulting to GUI.")
        launch_gui(config)


if __name__ == "__main__":
    main()
