import sys
from loguru import logger
from pathlib import Path


def setup_logger(log_to_file=False, log_dir="logs", log_filename="climaus.log"):
    logger.remove()

    logger.add(
        sys.stdout,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path / log_filename,
            rotation="10 MB",
            retention="7 days",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

    return logger
