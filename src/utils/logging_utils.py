import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(name: str, log_file: Optional[Path] = None, level=logging.INFO) -> logging.Logger:
    """
    Setup logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Path to log file. If None, only console logging is enabled.
        level: Logging level
    """
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger
