import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str = "system.log", level: str = "INFO") -> logging.Logger:
    """
    Setup a logger with console and file handlers.
    
    Args:
        name: Name of the logger (usually __name__)
        log_file: Path to the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    
    # Check if handlers already exist to avoid duplicates
    if logger.handlers:
        return logger
        
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (Rotating)
    try:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to setup file logging to {log_file}: {e}")
        
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with default configuration."""
    return setup_logger(name, level=os.getenv("LOG_LEVEL", "INFO"))
