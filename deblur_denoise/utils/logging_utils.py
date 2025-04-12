import logging
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Any

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

def setup_logger(name: str, log_file: str = "deblur_denoise.log") -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        log_file: Name of the log file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate logging
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_execution_time(logger: logging.Logger) -> Callable:
    """
    Decorator to log the execution time of a function.
    
    Args:
        logger: Logger instance to use for logging
        
    Returns:
        Decorated function with execution time logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.process_time()
            result = func(*args, **kwargs)
            end_time = time.process_time()
            execution_time = end_time - start_time
            
            logger.info(
                f"Function {func.__name__} executed in {execution_time:.4f} seconds"
            )
            return result
        return wrapper
    return decorator

# Create default logger instance
logger = setup_logger("deblur_denoise") 