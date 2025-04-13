import os
import logging
import time
import pandas as pd
from functools import wraps
from pathlib import Path
from typing import Callable, Any, List, Optional, Dict

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Create results directory if it doesn't exist
results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)

CO_FILE_LOGLEVEL = logging.DEBUG if os.environ.get("CO_FILE_LOGLEVEL") == "DEBUG" else logging.INFO
CO_CONSOLE_LOGLEVEL = logging.DEBUG if os.environ.get("CO_CONSOLE_LOGLEVEL") == "DEBUG" else logging.INFO

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
    file_handler.setLevel(CO_FILE_LOGLEVEL)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(CO_CONSOLE_LOGLEVEL)
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

def save_loss_data(
    loss_list: List[float],
    algorithm_name: str,
    loss_function_name: str,
    parameters: Dict[str, Any],
    start_time: Optional[int] = None,
    filename: Optional[str] = None
) -> None:
    """
    Save loss data to a CSV file.
    
    Args:
        loss_list: List of loss values
        algorithm_name: Name of the algorithm used
        loss_function_name: Name of the loss function used
        parameters: Dictionary of algorithm parameters
        start_time: Start time of the algorithm (timestamp)
        filename: Optional custom filename
    """
    # Get current timestamp if not provided
    current_time = time.time()
    start_time = start_time or current_time
    
    # Create dataframe
    df = pd.DataFrame(loss_list, columns=['loss'])
    df['algorithm'] = algorithm_name
    df['loss_function'] = loss_function_name
    df['time_taken'] = current_time - start_time
    df['start_time'] = start_time
    
    # Add all parameters
    for key, value in parameters.items():
        df[key] = value
    
    # Generate filename
    if filename is None:
        int_start_time = int(start_time)
        filename = f"{algorithm_name}_{loss_function_name}_{int_start_time}.csv"
    
    # Save to file
    file_path = results_dir / filename
    if file_path.exists():
        # If file exists, append without writing the header
        df.to_csv(file_path, mode='a', header=False, index=False)
        logger.info(f"Loss data appended to existing file {file_path}")
    else:
        # If file doesn't exist, create new file with header
        df.to_csv(file_path, index=False)
        logger.info(f"Loss data saved to new file {file_path}")
    
    return df

# Create default logger instance
logger = setup_logger("deblur_denoise") 