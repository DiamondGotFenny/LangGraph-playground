import logging
import os

def setup_logger(log_file: str = "vector_store_agent.log", log_to_console: bool = True) -> logging.Logger:
    """
    Sets up the logger to record logs to a specified file with UTF-8 encoding.

    Args:
        log_file (str): Path to the log file.
        log_to_console (bool): If True, also log to console (stderr).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the log directory exists
    log_directory = os.path.dirname(log_file)
    if log_directory and not os.path.exists(log_directory):
        os.makedirs(log_directory, exist_ok=True)

    # Configure the logger
    logger = logging.getLogger("VectorStoreAgentLogger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Reset handlers to keep behavior predictable across repeated imports/runs
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Create file handler which logs messages with UTF-8 encoding
    try:
        fh = logging.FileHandler(log_file, encoding='utf-8')
    except TypeError:
        # For Python versions < 3.9 where 'encoding' might not be supported
        fh = logging.FileHandler(log_file)
        fh.stream = open(log_file, 'a', encoding='utf-8')

    fh.setLevel(logging.INFO)

    # Define log message format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger