import logging
import sys

def setup_logger():
    """Sets up a logger that outputs to the console only."""
    logger = logging.getLogger("LFL")
    logger.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console Handler (Keep this for visibility)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger
