import logging
import os

def setup_logger(name="PersonCountingApp", log_file="logs/app.log", level=logging.INFO):
    """Sets up a console and file logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()
