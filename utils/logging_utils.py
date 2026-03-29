import logging
import os
from datetime import datetime

def setup_logger(name: str = "alzheimers_ai", log_dir: str = "logs") -> logging.Logger:
    
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(os.path.join(log_dir, f"run_{timestamp}.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
