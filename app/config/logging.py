import logging
import sys
from app.config.settings import settings

def setup_logging() -> None:
    """Configure structured logging for the application."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Avoid duplicate handlers if setup is called multiple times
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(name)s - %(filename)s:%(lineno)d - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

# Initialize and export a default logger
setup_logging()
logger = logging.getLogger("legal_contract_platform")
