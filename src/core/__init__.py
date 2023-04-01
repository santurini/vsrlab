import logging

import os
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd().parents[0]
LOGGING_DIR = Path(PROJECT_ROOT / 'storage')

logger.debug(f"Inferred project root: {PROJECT_ROOT}")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["LOGGING_DIR"] = str(LOGGING_DIR)
os.environ["CONFIG_PATH"] = str(CONFIG_PATH)
os.environ["HYDRA_FULL_ERROR"] = "1"