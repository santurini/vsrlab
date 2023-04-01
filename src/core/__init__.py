import logging
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=False, default='default', help='configuration path')
args = parser.parse_args()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path.cwd().parents[0]
LOGGING_DIR = Path(PROJECT_ROOT / 'storage')
CONFIG_PATH = Path(args.config_path)

logger.debug(f"Inferred project root: {PROJECT_ROOT}")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["LOGGING_DIR"] = str(LOGGING_DIR)
os.environ["CONFIG_PATH"] = str(CONFIG_PATH)
os.environ["HYDRA_FULL_ERROR"] = "1"