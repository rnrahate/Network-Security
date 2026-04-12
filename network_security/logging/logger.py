import logging
from datetime import datetime
from pathlib import Path

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
current_dir = Path(__file__).resolve().parent
log_dir = current_dir.parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = log_dir / LOG_FILE

logging.basicConfig(
    filename=str(LOG_FILE_PATH),
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(levelname)s - %(message)s"
)
