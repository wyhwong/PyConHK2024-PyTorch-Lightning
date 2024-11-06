import os
from typing import Optional


STREAMING_LOG_LEVEL: int = int(os.getenv("STREAMING_LOG_LEVEL", "20"))
FILE_LOG_LEVEL: int = int(os.getenv("FILE_LOG_LEVEL", "10"))
LOG_FILEPATH: Optional[str] = os.getenv("LOG_FILEPATH")
LOG_FMT = os.getenv("LOG_FMT", "%(asctime)s [%(name)s | %(levelname)s]: %(message)s")
LOG_DATEFMT = os.getenv("LOG_DATEFMT", "%Y-%m-%dT%H:%M:%SZ")

SLACK_APP_TOKEN: str = os.getenv("SLACK_APP_TOKEN", "")
SLACK_BOT_TOKEN: str = os.getenv("SLACK_BOT_TOKEN", "")
