"""Step 7: Safe cleanup of temporary files.

Moves tmp/ contents to _trash/ with timestamp.
NEVER uses system rm — only Python shutil.move.
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path

from ..config import TMP_DIR, TRASH_DIR

logger = logging.getLogger(__name__)


def cleanup_tmp():
    """Move all contents of tmp/ to _trash/ with timestamp.

    Creates a fresh empty tmp/ directory after.
    """
    if not TMP_DIR.exists():
        logger.info("tmp/ does not exist, nothing to clean")
        return

    items = list(TMP_DIR.iterdir())
    if not items:
        logger.info("tmp/ is empty, nothing to clean")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_sub = TRASH_DIR / f"tmp_cleanup_{ts}"
    trash_sub.mkdir(parents=True, exist_ok=True)

    for item in items:
        dest = trash_sub / item.name
        logger.info("Moving %s → %s", item.name, dest)
        shutil.move(str(item), str(dest))

    # Re-create empty tmp
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Cleanup complete: moved %d items to %s", len(items), trash_sub)
