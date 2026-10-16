"""Safe file operations — all moves/deletes via Python shutil, NEVER system rm."""

import logging
import shutil
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional


def safe_move(src: Path, dst: Path) -> Path:
    """Move a file or directory using shutil.move. Creates parent dirs as needed."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))
    return dst


def safe_move_to_trash(src: Path, trash_root: Path) -> Path:
    """Move src to trash_root/ with a timestamped subdirectory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_sub = trash_root / f"{src.name}_{ts}"
    trash_sub.parent.mkdir(parents=True, exist_ok=True)
    return safe_move(src, trash_sub)


def read_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(filepath: Path, data: dict, indent: int = 2):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_text(filepath: Path) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def write_text(filepath: Path, content: str):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def rename_artifacts(json_path: Path) -> dict:
 """Rename all filesystem artifacts to match the JSON's metadata.

 After a year correction in the JSON, the PDF, cnRead, and image dirs
 may still have the old UNKN name. This function syncs them.

 Args:
     json_path: Path to a *_v2.json file with meta.book and meta.year

 Returns:
     dict with keys: renamed, skipped, errors
 """
 from ..config import DONE_PDF_DIR, CNREAD_DIR, IMAGES_DIR, TRASH_DIR
 import re

 stats = {"renamed": 0, "skipped": 0, "errors": 0}

 if not json_path.exists():
     return stats

 try:
     data = read_json(json_path)
 except Exception:
     return stats

 meta = data.get("meta", {}) if isinstance(data, dict) else {}
 book = meta.get("book", "")
 year = meta.get("year", "")

 if not book or not year or year in ("UNKN", "Unknown"):
     return stats

 # The correct base name from metadata
 correct_base = f"{book}_{year}"

 # The v2 filename tells us the old base name
 v2_name = json_path.stem  # e.g. "Foo_UNKN_v2"
 # Remove _v2 suffix
 old_base = re.sub(r"_v2$", "", v2_name)

 if old_base == correct_base:
     stats["skipped"] += 1
     return stats

 logger = logging.getLogger(__name__)

 # ── 1. DonePDF/{old_base}.pdf → {correct_base}.pdf ──
 for ext in [".pdf", ".PDF"]:
     src = DONE_PDF_DIR / f"{old_base}{ext}"
     if src.exists():
         dst = DONE_PDF_DIR / f"{correct_base}{ext}"
         try:
             safe_move(src, dst)
             logger.info("Renamed PDF: %s → %s", src.name, dst.name)
             stats["renamed"] += 1
         except Exception as e:
             logger.warning("Failed to rename PDF %s: %s", src.name, e)
             stats["errors"] += 1
         break

 # ── 2. cnRead/{old_base}.md and .bib ──
 for ext in [".md", ".bib"]:
     src = CNREAD_DIR / f"{old_base}{ext}"
     if src.exists():
         dst = CNREAD_DIR / f"{correct_base}{ext}"
         try:
             safe_move(src, dst)
             logger.info("Renamed cnRead: %s → %s", src.name, dst.name)
             stats["renamed"] += 1
         except Exception as e:
             logger.warning("Failed to rename cnRead %s: %s", src.name, e)
             stats["errors"] += 1

 # ── 3. images/{old_base}_images/ → {correct_base}_images/ ──
 for suffix in ["_images", "_Images", "_img"]:
     src = IMAGES_DIR / f"{old_base}{suffix}"
     if src.exists() and src.is_dir():
         dst = IMAGES_DIR / f"{correct_base}{suffix}"
         try:
             if dst.exists():
                 # Merge: move files into existing dir
                 for item in src.iterdir():
                     target = dst / item.name
                     if not target.exists():
                         safe_move(item, target)
                 src.rmdir()
             else:
                 safe_move(src, dst)
             logger.info("Renamed images: %s → %s", src.name, dst.name)
             stats["renamed"] += 1
         except Exception as e:
             logger.warning("Failed to rename images %s: %s", src.name, e)
             stats["errors"] += 1
         break

 return stats


def sync_all_artifacts(json_dir: Path) -> dict:
 """Scan all _v2.json files and rename artifacts to match metadata.

 Call this after fixing years to ensure all filenames are consistent.

 Args:
     json_dir: Directory containing *_v2.json files

 Returns:
     Aggregated stats dict
 """
 import re
 totals = {"renamed": 0, "skipped": 0, "errors": 0}
 for f in sorted(json_dir.glob("*_v2.json")):
     stats = rename_artifacts(f)
     for k in totals:
         totals[k] += stats[k]
 return totals
