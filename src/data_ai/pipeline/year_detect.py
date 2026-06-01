import re
from datetime import datetime
from pathlib import Path


YEAR_PATTERN = re.compile(r"(19|20)\d{2}")


def _get_mtime_year(file_path: Path) -> int | None:
    """Get year from file modification time."""
    try:
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).year
    except (OSError, ValueError):
        return None


def _get_current_year() -> int:
    """Get current year."""
    return datetime.now().year


def detect_year(file_path: Path) -> int:
    """
    Detect year from file.

    Priority:
    1. Year in filename (e.g., rechnung_2024.pdf)
    2. Year in path (e.g., /archive/2024/file.pdf)
    3. File modification time
    4. Current year as fallback
    """
    # Try filename first
    filename_match = YEAR_PATTERN.search(file_path.name)
    if filename_match:
        return int(filename_match.group())

    # Try full path
    path_match = YEAR_PATTERN.search(str(file_path))
    if path_match:
        return int(path_match.group())

    # Try mtime
    mtime_year = _get_mtime_year(file_path)
    if mtime_year:
        return mtime_year

    # Fallback to current year
    return _get_current_year()
