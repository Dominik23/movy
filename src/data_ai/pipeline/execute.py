# src/data_ai/pipeline/execute.py
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

console = Console()


def execute_move(source: Path, target_dir: Path) -> bool:
    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / source.name

        # Handle duplicate filenames
        if target_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = source.stem
            suffix = source.suffix
            target_path = target_dir / f"{stem}_{timestamp}{suffix}"

        shutil.move(str(source), str(target_path))
        return True
    except Exception as e:
        console.print(f"[red]Error moving file: {e}[/red]")
        return False


def prompt_for_category(
    file_path: Path,
    matches: list[tuple[str, float]],
) -> Optional[str]:
    console.print(f"\n[yellow]? {file_path.name}[/yellow] — uncertain")
    console.print()

    for i, (category, score) in enumerate(matches[:5], 1):
        console.print(f"  [{i}] {category} ({score:.0%})")

    console.print("  [s] Skip")
    console.print("  [q] Abort")
    console.print()

    choice = Prompt.ask("Selection")

    if choice.lower() == "s":
        return None
    if choice.lower() == "q":
        raise KeyboardInterrupt("User aborted")

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(matches):
            return matches[idx][0]
    except ValueError:
        pass

    console.print("[red]Invalid choice, skipping[/red]")
    return None


def execute_copy(
    source: Path,
    target_dir: Path,
    log_file: Optional[Path] = None,
) -> Optional[Path]:
    """
    Copy file to target directory (preserves original).
    Returns target path on success, None on failure.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / source.name

        # Handle duplicate filenames
        if target_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = source.stem
            suffix = source.suffix
            target_path = target_dir / f"{stem}_{timestamp}{suffix}"

        shutil.copy2(str(source), str(target_path))

        # Write log entry
        if log_file:
            log_entry = {
                "source": str(source),
                "target": str(target_path),
                "timestamp": datetime.now().isoformat(),
            }
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return target_path

    except Exception as e:
        console.print(f"[red]Error copying file: {e}[/red]")
        return None


def sanitize_folder_name(name: str) -> str:
    """
    Sanitize a string to be a valid folder name.
    """
    # Replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    # Fallback if empty
    if not sanitized:
        sanitized = "Unnamed"
    return sanitized
