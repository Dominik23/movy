# src/data_ai/cli.py
import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from data_ai.config import Config, load_config, get_default_config_path, create_default_config

app = typer.Typer(
    name="data-ai",
    help="Intelligent file organizer using semantic similarity",
)
console = Console()

SCAN_RESULT_FILE = Path.home() / ".cache" / "data-ai" / "last_scan.json"


def get_config(config_path: Optional[Path]) -> "Config":
    from data_ai.config import Config

    path = config_path or get_default_config_path()
    if not path.exists():
        console.print(f"[red]Config not found: {path}[/red]")
        console.print("Run [green]data-ai init[/green] to create a config file")
        raise typer.Exit(1)

    return load_config(path)


@app.command()
def init(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Create a default config file."""
    path = config_path or get_default_config_path()

    if path.exists():
        console.print(f"[yellow]Config already exists: {path}[/yellow]")
        raise typer.Exit(1)

    create_default_config(path)
    console.print(f"[green]Created config at: {path}[/green]")
    console.print("Edit this file to define your categories and keywords.")


@app.command()
def config(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Show current config."""
    cfg = get_config(config_path)

    console.print("\n[bold]Settings:[/bold]")
    console.print(f"  Ollama model: {cfg.settings.ollama_model}")
    console.print(f"  Vision model: {cfg.settings.vision_model}")
    console.print(f"  Threshold: {cfg.settings.similarity_threshold}")
    console.print(f"  Inbox: {cfg.settings.inbox}")


@app.command()
def categories(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """List all categories with keywords."""
    cfg = get_config(config_path)

    table = Table(title="Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Keywords", style="green")
    table.add_column("Examples", style="yellow")

    for name, cat in cfg.categories.items():
        keywords = ", ".join(cat.keywords[:3])
        if len(cat.keywords) > 3:
            keywords += f" (+{len(cat.keywords) - 3})"

        examples = str(len(cat.examples))
        table.add_row(name, keywords, examples)

    console.print(table)


@app.command()
def sort(
    inbox: Optional[Path] = typer.Argument(
        None, help="Directory to sort (uses config inbox if not specified)"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    target: Optional[Path] = typer.Option(
        None, "--target", "-t", help="Target base directory (default: parent of inbox)"
    ),
) -> None:
    """Sort files from inbox into categories."""
    from data_ai.pipeline import process_file

    cfg = get_config(config_path)

    source_dir = inbox or Path(cfg.settings.inbox)
    if not source_dir.exists():
        console.print(f"[red]Inbox not found: {source_dir}[/red]")
        raise typer.Exit(1)

    target_base = target or source_dir.parent

    files = [f for f in source_dir.iterdir() if f.is_file()]

    if not files:
        console.print("[yellow]No files to sort[/yellow]")
        return

    console.print(f"[bold]Sorting {len(files)} files...[/bold]\n")

    success = 0
    failed = 0

    for file_path in files:
        try:
            if process_file(file_path, cfg, target_base, config_path=config_path or get_default_config_path()):
                success += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            console.print("\n[yellow]Aborted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error processing {file_path.name}: {e}[/red]")
            failed += 1

    console.print(f"\n[bold]Done:[/bold] {success} sorted, {failed} skipped/failed")


@app.command()
def scan(
    inbox: Optional[Path] = typer.Argument(
        None, help="Directory to scan (uses config inbox if not specified)"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    target: Optional[Path] = typer.Option(
        None, "--target", "-t", help="Target base directory"
    ),
) -> None:
    """Scan files and show what would be sorted (dry run)."""
    from data_ai.pipeline import process_file

    cfg = get_config(config_path)

    source_dir = inbox or Path(cfg.settings.inbox)
    if not source_dir.exists():
        console.print(f"[red]Inbox not found: {source_dir}[/red]")
        raise typer.Exit(1)

    target_base = target or source_dir.parent

    files = [f for f in source_dir.iterdir() if f.is_file()]

    if not files:
        console.print("[yellow]No files to scan[/yellow]")
        return

    console.print(f"[bold]Scanning {len(files)} files...[/bold]\n")

    scan_results = []

    for file_path in files:
        try:
            process_file(file_path, cfg, target_base, dry_run=True)
            scan_results.append({
                "source": str(file_path),
                "target_base": str(target_base),
            })
        except Exception as e:
            console.print(f"[red]Error scanning {file_path.name}: {e}[/red]")

    # Save scan results
    SCAN_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCAN_RESULT_FILE.write_text(json.dumps({
        "config_path": str(config_path or get_default_config_path()),
        "files": scan_results,
    }))

    console.print(f"\n[dim]Run [green]data-ai apply[/green] to execute[/dim]")


@app.command()
def apply() -> None:
    """Execute the last scan."""
    from data_ai.pipeline import process_file

    if not SCAN_RESULT_FILE.exists():
        console.print("[red]No scan results found. Run [green]data-ai scan[/green] first.[/red]")
        raise typer.Exit(1)

    data = json.loads(SCAN_RESULT_FILE.read_text())
    config_path = Path(data["config_path"])
    files = data["files"]

    if not files:
        console.print("[yellow]No files in scan results[/yellow]")
        return

    cfg = load_config(config_path)

    console.print(f"[bold]Applying to {len(files)} files...[/bold]\n")

    success = 0
    failed = 0

    for item in files:
        source = Path(item["source"])
        target_base = Path(item["target_base"])

        if not source.exists():
            console.print(f"[yellow]Skipping (not found): {source.name}[/yellow]")
            failed += 1
            continue

        try:
            if process_file(source, cfg, target_base, config_path=config_path):
                success += 1
            else:
                failed += 1
        except KeyboardInterrupt:
            console.print("\n[yellow]Aborted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            failed += 1

    # Clear scan results
    SCAN_RESULT_FILE.unlink(missing_ok=True)

    console.print(f"\n[bold]Done:[/bold] {success} sorted, {failed} skipped/failed")


class SortHandler(FileSystemEventHandler):
    def __init__(self, config: Config, target_base: Path, config_path: Path):
        self.config = config
        self.target_base = target_base
        self.config_path = config_path
        self._processing = set()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return

        from data_ai.pipeline import process_file

        file_path = Path(event.src_path)

        # Avoid processing the same file twice
        if file_path in self._processing:
            return

        self._processing.add(file_path)

        # Wait a moment for file to be fully written
        time.sleep(0.5)

        try:
            console.print(f"\n[bold]New file:[/bold] {file_path.name}")
            process_file(file_path, self.config, self.target_base, config_path=self.config_path)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
        finally:
            self._processing.discard(file_path)


@app.command()
def watch(
    inbox: Optional[Path] = typer.Argument(
        None, help="Directory to watch (uses config inbox if not specified)"
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
    target: Optional[Path] = typer.Option(
        None, "--target", "-t", help="Target base directory"
    ),
) -> None:
    """Watch a directory and sort new files automatically."""
    cfg = get_config(config_path)

    source_dir = inbox or Path(cfg.settings.inbox)
    if not source_dir.exists():
        console.print(f"[red]Inbox not found: {source_dir}[/red]")
        raise typer.Exit(1)

    target_base = target or source_dir.parent

    console.print(f"[bold]Watching:[/bold] {source_dir}")
    console.print(f"[bold]Target:[/bold] {target_base}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    handler = SortHandler(cfg, target_base, config_path or get_default_config_path())
    observer = Observer()
    observer.schedule(handler, str(source_dir), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[yellow]Stopped watching[/yellow]")

    observer.join()


@app.command("test")
def test_file(
    file_path: Path = typer.Argument(..., help="File to test"),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config file path"
    ),
) -> None:
    """Test classification of a single file without moving it."""
    from data_ai.pipeline import build_category_embeddings
    from data_ai.pipeline.extract import extract_stage
    from data_ai.pipeline.embed import embed_stage
    from data_ai.pipeline.match import match_stage

    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)

    cfg = get_config(config_path)

    console.print(f"[bold]Testing:[/bold] {file_path.name}\n")

    # Extract
    console.print("[dim]Extracting text...[/dim]")
    text = extract_stage(file_path, vision_model=cfg.settings.vision_model)
    if not text:
        console.print("[red]Could not extract text from file[/red]")
        raise typer.Exit(1)

    console.print(f"[green]Extracted {len(text)} characters[/green]")
    console.print(f"[dim]Preview: {text[:100]}...[/dim]\n")

    # Embed
    console.print("[dim]Creating embedding...[/dim]")
    file_vector = embed_stage(text, model=cfg.settings.ollama_model)
    console.print(f"[green]Created {len(file_vector)}-dim vector[/green]\n")

    # Match
    console.print("[dim]Matching against categories...[/dim]")
    category_embeddings = build_category_embeddings(cfg, cfg.settings.ollama_model)

    # Get all scores
    from data_ai.utils.similarity import cosine_similarity

    scores = []
    for cat_emb in category_embeddings:
        score = cosine_similarity(file_vector, cat_emb.vector)
        scores.append((cat_emb.name, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    console.print("\n[bold]Results:[/bold]")
    for category, score in scores:
        threshold = cfg.settings.similarity_threshold
        if score >= threshold:
            console.print(f"  [green]✓ {category}: {score:.1%}[/green]")
        else:
            console.print(f"  [dim]✗ {category}: {score:.1%}[/dim]")

    best = scores[0]
    threshold = cfg.settings.similarity_threshold

    console.print()
    if best[1] >= threshold:
        console.print(f"[bold green]Would sort to: {best[0]}[/bold green]")
    else:
        console.print(f"[bold yellow]Would prompt user (best: {best[0]} at {best[1]:.1%})[/bold yellow]")


if __name__ == "__main__":
    app()
