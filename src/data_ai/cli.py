# src/data_ai/cli.py
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from data_ai.config import load_config, get_default_config_path, create_default_config

app = typer.Typer(
    name="data-ai",
    help="Intelligent file organizer using semantic similarity",
)
console = Console()


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


if __name__ == "__main__":
    app()
