import shutil
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from data_ai.pipeline.year_detect import detect_year
from data_ai.pipeline.extract_v2 import extract_text, SUPPORTED_EXTENSIONS
from data_ai.pipeline.cluster_v2 import cluster_documents, get_topic_keywords
from data_ai.pipeline.naming import generate_cluster_name


console = Console()


def scan_files(input_dir: Path) -> list[Path]:
    """Scan directory for supported files."""
    files = []
    for file_path in input_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file_path)
    return files


def group_by_year(files: list[Path]) -> dict[int, list[Path]]:
    """Group files by detected year."""
    groups: dict[int, list[Path]] = {}
    for file_path in files:
        year = detect_year(file_path)
        if year not in groups:
            groups[year] = []
        groups[year].append(file_path)
    return groups


def process_year_batch(
    files: list[Path],
    min_topic_size: int = 10,
    model: str = "llama3.2",
    progress: Progress | None = None,
    task_id: int | None = None,
) -> dict[str, list[Path]]:
    """
    Process a batch of files for one year.

    Returns: Dict mapping cluster_name to list of files
    """
    # Extract text from all files
    documents: list[tuple[Path, str]] = []

    for i, file_path in enumerate(files):
        if progress and task_id is not None:
            progress.update(task_id, advance=1)

        text = extract_text(file_path)
        if text:
            documents.append((file_path, text))

    if not documents:
        return {"_Sonstiges": files}

    # Cluster documents
    clusters, topic_model = cluster_documents(documents, min_topic_size=min_topic_size)

    if not clusters:
        return {"_Sonstiges": files}

    # Generate names for each cluster
    result: dict[str, list[Path]] = {}

    for topic_id, topic_files in clusters.items():
        keywords = get_topic_keywords(topic_model, topic_id)
        sample_names = [f.name for f in topic_files[:5]]

        name = generate_cluster_name(keywords, sample_names, model=model)

        # Handle duplicate names
        original_name = name
        counter = 1
        while name in result:
            name = f"{original_name}_{counter}"
            counter += 1

        result[name] = topic_files

    return result


def copy_files(
    clusters: dict[int, dict[str, list[Path]]],
    output_dir: Path,
    dry_run: bool = False,
) -> int:
    """
    Copy files to output directory structure.

    Args:
        clusters: Dict of year -> cluster_name -> files
        output_dir: Target directory
        dry_run: If True, don't actually copy

    Returns:
        Number of files copied
    """
    count = 0

    for year, year_clusters in sorted(clusters.items()):
        year_dir = output_dir / str(year)

        for cluster_name, files in year_clusters.items():
            cluster_dir = year_dir / cluster_name

            if not dry_run:
                cluster_dir.mkdir(parents=True, exist_ok=True)

            for file_path in files:
                target = cluster_dir / file_path.name

                # Handle duplicates
                if target.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    counter = 1
                    while target.exists():
                        target = cluster_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                if not dry_run:
                    shutil.copy2(file_path, target)

                count += 1

    return count


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    min_topic_size: int = 10,
    model: str = "llama3.2",
    dry_run: bool = False,
) -> None:
    """
    Run the complete pipeline.

    1. Scan for files
    2. Group by year
    3. Process each year (extract, cluster, name)
    4. Copy to output
    """
    console.print(f"\n[bold]Scanning {input_dir}...[/bold]")
    files = scan_files(input_dir)
    console.print(f"Found [green]{len(files)}[/green] files\n")

    if not files:
        console.print("[yellow]No files to process.[/yellow]")
        return

    console.print("[bold]Detecting years...[/bold]")
    year_groups = group_by_year(files)

    for year in sorted(year_groups.keys()):
        console.print(f"  {year}: [cyan]{len(year_groups[year])}[/cyan] files")
    console.print()

    all_clusters: dict[int, dict[str, list[Path]]] = {}

    for year in sorted(year_groups.keys()):
        year_files = year_groups[year]
        console.print(f"[bold]Processing {year} ({len(year_files)} files)...[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting & clustering", total=len(year_files))

            clusters = process_year_batch(
                year_files,
                min_topic_size=min_topic_size,
                model=model,
                progress=progress,
                task_id=task,
            )

        all_clusters[year] = clusters

        console.print(f"  Found [green]{len(clusters)}[/green] topics:")
        for name, topic_files in sorted(clusters.items(), key=lambda x: -len(x[1])):
            console.print(f"    - {name}: {len(topic_files)} files")
        console.print()

    console.print(f"[bold]{'Would copy' if dry_run else 'Copying'} files to {output_dir}...[/bold]")
    copied = copy_files(all_clusters, output_dir, dry_run=dry_run)

    if dry_run:
        console.print(f"\n[yellow]Dry run:[/yellow] Would copy {copied} files")
    else:
        console.print(f"\n[green]Done![/green] Copied {copied} files")
