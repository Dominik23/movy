# src/data_ai/cli_v2.py
"""
data-ai CLI v2

The `run` command uses the new v2 pipeline (Docling + BERTopic).
Legacy commands (init, status, scan, cluster, review, apply) use the old pipeline and require Qdrant.
"""
import time
import webbrowser
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from uuid import uuid4

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table

# New v2 pipeline import (no heavy dependencies)
from data_ai.pipeline.run import run_pipeline

# Lazy imports for legacy commands that need Qdrant
if TYPE_CHECKING:
    from data_ai.config import load_config, get_default_config_path, create_default_config
    from data_ai.storage import QdrantStore, Document, Cluster, DocumentStatus, ClusterStatus
    from data_ai.pipeline.extract import scan_folder, extract_stage
    from data_ai.pipeline.embed import embed_stage
    from data_ai.pipeline.cluster import cluster_documents
    from data_ai.pipeline.naming import generate_cluster_name
    from data_ai.pipeline.review import generate_review_html
    from data_ai.pipeline.execute import execute_copy, sanitize_folder_name
    from data_ai.utils.similarity import compute_variance


def _import_legacy():
    """Lazy import legacy modules (requires Qdrant and old dependencies)."""
    global load_config, get_default_config_path, create_default_config
    global QdrantStore, Document, Cluster, DocumentStatus, ClusterStatus
    global scan_folder, extract_stage, embed_stage, cluster_documents
    global generate_cluster_name, generate_review_html, execute_copy, sanitize_folder_name
    global compute_variance

    from data_ai.config import load_config, get_default_config_path, create_default_config
    from data_ai.storage import QdrantStore, Document, Cluster, DocumentStatus, ClusterStatus
    from data_ai.pipeline.extract import scan_folder, extract_stage
    from data_ai.pipeline.embed import embed_stage
    from data_ai.pipeline.cluster import cluster_documents
    from data_ai.pipeline.naming import generate_cluster_name as _gen_name
    generate_cluster_name = _gen_name
    from data_ai.pipeline.review import generate_review_html
    from data_ai.pipeline.execute import execute_copy, sanitize_folder_name
    from data_ai.utils.similarity import compute_variance

app = typer.Typer(
    name="data-ai",
    help="Intelligent file organizer using vector clustering",
)
console = Console()


def get_store(config_path: Optional[Path] = None) -> "QdrantStore":
    _import_legacy()
    from data_ai.config import load_config, get_default_config_path
    from data_ai.storage import QdrantStore
    path = config_path or get_default_config_path()
    if path.exists():
        cfg = load_config(path)
        return QdrantStore(url=cfg.settings.qdrant_url, prefix=cfg.settings.qdrant_collection_prefix)
    return QdrantStore()


@app.command()
def init(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    db_url: str = typer.Option("localhost:6333", "--db-url"),
) -> None:
    """Initialize config and verify Qdrant connection."""
    _import_legacy()
    from data_ai.config import get_default_config_path, create_default_config
    from data_ai.storage import QdrantStore
    path = config_path or get_default_config_path()

    if not path.exists():
        create_default_config(path)
        console.print(f"[green]Created config at: {path}[/green]")

    # Test Qdrant connection
    try:
        store = QdrantStore(url=db_url)
        console.print(f"[green]Connected to Qdrant at {db_url}[/green]")
    except Exception as e:
        console.print(f"[red]Could not connect to Qdrant: {e}[/red]")
        console.print("Make sure Qdrant is running: docker-compose up -d")
        raise typer.Exit(1)


@app.command()
def status(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Show current pipeline status."""
    store = get_store(config_path)

    docs = store.get_all_documents()
    clusters = store.get_all_clusters()

    # Count by status
    doc_status_counts: dict[str, int] = {}
    for doc in docs:
        status_val = doc.status.value if hasattr(doc.status, 'value') else doc.status
        doc_status_counts[status_val] = doc_status_counts.get(status_val, 0) + 1

    cluster_status_counts: dict[str, int] = {}
    for cluster in clusters:
        status_val = cluster.status.value if hasattr(cluster.status, 'value') else cluster.status
        cluster_status_counts[status_val] = cluster_status_counts.get(status_val, 0) + 1

    console.print("\n[bold]Documents:[/bold]")
    table = Table()
    table.add_column("Status")
    table.add_column("Count", justify="right")

    for status_val, count in sorted(doc_status_counts.items()):
        table.add_row(status_val, str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{len(docs)}[/bold]")

    console.print(table)

    console.print("\n[bold]Clusters:[/bold]")
    table = Table()
    table.add_column("Status")
    table.add_column("Count", justify="right")

    for status_val, count in sorted(cluster_status_counts.items()):
        table.add_row(status_val, str(count))
    table.add_row("[bold]Total[/bold]", f"[bold]{len(clusters)}[/bold]")

    console.print(table)


@app.command()
def reset(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    confirm: bool = typer.Option(False, "--confirm", help="Confirm reset"),
) -> None:
    """Reset all data in Qdrant."""
    if not confirm:
        console.print("[red]This will delete all documents and clusters![/red]")
        console.print("Run with [green]--confirm[/green] to proceed")
        raise typer.Exit(1)

    store = get_store(config_path)
    store.reset()

    console.print("[green]Database reset complete[/green]")


@app.command()
def scan(
    folder: Path = typer.Argument(..., help="Folder to scan"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    trash_dir: Optional[Path] = typer.Option(None, "--trash", "-t", help="Move unsupported files here"),
) -> None:
    """Scan folder, extract text, create embeddings, and store in Qdrant."""
    if not folder.exists():
        console.print(f"[red]Folder does not exist: {folder}[/red]")
        raise typer.Exit(1)

    path = config_path or get_default_config_path()
    if path.exists():
        cfg = load_config(path)
    else:
        from data_ai.config import Settings
        cfg = type('Config', (), {'settings': Settings()})()

    store = get_store(config_path)

    # Step 1: Scan folder
    console.print(f"[blue]Scanning {folder}...[/blue]")
    supported_files, trash_log = scan_folder(folder, trash_dir)

    if trash_log:
        console.print(f"[yellow]Moved {len(trash_log)} unsupported files to trash[/yellow]")

    if not supported_files:
        console.print("[yellow]No supported files found[/yellow]")
        return

    console.print(f"[green]Found {len(supported_files)} supported files[/green]")

    # Step 2: Extract text and create embeddings
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files...", total=len(supported_files))

        for file_path in supported_files:
            progress.update(task, description=f"Processing {file_path.name}...")

            # Extract text
            text = extract_stage(file_path, vision_model=cfg.settings.vision_model)

            if not text:
                console.print(f"[yellow]Could not extract text from {file_path.name}[/yellow]")
                progress.advance(task)
                continue

            # Truncate summary if needed
            summary = text[:cfg.settings.summary_length] if len(text) > cfg.settings.summary_length else text

            # Create embedding
            try:
                vector = embed_stage(text, model=cfg.settings.ollama_model)
            except Exception as e:
                console.print(f"[red]Embedding failed for {file_path.name}: {e}[/red]")
                progress.advance(task)
                continue

            # Create document
            doc = Document(
                id=store.generate_id(),
                source_path=str(file_path.absolute()),
                file_type=file_path.suffix.lower(),
                file_size=file_path.stat().st_size,
                summary=summary,
                status=DocumentStatus.EMBEDDED,
                vector=vector,
            )

            store.upsert_document(doc)
            progress.advance(task)

            # Small delay to avoid overwhelming Ollama
            time.sleep(0.2)

    # Show final count
    docs = store.get_documents_by_status(DocumentStatus.EMBEDDED)
    console.print(f"[green]Processed {len(docs)} documents, ready for clustering[/green]")


@app.command()
def cluster(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    min_cluster_size: Optional[int] = typer.Option(None, "--min-cluster-size", help="Minimum cluster size"),
    recluster: bool = typer.Option(False, "--recluster", help="Delete existing clusters and recluster"),
) -> None:
    """Run clustering on embedded documents and generate names."""
    path = config_path or get_default_config_path()
    if path.exists():
        cfg = load_config(path)
    else:
        from data_ai.config import Settings
        cfg = type('Config', (), {'settings': Settings()})()

    store = get_store(config_path)

    # Delete existing clusters if reclustering
    if recluster:
        console.print("[yellow]Deleting existing clusters...[/yellow]")
        store.delete_all_clusters()

    # Get embedded documents
    docs = store.get_documents_by_status(DocumentStatus.EMBEDDED)

    if not docs:
        console.print("[yellow]No embedded documents found. Run 'scan' first.[/yellow]")
        return

    console.print(f"[blue]Clustering {len(docs)} documents...[/blue]")

    # Extract vectors
    vectors = [doc.vector for doc in docs if doc.vector]
    doc_ids = [doc.id for doc in docs if doc.vector]

    if len(vectors) < 2:
        console.print("[yellow]Need at least 2 documents for clustering[/yellow]")
        return

    # Get clustering parameters
    effective_min_cluster_size = min_cluster_size or cfg.settings.min_cluster_size
    umap_components = cfg.settings.umap_components

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running UMAP + HDBSCAN clustering...", total=None)

        # Run clustering
        labels, centroids, outlier_indices = cluster_documents(
            vectors,
            min_cluster_size=effective_min_cluster_size,
            umap_n_components=umap_components,
        )

        progress.update(task, description="Creating cluster records...")

        # Group documents by cluster
        cluster_docs: dict[int, list[tuple[str, Document]]] = {}
        outlier_docs: list[tuple[str, Document]] = []

        for idx, (doc_id, label) in enumerate(zip(doc_ids, labels)):
            doc = next(d for d in docs if d.id == doc_id)
            if label == -1:
                outlier_docs.append((doc_id, doc))
            else:
                if label not in cluster_docs:
                    cluster_docs[label] = []
                cluster_docs[label].append((doc_id, doc))

        progress.update(task, description="Generating cluster names...")

        # Create regular clusters
        for cluster_idx, doc_list in cluster_docs.items():
            cluster_id = store.generate_id()

            # Assign documents to cluster
            for doc_id, _ in doc_list:
                store.update_document_cluster(doc_id, cluster_id)

            # Generate name from summaries
            summaries = [doc.summary for _, doc in doc_list]
            name = generate_cluster_name(summaries, model=cfg.settings.chat_model)

            # Get centroid
            centroid = centroids[cluster_idx] if cluster_idx < len(centroids) else []

            # Calculate variance
            vectors_in_cluster = [doc.vector for _, doc in doc_list if doc.vector]
            variance = compute_variance(vectors_in_cluster) if vectors_in_cluster else 0.0

            cluster_record = Cluster(
                id=cluster_id,
                name=name,
                doc_count=len(doc_list),
                variance=variance,
                centroid=centroid,
                status=ClusterStatus.PROPOSED,
            )

            store.upsert_cluster(cluster_record)

        # Create outlier cluster if there are outliers
        if outlier_docs:
            outlier_cluster_id = store.generate_id()

            for doc_id, _ in outlier_docs:
                store.update_document_cluster(doc_id, outlier_cluster_id)

            outlier_cluster = Cluster(
                id=outlier_cluster_id,
                name="Nicht zuordenbar",
                doc_count=len(outlier_docs),
                variance=0.0,
                centroid=[],
                status=ClusterStatus.OUTLIER,
            )

            store.upsert_cluster(outlier_cluster)

    # Show results
    clusters = store.get_all_clusters()
    regular_clusters = [c for c in clusters if c.status != ClusterStatus.OUTLIER]
    outlier_cluster = next((c for c in clusters if c.status == ClusterStatus.OUTLIER), None)

    console.print(f"\n[green]Created {len(regular_clusters)} clusters:[/green]\n")

    table = Table()
    table.add_column("Name")
    table.add_column("Documents", justify="right")
    table.add_column("Variance", justify="right")

    for c in sorted(regular_clusters, key=lambda x: x.doc_count, reverse=True):
        table.add_row(c.name, str(c.doc_count), f"{c.variance:.2f}")

    console.print(table)

    if outlier_cluster:
        console.print(f"\n[yellow]Outliers: {outlier_cluster.doc_count} documents not assignable[/yellow]")


def _interactive_review(store: "QdrantStore") -> None:
    """Interactive TUI for reviewing and editing clusters."""
    while True:
        console.clear()
        clusters = store.get_all_clusters()
        clusters = sorted(clusters, key=lambda c: c.doc_count, reverse=True)

        # Display cluster table
        table = Table(title="Cluster Review")
        table.add_column("#", justify="right", style="cyan")
        table.add_column("Name")
        table.add_column("Docs", justify="right")
        table.add_column("Status", justify="center")

        for idx, cluster in enumerate(clusters, 1):
            status_style = "green" if cluster.status == ClusterStatus.APPROVED else "yellow"
            table.add_row(
                str(idx),
                cluster.name,
                str(cluster.doc_count),
                f"[{status_style}]{cluster.status.value}[/{status_style}]",
            )

        console.print(table)
        console.print()
        console.print("[dim]Enter number to edit, [a]pprove all, [q]uit and save[/dim]")

        choice = Prompt.ask("Choice").strip().lower()

        if choice == "q":
            console.print("[green]Changes saved.[/green]")
            break
        elif choice == "a":
            for cluster in clusters:
                if cluster.status == ClusterStatus.PROPOSED:
                    store.update_cluster_status(cluster.id, ClusterStatus.APPROVED)
            console.print("[green]All clusters approved.[/green]")
            continue

        # Check if it's a number
        try:
            idx = int(choice)
            if 1 <= idx <= len(clusters):
                _edit_cluster(store, clusters[idx - 1])
            else:
                console.print("[red]Invalid number[/red]")
                Prompt.ask("Press Enter to continue")
        except ValueError:
            console.print("[red]Invalid input[/red]")
            Prompt.ask("Press Enter to continue")


def _edit_cluster(store: "QdrantStore", cluster: "Cluster") -> None:
    """Show cluster details and allow editing."""
    while True:
        console.clear()

        # Get sample documents
        docs = store.get_documents_by_cluster(cluster.id)
        sample_files = [Path(doc.source_path).name for doc in docs[:10]]

        # Build content
        content = "[bold]Sample files:[/bold]\n"
        for fname in sample_files:
            content += f"  - {fname}\n"
        if len(docs) > 10:
            content += f"  [dim]... and {len(docs) - 10} more[/dim]\n"

        status_style = "green" if cluster.status == ClusterStatus.APPROVED else "yellow"
        content += f"\n[bold]Status:[/bold] [{status_style}]{cluster.status.value}[/{status_style}]"

        panel = Panel(
            content,
            title=f"Cluster: {cluster.name} ({cluster.doc_count} documents)",
            border_style="blue",
        )
        console.print(panel)
        console.print()
        console.print("[dim][a]pprove  [r]ename  [s]kip  [b]ack[/dim]")

        choice = Prompt.ask("Action").strip().lower()

        if choice == "b":
            break
        elif choice == "a":
            store.update_cluster_status(cluster.id, ClusterStatus.APPROVED)
            # Refresh cluster object
            clusters = store.get_all_clusters()
            for c in clusters:
                if c.id == cluster.id:
                    cluster = c
                    break
            console.print("[green]Cluster approved.[/green]")
        elif choice == "r":
            new_name = Prompt.ask("New name", default=cluster.name)
            if new_name and new_name != cluster.name:
                store.update_cluster_name(cluster.id, new_name)
                # Refresh cluster object
                clusters = store.get_all_clusters()
                for c in clusters:
                    if c.id == cluster.id:
                        cluster = c
                        break
                console.print(f"[green]Renamed to: {new_name}[/green]")
        elif choice == "s":
            # Skip means set back to proposed (won't be applied if there are approved clusters)
            store.update_cluster_status(cluster.id, ClusterStatus.PROPOSED)
            clusters = store.get_all_clusters()
            for c in clusters:
                if c.id == cluster.id:
                    cluster = c
                    break
            console.print("[yellow]Cluster marked as skipped (proposed).[/yellow]")


@app.command()
def review(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output HTML file path"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open in browser"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive TUI mode"),
) -> None:
    """Generate and optionally open HTML review."""
    path = config_path or get_default_config_path()
    if path.exists():
        cfg = load_config(path)
    else:
        from data_ai.config import Settings
        cfg = type('Config', (), {'settings': Settings()})()

    store = get_store(config_path)

    clusters = store.get_all_clusters()

    if not clusters:
        console.print("[yellow]No clusters found. Run 'cluster' first.[/yellow]")
        return

    # Interactive mode
    if interactive:
        _interactive_review(store)
        return

    # Build cluster_docs map
    cluster_docs: dict[str, list[str]] = {}
    for cluster in clusters:
        docs = store.get_documents_by_cluster(cluster.id)
        cluster_docs[cluster.id] = [Path(doc.source_path).name for doc in docs]

    # Generate HTML
    output_path = output or Path(cfg.settings.review_html)

    console.print(f"[blue]Generating review at {output_path}...[/blue]")
    generate_review_html(clusters, cluster_docs, output_path)

    console.print(f"[green]Review generated: {output_path}[/green]")

    if open_browser:
        webbrowser.open(f"file://{output_path.absolute()}")


@app.command()
def apply(
    target: Path = typer.Argument(..., help="Target directory for organized files"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    log_file: Optional[Path] = typer.Option(None, "--log", "-l", help="Log file for copy operations"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would be copied without copying"),
) -> None:
    """Copy files to target directory based on cluster assignments."""
    path = config_path or get_default_config_path()
    if path.exists():
        cfg = load_config(path)
    else:
        from data_ai.config import Settings
        cfg = type('Config', (), {'settings': Settings()})()

    store = get_store(config_path)

    clusters = store.get_all_clusters()
    approved_clusters = [c for c in clusters if c.status == ClusterStatus.APPROVED]
    outlier_clusters = [c for c in clusters if c.status == ClusterStatus.OUTLIER]

    if not approved_clusters:
        # If no approved clusters, use all proposed ones
        approved_clusters = [c for c in clusters if c.status == ClusterStatus.PROPOSED]
        if not approved_clusters and not outlier_clusters:
            console.print("[yellow]No clusters found. Run 'cluster' first.[/yellow]")
            return
        if approved_clusters:
            console.print("[yellow]No approved clusters. Using proposed clusters.[/yellow]")

    # Add outlier clusters to the list to process
    all_clusters_to_apply = approved_clusters + outlier_clusters

    console.print(f"[blue]Applying {len(all_clusters_to_apply)} clusters to {target}...[/blue]")

    log_path = log_file or (Path(cfg.settings.log_file) if hasattr(cfg.settings, 'log_file') else None)

    copied_count = 0
    failed_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Copying files...", total=len(all_clusters_to_apply))

        for cluster in all_clusters_to_apply:
            progress.update(task, description=f"Processing {cluster.name}...")

            docs = store.get_documents_by_cluster(cluster.id)

            # Special folder for outliers
            if cluster.status == ClusterStatus.OUTLIER:
                folder_name = "_nicht_zuordenbar"
            else:
                folder_name = sanitize_folder_name(cluster.name)

            target_dir = target / folder_name

            for doc in docs:
                source = Path(doc.source_path)

                if not source.exists():
                    console.print(f"[yellow]Source not found: {source}[/yellow]")
                    failed_count += 1
                    continue

                if dry_run:
                    console.print(f"[dim]Would copy: {source.name} -> {target_dir}[/dim]")
                    copied_count += 1
                else:
                    result = execute_copy(source, target_dir, log_path)
                    if result:
                        copied_count += 1
                        store.update_document_status(doc.id, DocumentStatus.APPLIED)
                    else:
                        failed_count += 1

            # Update cluster status
            if not dry_run:
                store.update_cluster_status(cluster.id, ClusterStatus.APPLIED)

            progress.advance(task)

    if dry_run:
        console.print(f"\n[blue]Dry run: would copy {copied_count} files[/blue]")
    else:
        console.print(f"\n[green]Copied {copied_count} files, {failed_count} failed[/green]")


@app.command()
def run(
    input_dir: Path = typer.Argument(..., help="Input directory to process"),
    output: Path = typer.Option(None, "--output", "-o", help="Output directory"),
    min_topic_size: int = typer.Option(10, "--min-topic-size", help="Minimum documents per cluster"),
    model: str = typer.Option("llama3.2", "--model", "-m", help="Ollama model for naming"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would happen without copying"),
) -> None:
    """
    Process documents: detect year, extract text, cluster, and organize.

    Example:
        data-ai run /input --output /output
    """
    if not input_dir.exists():
        console.print(f"[red]Error:[/red] Input directory does not exist: {input_dir}")
        raise typer.Exit(1)

    output_dir = output or (input_dir.parent / "output")

    run_pipeline(
        input_dir=input_dir,
        output_dir=output_dir,
        min_topic_size=min_topic_size,
        model=model,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
