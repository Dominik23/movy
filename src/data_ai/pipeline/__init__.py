# src/data_ai/pipeline/__init__.py
from pathlib import Path
from typing import Optional

from rich.console import Console

from data_ai.config import Config, get_default_config_path, add_keywords_to_config
from data_ai.pipeline.extract import extract_stage
from data_ai.pipeline.embed import embed_stage
from data_ai.pipeline.match import match_stage, MatchResult, CategoryEmbedding
from data_ai.pipeline.execute import execute_move, prompt_for_category
from data_ai.utils.similarity import average_vectors

console = Console()

# Cache for category embeddings
_category_embeddings_cache: dict[str, list[CategoryEmbedding]] = {}


def build_category_embeddings(
    config: Config,
    model: str = "nomic-embed-text",
) -> list[CategoryEmbedding]:
    cache_key = f"{id(config)}_{model}"

    if cache_key in _category_embeddings_cache:
        return _category_embeddings_cache[cache_key]

    embeddings = []

    for name, category in config.categories.items():
        vectors = []

        # Embed keywords
        for keyword in category.keywords:
            vec = embed_stage(keyword, model=model)
            vectors.append(vec)

        # Embed example documents
        for example_path in category.examples:
            path = Path(example_path)
            if path.exists():
                text = extract_stage(path)
                if text:
                    vec = embed_stage(text, model=model)
                    vectors.append(vec)

        if vectors:
            avg_vector = average_vectors(vectors)
            embeddings.append(CategoryEmbedding(name=name, vector=avg_vector))

    _category_embeddings_cache[cache_key] = embeddings
    return embeddings


def process_file(
    file_path: Path,
    config: Config,
    target_base: Path,
    dry_run: bool = False,
    config_path: Optional[Path] = None,
) -> bool:
    # Step 1: Extract
    text = extract_stage(file_path, vision_model=config.settings.vision_model)
    if not text:
        console.print(f"[yellow]Skipping {file_path.name}: no text extracted[/yellow]")
        return False

    # Step 2: Embed
    file_vector = embed_stage(text, model=config.settings.ollama_model)

    # Step 3: Match
    category_embeddings = build_category_embeddings(config, config.settings.ollama_model)
    match_result = match_stage(
        file_vector,
        category_embeddings,
        threshold=config.settings.similarity_threshold,
    )

    # Step 4: Execute
    if match_result:
        target_dir = target_base / match_result.category

        # Step 5: Auto-learn if confidence is high enough
        if (
            config.settings.auto_learn
            and match_result.confidence >= config.settings.learning_threshold
            and not dry_run
        ):
            _learn_from_document(
                text=text,
                category=match_result.category,
                config=config,
                config_path=config_path or get_default_config_path(),
            )

        if dry_run:
            console.print(
                f"[green]{file_path.name}[/green] → "
                f"[blue]{match_result.category}[/blue] ({match_result.confidence:.0%})"
            )
            return True
        return execute_move(file_path, target_dir)
    else:
        # Below threshold - get all matches for prompt
        all_matches = []
        for cat_emb in category_embeddings:
            from data_ai.utils.similarity import cosine_similarity
            score = cosine_similarity(file_vector, cat_emb.vector)
            all_matches.append((cat_emb.name, score))
        all_matches.sort(key=lambda x: x[1], reverse=True)

        if dry_run:
            best = all_matches[0] if all_matches else ("unknown", 0.0)
            console.print(
                f"[yellow]{file_path.name}[/yellow] → "
                f"[dim]uncertain (best: {best[0]} {best[1]:.0%})[/dim]"
            )
            return False

        chosen = prompt_for_category(file_path, all_matches)
        if chosen:
            # Learn from user choice
            if config.settings.auto_learn and not dry_run:
                _learn_from_document(
                    text=text,
                    category=chosen,
                    config=config,
                    config_path=config_path or get_default_config_path(),
                )
            target_dir = target_base / chosen
            return execute_move(file_path, target_dir)
        return False


def _learn_from_document(
    text: str,
    category: str,
    config: Config,
    config_path: Path,
) -> None:
    """Extract and save new keywords from a successfully matched document."""
    from data_ai.providers.ollama import extract_keywords

    existing_keywords = config.categories[category].keywords

    console.print(f"[dim]Learning from document...[/dim]")

    new_keywords = extract_keywords(
        text=text,
        category=category,
        existing_keywords=existing_keywords,
        model=config.settings.chat_model,
    )

    if new_keywords:
        add_keywords_to_config(config_path, category, new_keywords)
        console.print(f"[green]Learned new keywords:[/green] {', '.join(new_keywords)}")

        # Clear cache so new keywords are used
        _category_embeddings_cache.clear()
