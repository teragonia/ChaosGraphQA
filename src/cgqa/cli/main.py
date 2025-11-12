"""Main CLI entry point for ChaosGraphQA."""

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import click
from rich import print as rprint
from rich.console import Console
from rich.progress import track
from rich.table import Table

from ..evaluators.ground_truth import GroundTruthVerifier
from ..generators import GeneratorType
from ..generators.conflicting import ConflictingGenerator
from ..generators.hierarchical import HierarchicalGenerator
from ..generators.multihop import MultiHopGenerator
from ..generators.temporal import TemporalGenerator
from ..generators.weighted import WeightedGenerator
from ..generators.base_generator import BaseGraphGenerator
from ..llm.evaluation.llm_evaluator import LLMEvaluator
from ..llm.evaluation.provider_factory import ProviderFactory
from ..models.graph import KnowledgeGraph
from ..models.question import QuestionType
from ..questions.templates import QuestionGenerator
from ..utils.directory_manager import DirectoryManager, get_default_directory_manager

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """ChaosGraphQA (CGQA)

    A comprehensive benchmark for testing reasoning capabilities of LLMs
    using dynamically generated knowledge graphs.
    """
    pass


@cli.command()
@click.option(
    "--generator-type",
    type=click.Choice(
        ["multihop", "hierarchical", "temporal", "weighted", "conflicting"]
    ),
    default="multihop",
    help="Type of reasoning to generate",
)
@click.option(
    "--complexity", type=click.IntRange(1, 4), default=1, help="Complexity level (1-4)"
)
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility")
@click.option(
    "--num-questions", type=int, default=5, help="Number of questions to generate"
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path (default: auto-generated in organized structure)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Base output directory (default: ./cgqa_outputs)",
)
@click.option("--verify", is_flag=True, help="Verify ground truth answers")
@click.option(
    "--output-info",
    type=click.Path(),
    default=None,
    help="Write benchmark path info to this file (for programmatic access)",
)
def generate(
    generator_type: str,
    complexity: int,
    seed: Optional[int],
    num_questions: int,
    output: Optional[str],
    output_dir: Optional[str],
    verify: bool,
    output_info: Optional[str],
) -> None:
    """Generate a benchmark dataset."""

    # Initialize directory manager
    dir_manager: DirectoryManager = (
        DirectoryManager(output_dir) if output_dir else get_default_directory_manager()
    )

    console.print(f"[bold blue]Generating {generator_type} benchmark...[/bold blue]")
    console.print(f"Complexity: {complexity}, Questions: {num_questions}")
    console.print(f"Output directory: [dim]{dir_manager.base_dir}[/dim]")

    if seed:
        console.print(f"Using seed: {seed}")

    try:
        # Generate knowledge graph
        if generator_type == "multihop":
            generator: GeneratorType = MultiHopGenerator(complexity_level=complexity, seed=seed)
        elif generator_type == "hierarchical":
            generator = HierarchicalGenerator(complexity_level=complexity, seed=seed)
        elif generator_type == "temporal":
            generator = TemporalGenerator(complexity_level=complexity, seed=seed)
        elif generator_type == "weighted":
            generator = WeightedGenerator(complexity_level=complexity, seed=seed)
        elif generator_type == "conflicting":
            generator = ConflictingGenerator(complexity_level=complexity, seed=seed)
        else:
            console.print(f"[red]Error: Unknown generator type: {generator_type}[/red]")
            sys.exit(1)

        with console.status("[bold green]Generating knowledge graph..."):
            kg = generator.generate()

        console.print(
            f"[green]✓[/green] Generated graph with {len(kg.entities)} entities and {len(kg.relationships)} relationships"
        )

        # Generate questions
        question_gen = QuestionGenerator(seed=seed)

        # Map generator type to question type
        question_type_mapping = {
            "multihop": QuestionType.MULTIHOP,
            "hierarchical": QuestionType.HIERARCHICAL,
            "temporal": QuestionType.TEMPORAL,
            "weighted": QuestionType.WEIGHTED,
            "conflicting": QuestionType.CONFLICTING,
        }

        target_question_type = question_type_mapping[generator_type]

        with console.status("[bold green]Generating questions..."):
            question_set = question_gen.generate_questions(
                kg,
                question_types=[target_question_type],
                num_questions_per_type=num_questions,
                complexity_levels=[complexity],
            )

        console.print(
            f"[green]✓[/green] Generated {len(question_set.questions)} questions"
        )

        # Verify ground truth if requested
        if verify:
            with console.status("[bold green]Verifying ground truth..."):
                verifier = GroundTruthVerifier(kg)
                verification_results = verifier.verify_question_set(
                    question_set.questions
                )

            validity_rate = verification_results["validity_rate"]
            if validity_rate == 1.0:
                console.print(f"[green]✓[/green] All questions verified successfully")
            else:
                console.print(
                    f"[yellow]⚠[/yellow] {validity_rate:.1%} of questions verified successfully"
                )

        # Determine output path
        if output:
            output_path = Path(output)
        else:
            output_path = dir_manager.get_benchmark_path(
                generator_type=generator_type, complexity=complexity, seed=seed
            )

        # Save to file
        benchmark_data = {
            "metadata": {
                "generator_type": generator_type,
                "complexity_level": complexity,
                "seed": seed,
                "num_questions": len(question_set.questions),
                "verified": verify,
                "output_path": str(output_path),
            },
            "knowledge_graph": kg.model_dump(),
            "questions": question_set.model_dump(),
        }

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(benchmark_data, f, indent=2, default=str)

        console.print(f"[green]✓[/green] Benchmark saved to [bold]{output_path}[/bold]")
        console.print(f"[dim]   Organized under: {output_path.parent}[/dim]")

        # Write output info to file if requested
        if output_info:
            info_data = {
                "benchmark_path": str(output_path),
                "generator_type": generator_type,
                "complexity": complexity,
                "seed": seed,
                "num_questions": len(question_set.questions),
            }
            with open(output_info, "w") as f:
                json.dump(info_data, f, indent=2)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("benchmark_file", type=click.Path(exists=True))
@click.option(
    "--model",
    required=True,
    help="Model to evaluate (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet', 'gemini/gemini-1.5-flash')",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output file for results (default: auto-generated in organized structure)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Base output directory (default: ./cgqa_outputs)",
)
@click.option(
    "--api-key", help="API key for the LLM provider (or set environment variable)"
)
@click.option(
    "--temperature", type=float, default=0.1, help="Sampling temperature (0.0-2.0)"
)
@click.option("--max-tokens", type=int, default=1000, help="Maximum tokens to generate")
@click.option(
    "--no-context",
    is_flag=True,
    help="Don't include knowledge graph context in prompts",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Process questions in batches (for rate limiting)",
)
@click.option(
    "--output-info",
    type=click.Path(),
    default=None,
    help="Write result path info to this file (for programmatic access)",
)
def evaluate(
    benchmark_file: str,
    model: str,
    output: Optional[str],
    output_dir: Optional[str],
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    no_context: bool,
    batch_size: Optional[int],
    output_info: Optional[str],
):
    """Evaluate a model on a benchmark dataset."""

    # Initialize directory manager
    dir_manager: DirectoryManager = (
        DirectoryManager(output_dir) if output_dir else get_default_directory_manager()
    )

    console.print(f"[bold blue]Evaluating benchmark: {benchmark_file}[/bold blue]")
    console.print(f"Model: [bold]{model}[/bold]")
    console.print(f"Output directory: [dim]{dir_manager.base_dir}[/dim]")

    try:
        # Load benchmark
        with open(benchmark_file, "r") as f:
            benchmark_data = json.load(f)

        # Reconstruct objects
        kg_data = benchmark_data["knowledge_graph"]
        questions_data = benchmark_data["questions"]

        # Create knowledge graph
        kg = KnowledgeGraph(**kg_data)

        # Create questions list
        from ..models.question import Question

        questions = [Question(**q_data) for q_data in questions_data["questions"]]

        console.print(f"Questions: {len(questions)}")
        console.print(
            f"Knowledge Graph: {len(kg.entities)} entities, {len(kg.relationships)} relationships"
        )

        # Create LLM evaluator
        with console.status("[bold green]Setting up LLM provider..."):
            provider_config = {
                "api_key": api_key,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            try:
                evaluator = LLMEvaluator.from_model_string(model, **provider_config)
                console.print(
                    f"[green]✓[/green] Connected to {evaluator.provider.provider_name}"
                )

                # Test connection
                if not evaluator.provider.test_connection():
                    console.print(
                        "[yellow]⚠[/yellow] Connection test failed, but continuing..."
                    )

            except Exception as e:
                console.print(f"[red]Error setting up LLM provider: {str(e)}[/red]")
                console.print("\n[bold]Available providers and examples:[/bold]")
                console.print("  OpenAI: --model openai/gpt-4o-mini")
                console.print("  Anthropic: --model anthropic/claude-3.5-sonnet")
                console.print("  Gemini: --model gemini/gemini-1.5-flash")
                console.print("  HuggingFace: --model huggingface/distilgpt2")
                console.print("\n[bold]Environment variables:[/bold]")
                console.print(
                    "  OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, HF_TOKEN"
                )
                sys.exit(1)

        # Run evaluation
        include_context = not no_context
        console.print(f"Context: {'Enabled' if include_context else 'Disabled'}")

        if batch_size:
            console.print(f"Batch size: {batch_size}")
            # Process in batches
            all_results = []
            for i in range(0, len(questions), batch_size):
                batch = questions[i : i + batch_size]
                console.print(
                    f"\n[bold]Processing batch {i//batch_size + 1}[/bold] ({len(batch)} questions)"
                )

                batch_summary = evaluator.evaluate_questions(
                    batch, kg, include_context=include_context, show_progress=True
                )
                all_results.extend(batch_summary.results)

                # Brief pause between batches
                if i + batch_size < len(questions):
                    console.print("Pausing between batches...")
                    time.sleep(1.0)

            # Create combined summary
            total_time = sum(r.response_time for r in all_results)
            summary = evaluator._create_summary(all_results, total_time)

        else:
            # Process all at once
            summary = evaluator.evaluate_questions(
                questions, kg, include_context=include_context, show_progress=True
            )

        # Display results
        console.print(f"\n[bold green]Evaluation Complete![/bold green]")
        console.print(f"Accuracy: {summary.accuracy:.1%}")
        console.print(f"Average Score: {summary.average_score:.3f}")
        console.print(f"Total Time: {summary.evaluation_time:.1f}s")

        if summary.total_tokens_used > 0:
            console.print(f"Tokens Used: {summary.total_tokens_used:,}")
            console.print(
                f"Avg Tokens/Question: {summary.average_tokens_per_question:.0f}"
            )

        if summary.error_rate > 0:
            console.print(f"[yellow]Error Rate: {summary.error_rate:.1%}[/yellow]")

        # Determine output path
        if output:
            output_path = Path(output)
        else:
            benchmark_name = Path(benchmark_file).stem
            output_path = dir_manager.get_evaluation_path(
                benchmark_name=benchmark_name, model_name=model
            )

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        evaluator.save_results(summary, output_path)
        console.print(f"[green]✓[/green] Results saved to [bold]{output_path}[/bold]")
        console.print(f"[dim]   Organized under: {output_path.parent}[/dim]")

        # Write output info to file if requested
        if output_info:
            info_data = {
                "result_path": str(output_path),
                "benchmark_file": benchmark_file,
                "model": model,
                "accuracy": summary.accuracy,
                "evaluation_time": summary.evaluation_time,
                "total_tokens": summary.total_tokens_used,
            }
            with open(output_info, "w") as f:
                json.dump(info_data, f, indent=2)

        # Show breakdown by complexity if available
        if summary.by_complexity:
            console.print("\n[bold]Performance by Complexity:[/bold]")
            for level in sorted(summary.by_complexity.keys()):
                stats = summary.by_complexity[level]
                console.print(
                    f"  Level {level}: {stats['accuracy']:.1%} accuracy ({stats['count']} questions)"
                )

    except KeyboardInterrupt:
        console.print("\n[yellow]Evaluation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


@cli.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--format",
    type=click.Choice(["table", "json", "html"]),
    default="table",
    help="Output format",
)
def analyze(results_file: str, format: str):
    """Analyze evaluation results."""

    console.print(f"[bold blue]Analyzing results: {results_file}[/bold blue]")

    try:
        with open(results_file, "r") as f:
            results_data = json.load(f)

        if format == "table":
            _display_results_table(results_data)
        else:
            console.print(f"[yellow]{format} format not yet implemented[/yellow]")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


def _display_results_table(results_data):
    """Display results in a table format."""
    table = Table(title="Evaluation Results")

    table.add_column("Metric", justify="left")
    table.add_column("Value", justify="right")

    # Add sample data (replace with actual results parsing)
    table.add_row("Total Questions", "50")
    table.add_row("Correct Answers", "42")
    table.add_row("Accuracy", "84.0%")
    table.add_row("Avg. Confidence", "0.76")

    console.print(table)


@cli.command()
@click.argument("benchmark_file", type=click.Path(exists=True))
def info(benchmark_file: str):
    """Show information about a benchmark dataset."""

    try:
        with open(benchmark_file, "r") as f:
            benchmark_data = json.load(f)

        metadata = benchmark_data.get("metadata", {})
        kg_data = benchmark_data.get("knowledge_graph", {})
        questions_data = benchmark_data.get("questions", {})

        # Display metadata
        console.print(f"[bold]Benchmark Information[/bold]")
        console.print(f"Generator Type: {metadata.get('generator_type', 'Unknown')}")
        console.print(
            f"Complexity Level: {metadata.get('complexity_level', 'Unknown')}"
        )
        console.print(f"Seed: {metadata.get('seed', 'Random')}")
        console.print(f"Verified: {'Yes' if metadata.get('verified') else 'No'}")
        console.print()

        # Graph stats
        console.print(f"[bold]Knowledge Graph[/bold]")
        console.print(f"Entities: {len(kg_data.get('entities', {}))}")
        console.print(f"Relationships: {len(kg_data.get('relationships', []))}")
        console.print()

        # Question stats
        questions = questions_data.get("questions", [])
        console.print(f"[bold]Questions[/bold]")
        console.print(f"Total: {len(questions)}")

        # Question types breakdown
        type_counts = {}
        complexity_counts = {}

        for q in questions:
            q_type = q.get("question_type", "unknown")
            complexity = q.get("complexity_level", 0)

            type_counts[q_type] = type_counts.get(q_type, 0) + 1
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        if type_counts:
            console.print("By Type:")
            for q_type, count in type_counts.items():
                console.print(f"  {q_type}: {count}")

        if complexity_counts:
            console.print("By Complexity:")
            for level, count in sorted(complexity_counts.items()):
                console.print(f"  Level {level}: {count}")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Base output directory to list (default: ./cgqa_outputs)",
)
def list_files(output_dir: Optional[str]):
    """List organized benchmark and evaluation files."""

    dir_manager: DirectoryManager = (
        DirectoryManager(output_dir) if output_dir else get_default_directory_manager()
    )

    console.print(f"[bold]ChaosGraphQA Files[/bold]")
    console.print(f"Directory: {dir_manager.base_dir}\n")

    # List benchmarks
    benchmarks = dir_manager.list_benchmarks()
    if benchmarks:
        console.print("[bold blue]📁 Benchmarks[/bold blue]")
        for generator_type, complexities in benchmarks.items():
            console.print(f"  {generator_type}:")
            for complexity, files in complexities.items():
                console.print(f"    {complexity}: {len(files)} files")
                for file in files[:3]:  # Show first 3 files
                    console.print(f"      • {file}")
                if len(files) > 3:
                    console.print(f"      ... and {len(files) - 3} more")
        console.print()

    # List evaluations
    evaluations = dir_manager.list_evaluations()
    if evaluations:
        console.print("[bold green]📊 Evaluations[/bold green]")
        for benchmark, models in evaluations.items():
            console.print(f"  {benchmark}:")
            for model, files in models.items():
                console.print(f"    {model}: {len(files)} results")
        console.print()

    # Directory info
    structure_info = dir_manager.get_structure_info()
    console.print("[bold yellow]📈 Directory Statistics[/bold yellow]")
    console.print(f"  Total size: {structure_info['total_size_mb']} MB")
    for dir_name, stats in structure_info["directories"].items():
        console.print(f"  {dir_name}: {stats['files']} files, {stats['size_mb']} MB")


@cli.command()
@click.option(
    "--max-age", type=int, default=24, help="Maximum age in hours for temp file cleanup"
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Base output directory (default: ./cgqa_outputs)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be cleaned up without actually deleting",
)
def cleanup(max_age: int, output_dir: Optional[str], dry_run: bool):
    """Clean up old temporary files and organize outputs."""

    dir_manager: DirectoryManager = (
        DirectoryManager(output_dir) if output_dir else get_default_directory_manager()
    )

    console.print(f"[bold]Cleaning up ChaosGraphQA files[/bold]")
    console.print(f"Directory: {dir_manager.base_dir}")
    console.print(f"Max age: {max_age} hours")

    if dry_run:
        console.print("[yellow]DRY RUN - No files will be deleted[/yellow]\n")
    else:
        console.print()

    if not dry_run:
        cleaned_count = dir_manager.cleanup_temp(max_age)
        if cleaned_count > 0:
            console.print(
                f"[green]✓[/green] Cleaned up {cleaned_count} temporary files"
            )
        else:
            console.print("[dim]No temporary files to clean up[/dim]")
    else:
        temp_dir = dir_manager.base_dir / "temp"
        if temp_dir.exists():
            import time

            current_time = time.time()
            max_age_seconds = max_age * 3600

            old_files = []
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        old_files.append(file_path)

            if old_files:
                console.print(
                    f"[yellow]Would clean up {len(old_files)} files:[/yellow]"
                )
                for file_path in old_files[:10]:  # Show first 10
                    console.print(f"  • {file_path.name}")
                if len(old_files) > 10:
                    console.print(f"  ... and {len(old_files) - 10} more")
            else:
                console.print("[dim]No temporary files to clean up[/dim]")

    # Show structure info
    structure_info = dir_manager.get_structure_info()
    console.print(f"\n[bold]Directory Status:[/bold]")
    console.print(f"Total size: {structure_info['total_size_mb']} MB")


@cli.command()
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Base output directory (default: ./cgqa_outputs)",
)
def init_structure(output_dir: Optional[str]):
    """Initialize organized directory structure."""

    dir_manager: DirectoryManager = (
        DirectoryManager(output_dir) if output_dir else get_default_directory_manager()
    )

    console.print(f"[bold]Initializing ChaosGraphQA directory structure[/bold]")
    console.print(f"Location: {dir_manager.base_dir}\n")

    # Create README
    dir_manager.create_directory_readme()

    console.print("[green]✓[/green] Directory structure initialized")
    console.print("[green]✓[/green] README.md created")

    structure_info = dir_manager.get_structure_info()
    console.print("\n[bold]Created directories:[/bold]")
    for dir_name in structure_info["directories"].keys():
        dir_path = dir_manager.base_dir / dir_name
        console.print(f"  📁 {dir_name}/ ({dir_path})")

    console.print(
        "\n[dim]Files will now be automatically organized into these directories.[/dim]"
    )


@cli.command()
def list_models():
    """List available LLM models by provider."""

    console.print("[bold]Available LLM Models[/bold]\n")

    supported_models = ProviderFactory.list_supported_models()

    for provider_name, models in supported_models.items():
        # Test provider availability
        is_available, status = ProviderFactory.test_provider_availability(provider_name)

        status_icon = "[green]✓[/green]" if is_available else "[red]✗[/red]"
        console.print(f"{status_icon} [bold]{provider_name.upper()}[/bold]")

        if not is_available:
            console.print(f"  [dim]{status}[/dim]")

        if models and is_available:
            # Show first few models as examples
            example_models = models[:5]
            for model in example_models:
                console.print(f"  • {model}")

            if len(models) > 5:
                console.print(f"  ... and {len(models) - 5} more")

        console.print()

    console.print("[bold]Usage Examples:[/bold]")
    console.print("  cgqa evaluate benchmark.json --model openai/gpt-4o-mini")
    console.print("  cgqa evaluate benchmark.json --model anthropic/claude-3.5-sonnet")
    console.print("  cgqa evaluate benchmark.json --model gemini/gemini-1.5-flash")
    console.print("  cgqa evaluate benchmark.json --model huggingface/distilgpt2")


@cli.command()
@click.option(
    "--model", required=True, help="Model to test (e.g., 'openai/gpt-4o-mini')"
)
@click.option("--api-key", help="API key for the provider")
def test_model(model: str, api_key: Optional[str]):
    """Test connection to an LLM provider."""

    console.print(f"[bold blue]Testing model: {model}[/bold blue]")

    try:
        # Create provider
        provider_config = {}
        if api_key:
            provider_config["api_key"] = api_key

        with console.status("[bold green]Setting up provider..."):
            evaluator = LLMEvaluator.from_model_string(model, **provider_config)

        console.print(f"[green]✓[/green] Provider: {evaluator.provider.provider_name}")
        console.print(f"[green]✓[/green] Model: {evaluator.provider.config.model_name}")

        # Test connection
        with console.status("[bold green]Testing connection..."):
            connection_ok = evaluator.provider.test_connection()

        if connection_ok:
            console.print("[green]✓[/green] Connection successful")

            # Show model info
            model_info = evaluator.provider.get_model_info()
            console.print(f"\n[bold]Model Information:[/bold]")
            console.print(f"Max tokens: {model_info.get('max_tokens', 'Unknown')}")
            console.print(f"Temperature: {model_info.get('temperature', 'Unknown')}")

            provider_specific = model_info.get("provider_specific", {})
            if provider_specific:
                console.print(
                    f"Context window: {provider_specific.get('context_window', 'Unknown')}"
                )
                console.print(
                    f"Training cutoff: {provider_specific.get('training_data_cutoff', 'Unknown')}"
                )
        else:
            console.print("[red]✗[/red] Connection failed")
            console.print("Check your API key and network connection")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
