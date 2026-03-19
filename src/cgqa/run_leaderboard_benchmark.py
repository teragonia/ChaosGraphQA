"""
Comprehensive benchmarking script for ChaosGraphQA leaderboard.
Tests multiple models across all reasoning types and complexity levels.
Runs each configuration 3 times and calculates mean and standard deviation.
"""

import json
import os
import statistics
import subprocess
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TypedDict

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ModelProgress(TypedDict):
    """Type definition for model progress tracking."""

    completed: int
    total: int
    completion_time: Optional[float]


# Models to benchmark
MODELS = [
    # Claude models
    "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-7-sonnet-20250219",
    "anthropic/claude-3-5-haiku-20241022",
    # OpenAI models
    "openai/gpt-5.2",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-4o",
    # Gemini models
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-3-pro-preview",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.0-flash",
    # HuggingFace models
    "huggingface/HuggingFaceTB/SmolLM3-3B",
]

# Benchmark configurations
REASONING_TYPES = [
    "multihop",
    "hierarchical",
    "temporal",
    "weighted",
    "conflicting",
]

COMPLEXITY_LEVELS = [1, 2, 3, 4]
NUM_QUESTIONS = 10  # Questions per complexity level
NUM_RUNS = 3  # Run each configuration 3 times

# Timeout configuration: 3 minutes per question = 60 minutes for 20 questions
TIMEOUT_PER_EVALUATION = 600  # 10 minutes (3 min/question × 10 questions)
MAX_RETRIES = 2  # Number of times to retry a timed-out evaluation
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N model completions
CHECKPOINT_FILE = "leaderboard_checkpoint.json"  # Checkpoint filename


class DisplayManager:
    """Manages the display with fixed progress bars at bottom and scrolling results above."""

    def __init__(self, models: List[str], total_tasks: int, start_time: float):
        self.models = models
        self.total_tasks = total_tasks
        self.start_time = start_time
        self.completed_tasks_count = 0
        self.model_progress: Dict[str, ModelProgress] = {
            model: {"completed": 0, "total": 0, "completion_time": None}
            for model in models
        }
        # Scale results log with number of models: keep it readable but not too small
        results_log_size = max(8, len(models) // 2)
        self.results_log: Deque[Tuple[str, str]] = deque(maxlen=results_log_size)
        self.lock = Lock()
        self.console = Console() if RICH_AVAILABLE else None

    def add_result(
        self,
        model: str,
        reasoning_type: str,
        complexity: int,
        run_num: int,
        accuracy: float,
        score: float,
    ) -> None:
        """Add a successful result to the log."""
        with self.lock:
            model_short = model.split("/")[-1].replace(".", "_").replace("-", "_")
            result_line = (
                f"✓ [{model_short}] {reasoning_type} C{complexity} Run{run_num}: "
                f"Acc={accuracy:.2%}, Score={score:.3f}"
            )
            self.results_log.append(("success", result_line))
            self.completed_tasks_count += 1

    def add_error(self, model: str, message: str) -> None:
        """Add an error message to the log."""
        with self.lock:
            model_short = model.split("/")[-1].replace(".", "_").replace("-", "_")
            result_line = f"✗ [{model_short}] {message}"
            self.results_log.append(("error", result_line))

    def add_info(self, message: str) -> None:
        """Add an info message to the log."""
        with self.lock:
            self.results_log.append(("info", message))

    def update_model_progress(self, model: str, completed: int, total: int) -> None:
        """Update progress for a specific model."""
        with self.lock:
            self.model_progress[model]["completed"] = completed
            self.model_progress[model]["total"] = total

    def increment_model_progress(self, model: str) -> None:
        """Increment completed count for a model."""
        with self.lock:
            self.model_progress[model]["completed"] += 1

    def mark_model_complete(self, model: str) -> None:
        """Mark a model as complete and record its completion time."""
        with self.lock:
            elapsed = time.time() - self.start_time
            self.model_progress[model]["completion_time"] = elapsed

    def set_model_totals(self, model_tasks: Dict[str, List]) -> None:
        """Set total tasks for each model."""
        with self.lock:
            for model, tasks in model_tasks.items():
                self.model_progress[model]["total"] = len(tasks)

    def render(self) -> Optional[Layout]:
        """Render the full display layout."""
        if not RICH_AVAILABLE:
            return None

        layout = Layout()

        # Calculate dynamic ratio based on number of models
        # Progress needs: 3 (borders/header) + len(models) lines
        # Results gets remaining space, minimum 5 lines for readability
        progress_ratio = 3 + len(self.models)
        results_ratio = max(
            5, len(self.models) // 2
        )  # Scale with models but keep readable

        layout.split_column(
            Layout(name="results", ratio=results_ratio),
            Layout(name="progress", ratio=progress_ratio),
        )

        # Render results section
        with self.lock:
            results_lines = []
            for result_type, line in self.results_log:
                if result_type == "success":
                    results_lines.append(Text(line, style="green"))
                elif result_type == "error":
                    results_lines.append(Text(line, style="red"))
                else:
                    results_lines.append(Text(line, style="cyan"))

            if not results_lines:
                results_lines.append(Text("Waiting for results...", style="dim"))

            results_panel = Panel(
                Group(*results_lines),
                title="[bold cyan]Recent Results[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
            layout["results"].update(results_panel)

            # Render progress section
            elapsed = time.time() - self.start_time
            rate = self.completed_tasks_count / elapsed if elapsed > 0 else 0
            remaining_time = (
                (self.total_tasks - self.completed_tasks_count) / rate
                if rate > 0
                else 0
            )

            progress_table = Table(
                show_header=True, header_style="bold magenta", box=None, padding=(0, 1)
            )
            progress_table.add_column("Model", style="cyan", width=40)
            progress_table.add_column("Progress", width=25)
            progress_table.add_column("Status", width=18)

            # Sort models by progress
            sorted_models = sorted(
                self.models,
                key=lambda m: (
                    self.model_progress[m]["completed"]
                    / max(self.model_progress[m]["total"], 1)
                ),
                reverse=True,
            )

            for model in sorted_models:
                model_short = model.split("/")[-1]
                prog = self.model_progress[model]

                # Always show all models, even if total is 0
                if prog["total"] == 0:
                    # Model has no tasks assigned
                    progress_table.add_row(
                        model_short, "[░░░░░░░░░░░░░░░] 0/0 (0.0%)", "⏭️  Skipped"
                    )
                else:
                    pct = prog["completed"] / prog["total"] * 100
                    bar_width = 15
                    filled = int(bar_width * prog["completed"] / prog["total"])
                    bar = "█" * filled + "░" * (bar_width - filled)

                    if prog["completed"] == prog["total"]:
                        # Model is complete - show completion time
                        if prog["completion_time"] is not None:
                            completion_mins = prog["completion_time"] / 60
                            status = f"✅ Done ({completion_mins:.1f}m)"
                        else:
                            status = "✅ Complete"
                    else:
                        status = "🔄 Running"

                    progress_table.add_row(
                        model_short,
                        f"[{bar}] {prog['completed']}/{prog['total']} ({pct:.1f}%)",
                        status,
                    )

            # Overall stats
            overall_pct = (
                (self.completed_tasks_count / self.total_tasks * 100)
                if self.total_tasks > 0
                else 0
            )
            progress_title = (
                f"[bold magenta]Overall Progress[/bold magenta]: "
                f"{self.completed_tasks_count}/{self.total_tasks} ({overall_pct:.1f}%) | "
                f"⏱️  {elapsed/60:.1f}min | "
                f"ETA: {remaining_time/60:.1f}min"
            )

            progress_panel = Panel(
                progress_table,
                title=progress_title,
                border_style="magenta",
                padding=(1, 2),
            )
            layout["progress"].update(progress_panel)

        return layout


def load_results(results_file: str) -> Optional[dict]:
    """Load evaluation results from JSON file."""
    try:
        with open(results_file, "r") as f:
            data = json.load(f)
            return {
                "accuracy": data["metrics"]["accuracy"],
                "average_score": data["metrics"]["average_score"],
                "total_tokens": data["metrics"].get("total_tokens_used", 0),
                "error_rate": data["metrics"].get("error_rate", 0),
            }
    except Exception as e:
        print(f"    ✗ Failed to load results: {e}")
        return None


def run_single_evaluation_with_benchmark(
    model: str,
    reasoning_type: str,
    complexity: int,
    run_num: int,
    benchmark_file: str,
    display_manager: Optional["DisplayManager"] = None,
) -> Optional[dict]:
    """Run evaluation for a model using a pre-generated benchmark file.

    Args:
        model: Model identifier
        reasoning_type: Type of reasoning being tested
        complexity: Complexity level (1-4)
        run_num: Run number (1-3)
        benchmark_file: Path to the shared benchmark file
        display_manager: Display manager for output

    Returns:
        Dictionary with evaluation results, or None on failure
    """
    model_short = model.split("/")[-1].replace(".", "_").replace("-", "_")
    timeout = TIMEOUT_PER_EVALUATION

    # Create structured output directories
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = (
        Path("results") / date_str / model_short / reasoning_type / f"c{complexity}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retry logic for timeouts
    for attempt in range(MAX_RETRIES + 1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        results_file = str(output_dir / f"results_run{run_num}_{timestamp}.json")

        eval_cmd = [
            "cgqa",
            "evaluate",
            benchmark_file,
            "--model",
            model,
            "--output",
            results_file,
        ]

        try:
            subprocess.run(
                eval_cmd, check=True, capture_output=True, text=True, timeout=timeout
            )

            # Load results
            results = load_results(results_file)

            if results and display_manager:
                if attempt > 0:
                    # Indicate this was a retry that succeeded
                    display_manager.add_result(
                        model,
                        reasoning_type,
                        complexity,
                        run_num,
                        results["accuracy"],
                        results["average_score"],
                    )
                    display_manager.add_info(
                        f"✓ [{model_short}] Retry succeeded on attempt {attempt + 1} ({reasoning_type} C{complexity} Run{run_num})"
                    )
                else:
                    display_manager.add_result(
                        model,
                        reasoning_type,
                        complexity,
                        run_num,
                        results["accuracy"],
                        results["average_score"],
                    )

            return results

        except subprocess.TimeoutExpired:
            # Clean up results file on timeout
            Path(results_file).unlink(missing_ok=True)

            if attempt < MAX_RETRIES:
                # Not the last attempt - retry
                if display_manager:
                    display_manager.add_info(
                        f"⏱️  [{model_short}] Timeout on attempt {attempt + 1}/{MAX_RETRIES + 1}, retrying... ({reasoning_type} C{complexity} Run{run_num})"
                    )
                continue
            else:
                # Last attempt failed - give up
                if display_manager:
                    display_manager.add_error(
                        model,
                        f"Evaluation timed out after {MAX_RETRIES + 1} attempts ({reasoning_type} C{complexity} Run{run_num})",
                    )
                return None

        except subprocess.CalledProcessError as e:
            # Don't retry on non-timeout errors
            if display_manager:
                display_manager.add_error(
                    model,
                    f"Evaluation failed: {reasoning_type} C{complexity} Run{run_num} - {str(e.stderr[:100])}",
                )
            # Clean up results file on failure
            Path(results_file).unlink(missing_ok=True)
            return None

    return None


def calculate_statistics(runs: List[dict]) -> Optional[dict]:
    """Calculate mean and standard deviation from multiple runs."""
    if not runs or len(runs) == 0:
        return None

    accuracies = [r["accuracy"] for r in runs]
    scores = [r["average_score"] for r in runs]
    tokens = [r["total_tokens"] for r in runs]
    error_rates = [r["error_rate"] for r in runs]

    return {
        "accuracy_mean": statistics.mean(accuracies),
        "accuracy_stdev": statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
        "score_mean": statistics.mean(scores),
        "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
        "tokens_mean": statistics.mean(tokens),
        "tokens_stdev": statistics.stdev(tokens) if len(tokens) > 1 else 0,
        "error_rate_mean": statistics.mean(error_rates),
        "num_runs": len(runs),
    }


def save_checkpoint(
    all_results: Dict,
    completed: int,
    total: int,
    start_time: float,
    completed_tasks: set,
) -> None:
    """Save progress checkpoint to resume interrupted runs.

    Args:
        all_results: Dictionary of results by model and config
        completed: Number of completed tasks
        total: Total number of tasks
        start_time: Timestamp when execution started
        completed_tasks: Set of completed task identifiers
    """
    checkpoint_data = {
        "timestamp": datetime.now().isoformat(),
        "completed": completed,
        "total": total,
        "elapsed_seconds": time.time() - start_time,
        "results": all_results,
        "completed_tasks": list(completed_tasks),
        "configuration": {
            "models": MODELS,
            "reasoning_types": REASONING_TYPES,
            "complexity_levels": COMPLEXITY_LEVELS,
            "num_questions": NUM_QUESTIONS,
            "num_runs": NUM_RUNS,
        },
    }

    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
    except Exception as e:
        print(f"⚠️  Warning: Failed to save checkpoint: {e}")


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint from previous interrupted run.

    Returns:
        Dictionary with checkpoint data, or None if no checkpoint exists
    """
    checkpoint_path = Path(CHECKPOINT_FILE)
    if not checkpoint_path.exists():
        return None

    try:
        with open(CHECKPOINT_FILE) as f:
            checkpoint_data: dict = json.load(f)

        # Validate checkpoint configuration matches current settings
        config = checkpoint_data.get("configuration", {})
        if (
            config.get("models") != MODELS
            or config.get("reasoning_types") != REASONING_TYPES
            or config.get("complexity_levels") != COMPLEXITY_LEVELS
            or config.get("num_questions") != NUM_QUESTIONS
            or config.get("num_runs") != NUM_RUNS
        ):
            print(
                "⚠️  Warning: Checkpoint configuration doesn't match current settings"
            )
            print("   Starting fresh run instead of resuming")
            return None

        return checkpoint_data
    except Exception as e:
        print(f"⚠️  Warning: Failed to load checkpoint: {e}")
        return None


def get_task_id(task: Dict) -> str:
    """Generate unique identifier for a task."""
    return f"{task['model']}_{task['reasoning_type']}_c{task['complexity']}_run{task['run_num']}"


def get_model_config_id(model: str, reasoning_type: str, complexity: int) -> str:
    """Generate unique identifier for a model-config combination."""
    return f"{model}_{reasoning_type}_c{complexity}"


def generate_shared_benchmark(
    reasoning_type: str,
    complexity: int,
    run_num: int,
    num_questions: int,
    display_manager: Optional["DisplayManager"] = None,
) -> Optional[str]:
    """Generate a shared benchmark file for a specific config.

    Returns:
        Path to the benchmark file, or None if generation failed
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    shared_benchmark_dir = (
        Path("results")
        / date_str
        / "_shared_benchmarks"
        / reasoning_type
        / f"c{complexity}"
    )
    shared_benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_file = str(
        shared_benchmark_dir / f"benchmark_run{run_num}_{timestamp}.json"
    )

    gen_cmd = [
        "cgqa",
        "generate",
        "--generator-type",
        reasoning_type,
        "--complexity",
        str(complexity),
        "--num-questions",
        str(num_questions),
        "--output",
        benchmark_file,
    ]

    try:
        subprocess.run(gen_cmd, check=True, capture_output=True, text=True)
        if display_manager:
            display_manager.add_info(
                f"📝 Generated benchmark: {reasoning_type} C{complexity} Run{run_num}"
            )
        return benchmark_file
    except subprocess.CalledProcessError as e:
        if display_manager:
            display_manager.add_error(
                "Benchmark",
                f"Generation failed: {reasoning_type} C{complexity} Run{run_num} - {e.stderr[:100]}",
            )
        return None


def run_model_tasks(
    model: str,
    tasks: List[Dict],
    benchmark_files: Dict[str, str],
    completed_tasks: set,
    display_manager: Optional["DisplayManager"],
    results_callback: Any,
) -> List[dict]:
    """Run all tasks for a single model with concurrent re-runs.

    Groups tasks by (reasoning_type, complexity) and runs all 3 re-runs concurrently.

    Args:
        model: Model identifier
        tasks: List of task dictionaries to run
        benchmark_files: Dict mapping task_key to benchmark file path
        completed_tasks: Set of already completed task IDs
        display_manager: Display manager for output
        results_callback: Function to call when a task completes

    Returns:
        List of result dictionaries
    """
    model_short = model.split("/")[-1].replace(".", "_").replace("-", "_")
    results = []

    if display_manager:
        display_manager.add_info(
            f"🚀 [{model_short}] Starting {len(tasks)} tasks (3-way concurrent re-runs)"
        )

    # Group tasks by (reasoning_type, complexity)
    from collections import defaultdict

    task_groups = defaultdict(list)
    for task in tasks:
        key = (task["reasoning_type"], task["complexity"])
        task_groups[key].append(task)

    # Process each group with concurrent re-runs
    for (reasoning_type, complexity), group_tasks in task_groups.items():
        # Filter out already completed tasks
        pending_tasks = [
            task for task in group_tasks if get_task_id(task) not in completed_tasks
        ]

        if not pending_tasks:
            if display_manager:
                display_manager.add_info(
                    f"⏭️  [{model_short}] Skipped: {reasoning_type} C{complexity} (all runs completed)"
                )
            continue

        if display_manager:
            display_manager.add_info(
                f"🔄 [{model_short}] Running {reasoning_type} C{complexity}: {len(pending_tasks)} re-runs concurrently"
            )

        # Run all re-runs for this group concurrently
        with ThreadPoolExecutor(max_workers=len(pending_tasks)) as executor:
            futures = {}
            for task in pending_tasks:
                task_key = f"{task['reasoning_type']}_c{task['complexity']}_run{task['run_num']}"
                benchmark_file = benchmark_files.get(task_key)

                if not benchmark_file:
                    if display_manager:
                        display_manager.add_error(model, f"No benchmark for {task_key}")
                    continue

                future = executor.submit(
                    run_single_evaluation_with_benchmark,
                    model,
                    task["reasoning_type"],
                    task["complexity"],
                    task["run_num"],
                    benchmark_file,
                    display_manager,
                )
                futures[future] = task

            # Collect results as they complete
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if result:
                        result_data = {"result": result, "task": task}
                        results.append(result_data)

                        # Notify main thread that task completed
                        results_callback(result_data, task)
                except Exception as e:
                    if display_manager:
                        display_manager.add_error(
                            model,
                            f"Task failed: {task['reasoning_type']} C{task['complexity']} Run{task['run_num']}: {e}",
                        )

    if display_manager:
        display_manager.add_info(
            f"✅ [{model_short}] Completed all tasks: {len(results)}/{len(tasks)} succeeded"
        )

    return results


def _main() -> None:
    """Run comprehensive benchmark suite with multiple runs in parallel."""

    print("=" * 80)
    print("ChaosGraphQA Leaderboard Benchmark - Per-Config Parallelization")
    print("=" * 80)
    print(f"\nModels: {len(MODELS)}")
    print(f"Reasoning types: {len(REASONING_TYPES)}")
    print(f"Complexity levels: {len(COMPLEXITY_LEVELS)}")
    print(f"Questions per level: {NUM_QUESTIONS}")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(
        f"Total evaluations: {len(MODELS) * len(REASONING_TYPES) * len(COMPLEXITY_LEVELS) * NUM_RUNS}"
    )
    print(f"\n⚡ Strategy: For each config, all {len(MODELS)} models run in parallel")
    print(
        f"   Configs are processed sequentially: {len(REASONING_TYPES)} types × {len(COMPLEXITY_LEVELS)} levels × {NUM_RUNS} runs"
    )
    print(
        f"   Estimated time: ~{len(REASONING_TYPES) * len(COMPLEXITY_LEVELS) * NUM_RUNS * 2} minutes"
    )
    print("=" * 80)

    # Check API keys
    required_keys = {
        "ANTHROPIC_API_KEY": any("anthropic" in m for m in MODELS),
        "OPENAI_API_KEY": any("openai" in m for m in MODELS),
        "GOOGLE_API_KEY": any("gemini" in m for m in MODELS),
        "HF_TOKEN": any("huggingface" in m for m in MODELS),
    }

    missing_keys = [
        key for key, needed in required_keys.items() if needed and not os.getenv(key)
    ]
    if missing_keys:
        print(f"\n⚠ Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some benchmarks may fail.\n")

    # Load checkpoint if it exists
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\n📂 Found checkpoint from {checkpoint['timestamp']}")
        print(
            f"   Resuming from {checkpoint['completed']}/{checkpoint['total']} completed"
        )
        print(
            f"   Previous elapsed time: {checkpoint['elapsed_seconds']/60:.1f} minutes"
        )

        all_results = checkpoint["results"]
        completed_tasks = set(checkpoint["completed_tasks"])
        completed = checkpoint["completed"]

        # Ask user if they want to resume
        response = input("\n   Resume from checkpoint? [Y/n]: ").strip().lower()
        if response and response != "y" and response != "yes":
            print("   Starting fresh run...")
            all_results = {model.split("/")[-1]: {} for model in MODELS}
            completed_tasks = set()
            completed = 0
            # Delete old checkpoint
            Path(CHECKPOINT_FILE).unlink(missing_ok=True)
        else:
            print("   Resuming from checkpoint...\n")
    else:
        all_results = {model.split("/")[-1]: {} for model in MODELS}
        completed_tasks = set()
        completed = 0

    # Prepare configurations (reasoning_type, complexity, run_num)
    # For each config, all models will run in parallel
    configs_to_run = []
    for reasoning_type in REASONING_TYPES:
        for complexity in COMPLEXITY_LEVELS:
            for run_num in range(1, NUM_RUNS + 1):
                # Check if this config has any incomplete models
                has_incomplete = False
                for model in MODELS:
                    task = {
                        "model": model,
                        "reasoning_type": reasoning_type,
                        "complexity": complexity,
                        "run_num": run_num,
                    }
                    if get_task_id(task) not in completed_tasks:
                        has_incomplete = True
                        break

                if has_incomplete:
                    configs_to_run.append(
                        {
                            "reasoning_type": reasoning_type,
                            "complexity": complexity,
                            "run_num": run_num,
                        }
                    )

    total_configs = len(REASONING_TYPES) * len(COMPLEXITY_LEVELS) * NUM_RUNS
    total_tasks = total_configs * len(MODELS)
    remaining_configs = len(configs_to_run)

    if not configs_to_run:
        print("\n✅ All evaluations already completed!")
        print("   Delete checkpoint file to start fresh run")
        return
    else:
        print(f"\n🚀 Starting sequential config execution...")
        print(f"   Configs remaining: {remaining_configs}/{total_configs}")
        print(
            f"   Total tasks: {remaining_configs * len(MODELS)} (all {len(MODELS)} models per config)"
        )
        print(f"   Already completed: {completed}/{total_tasks} individual tasks")
        print()

    start_time = time.time()
    results_lock = Lock()

    # Pre-generate all benchmark files
    print("\n📝 Pre-generating benchmark files...")
    benchmark_files = {}
    for i, config in enumerate(configs_to_run, 1):
        task_key = (
            f"{config['reasoning_type']}_c{config['complexity']}_run{config['run_num']}"
        )
        benchmark_file = generate_shared_benchmark(
            config["reasoning_type"],  # type: ignore
            config["complexity"],  # type: ignore
            config["run_num"],  # type: ignore
            NUM_QUESTIONS,
            None,  # No display manager during benchmark generation
        )
        if benchmark_file:
            benchmark_files[task_key] = benchmark_file

        # Progress indicator for benchmark generation
        print(
            f"\r   Generated: {i}/{len(configs_to_run)} benchmarks", end="", flush=True
        )

    print(
        f"\n\n✅ Generated {len(benchmark_files)}/{len(configs_to_run)} benchmark files"
    )

    # Build task lists for each model
    model_tasks: dict = {model: [] for model in MODELS}
    for config in configs_to_run:
        for model in MODELS:
            task = {
                "model": model,
                "reasoning_type": config["reasoning_type"],
                "complexity": config["complexity"],
                "run_num": config["run_num"],
            }
            if get_task_id(task) not in completed_tasks:
                model_tasks[model].append(task)

    # Create display manager
    display_manager = (
        DisplayManager(MODELS, total_tasks, start_time) if RICH_AVAILABLE else None
    )
    if display_manager:
        display_manager.set_model_totals(model_tasks)

    # Callback to handle completed tasks
    def handle_result(result_data: Any, task: Any) -> None:
        with results_lock:
            result = result_data["result"]
            model_key = task["model"].split("/")[-1]
            config_key = f"{task['reasoning_type']}_c{task['complexity']}"

            if config_key not in all_results[model_key]:
                all_results[model_key][config_key] = []

            all_results[model_key][config_key].append(result)
            completed_tasks.add(get_task_id(task))

            # Update model progress
            if display_manager:
                display_manager.increment_model_progress(task["model"])

            # Save checkpoint periodically
            if len(completed_tasks) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    all_results,
                    len(completed_tasks),
                    total_tasks,
                    start_time,
                    completed_tasks,
                )

    # Run all models in parallel, each working through its own task queue
    print(f"\n🚀 Starting {len(MODELS)} parallel model threads...")
    print(f"   Each model will work through its task queue independently\n")

    if RICH_AVAILABLE and display_manager:
        # Use Rich Live display with auto-updating
        import threading

        stop_refresh = threading.Event()

        def refresh_display(
            live_display: Live, display_mgr: DisplayManager, stop_event: Any
        ) -> None:
            """Background thread to refresh display periodically."""
            while not stop_event.is_set():
                rendered = display_mgr.render()
                if rendered is not None:
                    live_display.update(rendered)
                stop_event.wait(0.5)  # Update every 0.5 seconds

        with Live(display_manager.render(), console=display_manager.console) as live:
            # Start background refresh thread
            refresh_thread = threading.Thread(
                target=refresh_display,
                args=(live, display_manager, stop_refresh),
                daemon=True,
            )
            refresh_thread.start()

            with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
                futures = {}
                for model in MODELS:
                    if model_tasks[model]:  # Only submit if there are tasks
                        future = executor.submit(
                            run_model_tasks,
                            model,
                            model_tasks[model],
                            benchmark_files,
                            completed_tasks,
                            display_manager,
                            handle_result,
                        )
                        futures[future] = model

                # Wait for all models to complete
                for future in as_completed(futures):
                    model = futures[future]
                    model_short = model.split("/")[-1]
                    try:
                        future.result()
                        if display_manager:
                            display_manager.mark_model_complete(model)
                            display_manager.add_info(
                                f"🎉 [{model_short}] Finished all tasks!"
                            )
                    except Exception as e:
                        if display_manager:
                            display_manager.add_error(
                                model, f"Failed with exception: {e}"
                            )

            # Stop the refresh thread
            stop_refresh.set()
            refresh_thread.join(timeout=1.0)
    else:
        # Fallback: simple print-based display
        with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
            futures = {}
            for model in MODELS:
                if model_tasks[model]:  # Only submit if there are tasks
                    future = executor.submit(
                        run_model_tasks,
                        model,
                        model_tasks[model],
                        benchmark_files,
                        completed_tasks,
                        None,  # No display manager
                        handle_result,
                    )
                    futures[future] = model

            # Wait for all models to complete
            for future in as_completed(futures):
                model = futures[future]
                model_short = model.split("/")[-1]
                try:
                    future.result()
                    print(f"\n🎉 [{model_short}] Finished all tasks!")
                except Exception as e:
                    print(f"\n✗ [{model_short}] Failed with exception: {e}")

    print("\n" + "=" * 100)
    print("🎉 ALL MODELS COMPLETED!")
    print("=" * 100)

    # Clean up benchmark files
    print("\n🧹 Cleaning up benchmark files...")
    for benchmark_file in benchmark_files.values():
        Path(benchmark_file).unlink(missing_ok=True)

    # Calculate statistics for each model/config combination
    print("\n" + "=" * 80)
    print("Calculating statistics...")
    print("=" * 80)

    for model_key in all_results:
        for config_key in list(all_results[model_key].keys()):
            runs = all_results[model_key][config_key]
            if runs:
                stats = calculate_statistics(runs)
                all_results[model_key][config_key] = stats
            else:
                del all_results[model_key][config_key]

    # Generate comprehensive summary
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("LEADERBOARD SUMMARY")
    print("=" * 80)

    # Calculate overall performance by model
    model_overall = {}
    for model_key, configs in all_results.items():
        if configs:
            accuracies = [stats["accuracy_mean"] for stats in configs.values()]
            scores = [stats["score_mean"] for stats in configs.values()]
            model_overall[model_key] = {
                "avg_accuracy": statistics.mean(accuracies),
                "avg_score": statistics.mean(scores),
                "num_configs": len(configs),
            }

    # Sort by average accuracy
    sorted_models = sorted(
        model_overall.items(), key=lambda x: x[1]["avg_accuracy"], reverse=True
    )

    print("\n🏆 Overall Rankings (by average accuracy):")
    for rank, (model_key, stats) in enumerate(sorted_models, 1):
        print(
            f"  {rank}. {model_key:40} {stats['avg_accuracy']:.2%} "
            f"(score: {stats['avg_score']:.3f}, configs: {stats['num_configs']})"
        )

    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results per model saved in: results/")

    # Save detailed results to JSON in structured directory
    date_str = datetime.now().strftime("%Y-%m-%d")
    summary_dir = Path("results") / date_str
    summary_dir.mkdir(parents=True, exist_ok=True)
    results_file = str(
        summary_dir / f"leaderboard_summary_{datetime.now().strftime('%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "configuration": {
                    "models": MODELS,
                    "reasoning_types": REASONING_TYPES,
                    "complexity_levels": COMPLEXITY_LEVELS,
                    "num_questions": NUM_QUESTIONS,
                    "num_runs": NUM_RUNS,
                },
                "results": all_results,
                "overall_rankings": {k: v for k, v in sorted_models},
                "elapsed_minutes": elapsed / 60,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Detailed results saved to: {results_file}")
    print("=" * 80)

    # Clean up checkpoint file after successful completion
    if Path(CHECKPOINT_FILE).exists():
        Path(CHECKPOINT_FILE).unlink()
        print(f"\n🧹 Cleaned up checkpoint file")


def main() -> None:
    try:
        _main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print(f"   Progress saved in checkpoint: {CHECKPOINT_FILE}")
        print(f"   Run script again to resume from checkpoint")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print(f"   Progress saved in checkpoint: {CHECKPOINT_FILE}")
        print(f"   Run script again to resume from checkpoint")
        raise
