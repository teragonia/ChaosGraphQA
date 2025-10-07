"""Directory management utilities for organizing ChaosGraphQA outputs."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class DirectoryManager:
    """Manages organized directory structure for ChaosGraphQA outputs."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """Initialize directory manager.

        Args:
            base_dir: Base directory for all outputs. Defaults to ./cgqa_outputs
        """
        if base_dir is None:
            base_dir = Path.cwd() / "cgqa_outputs"

        self.base_dir = Path(base_dir)
        self._ensure_base_structure()

    def _ensure_base_structure(self) -> None:
        """Create the basic directory structure."""
        subdirs = [
            "benchmarks",  # Generated benchmark files
            "evaluations",  # Evaluation results
            "analysis",  # Analysis outputs
            "models",  # Model-specific outputs
            "logs",  # Log files
            "temp",  # Temporary files
        ]

        for subdir in subdirs:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

    def get_benchmark_path(
        self,
        generator_type: str,
        complexity: int,
        seed: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Get organized path for benchmark files.

        Args:
            generator_type: Type of reasoning generator
            complexity: Complexity level
            seed: Random seed (if used)
            filename: Custom filename (optional)

        Returns:
            Path object for the benchmark file
        """
        subdir = (
            self.base_dir / "benchmarks" / generator_type / f"complexity_{complexity}"
        )
        subdir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            seed_suffix = f"_seed_{seed}" if seed else ""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{generator_type}_c{complexity}{seed_suffix}_{timestamp}.json"

        return subdir / filename

    def get_evaluation_path(
        self, benchmark_name: str, model_name: str, filename: Optional[str] = None
    ) -> Path:
        """Get organized path for evaluation results.

        Args:
            benchmark_name: Name/ID of the benchmark
            model_name: Name of the evaluated model
            filename: Custom filename (optional)

        Returns:
            Path object for the evaluation results
        """
        # Clean model name for filesystem
        model_clean = model_name.replace("/", "_").replace(":", "_")
        benchmark_clean = Path(benchmark_name).stem  # Remove extension

        subdir = self.base_dir / "evaluations" / benchmark_clean / model_clean
        subdir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        return subdir / filename

    def get_analysis_path(
        self, analysis_type: str = "general", filename: Optional[str] = None
    ) -> Path:
        """Get organized path for analysis outputs.

        Args:
            analysis_type: Type of analysis
            filename: Custom filename (optional)

        Returns:
            Path object for the analysis file
        """
        subdir = self.base_dir / "analysis" / analysis_type
        subdir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_{timestamp}.html"

        return subdir / filename

    def get_model_specific_path(
        self, model_name: str, subtype: str = "general", filename: Optional[str] = None
    ) -> Path:
        """Get path for model-specific outputs.

        Args:
            model_name: Name of the model
            subtype: Subtype of output (e.g., 'performance', 'comparisons')
            filename: Custom filename (optional)

        Returns:
            Path object for the model-specific file
        """
        model_clean = model_name.replace("/", "_").replace(":", "_")
        subdir = self.base_dir / "models" / model_clean / subtype
        subdir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{subtype}_{timestamp}.json"

        return subdir / filename

    def get_log_path(self, log_type: str = "general") -> Path:
        """Get path for log files.

        Args:
            log_type: Type of log

        Returns:
            Path object for the log file
        """
        subdir = self.base_dir / "logs"
        subdir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"{log_type}_{timestamp}.log"

        return subdir / filename

    def get_temp_path(self, filename: str) -> Path:
        """Get path for temporary files.

        Args:
            filename: Temporary filename

        Returns:
            Path object for the temporary file
        """
        subdir = self.base_dir / "temp"
        subdir.mkdir(parents=True, exist_ok=True)

        return subdir / filename

    def cleanup_temp(self, max_age_hours: int = 24) -> int:
        """Clean up old temporary files.

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            Number of files cleaned up
        """
        temp_dir = self.base_dir / "temp"
        if not temp_dir.exists():
            return 0

        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        cleaned_count = 0
        for file_path in temp_dir.iterdir():
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    cleaned_count += 1

        return cleaned_count

    def list_benchmarks(self) -> dict:
        """List all available benchmarks organized by type and complexity.

        Returns:
            Dictionary with benchmark organization
        """
        benchmarks = {}
        benchmarks_dir = self.base_dir / "benchmarks"

        if not benchmarks_dir.exists():
            return benchmarks

        for generator_dir in benchmarks_dir.iterdir():
            if generator_dir.is_dir():
                generator_type = generator_dir.name
                benchmarks[generator_type] = {}

                for complexity_dir in generator_dir.iterdir():
                    if complexity_dir.is_dir() and complexity_dir.name.startswith(
                        "complexity_"
                    ):
                        complexity = complexity_dir.name
                        files = [
                            f.name
                            for f in complexity_dir.iterdir()
                            if f.suffix == ".json"
                        ]
                        benchmarks[generator_type][complexity] = files

        return benchmarks

    def list_evaluations(self) -> dict:
        """List all evaluation results organized by benchmark and model.

        Returns:
            Dictionary with evaluation organization
        """
        evaluations = {}
        evaluations_dir = self.base_dir / "evaluations"

        if not evaluations_dir.exists():
            return evaluations

        for benchmark_dir in evaluations_dir.iterdir():
            if benchmark_dir.is_dir():
                benchmark_name = benchmark_dir.name
                evaluations[benchmark_name] = {}

                for model_dir in benchmark_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        files = [
                            f.name for f in model_dir.iterdir() if f.suffix == ".json"
                        ]
                        evaluations[benchmark_name][model_name] = files

        return evaluations

    def get_structure_info(self) -> dict:
        """Get information about the current directory structure.

        Returns:
            Dictionary with structure statistics
        """
        info = {
            "base_directory": str(self.base_dir),
            "total_size_mb": 0,
            "directories": {},
            "file_counts": {},
        }

        for subdir_name in [
            "benchmarks",
            "evaluations",
            "analysis",
            "models",
            "logs",
            "temp",
        ]:
            subdir = self.base_dir / subdir_name
            if subdir.exists():
                file_count = sum(1 for _ in subdir.rglob("*") if _.is_file())
                dir_count = sum(1 for _ in subdir.rglob("*") if _.is_dir())

                # Calculate total size
                total_size = sum(
                    f.stat().st_size for f in subdir.rglob("*") if f.is_file()
                )

                info["directories"][subdir_name] = {
                    "files": file_count,
                    "subdirectories": dir_count,
                    "size_mb": round(total_size / (1024 * 1024), 2),
                }
                info["total_size_mb"] += info["directories"][subdir_name]["size_mb"]

        info["total_size_mb"] = round(info["total_size_mb"], 2)
        return info

    def create_directory_readme(self) -> None:
        """Create README file explaining the directory structure."""
        readme_content = """# ChaosGraphQA Output Directory Structure

This directory contains all outputs from ChaosGraphQA benchmark generation and evaluation.

## Directory Structure

```
cgqa_outputs/
├── benchmarks/          # Generated benchmark datasets
│   ├── multihop/        # Multi-hop reasoning benchmarks
│   │   └── complexity_1/
│   │       └── benchmark_multihop_c1_20240101_120000.json
│   ├── hierarchical/    # Hierarchical reasoning benchmarks
│   ├── temporal/        # Temporal reasoning benchmarks
│   ├── weighted/        # Weighted reasoning benchmarks
│   └── conflicting/     # Conflicting information benchmarks
│
├── evaluations/         # Evaluation results
│   ├── benchmark_name/  # Organized by benchmark
│   │   ├── openai_gpt-4o/
│   │   │   └── results_20240101_120000.json
│   │   └── anthropic_claude-3.5-sonnet/
│   │       └── results_20240101_120000.json
│
├── analysis/            # Analysis outputs and reports
│   ├── performance/     # Performance analysis
│   ├── comparisons/     # Model comparisons
│   └── general/         # General analysis
│
├── models/             # Model-specific outputs
│   ├── openai_gpt-4o/
│   │   ├── performance/
│   │   └── comparisons/
│
├── logs/               # Application logs
│   ├── general_20240101.log
│   └── evaluation_20240101.log
│
└── temp/               # Temporary files (auto-cleaned)
```

## File Naming Conventions

### Benchmarks
- Format: `benchmark_{generator_type}_c{complexity}_seed_{seed}_{timestamp}.json`
- Example: `benchmark_temporal_c2_seed_42_20240101_120000.json`

### Evaluations  
- Format: `results_{timestamp}.json`
- Organized in: `evaluations/{benchmark_name}/{model_name}/`

### Analysis
- Format: `{analysis_type}_{timestamp}.html`
- Various formats supported (JSON, HTML, CSV)

## Usage

The directory structure is automatically created and managed by ChaosGraphQA. Files are organized to:

- **Prevent clutter** in the working directory
- **Enable easy navigation** between related files
- **Support batch operations** on similar file types
- **Maintain version history** with timestamps
- **Allow quick cleanup** of temporary files

## Cleanup

Temporary files are automatically cleaned up after 24 hours. Use the CLI command:
```bash
cgqa cleanup --max-age 24
```
"""

        readme_path = self.base_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)


def get_default_directory_manager() -> DirectoryManager:
    """Get a default directory manager instance."""
    return DirectoryManager()
