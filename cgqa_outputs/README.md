# ChaosGraphQA Output Directory Structure

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
