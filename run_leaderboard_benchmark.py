#!/usr/bin/env python3
"""
Comprehensive benchmarking script for ChaosGraphQA leaderboard.
Tests multiple models across all reasoning types and complexity levels.
Runs each configuration 3 times and calculates mean and standard deviation.
"""

import os
import subprocess
import time
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Models to benchmark
MODELS = [
    # Claude models
    "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-5-sonnet-20241022",
    "anthropic/claude-3-5-haiku-20241022",
    # OpenAI models
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-4.1",
    "openai/gpt-4o",
    # Gemini models
    "gemini/gemini-2.5-pro",
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
NUM_QUESTIONS = 20  # Questions per complexity level
NUM_RUNS = 3  # Run each configuration 3 times
MAX_PARALLEL_WORKERS = 16  # Maximum parallel evaluations

def load_results(results_file: str) -> dict:
    """Load evaluation results from JSON file."""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
            return {
                'accuracy': data['metrics']['accuracy'],
                'average_score': data['metrics']['average_score'],
                'total_tokens': data['metrics'].get('total_tokens_used', 0),
                'error_rate': data['metrics'].get('error_rate', 0),
            }
    except Exception as e:
        print(f"    ✗ Failed to load results: {e}")
        return None

def run_single_evaluation(model: str, reasoning_type: str, complexity: int,
                         num_questions: int, run_num: int, print_lock: Lock = None) -> dict:
    """Run a single benchmark evaluation."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Add microseconds for uniqueness
    model_short = model.split('/')[-1].replace('.', '_').replace('-', '_')

    # Generate benchmark
    benchmark_file = f"benchmark_{reasoning_type}_c{complexity}_{model_short}_run{run_num}_{timestamp}.json"

    gen_cmd = [
        "cgqa", "generate",
        "--generator-type", reasoning_type,
        "--complexity", str(complexity),
        "--num-questions", str(num_questions),
        "--output", benchmark_file,
    ]

    try:
        subprocess.run(gen_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if print_lock:
            with print_lock:
                print(f"      ✗ [{model_short}] Generation failed: {e.stderr[:100]}")
        return None

    # Evaluate with model
    results_file = f"results_{reasoning_type}_c{complexity}_{model_short}_run{run_num}_{timestamp}.json"

    eval_cmd = [
        "cgqa", "evaluate",
        benchmark_file,
        "--model", model,
        "--output", results_file,
    ]

    try:
        subprocess.run(eval_cmd, check=True, capture_output=True, text=True, timeout=600)

        # Load results
        results = load_results(results_file)

        if results and print_lock:
            with print_lock:
                print(f"      ✓ [{model_short}] {reasoning_type} C{complexity} Run{run_num}: "
                      f"Acc={results['accuracy']:.2%}, Score={results['average_score']:.3f}")

        # Clean up benchmark file
        Path(benchmark_file).unlink(missing_ok=True)

        return results

    except subprocess.TimeoutExpired:
        if print_lock:
            with print_lock:
                print(f"      ✗ [{model_short}] Evaluation timed out")
        return None
    except subprocess.CalledProcessError as e:
        if print_lock:
            with print_lock:
                print(f"      ✗ [{model_short}] Evaluation failed: {e.stderr[:100]}")
        return None

def calculate_statistics(runs: List[dict]) -> dict:
    """Calculate mean and standard deviation from multiple runs."""
    if not runs or len(runs) == 0:
        return None

    accuracies = [r['accuracy'] for r in runs]
    scores = [r['average_score'] for r in runs]
    tokens = [r['total_tokens'] for r in runs]
    error_rates = [r['error_rate'] for r in runs]

    return {
        'accuracy_mean': statistics.mean(accuracies),
        'accuracy_stdev': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
        'score_mean': statistics.mean(scores),
        'score_stdev': statistics.stdev(scores) if len(scores) > 1 else 0,
        'tokens_mean': statistics.mean(tokens),
        'tokens_stdev': statistics.stdev(tokens) if len(tokens) > 1 else 0,
        'error_rate_mean': statistics.mean(error_rates),
        'num_runs': len(runs),
    }

def main():
    """Run comprehensive benchmark suite with multiple runs in parallel."""

    print("=" * 80)
    print("ChaosGraphQA Leaderboard Benchmark - Parallel Multi-Run Analysis")
    print("=" * 80)
    print(f"\nModels: {len(MODELS)}")
    print(f"Reasoning types: {len(REASONING_TYPES)}")
    print(f"Complexity levels: {len(COMPLEXITY_LEVELS)}")
    print(f"Questions per level: {NUM_QUESTIONS}")
    print(f"Runs per configuration: {NUM_RUNS}")
    print(f"Parallel workers: {MAX_PARALLEL_WORKERS}")
    print(f"Total evaluations: {len(MODELS) * len(REASONING_TYPES) * len(COMPLEXITY_LEVELS) * NUM_RUNS}")
    print(f"\nEstimated time with parallelization: ~{len(MODELS) * len(REASONING_TYPES) * len(COMPLEXITY_LEVELS) * NUM_RUNS * 2 // MAX_PARALLEL_WORKERS} minutes")
    print("=" * 80)

    # Check API keys
    required_keys = {
        "ANTHROPIC_API_KEY": any("anthropic" in m for m in MODELS),
        "OPENAI_API_KEY": any("openai" in m for m in MODELS),
        "GOOGLE_API_KEY": any("gemini" in m for m in MODELS),
        "HF_TOKEN": any("huggingface" in m for m in MODELS),
    }

    missing_keys = [key for key, needed in required_keys.items() if needed and not os.getenv(key)]
    if missing_keys:
        print(f"\n⚠ Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some benchmarks may fail.\n")

    # Prepare all evaluation tasks
    tasks = []
    for model in MODELS:
        for reasoning_type in REASONING_TYPES:
            for complexity in COMPLEXITY_LEVELS:
                for run_num in range(1, NUM_RUNS + 1):
                    tasks.append({
                        'model': model,
                        'reasoning_type': reasoning_type,
                        'complexity': complexity,
                        'run_num': run_num,
                    })

    print(f"\n🚀 Starting parallel execution with {MAX_PARALLEL_WORKERS} workers...\n")

    start_time = time.time()
    print_lock = Lock()

    # Store results by model and config
    all_results = {model.split('/')[-1]: {} for model in MODELS}
    completed = 0
    total = len(tasks)

    # Execute tasks in parallel
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                run_single_evaluation,
                task['model'],
                task['reasoning_type'],
                task['complexity'],
                NUM_QUESTIONS,
                task['run_num'],
                print_lock
            ): task
            for task in tasks
        }

        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1

            try:
                result = future.result()

                # Store result
                model_key = task['model'].split('/')[-1]
                config_key = f"{task['reasoning_type']}_c{task['complexity']}"

                if config_key not in all_results[model_key]:
                    all_results[model_key][config_key] = []

                if result:
                    all_results[model_key][config_key].append(result)

            except Exception as e:
                with print_lock:
                    print(f"      ✗ Task failed with exception: {e}")

            # Progress update
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total - completed) / rate if rate > 0 else 0

            with print_lock:
                print(f"\n[{completed}/{total}] Progress: {completed/total*100:.1f}% | "
                      f"Elapsed: {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")

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
            accuracies = [stats['accuracy_mean'] for stats in configs.values()]
            scores = [stats['score_mean'] for stats in configs.values()]
            model_overall[model_key] = {
                'avg_accuracy': statistics.mean(accuracies),
                'avg_score': statistics.mean(scores),
                'num_configs': len(configs),
            }

    # Sort by average accuracy
    sorted_models = sorted(model_overall.items(),
                          key=lambda x: x[1]['avg_accuracy'],
                          reverse=True)

    print("\n🏆 Overall Rankings (by average accuracy):")
    for rank, (model_key, stats) in enumerate(sorted_models, 1):
        print(f"  {rank}. {model_key:40} {stats['avg_accuracy']:.2%} "
              f"(score: {stats['avg_score']:.3f}, configs: {stats['num_configs']})")

    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results per model saved in: cgqa_outputs/evaluations/")

    # Save detailed results to JSON
    results_file = f"leaderboard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'models': MODELS,
                'reasoning_types': REASONING_TYPES,
                'complexity_levels': COMPLEXITY_LEVELS,
                'num_questions': NUM_QUESTIONS,
                'num_runs': NUM_RUNS,
            },
            'results': all_results,
            'overall_rankings': {k: v for k, v in sorted_models},
            'elapsed_minutes': elapsed/60,
        }, f, indent=2)

    print(f"\n✅ Detailed results saved to: {results_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
