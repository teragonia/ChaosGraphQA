#!/usr/bin/env python3
"""
KGRB Leaderboard Evaluation - Fixed Version

This script evaluates specified models across all reasoning types and creates
a comprehensive leaderboard with detailed performance metrics.
"""

import subprocess
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

# API Keys
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Target models
MODELS = [
    {
        "id": "openai/gpt-4o-2024-11-20",
        "name": "GPT-4o (Nov 2024)",
        "provider": "openai",
        "api_key": OPENAI_API_KEY
    },
    {
        "id": "openai/gpt-4.1-2025-04-14",
        "name": "GPT-4.1 (Apr 2025)", 
        "provider": "openai",
        "api_key": OPENAI_API_KEY
    },
    {
        "id": "anthropic/claude-opus-4-20250514",
        "name": "Claude Opus 4",
        "provider": "anthropic", 
        "api_key": ANTHROPIC_API_KEY
    },
    {
        "id": "anthropic/claude-sonnet-4-20250514",
        "name": "Claude Sonnet 4",
        "provider": "anthropic",
        "api_key": ANTHROPIC_API_KEY
    },
    {
        "id": "anthropic/claude-3-7-sonnet-20250219",
        "name": "Claude 3.7 Sonnet",
        "provider": "anthropic",
        "api_key": ANTHROPIC_API_KEY
    }
]

REASONING_TYPES = ["multihop", "hierarchical", "temporal", "weighted", "conflicting"]

# Configuration for multiple runs
NUM_RUNS = 3  # Number of runs to average over
BASE_SEEDS = [42, 123, 456]  # Different seeds for each run
MAX_PARALLEL_RUNS = 3  # Number of parallel runs (be careful with API rate limits)
ENABLE_PARALLEL = True  # Set to False to run sequentially (safer for API limits)

def generate_benchmark_with_seed(reasoning_type, seed, run_number):
    """Generate a benchmark with a specific seed."""
    benchmark_file = f"benchmark_{reasoning_type}_run{run_number}.json"
    
    cmd = [
        "python", "-m", "kgrb.cli.main", "generate",
        "--generator-type", reasoning_type,
        "--complexity", "2",
        "--num-questions", "10", 
        "--seed", str(seed),
        "--output", benchmark_file,
        "--verify"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to generate {benchmark_file}")
        return None
    
    return benchmark_file

def run_single_evaluation(model, reasoning_type, run_number, benchmark_file):
    """Run a single evaluation with a specific benchmark."""
    result_file = f"results_{model['id'].replace('/', '_')}_{reasoning_type}_run{run_number}.json"
    
    cmd = [
        "python", "-m", "kgrb.cli.main", "evaluate",
        benchmark_file,
        "--model", model["id"],
        "--api-key", model["api_key"],
        "--output", result_file,
        "--temperature", "0.1",
        "--max-tokens", "1500"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Success if return code is 0, regardless of stderr warnings
    if result.returncode != 0:
        print(f"❌ Evaluation failed with return code {result.returncode}")
        if result.stderr and "RuntimeWarning" not in result.stderr:
            print(f"   Error: {result.stderr.strip()[-200:]}")
        return None
    
    # Parse results from output file
    try:
        if not Path(result_file).exists():
            print(f"❌ Result file not created: {result_file}")
            return None
            
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        metrics = data.get("metrics", {})
        metadata = data.get("metadata", {})
        
        eval_result = {
            "model_id": model["id"],
            "model_name": model["name"],
            "reasoning_type": reasoning_type,
            "run_number": run_number,
            "accuracy": metrics.get("accuracy", 0.0),
            "average_score": metrics.get("average_score", 0.0),
            "total_tokens": metrics.get("total_tokens_used", 0),
            "evaluation_time": metadata.get("evaluation_time", 0.0),
            "error_rate": metrics.get("error_rate", 0.0),
            "questions_total": metadata.get("total_questions", 0),
            "result_file": result_file
        }
        
        return eval_result
        
    except Exception as e:
        print(f"❌ Failed to parse results: {e}")
        return None

def run_single_run_task(model, reasoning_type, run_num, seed, lock=None):
    """Run a single evaluation run as a parallelizable task."""
    thread_name = threading.current_thread().name
    
    if lock:
        with lock:
            print(f"   🔄 [Thread-{thread_name}] Run {run_num}/{NUM_RUNS} (seed: {seed})")
    
    # Generate benchmark for this run
    benchmark_file = generate_benchmark_with_seed(reasoning_type, seed, run_num)
    if not benchmark_file:
        if lock:
            with lock:
                print(f"   ❌ [Thread-{thread_name}] Failed to generate benchmark for run {run_num}")
        return None
    
    # Run evaluation
    result = run_single_evaluation(model, reasoning_type, run_num, benchmark_file)
    if result:
        if lock:
            with lock:
                print(f"   ✅ [Thread-{thread_name}] Run {run_num}: {result['accuracy']:.1%} accuracy, {result['total_tokens']:,} tokens")
    else:
        if lock:
            with lock:
                print(f"   ❌ [Thread-{thread_name}] Failed run {run_num}")
    
    return result

def run_averaged_evaluation_parallel(model, reasoning_type):
    """Run multiple evaluations in parallel and return averaged results."""
    print(f"🧠 Evaluating {model['name']} on {reasoning_type} reasoning ({NUM_RUNS} parallel runs)...")
    
    # Create thread lock for synchronized printing
    print_lock = threading.Lock()
    
    # Prepare run tasks
    run_tasks = []
    for run_num in range(1, NUM_RUNS + 1):
        seed = BASE_SEEDS[run_num - 1] if run_num - 1 < len(BASE_SEEDS) else BASE_SEEDS[0] + run_num * 100
        run_tasks.append((model, reasoning_type, run_num, seed, print_lock))
    
    # Execute runs in parallel
    run_results = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_RUNS) as executor:
        # Submit all tasks
        future_to_run = {
            executor.submit(run_single_run_task, *task): task[2] 
            for task in run_tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_run):
            run_num = future_to_run[future]
            try:
                result = future.result()
                if result:
                    run_results.append(result)
            except Exception as exc:
                with print_lock:
                    print(f"   ❌ Run {run_num} generated exception: {exc}")
    
    if not run_results:
        return None
    
    # Calculate averaged metrics
    avg_result = {
        "model_id": model["id"],
        "model_name": model["name"],
        "reasoning_type": reasoning_type,
        "runs_completed": len(run_results),
        "accuracy": sum(r["accuracy"] for r in run_results) / len(run_results),
        "accuracy_std": calculate_std_dev([r["accuracy"] for r in run_results]),
        "average_score": sum(r["average_score"] for r in run_results) / len(run_results),
        "total_tokens": sum(r["total_tokens"] for r in run_results),
        "avg_tokens_per_run": sum(r["total_tokens"] for r in run_results) / len(run_results),
        "evaluation_time": sum(r["evaluation_time"] for r in run_results),
        "avg_time_per_run": sum(r["evaluation_time"] for r in run_results) / len(run_results),
        "error_rate": sum(r["error_rate"] for r in run_results) / len(run_results),
        "questions_total": run_results[0]["questions_total"],
        "individual_runs": run_results
    }
    
    print(f"✅ {model['name']}: {avg_result['accuracy']:.1%} ± {avg_result['accuracy_std']:.1%} accuracy ({len(run_results)}/{NUM_RUNS} runs)")
    return avg_result

def run_averaged_evaluation_sequential(model, reasoning_type):
    """Run multiple evaluations sequentially and return averaged results."""
    print(f"🧠 Evaluating {model['name']} on {reasoning_type} reasoning ({NUM_RUNS} sequential runs)...")
    
    run_results = []
    
    for run_num in range(1, NUM_RUNS + 1):
        seed = BASE_SEEDS[run_num - 1] if run_num - 1 < len(BASE_SEEDS) else BASE_SEEDS[0] + run_num * 100
        
        print(f"   🔄 Run {run_num}/{NUM_RUNS} (seed: {seed})")
        
        # Generate benchmark for this run
        benchmark_file = generate_benchmark_with_seed(reasoning_type, seed, run_num)
        if not benchmark_file:
            print(f"   ❌ Failed to generate benchmark for run {run_num}")
            continue
        
        # Run evaluation
        result = run_single_evaluation(model, reasoning_type, run_num, benchmark_file)
        if result:
            run_results.append(result)
            print(f"   ✅ Run {run_num}: {result['accuracy']:.1%} accuracy, {result['total_tokens']:,} tokens")
        else:
            print(f"   ❌ Failed run {run_num}")
        
        # Brief pause between runs
        if run_num < NUM_RUNS:
            time.sleep(1)
    
    if not run_results:
        return None
    
    # Calculate averaged metrics
    avg_result = {
        "model_id": model["id"],
        "model_name": model["name"],
        "reasoning_type": reasoning_type,
        "runs_completed": len(run_results),
        "accuracy": sum(r["accuracy"] for r in run_results) / len(run_results),
        "accuracy_std": calculate_std_dev([r["accuracy"] for r in run_results]),
        "average_score": sum(r["average_score"] for r in run_results) / len(run_results),
        "total_tokens": sum(r["total_tokens"] for r in run_results),
        "avg_tokens_per_run": sum(r["total_tokens"] for r in run_results) / len(run_results),
        "evaluation_time": sum(r["evaluation_time"] for r in run_results),
        "avg_time_per_run": sum(r["evaluation_time"] for r in run_results) / len(run_results),
        "error_rate": sum(r["error_rate"] for r in run_results) / len(run_results),
        "questions_total": run_results[0]["questions_total"],
        "individual_runs": run_results
    }
    
    print(f"✅ {model['name']}: {avg_result['accuracy']:.1%} ± {avg_result['accuracy_std']:.1%} accuracy ({len(run_results)}/{NUM_RUNS} runs)")
    return avg_result

def run_averaged_evaluation(model, reasoning_type):
    """Run multiple evaluations and return averaged results."""
    if ENABLE_PARALLEL:
        return run_averaged_evaluation_parallel(model, reasoning_type)
    else:
        return run_averaged_evaluation_sequential(model, reasoning_type)

def calculate_std_dev(values):
    """Calculate standard deviation of a list of values."""
    if len(values) <= 1:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5

def create_leaderboard(results):
    """Create the leaderboard markdown."""
    if not results:
        return "# No results to display"
    
    # Calculate model aggregates
    model_stats = {}
    for result in results:
        model = result["model_name"]
        if model not in model_stats:
            model_stats[model] = {
                "accuracies": [],
                "accuracy_stds": [],
                "scores": [],
                "tokens": [],
                "times": [],
                "types": [],
                "runs_completed": []
            }
        
        model_stats[model]["accuracies"].append(result["accuracy"])
        model_stats[model]["accuracy_stds"].append(result["accuracy_std"])
        model_stats[model]["scores"].append(result["average_score"])
        model_stats[model]["tokens"].append(result["total_tokens"])
        model_stats[model]["times"].append(result["evaluation_time"])
        model_stats[model]["types"].append(result["reasoning_type"])
        model_stats[model]["runs_completed"].append(result["runs_completed"])
    
    # Calculate averages
    for model in model_stats:
        data = model_stats[model]
        data["avg_accuracy"] = sum(data["accuracies"]) / len(data["accuracies"])
        data["avg_accuracy_std"] = sum(data["accuracy_stds"]) / len(data["accuracy_stds"])
        data["avg_score"] = sum(data["scores"]) / len(data["scores"])
        data["total_tokens"] = sum(data["tokens"])
        data["avg_time"] = sum(data["times"]) / len(data["times"])
        data["types_completed"] = len(data["types"])
        data["total_runs"] = sum(data["runs_completed"])
    
    # Calculate total runs for reporting
    total_expected_runs = len(results) * NUM_RUNS
    total_completed_runs = sum(sum(model_data["runs_completed"]) for model_data in model_stats.values())
    
    # Create markdown
    timestamp = time.strftime("%Y-%m-%d %H:%M UTC")
    
    markdown = f"""## 🏆 KGRB Leaderboard (Multi-Run Averaged)

*Generated: {timestamp}*  
*Models: GPT-4o (Nov 2024), GPT-4.1 (Apr 2025), Claude Opus 4, Claude Sonnet 4, Claude 3.7 Sonnet*  
*Configuration: Complexity 2, 10 questions per type, {NUM_RUNS} runs per evaluation*  
*Seeds: {BASE_SEEDS} (averaged for statistical significance)*

### 🥇 Overall Performance

| Rank | Model | Avg Accuracy | Std Dev | Avg Score | Total Tokens | Avg Time (s) | Types | Runs |
|------|-------|--------------|---------|-----------|--------------|-------------|-------|------|
"""
    
    # Sort by average accuracy
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["avg_accuracy"], reverse=True)
    
    for rank, (model_name, stats) in enumerate(sorted_models, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        markdown += f"| {emoji}{rank} | **{model_name}** | {stats['avg_accuracy']:.1%} | ±{stats['avg_accuracy_std']:.1%} | {stats['avg_score']:.3f} | {stats['total_tokens']:,} | {stats['avg_time']:.1f} | {stats['types_completed']}/5 | {stats['total_runs']}/{stats['types_completed'] * NUM_RUNS} |\n"
    
    # Performance by reasoning type
    markdown += "\n### 📊 Performance by Reasoning Type\n\n"
    
    for reasoning_type in REASONING_TYPES:
        type_results = [r for r in results if r["reasoning_type"] == reasoning_type]
        if not type_results:
            continue
            
        type_results.sort(key=lambda x: x["accuracy"], reverse=True)
        
        markdown += f"#### {reasoning_type.title()} Reasoning\n\n"
        markdown += "| Rank | Model | Accuracy | Std Dev | Score | Tokens | Time (s) | Runs |\n"
        markdown += "|------|-------|----------|---------|-------|--------|---------|----- |\n"
        
        for rank, result in enumerate(type_results, 1):
            emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
            markdown += f"| {emoji}{rank} | **{result['model_name']}** | {result['accuracy']:.1%} | ±{result['accuracy_std']:.1%} | {result['average_score']:.3f} | {result['total_tokens']:,} | {result['evaluation_time']:.1f} | {result['runs_completed']}/{NUM_RUNS} |\n"
        
        markdown += "\n"
    
    # Add insights
    if sorted_models:
        best_model = sorted_models[0][0]
        best_accuracy = sorted_models[0][1]["avg_accuracy"]
        
        best_std = sorted_models[0][1]["avg_accuracy_std"]
        
        markdown += f"""### 🔍 Key Insights

🎯 **Champion**: {best_model} leads with {best_accuracy:.1%} ± {best_std:.1%} average accuracy  
📊 **Total Evaluations**: {total_completed_runs}/{total_expected_runs} runs completed successfully  
🎲 **Statistical Robustness**: Results averaged over {NUM_RUNS} runs with different seeds  
⚙️ **Configuration**: All evaluations used identical settings for fair comparison  
📈 **Variance**: Standard deviation shows consistency across different test sets

### 📋 Methodology

- **Multiple Runs**: Each model-reasoning type combination tested {NUM_RUNS} times  
- **Different Seeds**: Seeds {BASE_SEEDS} ensure diverse test cases  
- **Statistical Significance**: Standard deviation indicates result reliability  
- **Reproducible**: Different seeds prevent overfitting to specific test cases  

This leaderboard demonstrates authentic reasoning performance through dynamic graph generation with statistical averaging that prevents memorization while ensuring reproducible and reliable results.
"""
    
    return markdown

def main():
    """Main execution function."""
    execution_mode = "Parallel" if ENABLE_PARALLEL else "Sequential"
    print(f"🚀 KGRB Leaderboard Generation (Multi-Run Averaged with {execution_mode} Execution)")
    print(f"🎯 Testing {len(MODELS)} models on {len(REASONING_TYPES)} reasoning types")
    print(f"🎲 Running {NUM_RUNS} evaluations per model-type combination with seeds {BASE_SEEDS}")
    if ENABLE_PARALLEL:
        print(f"⚡ Parallel execution: Up to {MAX_PARALLEL_RUNS} concurrent runs per evaluation")
    else:
        print("🔄 Sequential execution: One run at a time (safer for API limits)")
    print("📋 Models: GPT-4o (Nov 2024), GPT-4.1 (Apr 2025), Claude Opus 4, Claude Sonnet 4, Claude 3.7 Sonnet\n")
    
    # Run evaluations
    results = []
    total_evals = len(MODELS) * len(REASONING_TYPES)
    current_eval = 0
    
    for model in MODELS:
        print(f"\n🔍 Testing {model['name']}")
        
        for reasoning_type in REASONING_TYPES:
            current_eval += 1
            print(f"   📊 Progress: {current_eval}/{total_evals}")
            
            try:
                result = run_averaged_evaluation(model, reasoning_type)
                if result:
                    results.append(result)
                
                # Rate limiting between model-type combinations
                if current_eval < total_evals:
                    print("   ⏱️  Pausing 5 seconds before next evaluation...")
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                print(f"\n⏹️  Stopped by user after {len(results)} evaluations")
                break
            except Exception as e:
                print(f"   ❌ Unexpected error: {e}")
                continue
    
    if not results:
        print("❌ No successful evaluations completed")
        return
    
    print(f"\n✅ Completed {len(results)}/{total_evals} evaluations")
    
    # Create and save leaderboard
    markdown = create_leaderboard(results)
    
    leaderboard_file = "LEADERBOARD.md"
    with open(leaderboard_file, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    results_file = "leaderboard_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "results": results,
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "total_evaluations": len(results),
                "models": [m["id"] for m in MODELS],
                "reasoning_types": REASONING_TYPES
            }
        }, f, indent=2)
    
    print(f"\n🏆 LEADERBOARD CREATED!")
    print(f"📄 Leaderboard: {leaderboard_file}")
    print(f"💾 Raw results: {results_file}")
    
    # Show summary
    model_summary = {}
    for result in results:
        model = result["model_name"]
        if model not in model_summary:
            model_summary[model] = []
        model_summary[model].append(result["accuracy"])
    
    print(f"\n📊 FINAL RANKINGS:")
    model_avgs = [(model, sum(accs)/len(accs)) for model, accs in model_summary.items()]
    model_avgs.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (model_name, avg_acc) in enumerate(model_avgs, 1):
        emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📍"
        completed_types = len(model_summary[model_name])
        print(f"{emoji} {rank}. {model_name}: {avg_acc:.1%} avg accuracy ({completed_types}/5 types)")

if __name__ == "__main__":
    main()