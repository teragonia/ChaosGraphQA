## 🏆 ChaosGraphQA Leaderboard (Multi-Run Averaged)

*Generated: 2025-08-07 15:08 UTC*  
*Models: GPT-4o (Nov 2024), GPT-4.1 (Apr 2025), Claude Opus 4, Claude Sonnet 4, Claude 3.7 Sonnet*  
*Configuration: Complexity 2, 10 questions per type, 3 runs per evaluation*  
*Seeds: [42, 123, 456] (averaged for statistical significance)*

### 🥇 Overall Performance

| Rank | Model | Avg Accuracy | Std Dev | Avg Score | Total Tokens | Avg Time (s) | Types | Runs |
|------|-------|--------------|---------|-----------|--------------|-------------|-------|------|
| 🥇1 | **Claude Opus 4** | 62.7% | ±24.0% | 0.638 | 106,581 | 204.7 | 5/5 | 15/15 |
| 🥈2 | **Claude Sonnet 4** | 54.0% | ±20.7% | 0.564 | 121,435 | 156.6 | 5/5 | 15/15 |
| 🥉3 | **Claude 3.7 Sonnet** | 51.3% | ±27.9% | 0.533 | 92,588 | 70.3 | 5/5 | 15/15 |
|   4 | **GPT-4o (Nov 2024)** | 50.7% | ±22.9% | 0.534 | 63,791 | 18.8 | 5/5 | 15/15 |
|   5 | **GPT-4.1 (Apr 2025)** | 45.3% | ±21.9% | 0.488 | 64,479 | 25.2 | 5/5 | 15/15 |

### 📊 Performance by Reasoning Type

#### Multihop Reasoning

| Rank | Model | Accuracy | Std Dev | Score | Tokens | Time (s) | Runs |
|------|-------|----------|---------|-------|--------|---------|----- |
| 🏆1 | **Claude Opus 4** | 76.7% | ±5.8% | 0.767 | 24,299 | 268.2 | 3/3 |
| 🥈2 | **Claude Sonnet 4** | 73.3% | ±5.8% | 0.733 | 27,695 | 177.3 | 3/3 |
| 🥉3 | **Claude 3.7 Sonnet** | 73.3% | ±15.3% | 0.733 | 19,274 | 64.5 | 3/3 |
| 4 | **GPT-4.1 (Apr 2025)** | 66.7% | ±11.5% | 0.667 | 13,749 | 31.3 | 3/3 |
| 5 | **GPT-4o (Nov 2024)** | 50.0% | ±10.0% | 0.500 | 13,561 | 19.1 | 3/3 |

#### Hierarchical Reasoning

| Rank | Model | Accuracy | Std Dev | Score | Tokens | Time (s) | Runs |
|------|-------|----------|---------|-------|--------|---------|----- |
| 🏆1 | **GPT-4o (Nov 2024)** | 66.7% | ±28.9% | 0.725 | 4,029 | 16.9 | 3/3 |
| 🥈2 | **Claude Sonnet 4** | 66.7% | ±30.6% | 0.717 | 8,458 | 88.2 | 3/3 |
| 🥉3 | **Claude Opus 4** | 63.3% | ±32.1% | 0.658 | 7,016 | 122.4 | 3/3 |
| 4 | **Claude 3.7 Sonnet** | 63.3% | ±32.1% | 0.658 | 7,083 | 58.0 | 3/3 |
| 5 | **GPT-4.1 (Apr 2025)** | 60.0% | ±30.0% | 0.683 | 4,024 | 25.8 | 3/3 |

#### Temporal Reasoning

| Rank | Model | Accuracy | Std Dev | Score | Tokens | Time (s) | Runs |
|------|-------|----------|---------|-------|--------|---------|----- |
| 🏆1 | **Claude Opus 4** | 60.0% | ±36.1% | 0.600 | 14,097 | 98.9 | 3/3 |
| 🥈2 | **Claude Sonnet 4** | 60.0% | ±36.1% | 0.600 | 16,866 | 99.0 | 3/3 |
| 🥉3 | **Claude 3.7 Sonnet** | 56.7% | ±30.6% | 0.567 | 14,334 | 55.6 | 3/3 |
| 4 | **GPT-4o (Nov 2024)** | 53.3% | ±35.1% | 0.533 | 10,655 | 18.1 | 3/3 |
| 5 | **GPT-4.1 (Apr 2025)** | 53.3% | ±35.1% | 0.533 | 10,655 | 27.2 | 3/3 |

#### Weighted Reasoning

| Rank | Model | Accuracy | Std Dev | Score | Tokens | Time (s) | Runs |
|------|-------|----------|---------|-------|--------|---------|----- |
| 🏆1 | **GPT-4o (Nov 2024)** | 40.0% | ±17.3% | 0.480 | 19,392 | 22.5 | 3/3 |
| 🥈2 | **GPT-4.1 (Apr 2025)** | 36.7% | ±15.3% | 0.456 | 19,763 | 22.0 | 3/3 |
| 🥉3 | **Claude Opus 4** | 36.7% | ±5.8% | 0.400 | 33,141 | 293.0 | 3/3 |
| 4 | **Claude 3.7 Sonnet** | 36.7% | ±15.3% | 0.441 | 29,098 | 108.1 | 3/3 |
| 5 | **Claude Sonnet 4** | 33.3% | ±5.8% | 0.406 | 35,781 | 192.9 | 3/3 |

#### Conflicting Reasoning

| Rank | Model | Accuracy | Std Dev | Score | Tokens | Time (s) | Runs |
|------|-------|----------|---------|-------|--------|---------|----- |
| 🏆1 | **Claude Opus 4** | 76.7% | ±40.4% | 0.767 | 28,028 | 241.1 | 3/3 |
| 🥈2 | **GPT-4o (Nov 2024)** | 43.3% | ±23.1% | 0.433 | 16,154 | 17.2 | 3/3 |
| 🥉3 | **Claude Sonnet 4** | 36.7% | ±25.2% | 0.367 | 32,635 | 225.7 | 3/3 |
| 4 | **Claude 3.7 Sonnet** | 26.7% | ±46.2% | 0.267 | 22,799 | 65.3 | 3/3 |
| 5 | **GPT-4.1 (Apr 2025)** | 10.0% | ±17.3% | 0.100 | 16,288 | 20.0 | 3/3 |

### 🔍 Key Insights

🎯 **Champion**: Claude Opus 4 leads with 62.7% ± 24.0% average accuracy  
📊 **Total Evaluations**: 75/75 runs completed successfully  
🎲 **Statistical Robustness**: Results averaged over 3 runs with different seeds  
⚙️ **Configuration**: All evaluations used identical settings for fair comparison  
📈 **Variance**: Standard deviation shows consistency across different test sets

### 📋 Methodology

- **Multiple Runs**: Each model-reasoning type combination tested 3 times  
- **Different Seeds**: Seeds [42, 123, 456] ensure diverse test cases  
- **Statistical Significance**: Standard deviation indicates result reliability  
- **Reproducible**: Different seeds prevent overfitting to specific test cases  

This leaderboard demonstrates authentic reasoning performance through dynamic graph generation with statistical averaging that prevents memorization while ensuring reproducible and reliable results.

### 📁 File Organization

All benchmark and evaluation files are automatically organized in the `cgqa_outputs/` directory structure:
- **Benchmarks**: `benchmarks/{reasoning_type}/complexity_{level}/`
- **Evaluations**: `evaluations/{benchmark_name}/{model_name}/`
- **Leaderboards**: `analysis/leaderboard/`

This prevents root directory clutter and enables easy batch operations on related files.
