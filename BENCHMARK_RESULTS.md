# ChaosGraphQA Benchmark Results

**Date**: January 14, 2026
**Models Evaluated**: 13
**Total Runs**: 780 (13 models × 5 reasoning types × 4 complexity levels × 3 runs)
**Total Question-Evaluations**: 7,218

---

## 🏆 Overall Leaderboard Rankings

| Rank | Model | Accuracy | Avg Tokens |
|------|-------|----------|------------|
| 🥇 1 | **gemini-3-pro-preview** | **90.6%** | 51,255 |
| 🥈 2 | **claude-sonnet-4-5 (20250929)** | **88.7%** | 39,512 |
| 🥉 3 | **gemini-3-flash-preview** | **88.4%** | 35,574 |
| 4 | claude-sonnet-4 (20250514) | 87.4% | 38,984 |
| 5 | gemini-2.5-flash | 82.3% | 118,116 |
| 6 | claude-3.7-sonnet (20250219) | 81.3% | 28,448 |
| 7 | gpt-5.2 | 80.9% | 29,511 |
| 8 | gpt-5-mini | 77.9% | 26,383 |
| 9 | gemini-2.0-flash | 75.6% | 38,623 |
| 10 | gpt-4o | 74.7% | 31,036 |
| 11 | claude-3.5-haiku (20241022) | 70.2% | 32,208 |
| 12 | gpt-5-nano | 50.3% | 23,489 |
| 13 | SmolLM3-3B | 38.0% | 12,544 |

---

## 📊 Key Insights

### Top Performers
- **gemini-3-pro-preview** leads with 90.6% accuracy, strong across nearly every reasoning type
- **claude-sonnet-4-5** is a close 2nd at 88.7% and tops the multihop category
- **gemini-3-flash-preview** rounds out the top 3 at 88.4% while being the most reliable model overall

### Claude Models
- **Claude Sonnet 4.5** (88.7%) and **Claude Sonnet 4** (87.4%) sit in the top tier
- **Claude 3.7 Sonnet** holds 81.3%; **Claude 3.5 Haiku** trails the family at 70.2%
- Strongest showing in multihop reasoning, where Sonnet 4.5 ranks 1st

### Gemini Models
- **Gemini 3 Pro/Flash Preview** take 1st and 3rd overall
- **Gemini 2.5 Flash** performs well at 82.3% but is by far the most token-hungry model (118K avg)
- **Gemini 2.0 Flash** is solid at 75.6% and notably strong on temporal reasoning

### Small / Efficient Models
- **gpt-5-nano** (50.3%) and **SmolLM3-3B** (38.0%) anchor the bottom of the table
- **SmolLM3-3B** is the most token-efficient model (0.03 accuracy per 1K tokens), impressive for a 3B-parameter model despite the accuracy gap

---

## 🧠 Performance by Reasoning Type

Easiest type overall: **Hierarchical** (88.1% avg). Hardest type overall: **Weighted** (68.3% avg).

### Conflicting Reasoning
| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | gemini-3-pro-preview | 99.2% |
| 2 | claude-sonnet-4 (20250514) | 95.8% |
| 3 | gemini-2.5-flash | 94.2% |
| 4 | gemini-3-flash-preview | 92.5% |
| 5 | claude-sonnet-4-5 (20250929) | 90.0% |

### Hierarchical Reasoning
| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | gemini-3-flash-preview | 100.0% |
| 2 | gpt-5-mini | 99.2% |
| 3 | claude-sonnet-4-5 (20250929) | 99.2% |
| 4 | gpt-4o | 95.0% |
| 5 | gemini-3-pro-preview | 93.3% |

### Multihop Reasoning
| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | claude-sonnet-4-5 (20250929) | 92.5% |
| 2 | gemini-3-pro-preview | 91.7% |
| 3 | claude-sonnet-4 (20250514) | 90.8% |
| 4 | gemini-3-flash-preview | 84.2% |
| 5 | claude-3.7-sonnet (20250219) | 75.8% |

### Temporal Reasoning
| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | gpt-5.2 | 85.3% |
| 2 | claude-3.7-sonnet (20250219) | 84.6% |
| 3 | gemini-2.0-flash | 84.4% |
| 4 | claude-sonnet-4 (20250514) | 83.8% |
| 5 | gemini-2.5-flash | 82.0% |

### Weighted Reasoning
| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | gemini-3-pro-preview | 87.5% |
| 2 | gemini-3-flash-preview | 84.2% |
| 3 | claude-sonnet-4-5 (20250929) | 80.0% |
| 4 | claude-sonnet-4 (20250514) | 76.7% |
| 5 | claude-3.7-sonnet (20250219) | 75.8% |

---

## 🎯 Reliability Analysis

Coefficient of variation (CV) across runs — lower is more consistent.

| Rank | Model | CV |
|------|-------|----|
| 1 | gemini-3-flash-preview | 0.089 |
| 2 | claude-sonnet-4 (20250514) | 0.097 |
| 3 | gemini-3-pro-preview | 0.105 |
| 4 | claude-sonnet-4-5 (20250929) | 0.105 |
| 5 | gemini-2.5-flash | 0.140 |
| … | … | … |
| 12 | gpt-5-nano | 0.414 |
| 13 | SmolLM3-3B | 0.437 |

---

## 🔬 Statistical Notes

- **Bonferroni-corrected significance level**: α = 0.000641
- **Significant pairwise differences**: 22
- **Largest effect size**: SmolLM3-3B vs gemini-3-pro-preview, |Cohen's d| = 3.33 (large)

---

## 🧠 Benchmark Configuration

### Reasoning Types (5 total)
1. **Multi-hop**: Path-finding across 2–6 relationship hops
2. **Hierarchical**: Taxonomy and inheritance reasoning
3. **Temporal**: Time-based causal chains and sequences
4. **Weighted**: Probabilistic relationships with confidence scores
5. **Conflicting**: Contradiction detection and consistency checking

### Complexity Levels (4 total)
- **C1**: Simple graphs, basic questions
- **C2**: Medium complexity
- **C3**: Advanced complexity
- **C4**: Maximum difficulty with large graphs

### Evaluation Details
- **3 runs per configuration** for statistical reliability
- **Mean ± Standard Deviation** calculated across runs
- Ground truth verification using graph algorithms (BFS/DFS, cycle detection)

---

## 📁 Data Location

Generated files from the experiment run can be found in:

- **Overall Leaderboard**: `analysis_output/figures/01_overall_leaderboard.csv`
- **Per-Question Results**: `analysis_output/data/individual_results.csv`
- **Full Analysis Report**: `analysis_output/section_7_report.md`
- **Figures**: `analysis_output/figures/`

---

## 🔬 Methodology

This benchmark uses ChaosGraphQA's anti-benchmaxxing design:
- **Dynamic Generation**: Unique graphs and questions per run
- **Algorithmic Verification**: Graph algorithms validate answers
- **Template Variability**: 30+ question templates prevent pattern matching
- **Complexity Scaling**: 4 difficulty levels test adaptability
- **Multiple Runs**: Statistical averaging reduces variance

For detailed methodology, see the [main README](README.md).

---

*Generated from ChaosGraphQA Leaderboard Benchmark - January 14, 2026*
