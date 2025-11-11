# ChaosGraphQA Benchmark Results

**Date**: November 10, 2025
**Total Evaluations**: 720 (12 models × 5 reasoning types × 4 complexity levels × 3 runs)
**Questions per Evaluation**: 20

---

## 🏆 Overall Leaderboard Rankings

| Rank | Model | Accuracy | Score | Configs |
|------|-------|----------|-------|---------|
| 🥇 1 | **GPT-5** | **79.0%** | **0.795** | 20 |
| 🥈 2 | **GPT-5-mini** | **75.0%** | **0.763** | 20 |
| 🥉 3 | **GPT-4.1** | **71.8%** | **0.731** | 20 |
| 4 | Claude Sonnet 4.5 (20250929) | 69.7% | 0.716 | 20 |
| 5 | Claude Sonnet 4 (20250514) | 69.6% | 0.713 | 20 |
| 6 | Claude 3.5 Sonnet (20241022) | 68.4% | 0.701 | 20 |
| 7 | GPT-4o | 64.6% | 0.673 | 20 |
| 8 | Gemini 2.0 Flash | 63.6% | 0.665 | 20 |
| 9 | Claude 3.5 Haiku (20241022) | 60.5% | 0.627 | 20 |
| 10 | Gemini 2.5 Flash | 45.0% | 0.451 | 20 |
| 11 | SmolLM3-3B | 41.2% | 0.482 | 20 |
| 12 | Gemini 2.5 Pro | 26.3% | 0.275 | 20 |

---

## 📊 Key Insights

### Top Performers
- **GPT-5** dominates with 79% accuracy, showing strong reasoning across all types
- **GPT-5-mini** achieves impressive 75% accuracy at likely lower cost
- **GPT-4.1** rounds out the top 3 with 71.8% accuracy

### Claude Models
- **Claude Sonnet 4.5** (latest) achieves 69.7%, placing 4th overall
- Claude models show consistent performance (68-70% range) across versions
- Strong showing in complex reasoning tasks

### Gemini Models
- **Gemini 2.0 Flash** performs reasonably at 63.6% (8th place)
- **Gemini 2.5 Flash** significantly lower at 45% (10th place)
- **Gemini 2.5 Pro** unexpectedly lowest at 26.3% - possible API/parsing issues

### Small Models
- **SmolLM3-3B** achieves 41.2% accuracy despite being only 3B parameters
- Impressive given size constraints compared to frontier models

---

## 🧠 Benchmark Configuration

### Reasoning Types (5 total)
1. **Multi-hop**: Path-finding across 2-6 relationship hops
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
- **20 questions** per evaluation
- **Mean ± Standard Deviation** calculated across runs
- Ground truth verification using graph algorithms (BFS/DFS, cycle detection)

---

## 💡 Notable Observations

1. **Clear Performance Tiers**:
   - Tier 1 (GPT-5 family): 72-79%
   - Tier 2 (Claude 4.x, Claude 3.5 Sonnet): 68-70%
   - Tier 3 (GPT-4o, Gemini 2.0): 64-66%
   - Tier 4 (Others): <60%

2. **Model Size vs Performance**:
   - Frontier models significantly outperform smaller alternatives
   - SmolLM3-3B shows promise but gap remains substantial

3. **Consistency**:
   - All models completed 20/20 configurations (5 types × 4 levels)
   - Statistical reliability from 3-run averaging

4. **Anti-Benchmaxxing Success**:
   - Dynamically generated graphs prevent memorization
   - Wide performance spread indicates genuine reasoning assessment

---

## 📁 Data Location

Generated files from the experiment run will be found in:

- **Full Results**: `results/{date}/leaderboard_summary_{number}.json`
- **Per-Model Detailed Results**: `results/{date}/{model_name}/`
- **Checkpoint Data**: `leaderboard_checkpoint.json` (removed on successful completion)

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

*Generated from ChaosGraphQA Leaderboard Benchmark - November 10, 2025*
