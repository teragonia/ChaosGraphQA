# ChaosGraphQA Benchmark Results Analysis
**Generated:** 2026-01-14 21:19:33
---

## Executive Summary

- **Top Overall Performer:** gemini-3-pro-preview (90.6% accuracy)
- **Hardest Reasoning Type:** Weighted (68.3% avg accuracy)
- **Easiest Reasoning Type:** Hierarchical (88.1% avg accuracy)
- **Most Reliable Model:** gemini_3_flash_preview (CV = 0.089)
- **Most Efficient Model:** SmolLM3-3B (0.03 accuracy per 1K tokens)

## Overall Leaderboard

| Rank | Model | Accuracy | 95% CI | Avg Tokens |
|------|-------|----------|--------|------------|
| 1 | gemini-3-pro-preview | 0.906 | [nan, nan] | 51255 |
| 2 | claude-sonnet-4-5-20250929 | 0.887 | [nan, nan] | 39512 |
| 3 | gemini-3-flash-preview | 0.884 | [nan, nan] | 35574 |
| 4 | claude-sonnet-4-20250514 | 0.874 | [nan, nan] | 38984 |
| 5 | gemini-2.5-flash | 0.823 | [nan, nan] | 118116 |
| 6 | claude-3-7-sonnet-20250219 | 0.813 | [nan, nan] | 28448 |
| 7 | gpt-5.2 | 0.809 | [nan, nan] | 29511 |
| 8 | gpt-5-mini | 0.779 | [nan, nan] | 26383 |
| 9 | gemini-2.0-flash | 0.756 | [nan, nan] | 38623 |
| 10 | gpt-4o | 0.747 | [nan, nan] | 31036 |
| 11 | claude-3-5-haiku-20241022 | 0.702 | [nan, nan] | 32208 |
| 12 | gpt-5-nano | 0.503 | [nan, nan] | 23489 |
| 13 | SmolLM3-3B | 0.380 | [nan, nan] | 12544 |

## Performance by Reasoning Type

### Conflicting Reasoning

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | gemini-3-pro-preview | 0.992 |
| 2 | claude-sonnet-4-20250514 | 0.958 |
| 3 | gemini-2.5-flash | 0.942 |
| 4 | gemini-3-flash-preview | 0.925 |
| 5 | claude-sonnet-4-5-20250929 | 0.900 |
| 6 | claude-3-7-sonnet-20250219 | 0.817 |
| 7 | gpt-5.2 | 0.817 |
| 8 | gpt-5-mini | 0.808 |
| 9 | gemini-2.0-flash | 0.750 |
| 10 | gpt-4o | 0.742 |
| 11 | claude-3-5-haiku-20241022 | 0.708 |
| 12 | SmolLM3-3B | 0.342 |
| 13 | gpt-5-nano | 0.300 |

### Hierarchical Reasoning

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | gemini-3-flash-preview | 1.000 |
| 2 | gpt-5-mini | 0.992 |
| 3 | claude-sonnet-4-5-20250929 | 0.992 |
| 4 | gpt-4o | 0.950 |
| 5 | gemini-3-pro-preview | 0.933 |
| 6 | gpt-5.2 | 0.928 |
| 7 | claude-sonnet-4-20250514 | 0.900 |
| 8 | claude-3-7-sonnet-20250219 | 0.883 |
| 9 | gemini-2.0-flash | 0.879 |
| 10 | claude-3-5-haiku-20241022 | 0.873 |
| 11 | gemini-2.5-flash | 0.863 |
| 12 | SmolLM3-3B | 0.634 |
| 13 | gpt-5-nano | 0.622 |

### Multihop Reasoning

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | claude-sonnet-4-5-20250929 | 0.925 |
| 2 | gemini-3-pro-preview | 0.917 |
| 3 | claude-sonnet-4-20250514 | 0.908 |
| 4 | gemini-3-flash-preview | 0.842 |
| 5 | claude-3-7-sonnet-20250219 | 0.758 |
| 6 | gpt-5.2 | 0.758 |
| 7 | gemini-2.5-flash | 0.750 |
| 8 | gpt-5-mini | 0.664 |
| 9 | gpt-4o | 0.650 |
| 10 | gemini-2.0-flash | 0.642 |
| 11 | claude-3-5-haiku-20241022 | 0.633 |
| 12 | gpt-5-nano | 0.356 |
| 13 | SmolLM3-3B | 0.225 |

### Temporal Reasoning

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | gpt-5.2 | 0.853 |
| 2 | claude-3-7-sonnet-20250219 | 0.846 |
| 3 | gemini-2.0-flash | 0.844 |
| 4 | claude-sonnet-4-20250514 | 0.838 |
| 5 | gemini-2.5-flash | 0.820 |
| 6 | claude-sonnet-4-5-20250929 | 0.819 |
| 7 | gemini-3-flash-preview | 0.813 |
| 8 | gemini-3-pro-preview | 0.811 |
| 9 | gpt-5-mini | 0.795 |
| 10 | gpt-4o | 0.743 |
| 11 | claude-3-5-haiku-20241022 | 0.709 |
| 12 | gpt-5-nano | 0.669 |
| 13 | SmolLM3-3B | 0.391 |

### Weighted Reasoning

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | gemini-3-pro-preview | 0.875 |
| 2 | gemini-3-flash-preview | 0.842 |
| 3 | claude-sonnet-4-5-20250929 | 0.800 |
| 4 | claude-sonnet-4-20250514 | 0.767 |
| 5 | claude-3-7-sonnet-20250219 | 0.758 |
| 6 | gemini-2.5-flash | 0.742 |
| 7 | gpt-5.2 | 0.692 |
| 8 | gemini-2.0-flash | 0.667 |
| 9 | gpt-4o | 0.650 |
| 10 | gpt-5-mini | 0.633 |
| 11 | claude-3-5-haiku-20241022 | 0.584 |
| 12 | gpt-5-nano | 0.567 |
| 13 | SmolLM3-3B | 0.310 |

## Complexity Degradation Analysis

Linear regression slopes (accuracy ~ complexity) for each model × task:

**Conflicting Reasoning:**
- Steepest degradation: gemini-2.0-flash (slope = -0.107)
- Shallowest degradation: gemini-3-pro-preview (slope = 0.003)

**Hierarchical Reasoning:**
- Steepest degradation: SmolLM3-3B (slope = -0.114)
- Shallowest degradation: gemini-3-pro-preview (slope = 0.047)

**Multihop Reasoning:**
- Steepest degradation: claude-3-5-haiku-20241022 (slope = -0.240)
- Shallowest degradation: gemini-3-pro-preview (slope = -0.027)

**Temporal Reasoning:**
- Steepest degradation: gpt-5-nano (slope = 0.010)
- Shallowest degradation: gemini-3-pro-preview (slope = 0.060)

**Weighted Reasoning:**
- Steepest degradation: gpt-5-mini (slope = -0.160)
- Shallowest degradation: gemini-3-pro-preview (slope = -0.050)

## Statistical Significance

- Bonferroni-corrected significance level: α = 0.000641
- Number of significant pairwise differences: 22

## Effect Sizes

**Largest effect sizes (|Cohen's d|):**

- SmolLM3-3B vs gemini-3-pro-preview: |d| = 3.33 (large)
- SmolLM3-3B vs claude-sonnet-4-5-20250929: |d| = 3.15 (large)
- SmolLM3-3B vs gemini-3-flash-preview: |d| = 3.02 (large)
- SmolLM3-3B vs claude-sonnet-4-20250514: |d| = 2.98 (large)
- SmolLM3-3B vs gemini-2.5-flash: |d| = 2.56 (large)
- gemini-3-pro-preview vs gpt-5-nano: |d| = 2.54 (large)
- SmolLM3-3B vs gpt-5.2: |d| = 2.43 (large)
- claude-sonnet-4-5-20250929 vs gpt-5-nano: |d| = 2.38 (large)
- SmolLM3-3B vs claude-3-7-sonnet-20250219: |d| = 2.36 (large)
- gemini-3-flash-preview vs gpt-5-nano: |d| = 2.28 (large)

## Reliability Analysis

**Model reliability (coefficient of variation, lower is better):**

| Rank | Model | CV |
|------|-------|----|
| 1 | gemini_3_flash_preview | 0.089 |
| 2 | claude_sonnet_4_20250514 | 0.097 |
| 3 | gemini_3_pro_preview | 0.105 |
| 4 | claude_sonnet_4_5_20250929 | 0.105 |
| 5 | gemini_2_5_flash | 0.140 |
| 6 | gpt_5_2 | 0.148 |
| 7 | gemini_2_0_flash | 0.150 |
| 8 | gpt_5_mini | 0.160 |
| 9 | claude_3_7_sonnet_20250219 | 0.162 |
| 10 | claude_3_5_haiku_20241022 | 0.224 |
| 11 | gpt_4o | 0.225 |
| 12 | gpt_5_nano | 0.414 |
| 13 | SmolLM3_3B | 0.437 |

## Figures

All figures are saved in `figures/` subdirectory:

1. `01_overall_leaderboard.png` - Overall performance table
2. `02_task_leaderboard_*.png` - Task-specific leaderboards (5 files)
3. `07_heatmap.png` - Model × Task performance heatmap
4. `08_degradation_*.png` - Complexity degradation curves (5 files)
5. `13_efficiency_scatter.png` - Accuracy vs. token efficiency
6. `15_reliability_boxes.png` - Reliability box plots
7. `16_significance_matrix.png` - Statistical significance heatmap
8. `17_effect_sizes.png` - Effect size heatmap

