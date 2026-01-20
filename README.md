# ChaosGraphQA (CGQA)

A comprehensive benchmark for testing reasoning capabilities of Large Language Models using dynamically generated knowledge graphs. ChaosGraphQA creates unique, randomized graphs and questions that cannot be gamed or memorized, providing authentic assessments of LLM reasoning abilities.

## Features

- **🎲 Randomized Knowledge Graphs**: Generate unique graphs for each test run to prevent memorization
- **🧠 Five Reasoning Types**: Multi-hop, hierarchical, temporal, weighted, and conflicting information reasoning
- **📈 Complexity Scaling**: 4 difficulty levels (1-4) with automatic parameter adjustment
- **✅ Ground Truth Verification**: Algorithmic verification using graph algorithms (BFS/DFS, topological sorting, cycle detection)
- **🤖 Multi-LLM Support**: OpenAI GPT, Anthropic Claude, Google Gemini, and HuggingFace models
- **⚡ Anti-Benchmarking Design**: Dynamic content generation prevents optimization for specific test cases
- **🎯 Comprehensive Evaluation**: Token usage tracking, partial credit scoring, and detailed performance metrics

## Quick Start

### Installation

```bash
# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e .[dev,viz,llm]
```

### Basic Usage

```bash
# Generate benchmarks for different reasoning types
cgqa generate --generator-type multihop --complexity 2 --num-questions 10 --output benchmark.json
cgqa generate --generator-type hierarchical --complexity 3 --num-questions 5 --seed 42
cgqa generate --generator-type temporal --complexity 1 --num-questions 8 --verify
cgqa generate --generator-type weighted --complexity 4 --num-questions 6 --output weighted.json
cgqa generate --generator-type conflicting --complexity 2 --num-questions 7

# Show benchmark information
cgqa info benchmark.json

# List available LLM models
cgqa list-models

# Evaluate with different LLM providers
cgqa evaluate benchmark.json --model openai/gpt-4o-mini --output results.json
cgqa evaluate benchmark.json --model anthropic/claude-3.7-sonnet --output results.json
cgqa evaluate benchmark.json --model gemini/gemini-2.5-flash --output results.json
cgqa evaluate benchmark.json --model huggingface/HuggingFaceTB/SmolLM3-3B --output results.json

# Test LLM connection
cgqa test-model --model anthropic/claude-3.7-sonnet

# Advanced evaluation options
cgqa evaluate benchmark.json \
  --model anthropic/claude-4.5-sonnet \
  --temperature 0.1 \
  --max-tokens 1500 \
  --batch-size 5 \
  --no-context

# Organize and manage outputs
cgqa init-structure          # Initialize organized directory structure
cgqa list-files              # List all benchmarks and evaluations
cgqa cleanup                 # Clean up old temporary files
cgqa analyze results.json    # Analyze evaluation results
```

### Python API

```python
from cgqa.generators import (
    MultiHopGenerator, HierarchicalGenerator, TemporalGenerator,
    WeightedGenerator, ConflictingGenerator
)
from cgqa.questions import QuestionGenerator
from cgqa.models.question import QuestionType
from cgqa.llm.evaluation.llm_evaluator import LLMEvaluator

# Generate different types of knowledge graphs
multihop_gen = MultiHopGenerator(complexity_level=2, seed=42)
hierarchical_gen = HierarchicalGenerator(complexity_level=3, seed=123)
temporal_gen = TemporalGenerator(complexity_level=2, seed=456)
weighted_gen = WeightedGenerator(complexity_level=2, seed=789)
conflicting_gen = ConflictingGenerator(complexity_level=2, seed=101)

# Generate a hierarchical reasoning graph
kg = hierarchical_gen.generate()
print(f"Generated graph: {len(kg.entities)} entities, {len(kg.relationships)} relationships")

# Generate questions
question_gen = QuestionGenerator(seed=42)
questions = question_gen.generate_questions(
    kg,
    question_types=[QuestionType.HIERARCHICAL],
    num_questions_per_type=5,
    complexity_levels=[2, 3]
)

# Evaluate with multiple LLM providers
models_to_test = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.7-sonnet",
    "gemini/gemini-2.5-flash"
]

for model_name in models_to_test:
    print(f"\n=== Evaluating {model_name} ===")
    evaluator = LLMEvaluator.from_model_string(model_name)
    summary = evaluator.evaluate_questions(questions.questions, kg)

    print(f"Accuracy: {summary.accuracy:.1%}")
    print(f"Average Score: {summary.average_score:.3f}")
    print(f"Tokens Used: {summary.total_tokens_used:,}")
    print(f"Response Time: {summary.evaluation_time:.1f}s")

    # Save results
    evaluator.save_results(summary, f"results_{model_name.replace('/', '_')}.json")
```

### Configuration Files

Create YAML configuration files for consistent model settings:

```yaml
# configs/llm/my_openai.yaml
provider_name: "openai"
model_name: "gpt-4o-mini"
api_key: "${OPENAI_API_KEY}"
temperature: 0.1
max_tokens: 1000
rate_limit_delay: 0.1
```

### Environment Variables

Set your API keys:
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
export HF_TOKEN="your-token-here"
```

### Supported Models

ChaosGraphQA supports multiple LLM providers with the latest models:

**OpenAI**:
- GPT-5 family: `gpt-5`, `gpt-5-mini`, `gpt-5-nano` (400K context)
- GPT-4 family: `gpt-4.1`, `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
- o-series: `o3`, `o4-mini` (reasoning models, 200K context)

**Anthropic Claude**:
- Claude 4.5: `claude-4.5-opus`, `claude-4.5-sonnet`, `claude-4.5-haiku`
- Claude 4: `claude-4.1-opus`, `claude-4-opus`, `claude-4-sonnet`
- Claude 3.7: `claude-3.7-sonnet` (latest 3.x model)
- Claude 3.5: `claude-3.5-haiku`
- Claude 3: `claude-3-haiku`

**Google Gemini**:
- Gemini 2.5: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` (1M context)
- Gemini 2.0: `gemini-2.0-flash`

**HuggingFace**:
- Any HuggingFace model (e.g., `HuggingFaceTB/SmolLM3-3B`, `distilgpt2`)

Use `cgqa list-models` to see all available models and check provider availability.

### Output Directory Structure

ChaosGraphQA automatically organizes all outputs into a structured directory:

```
cgqa_outputs/              # Auto-generated, organized file structure
├── benchmarks/           # Generated benchmark files
│   └── {reasoning_type}/
│       └── complexity_{level}/
├── evaluations/          # Evaluation results
│   └── {benchmark_name}/
│       └── {model_name}/
├── analysis/             # Analysis outputs and reports
├── models/              # Model-specific outputs
├── logs/                # Application logs
└── temp/                # Temporary files (auto-cleaned)
```

Use `cgqa init-structure` to initialize this structure, `cgqa list-files` to browse files, and `cgqa cleanup` to remove old temporary files.

## Current Status

✅ **Phase 1 Complete - Core Foundation**:
- Project structure and packaging with pyproject.toml
- Core data models (Entity, Relationship, KnowledgeGraph, Question)
- Multi-hop reasoning generator with structured path generation
- Question template system with variable substitution
- Ground truth verification using graph algorithms
- Basic CLI interface with Rich console output
- Comprehensive test suite with pytest

✅ **Phase 2 Complete - LLM Integration**:
- **Multi-provider support**: OpenAI, Anthropic, Gemini, HuggingFace APIs
- **Flexible configuration**: YAML configs, environment variables, CLI options
- **Comprehensive evaluation**: Response validation, scoring, detailed metrics
- **Rate limiting**: Built-in delays and batch processing for API limits
- **Rich CLI**: Progress bars, error handling, detailed results display

✅ **Phase 3 Complete - Advanced Reasoning Types**:
- **Five reasoning generators**: Multi-hop, hierarchical, temporal, weighted, conflicting
- **Dynamic question generation**: 30+ question templates across all reasoning types
- **Complexity scaling**: Automatic parameter adjustment across 4 difficulty levels
- **Full CLI integration**: All generator types supported with comprehensive options
- **Organized output management**: Auto-organized directory structure with cleanup utilities
- **Robust error handling**: Validation, entity creation, and recovery mechanisms

🚧 **Future Enhancements**:
- Enhanced visualization dashboards and interactive reports
- Advanced statistical analysis and model comparison tools
- Additional reasoning types (causal, logical, mathematical)
- Performance optimization and result caching

## Architecture

```
cgqa/
├── models/                      # Core data structures
│   ├── graph.py                # Entity, Relationship, KnowledgeGraph
│   ├── question.py             # Question, Answer, QuestionSet, QuestionType
│   └── relationship_semantics.py  # Relationship semantic constraints
├── generators/                  # Knowledge graph generators
│   ├── base_generator.py       # Abstract base class with complexity scaling
│   ├── multihop.py            # Multi-hop path reasoning
│   ├── hierarchical.py        # Taxonomy and inheritance
│   ├── temporal.py            # Time-based and causal chains
│   ├── weighted.py            # Probabilistic relationships
│   └── conflicting.py         # Contradiction detection
├── questions/                   # Question generation system
│   ├── templates.py            # 30+ question templates across 5 reasoning types
│   └── validators.py           # Answer validation utilities
├── evaluators/                  # Ground truth verification
│   ├── ground_truth.py         # Main verification engine
│   └── graph_algorithms.py     # BFS/DFS, cycle detection, path finding
├── llm/                        # LLM integration and evaluation
│   ├── providers/             # Multi-provider LLM support
│   │   ├── base.py            # Abstract base provider
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── gemini_provider.py
│   │   └── huggingface_provider.py
│   └── evaluation/            # Evaluation engine and metrics
│       ├── llm_evaluator.py   # Main evaluation orchestrator
│       └── provider_factory.py  # Provider instantiation
├── utils/                      # Utility modules
│   └── directory_manager.py   # Organized output directory management
└── cli/                        # Command-line interface
    └── main.py                # Full-featured CLI with Rich output
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cgqa

# Run specific test file
pytest tests/test_models.py
```

## Development

```bash
# Install development dependencies
pip install -e .[dev]

# Run linting
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## Reasoning Types

All reasoning types are fully implemented and tested with comprehensive question templates.

### 🔗 Multi-hop Reasoning
Test LLMs' ability to traverse multi-step relationships and find complex connections.

**Capabilities:**
- Path finding between distant entities (2-6 hops)
- Shortest path computation with BFS verification
- Intermediate entity identification along paths
- Path existence verification with directed/undirected graphs
- Relationship-specific path constraints

**Sample Questions:**
- "What is the shortest path from Alice to the Research Department?"
- "How many steps does it take to get from Project Alpha to Database Server?"
- "What entities lie on the path from Marketing to the CEO?"

### 🏗️ Hierarchical Reasoning
Evaluate understanding of taxonomies, inheritance, and classification systems.

**Capabilities:**
- Multi-level taxonomies (animal kingdom, organizational charts, geographical hierarchies)
- Inheritance path traversal with "is_a" and "part_of" relationships
- Root ancestor identification and classification queries
- Compositional reasoning (system → subsystem → component)
- Dynamic hierarchy depth scaling (3-6 levels)

**Sample Questions:**
- "What is the root category that Lion belongs to in the taxonomy hierarchy?"
- "Does Software Engineer inherit from the Employee role?"
- "Show the complete inheritance path from Tiger to Animal."

### ⏰ Temporal Reasoning
Test comprehension of time-based relationships, event sequences, and causality.

**Capabilities:**
- Event timeline generation with realistic timestamps
- Causal chain reasoning (A causes B causes C)
- Temporal sequence analysis and ordering
- Duration-based calculations and constraints
- Before/after relationship validation

**Sample Questions:**
- "What events occur between Project Kickoff and Final Deadline?"
- "What is the final outcome when Budget Approval triggers a causal chain?"
- "How many steps are in the causal chain from Research to Product Launch?"

### ⚖️ Weighted Reasoning
Assess handling of probabilistic information, confidence scores, and uncertainty.

**Capabilities:**
- Confidence-weighted relationships (0.0-1.0 scores)
- Probabilistic path finding with confidence propagation
- Threshold-based queries and filtering
- High-confidence link identification
- Similarity and trust scoring between entities

**Sample Questions:**
- "How many relationships have confidence above 0.7?"
- "What is the highest confidence path from Alice to the Database?"
- "What is the confidence score for the 'likely_knows' relationship between Bob and Carol?"

### ⚡ Conflicting Information
Challenge LLMs to detect contradictions and reason about inconsistent data.

**Capabilities:**
- Direct contradictions (A is_friend_of B vs A is_enemy_of B)
- Transitive conflicts (A > B > C > A impossible cycles)
- Inheritance conflicts (X is_a Cat AND X is_a Dog)
- Temporal conflicts (A before B AND B before A)
- Exclusivity violations (X is_alive AND X is_dead)
- Consistent subgraph identification

**Sample Questions:**
- "Is there a contradiction in the relationship between Alice and Bob?"
- "Are the entities {Marketing, Sales, Engineering} part of a consistent subgraph?"
- "What type of conflict exists between the 'is_friend_of' and 'is_enemy_of' relationships?"

## Benchmark Results

Evaluated 13 state-of-the-art models across 780 evaluations. Performance ranged from 38.0% to 90.6%.

**Top performers**:
- Gemini-3-Pro-Preview: 90.6%
- Claude Sonnet 4.5: 88.7%
- Gemini-3-Flash-Preview: 88.4%

Run the full leaderboard:
```bash
cgqa_bm
```

## Example Results

Here's a sample evaluation showing how different LLMs perform across reasoning types:

### Multi-hop Reasoning (Complexity 2, 5 questions)
```
Model: anthropic/claude-3.7-sonnet
Accuracy: 60.0%
Average Score: 0.620
Total Time: 23.5s
Tokens Used: 4,251
Performance: ✓ Strong path-finding abilities, occasionally misses intermediate steps
```

### Hierarchical Reasoning (Complexity 3, 5 questions)
```
Model: openai/gpt-4o-mini
Accuracy: 80.0%
Average Score: 0.840
Total Time: 18.2s
Tokens Used: 3,847
Performance: ✓ Excellent taxonomy understanding, handles inheritance well
```

### Conflicting Information (Complexity 2, 5 questions)
```
Model: gemini/gemini-2.5-flash
Accuracy: 40.0%
Average Score: 0.420
Total Time: 15.8s
Tokens Used: 2,963
Performance: ⚠ Struggles with contradiction detection, needs improvement
```

### Key Insights from Evaluation:
- **Multi-hop reasoning** challenges vary significantly with graph complexity
- **Hierarchical reasoning** shows most consistent performance across models
- **Temporal reasoning** benefits from explicit timeline context
- **Weighted reasoning** performance correlates with mathematical reasoning ability
- **Conflicting information** detection remains challenging for most current LLMs

## Anti-Benchmaxxing Design

ChaosGraphQA prevents gaming and memorization through several key design principles:

🎲 **Dynamic Generation**: Each benchmark run creates unique graphs with different entity names, relationships, and structures. Using different seeds ensures different graphs, while the same seed guarantees reproducibility.

🔧 **Configurable Complexity**: Four complexity levels automatically adjust graph size, relationship density, and question difficulty, preventing optimization for specific configurations.

✅ **Algorithmic Ground Truth**: Answers are verified using graph algorithms (BFS/DFS, cycle detection, topological sorting), not pre-computed datasets, ensuring correctness regardless of graph structure.

🎯 **Template Variability**: 30+ question templates with variable substitution create countless unique question phrasings while maintaining consistent reasoning requirements.

🔀 **Seed-based Reproducibility**: While results are reproducible with the same seed, the massive space of possible configurations prevents systematic optimization.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
