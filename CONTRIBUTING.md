# Contributing to ChaosGraphQA

Thanks for your interest. Here's how to get set up and what we expect from contributions.

## Dev setup

```bash
git clone https://github.com/teragonia/ChaosGraphQA.git
cd ChaosGraphQA
pip install -e .[dev,viz,llm]
```

## Before opening a PR

Run the full check suite locally:

```bash
black src/ tests/          # formatting (line length 88)
isort src/ tests/          # import order
mypy src/                  # type checking (strict mode)
pytest                     # tests
pytest --cov=cgqa          # with coverage
```

The CI runs all four. PRs that fail any of these won't be merged.

**Type annotations are required.** All new code must satisfy `mypy` in strict mode (`disallow_untyped_defs = true`). If you're adding support for an optional dependency, add the appropriate `mypy` overrides in `pyproject.toml`.

## Adding a reasoning type

1. Create a generator in `src/cgqa/generators/` subclassing `BaseGenerator`.
2. Add at least 10 question templates in `src/cgqa/questions/templates.py`.
3. Add a verifier method in `src/cgqa/evaluators/ground_truth.py`.
4. Wire up the new type in the CLI (`src/cgqa/cli/main.py`).
5. Add tests covering generation, question output, and ground truth verification.

## Architecture overview

```
src/cgqa/
├── models/          # KnowledgeGraph, Entity, Relationship, Question
├── generators/      # One generator per reasoning type + abstract base
├── questions/       # Template system and answer validators
├── evaluators/      # Ground truth verification via graph algorithms
├── llm/             # Provider integrations and evaluation engine
│   ├── providers/   # openai, anthropic, gemini, huggingface
│   └── evaluation/  # LLMEvaluator, ProviderFactory
├── utils/           # Directory management
└── cli/             # Click-based CLI with Rich output
```

Ground truth is always algorithmic — answers are derived from NetworkX graph algorithms at question-generation time, not looked up from a dataset.

## PR process

1. Fork the repo and create a branch off `main`.
2. Keep PRs focused. One feature or fix per PR.
3. Add or update tests for any changed behaviour.
4. Make sure CI passes before requesting review.
5. Describe what changed and why in the PR body.
