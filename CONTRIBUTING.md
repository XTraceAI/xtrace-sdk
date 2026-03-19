# Contributing

Thank you for your interest in contributing to the XTrace SDK.

## Development Setup

```bash
git clone https://github.com/XTraceAI/xtrace-vec-sdk.git
cd xtrace-vec-sdk
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

The crypto tests (`test_paillier_*.py`, `test_goldwasser_micali_client.py`) run fully offline and do not require an XTrace account.

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting and [mypy](https://mypy-lang.org/) for static type checking:

```bash
ruff check src/xtrace_sdk/
python -m mypy src/xtrace_sdk/
```

Both must pass with no errors before submitting a PR.

## Pull Requests

- Keep changes focused — one concern per PR
- Add or update tests for any changed behaviour
- Do not commit `.env` files, execution context files, or any credentials
- Update `CHANGELOG.md` under an `[Unreleased]` heading

## Reporting Issues

Open an issue on GitHub with a minimal reproducible example.
