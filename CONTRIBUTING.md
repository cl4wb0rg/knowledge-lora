# Contributing

Thank you for your interest in contributing to **knowledge-lora**.

## Development setup

```bash
git clone https://github.com/cl4wb0rg/knowledge-lora.git
cd knowledge-lora
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ruff mypy
```

## Before opening a pull request

Run the linter and type checker locally:

```bash
ruff check scripts/
ruff format scripts/
mypy scripts/ --ignore-missing-imports
```

CI will fail if either tool reports errors.

## Code style

- **Python 3.11+** syntax throughout.
- **Type hints** on all public functions — `disallow_untyped_defs = true` is enforced.
- **`logging`** for all diagnostic output, never bare `print()` in production paths.
- **Atomic writes**: output files must be written to `.tmp` first and renamed on success.
- **No secrets in code** — credentials only via environment variables.

## Reporting issues

Please open a GitHub issue with:
- Python version and OS
- The exact command you ran
- The full error output

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0.
