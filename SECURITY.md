# SECURITY.md — knowledge-lora

This project follows the same workspace-wide security baseline as other
cl4wb0rg repositories.

---

## Visibility

This repository is **public**. Do not commit anything that reveals internal
infrastructure details, personal data, or credentials — even in commit messages.

---

## What is and is not committed

| Category | Status | Notes |
|---|---|---|
| Source code / scripts | ✅ committed | pipeline, training scripts, configs |
| Model configs (`.yaml`) | ✅ committed | hyperparameters only — no secrets |
| `.env` / secrets | ❌ never | gitignored; holds `HF_TOKEN` etc. |
| Training data | ❌ never | gitignored (`data/`) |
| Model weights / checkpoints | ❌ never | gitignored (`output/`) |
| Logs | ❌ never | gitignored (`*.log`, `logs/`) |
| Virtual environments | ❌ never | gitignored (`.venv/`, `.venv-vllm/`) |

---

## Pipeline auto-commit policy

`run_pipeline.sh` auto-commits and pushes **README.md** and
**`configs/sft_config.yaml`** after each pipeline stage.

Rules for auto-commit content:
- **No hostnames, IP addresses, or hardware identifiers** in committed files
  or commit messages.
- **No absolute local paths** — use relative paths in all configs and commit
  messages (e.g. `output/cpt/checkpoint-500`, not `/home/mvdb/...`).
- **No personal data** — training data sources must be public datasets only.
- **No tokens or API keys** — always load from `.env`, never hardcode.
- README status updates (step progress, loss values) are acceptable public
  information as they describe the open-source training process.

---

## Secrets handling

- All secrets (`HF_TOKEN`, `WANDB_API_KEY`, etc.) live exclusively in `.env`.
- `.env` is gitignored and must never be committed.
- `.env.example` may be committed as a template with placeholder values only.
- CI/CD (if added later) must use repository secrets, never inline values.

---

## Threat model

**Type:** ML training pipeline — offline batch job, no network services exposed.

- No credentials are stored in the repository.
- No user-controlled input reaches shell commands (no injection surface).
- External network calls: HuggingFace Hub (model download), GitHub (push).
  Both use token auth from `.env`.
- Training data is sourced from public datasets (Wikipedia DE); no PII.
- Model outputs (weights, checkpoints) remain local and are never pushed.

**`install.sh`:** Downloads and compiles third-party Python packages (including
`flash-attn` from source). Pin versions explicitly and review checksums when
updating dependencies. Do not run `install.sh` from untrusted forks.

**Dependencies:** Python packages managed via `pip` in `.venv` / `.venv-vllm`.
Run `pip-audit` to check for known vulnerabilities:
```bash
source .venv/bin/activate && pip-audit
```

---

## Vulnerability reporting

Report security issues **privately** — do not open a public issue for vulnerabilities.
Include repro steps, impact, and affected versions.
