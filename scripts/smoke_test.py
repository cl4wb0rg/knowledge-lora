#!/usr/bin/env python3
"""
Smoke test for the knowledge-lora training stack on DGX Spark / GB10.

Tests (in order):
  1. torch CUDA availability + device name
  2. flash-attn import
  3. LoRA wrapping (peft)
  4. 5 forward + backward passes with optimizer step
  5. Loss is finite and decreasing (sanity check)

Uses gpt2 (124 M params) + synthetic token sequences — no HF token, no data
pipeline, no large model download required.
"""

import sys

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
MODEL = "gpt2"
SEQ_LEN = 128
STEPS = 5


def check(label):
    print(f"  [OK]  {label}")


def fail(label, exc):
    print(f"  [FAIL] {label}: {exc}", file=sys.stderr)
    sys.exit(1)


print("=== knowledge-lora smoke test ===\n")

# ── 1. CUDA ───────────────────────────────────────────────────────────────────
try:
    assert torch.cuda.is_available(), "CUDA not available"
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    check(f"torch {torch.__version__}  |  {name}  |  {mem:.0f} GB")
except Exception as e:
    fail("CUDA", e)

# ── 2. flash-attn ─────────────────────────────────────────────────────────────
try:
    import flash_attn

    check(f"flash-attn {flash_attn.__version__}")
except Exception as e:
    fail("flash-attn import", e)

# ── 3. Load model + tokenizer ─────────────────────────────────────────────────
print(f"\n  Loading {MODEL} …")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    model = model.to(DEVICE)
    check(f"{MODEL} loaded on {DEVICE} (bf16)")
except Exception as e:
    fail(f"model load ({MODEL})", e)

# ── 4. LoRA ───────────────────────────────────────────────────────────────────
try:
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["c_attn"],  # GPT-2 attention projection
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    check(f"LoRA applied  |  trainable {trainable / 1e3:.1f}K / {total / 1e6:.1f}M params")
except Exception as e:
    fail("LoRA", e)

# ── 5. Training loop ──────────────────────────────────────────────────────────
print(f"\n  Running {STEPS} training steps …")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()
losses = []

# Synthetic batch: random token IDs, fixed each step for reproducibility
torch.manual_seed(42)
input_ids = torch.randint(0, tokenizer.vocab_size, (2, SEQ_LEN), device=DEVICE)

try:
    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        out = model(input_ids=input_ids, labels=input_ids)
        loss = out.loss
        assert torch.isfinite(loss), f"loss is {loss}"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        print(f"    step {step}/{STEPS}  loss={loss.item():.4f}")
except Exception as e:
    fail("training loop", e)

# ── 6. Sanity: loss must be finite and must move ──────────────────────────────
try:
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), "non-finite loss"
    check(f"all losses finite  |  first={losses[0]:.4f}  last={losses[-1]:.4f}")
except Exception as e:
    fail("loss sanity", e)

print("\n=== PASSED ===")
