# Project conventions & lessons

Guidance for working in this repo (ATIS structured-output fine-tuning experiment).

## Python environment (local)
- **Use `uv`, never plain `pip`.** Create envs with `uv venv`, install with `uv pip install <pkg>`,
  run tools with `uv run <cmd>` (e.g. `uv run jupyter notebook`). Do not activate the venv or call
  `.venv/bin/...` directly, and never use `pip` / `pip3` / `python -m pip`.

## Google Colab gotchas (hard-won)
- **NEVER `pip install -U torch` (or torchvision / torchaudio) on Colab.** Colab ships a
  correctly-paired torch/torchvision build; upgrading torch breaks that pairing and causes errors
  like `operator torchvision::nms does not exist`, which then surfaces as
  `Could not import module 'Qwen2ForCausalLM'`. Only install what's actually missing (e.g.
  `transformers`, `trl`, `peft`, `unsloth`); **torch and pandas are preinstalled**.
- **torchao**: Colab ships an old `torchao` that current PEFT rejects
  (`Found an incompatible version of torchao`). We don't use it, so the vanilla notebook does
  `!pip uninstall -y torchao` right after install. (The Unsloth notebook lets Unsloth pin its own
  deps; if you still hit it, `!pip install -U torchao` and restart.)
- **bf16 detection**: do **not** trust `torch.cuda.is_bf16_supported()` — it returns `True` on a
  T4 via slow emulation. Choose precision by compute capability instead:
  `bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8`
  (Ampere+ = native bf16; T4 is 7.5 → use fp16, which hits its tensor cores).
- Colab notebooks here are **self-contained**: they download ATIS and build the dataset in-notebook
  (no manual uploads). Keep them that way.

## Notebooks
- Edit `.ipynb` files with the **NotebookEdit** tool (the Edit tool refuses notebooks).
- After editing a notebook, **validate it**: `uv run python -c "import nbformat; nbformat.validate(nbformat.read('FILE.ipynb', as_version=4))"`.
- Keep the vanilla (`atis_finetune_colab.ipynb`) and Unsloth (`atis_finetune_colab_unsloth.ipynb`)
  notebooks **in sync** for everything except the model-load/LoRA/training cells, so they stay a
  fair comparison (same data, schema, hyperparameters, seed, eval).

## Unsloth + training (the two Unsloth notebooks)
- **Padding-free packing is auto-enabled** by Unsloth's `SFTTrainer` even with `packing=False`
  (log: `🦥 Unsloth: Padding-free auto-enabled`). It flattens each micro-batch of
  `per_device_train_batch_size` examples into ONE sequence — so don't reason about padding/throughput
  from stock-TRL defaults, and **don't shrink `MAX_SEQ_LENGTH` toward one example's length**: the cap
  must clear the *packed* length (batch × example), or packed sequences get truncated and the fused
  CE loss crashes. 1024 is safe headroom — it's a ceiling, not a per-step allocation.
- **`lora_dropout=0`** (not 0.05): any nonzero dropout disables Unsloth's fast LoRA kernels
  (log: `patched ... 0 QKV / 0 O / 0 MLP layers`) for ~no regularization gain at 2 epochs on a
  ~9M-param adapter. Kept 0 in all three notebooks (also keeps them identical for fair comparison).

## Judging a run
- **Don't trust train loss.** It's completion-only (prompt masked) over mostly-fixed JSON
  boilerplate, so it plateaus fast by learning the *template* while the value tokens barely move it.
  Judge by the **generation eval** (valid-JSON %, exact match, per-field accuracy) on the held-out
  split. Exact-match lands well below what the loss implies; **date fields** are usually the worst.

## Git + Colab workflow
- **GitHub `main` is the single source of truth.** Colab does **not** live-sync — to pull changes,
  re-open the notebook in Colab via **File → Open notebook → GitHub** (sync can lag a minute).
- Colab open/upload URL pattern:
  `https://colab.research.google.com/github/RodrigoVillatoro/atis-json-finetune/blob/main/<notebook>.ipynb`
- **Commit/push only when the user asks.** Commit messages end with the Co-Authored-By trailer.

## Repo facts
- **Gitignored (do not push):** `data/`, `.venv/`, `reference/`, `.env*`.
- Reproducibility: everything is seeded (`SEED = 42`); generation is greedy (`do_sample=False`);
  the train/test split is a fixed seeded 90/10 pool of ATIS's two CSVs.
- Schema (8 keys): `intent` (list ⊆ {flight, airfare, ground_service, airline}), `from_city`,
  `to_city`, `depart_date`, `depart_time`, `airline`, `round_trip`, `class_type`; `null` when a
  field isn't mentioned. Values are raw surface text (no canonicalization); multi-span values are
  joined (e.g. "june 12"). There is intentionally **no `return_date`** field (only ~0.3% of ATIS).
