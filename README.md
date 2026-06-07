# atis-json-finetune

Fine-tuning a tiny LLM (**Qwen2.5-0.5B-Instruct** + LoRA) to extract structured JSON from
airline-travel queries (the **ATIS** dataset). A small, educational, Colab-ready experiment in
supervised fine-tuning for structured output.

## What the model does

Given a natural-language airline query, the model returns a **fixed-schema JSON object** with the
user intent and the slots mentioned in the query (`null` when a field isn't mentioned):

```text
Query:  "i need a flight tomorrow from columbus to minneapolis"
```
```json
{
  "intent": ["flight"],
  "from_city": "columbus",
  "to_city": "minneapolis",
  "depart_date": "tomorrow",
  "depart_time": null,
  "airline": null,
  "round_trip": null,
  "class_type": null
}
```

**Schema**

| Field | Type | Notes |
|---|---|---|
| `intent` | list of strings | subset of `["flight", "airfare", "ground_service", "airline"]` |
| `from_city` | string / null | origin as written (usually a city) |
| `to_city` | string / null | destination as written |
| `depart_date` | string / null | e.g. `"tomorrow"`, `"monday"` |
| `depart_time` | string / null | e.g. `"morning"`, `"838 am"` |
| `airline` | string / null | airline name/code |
| `round_trip` | string / null | e.g. `"round trip"` |
| `class_type` | string / null | e.g. `"first class"` |

Values are the **surface text** copied from the query — no canonicalization (dates aren't resolved
to ISO, cities aren't mapped to IATA codes). That keeps the experiment focused on learning the
JSON structure; see [Next steps](#next-steps).

## How it works

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct` (tiny, fits a free/Pro Colab GPU).
- **Method:** LoRA (PEFT) supervised fine-tuning with TRL's `SFTTrainer`; only the JSON
  completion contributes to the loss (the prompt is masked).
- **Data:** [`tuetschek/atis`](https://huggingface.co/datasets/tuetschek/atis). Queries are kept
  only if every intent component is one of the four chosen intents; the fine-grained ATIS slots
  are folded into the 8 schema fields. The two predefined splits are pooled and re-split into a
  **seeded 90/10** train/test partition (reproducible across runs).
- **Evaluation:** JSON validity, exact-match, intent accuracy, and per-field accuracy on the test
  set, reported **before vs. after** fine-tuning.

## Run it in Colab

The notebook `atis_finetune_colab.ipynb` is **self-contained** — it downloads ATIS and builds the
dataset itself, so there's nothing to upload besides the notebook.

1. Open [Google Colab](https://colab.research.google.com) → **File → Upload notebook** →
   select `atis_finetune_colab.ipynb`.
2. **Enable the GPU:** `Runtime → Change runtime type → Hardware accelerator: GPU` (a T4 is enough).
3. **Run all:** `Runtime → Run all`.
   - The first cell (`pip install`) takes ~1–3 min.
   - You'll see baseline generations + metrics (**BEFORE**), then training (a few minutes), then
     the same generations + metrics (**AFTER**).
4. The LoRA adapter is saved to `./atis-qwen-lora`. To keep it past the session, uncomment the
   download/zip lines in the last cell or save it to Google Drive.

Everything is seeded (`SEED = 42`), so re-running reproduces the same split and training.

## Repo contents

| File | Purpose |
|---|---|
| `atis_finetune_colab.ipynb` | The self-contained Colab fine-tuning notebook (main artifact). |
| `atis_tag_analysis.ipynb` | Local EDA: intent/slot tag distributions that drove the schema. |
| `atis_to_json.ipynb` | Local ATIS → JSON conversion + validation. |

The local notebooks use `uv` for the environment:

```bash
uv venv
uv pip install jupyter pandas
uv run jupyter notebook
```

## Next steps

- Make fields **multi-valued** to handle multi-leg itineraries and time ranges.
- Add a **canonicalization layer**: resolve dates to ISO, cities/airports to IATA codes, and map
  `round_trip`/`class_type` to booleans/enums (e.g. `dateparser`/Duckling + a gazetteer + Pydantic).
- **Enforce** valid structured output via constrained decoding (e.g. Outlines/Guidance).
- Broaden intent/slot coverage; sweep model size and LoRA rank.
