#!/usr/bin/env python3
"""Benchmark + evaluate Ollama GGUF models on the ATIS structured-output task.

Runs queries ONE BY ONE (no batching) against the local Ollama server and reports,
per model: JSON validity, exact-match, intent + per-field accuracy, latency
(mean / median / p95), and generation throughput (tokens/sec). Prints a side-by-side
table and writes per-query CSVs.

The test set is rebuilt with the SAME seeded 90/10 ATIS split as the notebooks, so
results are directly comparable across models (and to the training-time eval).

Usage
-----
  # auto-register each model from a folder (Modelfile + .gguf inside):
  uv run python compare_ollama.py \
      --model qwen=~/Downloads/atis-qwen-unsloth \
      --model lfm2=~/Downloads/atis-lfm2

  # or use already-registered Ollama model names:
  uv run python compare_ollama.py --model qwen --model lfm2

  # optional flags:
  #   --limit N     only run the first N test queries (default: all)
  #   --out DIR     where to write per-query CSVs (default: ./bench)

Assumes each model was created with our Modelfile (system prompt + ChatML template
baked in), so we send only the raw user query as the prompt.
"""
import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.request

SEED = 42
TEST_FRACTION = 0.10
NUM_PREDICT = 128  # matches the notebooks' max_new_tokens
OLLAMA_URL = "http://localhost:11434/api/generate"
BASE = "https://huggingface.co/datasets/tuetschek/atis/resolve/main"
DATA_DIR = "data"  # gitignored; CSVs cached here

INTENT_ORDER = ["flight", "airfare", "ground_service", "airline"]
CHOSEN = set(INTENT_ORDER)
SCHEMA_FIELDS = ["from_city", "to_city", "depart_date", "depart_time",
                 "airline", "round_trip", "class_type"]
FIELD_MAP = {
    "fromloc": "from_city", "toloc": "to_city",
    "depart_date": "depart_date", "depart_time": "depart_time",
    "airline_name": "airline", "airline_code": "airline",
    "round_trip": "round_trip", "class_type": "class_type",
}


# --------------------------------------------------------------------------- data
def ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    for split in ("train", "test"):
        p = os.path.join(DATA_DIR, f"atis_{split}.csv")
        if not os.path.exists(p):
            print(f"downloading {p} ...")
            urllib.request.urlretrieve(f"{BASE}/atis_{split}.csv", p)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def normalize_intent(intent_str):
    parts = set(intent_str.split("+"))
    if not parts <= CHOSEN:
        return None
    return [i for i in INTENT_ORDER if i in parts]


def extract_spans(text, slots):
    spans, base, cur = [], None, []
    for tok, tag in zip(text.split(), slots.split()):
        if tag == "O":
            if base is not None:
                spans.append((base, " ".join(cur))); base, cur = None, []
            continue
        prefix, b = tag[0], tag[2:]
        if prefix == "B" or b != base:
            if base is not None:
                spans.append((base, " ".join(cur)))
            base, cur = b, [tok]
        else:
            cur.append(tok)
    if base is not None:
        spans.append((base, " ".join(cur)))
    return spans


def to_record(text, slots):
    acc = {f: [] for f in SCHEMA_FIELDS}
    for b, span_text in extract_spans(text, slots):
        field = FIELD_MAP.get(b.split(".")[0])
        if field is not None:
            acc[field].append(span_text)
    return {f: " ".join(v) if v else None for f, v in acc.items()}


def build(path):
    rows = []
    for r in read_csv(path):
        intent = normalize_intent(r["intent"])
        if intent is None:
            continue
        rows.append({"query": r["text"],
                     "output": {"intent": intent, **to_record(r["text"], r["slots"])}})
    return rows


def build_test_rows():
    ensure_data()
    all_rows = (build(os.path.join(DATA_DIR, "atis_train.csv"))
                + build(os.path.join(DATA_DIR, "atis_test.csv")))
    random.Random(SEED).shuffle(all_rows)
    n_test = round(len(all_rows) * TEST_FRACTION)
    return all_rows[:n_test]


# ------------------------------------------------------------------------- scoring
def parse_json(text):
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e < s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None


def _norm(v):
    return v.strip().lower() if isinstance(v, str) else v


def pct(xs, p):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
    return xs[k]


# -------------------------------------------------------------------------- ollama
def register_model(label, folder):
    folder = os.path.expanduser(folder)
    if not os.path.exists(os.path.join(folder, "Modelfile")):
        sys.exit(f"[{label}] no Modelfile found in {folder}")
    print(f"[{label}] ollama create from {folder} ...")
    subprocess.run(["ollama", "create", label, "-f", "Modelfile"], cwd=folder, check=True)


def ollama_generate(model, prompt):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "5m",
        "options": {"temperature": 0, "num_predict": NUM_PREDICT},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        resp = json.loads(r.read())
    return resp, time.perf_counter() - t0


def benchmark(label, test_rows, out_dir):
    print(f"[{label}] warming up ...")
    ollama_generate(label, test_rows[0]["query"])  # discard: includes model load

    n = len(test_rows)
    rows_out, lat, tps, gen_toks = [], [], [], []
    valid = exact = intent_ok = 0
    field_ok = {f: 0 for f in SCHEMA_FIELDS}

    for i, row in enumerate(test_rows):
        resp, wall = ollama_generate(label, row["query"])
        text = resp.get("response", "")
        ec = resp.get("eval_count") or 0
        ed = resp.get("eval_duration") or 0  # nanoseconds
        gen_tps_q = ec / (ed / 1e9) if ed else 0.0

        pred = parse_json(text)
        ok_json = pred is not None
        ok_intent = ok_exact = False
        if ok_json:
            valid += 1
            gold = row["output"]
            gi = set(gold["intent"])
            pi = set(pred.get("intent", [])) if isinstance(pred.get("intent"), list) else set()
            ok_intent = gi == pi
            intent_ok += ok_intent
            all_ok = ok_intent
            for f in SCHEMA_FIELDS:
                c = _norm(pred.get(f)) == _norm(gold[f])
                field_ok[f] += c
                if not c:
                    all_ok = False
            ok_exact = all_ok
            exact += ok_exact

        lat.append(wall)
        gen_toks.append(ec)
        if gen_tps_q:
            tps.append(gen_tps_q)
        rows_out.append({
            "query": row["query"],
            "gold": json.dumps(row["output"], ensure_ascii=False),
            "pred": text.strip(),
            "valid_json": int(ok_json),
            "exact": int(ok_exact),
            "intent_ok": int(ok_intent),
            "latency_s": round(wall, 4),
            "gen_tokens": ec,
            "gen_tps": round(gen_tps_q, 1),
        })
        print(f"\r[{label}] {i + 1}/{n}", end="", flush=True)
    print()

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"results_{label}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    return {
        "model": label,
        "n": n,
        "valid_json": valid / n,
        "exact_match": exact / n,
        "intent_acc": intent_ok / n,
        "field_acc": {f: field_ok[f] / n for f in SCHEMA_FIELDS},
        "lat_mean": statistics.mean(lat),
        "lat_median": statistics.median(lat),
        "lat_p95": pct(lat, 0.95),
        "gen_tps_mean": statistics.mean(tps) if tps else 0.0,
        "throughput_overall": sum(gen_toks) / sum(lat) if sum(lat) else 0.0,
        "csv": csv_path,
    }


# --------------------------------------------------------------------------- output
def print_table(summaries):
    labels = [s["model"] for s in summaries]
    w = 14

    def row(name, vals):
        print(f"  {name:<22}" + "".join(f"{v:>{w}}" for v in vals))

    print("\n" + "=" * (24 + w * len(labels)))
    print("  ATIS structured-output benchmark (query-by-query, temperature 0)")
    print("=" * (24 + w * len(labels)))
    row("metric", labels)
    print("  " + "-" * (22 + w * len(labels)))
    row("n", [s["n"] for s in summaries])
    print("  --- quality (% of n) ---")
    row("valid JSON", [f"{s['valid_json']:.1%}" for s in summaries])
    row("exact match", [f"{s['exact_match']:.1%}" for s in summaries])
    row("intent acc", [f"{s['intent_acc']:.1%}" for s in summaries])
    for f in SCHEMA_FIELDS:
        row(f"  {f}", [f"{s['field_acc'][f]:.1%}" for s in summaries])
    print("  --- latency (s/query) ---")
    row("mean", [f"{s['lat_mean']:.3f}" for s in summaries])
    row("median", [f"{s['lat_median']:.3f}" for s in summaries])
    row("p95", [f"{s['lat_p95']:.3f}" for s in summaries])
    print("  --- throughput (tok/s) ---")
    row("gen tok/s (mean)", [f"{s['gen_tps_mean']:.1f}" for s in summaries])
    row("overall tok/s", [f"{s['throughput_overall']:.1f}" for s in summaries])
    print("=" * (24 + w * len(labels)))
    for s in summaries:
        print(f"  per-query CSV: {s['csv']}")


def parse_model_arg(arg):
    """`label` (already-registered) or `label=/path/to/folder` (auto-register)."""
    if "=" in arg:
        label, folder = arg.split("=", 1)
        return label, folder
    return arg, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="label or label=/path/to/folder (repeatable)")
    ap.add_argument("--limit", type=int, default=None, help="only run first N queries")
    ap.add_argument("--out", default="bench", help="output dir for per-query CSVs")
    args = ap.parse_args()

    test_rows = build_test_rows()
    if args.limit:
        test_rows = test_rows[:args.limit]
    print(f"test queries: {len(test_rows)}\n")

    summaries = []
    for arg in args.model:
        label, folder = parse_model_arg(arg)
        if folder:
            register_model(label, folder)
        summaries.append(benchmark(label, test_rows, args.out))

    print_table(summaries)


if __name__ == "__main__":
    main()
