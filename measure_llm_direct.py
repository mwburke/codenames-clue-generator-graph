"""
Measure LLM-Direct yield, latency, tokens and cost on the same SALT scenarios
used in the v2 write-up, with one pinned model (no silent fallback).

Usage:
    export GOOGLE_API_KEY=...
    python measure_llm_direct.py --model gemini-2.5-flash-lite --n 50
    python measure_llm_direct.py --model gemini-2.5-flash-lite --n 50 --prompt-only

    --prompt-only   reproduce the published approach (JSON requested in the
                    prompt, parsed from text). Default uses response_schema.
    --prices FILE   JSON: {"<model>": {"input": $/1M, "output": $/1M}}.
                    Thinking tokens are billed at the output rate. Fill this
                    from the current Gemini pricing page; without it, the
                    script reports tokens only.
    --rpm N         client-side rate limit (free tier is low; default 10).

Writes one JSON line per scenario to runs/ and prints a summary you can paste
into the correction note.
"""
import argparse
import asyncio
import json
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone

from evaluation_framework import is_good_clue_word, load_salt_scenarios, precision_at_k
from llm_generator import LLMDirectGenerator


def cost_usd(call: dict, prices: dict) -> float | None:
    p = prices.get(call.get("served_model") or "")
    if not p or "prompt_tokens" not in call:
        return None
    out = call.get("output_tokens", 0) + call.get("thinking_tokens", 0)
    return (call["prompt_tokens"] * p["input"] + out * p["output"]) / 1_000_000


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--data", default="data/all.csv")
    ap.add_argument("--prompt-only", action="store_true")
    ap.add_argument("--prices", default=None)
    ap.add_argument("--rpm", type=float, default=10.0)
    args = ap.parse_args()

    api_key = os.environ["GOOGLE_API_KEY"]
    prices = json.load(open(args.prices)) if args.prices else {}
    scenarios = load_salt_scenarios(args.data)[: args.n]
    if not scenarios:
        raise SystemExit(f"No scenarios loaded from {args.data}")

    gen = LLMDirectGenerator(api_key, models=[args.model],
                             structured_output=not args.prompt_only)
    mode = "prompt-only" if args.prompt_only else "structured"
    os.makedirs("runs", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = f"runs/llm_direct_{args.model}_{mode}_{stamp}.jsonl"

    rows, gap = [], 60.0 / args.rpm
    with open(out_path, "w") as f:
        for i, s in enumerate(scenarios):
            t_start = time.time()
            clues = await gen.generate_clues(s["team"], s["opponent"], s["neutral"], s["assassin"])
            call = dict(gen.last_call)
            top = clues[0] if clues else None
            row = {
                "i": i, **call, "n_clues": len(clues),
                "top_clue": top["clue"] if top else None,
                "single_word": bool(top) and is_good_clue_word(top["clue"].split()[0]),
                "precision_at_k": (precision_at_k(top["targets"], s["human_clues"][0]["targets"])
                                   if top and s["human_clues"] else None),
                "cost_usd": cost_usd(call, prices),
            }
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            print(f"[{i+1}/{len(scenarios)}] {row['status']:<11} {row['top_clue']}", flush=True)
            await asyncio.sleep(max(0.0, gap - (time.time() - t_start)))

    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    lat = sorted(r["latency_s"] for r in rows if r["latency_s"] is not None)
    precs = [r["precision_at_k"] for r in ok if r["precision_at_k"] is not None]
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]
    mean = lambda xs: statistics.mean(xs) if xs else float("nan")

    summary = {
        "model": args.model, "mode": mode, "scenarios": n, "run_file": out_path,
        "yield": f"{len(ok)}/{n} ({len(ok)/n:.0%})",
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "precision_at_k": round(mean(precs), 3),
        "single_word_rate": f"{sum(r['single_word'] for r in rows)/n:.0%}",
        "latency_mean_s": round(mean(lat), 2),
        "latency_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 2) if lat else None,
        "prompt_tokens_mean": round(mean([r.get("prompt_tokens", 0) for r in ok])),
        "output_tokens_mean": round(mean([r.get("output_tokens", 0) for r in ok])),
        "thinking_tokens_mean": round(mean([r.get("thinking_tokens", 0) for r in ok])),
        "dropped_off_team_targets": sum(r["dropped_targets"] for r in rows),
        "cost_per_request_usd": round(mean(costs), 6) if costs else "set --prices",
    }
    print("\n--- LLM-Direct measurement ---")
    print(json.dumps(summary, indent=2))
    with open(out_path.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
