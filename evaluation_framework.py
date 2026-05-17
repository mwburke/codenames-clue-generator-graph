"""
Evaluation framework for the Codenames clue generators.

Runs all active generators against benchmark scenarios from two sources:
  - SALT-NLP dataset (all.csv)
  - Local game records (data/all_games.json)

Metrics per generator:
  - yield_rate:   fraction of scenarios that produced at least one clue
  - precision_at_k: how well the top clue's targets match the human clue's targets
  - single_word_rate: fraction of top clues that are single alphabetic words
  - avg_latency_s: average generation time per scenario
"""
import asyncio
import time
import json
import os
import re
import ast

from salt_ingestion import SaltDataIngestor
from production_generators import Neo4jClueGenerator, WikipediaWordMapper
from semantic_generator import SemanticClueGenerator
from llm_generator import LLMDirectGenerator


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_salt_scenarios(data_path="data/all.csv") -> list[dict]:
    """
    Load SALT-NLP scenarios.

    The SALT dataset does not separate opponent words from an assassin — the
    'black' group is all words to avoid.  We treat the entire black list as
    opponent words and set assassin to empty string, which is accurate to the
    dataset semantics and avoids the previous bug of picking black[0] as a
    hard-excluded assassin.
    """
    ingestor = SaltDataIngestor(data_path)
    raw = ingestor.get_scenarios()
    fixed = []
    for s in raw:
        fixed.append({
            "team":     s["team"],
            "opponent": s["black"],   # full black list = opponent, no special assassin
            "neutral":  s["neutral"],
            "assassin": "",
            "human_clues": s["human_clues"],
            "source": "salt",
        })
    return fixed


def load_game_scenarios(data_path="data/all_games.json") -> list[dict]:
    """Load scenarios from local recorded games."""
    if not os.path.exists(data_path):
        return []
    with open(data_path) as f:
        games = json.load(f)
    scenarios = []
    for game in games:
        board = game.get("board", [])
        color = game.get("active_color", "red")
        team     = [c["word"].lower() for c in board if c["color"] == color]
        opponent = [c["word"].lower() for c in board if c["color"] not in (color, "neutral", "assassin")]
        neutral  = [c["word"].lower() for c in board if c["color"] == "neutral"]
        assassin = next((c["word"].lower() for c in board if c["color"] == "assassin"), "")
        clues    = [{"word": cl["word"].lower(), "targets": [t.lower() for t in cl["targets"]]}
                    for cl in game.get("clues", [])]
        if team and clues:
            scenarios.append({
                "team": team, "opponent": opponent, "neutral": neutral,
                "assassin": assassin, "human_clues": clues, "source": "games",
            })
    return scenarios


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(predicted_targets: list, human_targets: list) -> float:
    if not human_targets or not predicted_targets:
        return 0.0
    k = len(human_targets)
    hits = len({t.lower() for t in predicted_targets[:k]} & {t.lower() for t in human_targets})
    return hits / k


def is_good_clue_word(word: str) -> bool:
    """Single all-alpha word — a proxy for 'valid Codenames clue'."""
    return word.isalpha() and " " not in word


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class MethodologyEvaluator:
    def __init__(self,
                 neo4j_uri="bolt://localhost:7687",
                 neo4j_user="neo4j",
                 neo4j_password="password",
                 salt_path="data/all.csv",
                 games_path="data/all_games.json"):

        salt = load_salt_scenarios(salt_path) if os.path.exists(salt_path) else []
        games = load_game_scenarios(games_path)
        self.scenarios = salt + games
        print(f"Loaded {len(salt)} SALT + {len(games)} game scenarios = {len(self.scenarios)} total")

        self.generators = {}
        api_key = os.getenv("GOOGLE_API_KEY", "")

        neo4j_gen = None
        try:
            neo4j_gen = Neo4jClueGenerator(neo4j_uri, neo4j_user, neo4j_password)
            if api_key:
                neo4j_gen.word_mapper = WikipediaWordMapper(neo4j_gen.driver, api_key)
                print("Neo4j generator ready (+ WikipediaWordMapper).")
            else:
                print("Neo4j generator ready (no word mapper — set GOOGLE_API_KEY to enable).")
            self.generators["Neo4j"] = neo4j_gen
        except Exception as e:
            print(f"Neo4j unavailable ({e}), skipping.")

        try:
            sem = SemanticClueGenerator()
            if sem.model:
                self.generators["Semantic"] = sem
                print("Semantic (GloVe) generator ready.")
        except Exception as e:
            print(f"Semantic generator unavailable ({e}), skipping.")

        if api_key:
            try:
                self.generators["LLM-Direct"] = LLMDirectGenerator(api_key)
                print("LLM-Direct generator ready.")
            except Exception as e:
                print(f"LLM-Direct generator unavailable ({e}), skipping.")

    def close(self):
        gen = self.generators.get("Neo4j")
        if gen:
            gen.close()

    async def run_comparison(self, max_scenarios: int = 50) -> dict:
        scenarios = self.scenarios[:max_scenarios]
        results = {
            name: {"yield": 0, "precision": [], "single_word": 0, "latency": []}
            for name in self.generators
        }

        for i, s in enumerate(scenarios):
            for name, gen in self.generators.items():
                t0 = time.time()
                try:
                    if asyncio.iscoroutinefunction(gen.generate_clues):
                        clues = await gen.generate_clues(
                            s["team"], s["opponent"], s["neutral"], s["assassin"]
                        )
                    else:
                        clues = gen.generate_clues(
                            s["team"], s["opponent"], s["neutral"], s["assassin"]
                        )
                except Exception as e:
                    print(f"  [{name}] error on scenario {i}: {e}")
                    clues = []

                elapsed = time.time() - t0
                results[name]["latency"].append(elapsed)

                if not clues:
                    continue

                results[name]["yield"] += 1
                top = clues[0]
                clue_word = top.get("clue", top.get("raw_clue", "")).upper().split()[0]

                if is_good_clue_word(clue_word):
                    results[name]["single_word"] += 1

                if s["human_clues"]:
                    human_targets = s["human_clues"][0].get("targets", [])
                    results[name]["precision"].append(
                        precision_at_k(top.get("targets", []), human_targets)
                    )

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(scenarios)} scenarios processed...")

        self._print_report(results, len(scenarios))
        return results

    def _print_report(self, results: dict, total: int):
        print("\n--- Evaluation Report ---")
        for name, r in results.items():
            avg_p = sum(r["precision"]) / len(r["precision"]) if r["precision"] else 0.0
            avg_t = sum(r["latency"]) / len(r["latency"]) if r["latency"] else 0.0
            sw_rate = r["single_word"] / total if total else 0.0
            yield_rate = r["yield"] / total if total else 0.0
            print(
                f"[{name:12s}] "
                f"Yield: {r['yield']}/{total} ({yield_rate:.0%})  "
                f"Precision@K: {avg_p:.3f}  "
                f"SingleWord: {sw_rate:.0%}  "
                f"Avg latency: {avg_t:.3f}s"
            )


async def main():
    evaluator = MethodologyEvaluator()
    try:
        await evaluator.run_comparison(max_scenarios=50)
    finally:
        evaluator.close()


if __name__ == "__main__":
    asyncio.run(main())
