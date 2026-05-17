"""
Grid search over scoring coefficients for Neo4jClueGenerator.

Hub penalty is now pre-baked into base_score by precompute_graph.py.
To tune hub_penalty_factor, re-run precompute_graph.py with --hub-penalty <value>
and then benchmark here.

Tunable at query time: team signal weight, opponent risk penalty, neutral penalty.
"""
import asyncio
import time
from production_generators import Neo4jClueGenerator
from salt_ingestion import SaltDataIngestor


def precision_at_k(predicted_targets, human_targets):
    if not human_targets or not predicted_targets: return 0.0
    k = len(human_targets)
    hits = len(set(predicted_targets[:k]) & set(human_targets))
    return hits / k


async def run_grid_search(data_path, max_scenarios=50):
    ingestor = SaltDataIngestor(data_path)
    scenarios = ingestor.get_scenarios()[:max_scenarios]

    # Query-time tunable weights (team signal, opp penalty, neutral penalty)
    configs = [
        {"team_weight": 20.0, "opp_penalty": 40.0, "neut_penalty": 15.0},
        {"team_weight": 30.0, "opp_penalty": 40.0, "neut_penalty": 15.0},
        {"team_weight": 30.0, "opp_penalty": 60.0, "neut_penalty": 15.0},
        {"team_weight": 40.0, "opp_penalty": 50.0, "neut_penalty": 10.0},
    ]

    best_config, max_precision = None, -1.0
    print(f"Grid search over {len(scenarios)} scenarios, {len(configs)} configs...")

    for cfg in configs:
        gen = Neo4jClueGenerator("bolt://localhost:7687", "neo4j", "password",
                                 team_weight=cfg["team_weight"],
                                 opp_penalty=cfg["opp_penalty"],
                                 neut_penalty=cfg["neut_penalty"])
        total_precision, count = 0.0, 0
        t0 = time.time()

        for i, s in enumerate(scenarios):
            try:
                clues = await gen.generate_clues(s["team"], s["opponent"], s["neutral"], s["assassin"])
                if clues:
                    total_precision += precision_at_k(clues[0]["targets"], s["team"])
                count += 1
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(scenarios)}...")
            except Exception as e:
                print(f"  Error on scenario {i}: {e}")

        avg = total_precision / count if count else 0
        print(f"{cfg} → Precision@K={avg:.4f}  ({time.time()-t0:.1f}s)")

        if avg > max_precision:
            max_precision, best_config = avg, cfg
        gen.close()

    print(f"\nBest: {best_config}  Precision@K={max_precision:.4f}")
    return best_config


if __name__ == "__main__":
    asyncio.run(run_grid_search("data/all.csv", max_scenarios=50))
