import itertools
import time
from production_generators import Neo4jClueGenerator
from salt_ingestion import SaltDataIngestor

def precision_at_k(predicted_targets, human_targets):
    if not human_targets or not predicted_targets: return 0.0
    k = len(human_targets)
    top_k_pred = set(predicted_targets[:k])
    human_set = set(human_targets)
    hits = len(top_k_pred.intersection(human_set))
    return hits / k

def run_grid_search(data_path, max_scenarios=50):
    ingestor = SaltDataIngestor(data_path)
    scenarios = ingestor.get_scenarios()[:max_scenarios]
    
    # Hyperparameters to tune
    hub_penalty_factors = [0.5, 1.0, 2.0]
    # We could also tune team_weight and opp_penalty if we made them parameters in Neo4jClueGenerator
    
    best_config = None
    max_precision = -1.0

    print(f"Starting Grid Search over {len(scenarios)} scenarios...")
    
    for hpf in hub_penalty_factors:
        print(f"\nTesting Hub Penalty Factor: {hpf}")
        gen = Neo4jClueGenerator("bolt://localhost:7687", "neo4j", "password", hub_penalty_factor=hpf)
        
        total_precision = 0.0
        count = 0
        
        t0 = time.time()
        for i, s in enumerate(scenarios):
            try:
                clues = gen.generate_clues(s['team'], s['opponent'], s['neutral'], s['assassin'], max_depth=1)
                if clues:
                    top = clues[0]
                    p_at_k = precision_at_k(top['targets'], s['team'])
                    total_precision += p_at_k
                count += 1
                if (i+1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(scenarios)}...")
            except Exception as e:
                print(f"  Error on scenario {i}: {e}")
        
        avg_precision = total_precision / count if count > 0 else 0
        elapsed = time.time() - t0
        print(f"Result for hpf={hpf}: Avg Precision@K = {avg_precision:.4f} (Time: {elapsed:.2f}s)")
        
        if avg_precision > max_precision:
            max_precision = avg_precision
            best_config = {"hub_penalty_factor": hpf}
        
        gen.close()

    print(f"\nBest Config Found: {best_config} with Precision@K {max_precision:.4f}")
    return best_config

if __name__ == "__main__":
    run_grid_search("data/all.csv", max_scenarios=50)
