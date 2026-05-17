import time, os
import pandas as pd
from salt_ingestion import SaltDataIngestor
from codenames_generator import CodenamesClueGenerator
from semantic_generator import SemanticClueGenerator

class MethodologyEvaluator:
    def __init__(self, dataset_path="data/all.csv"):
        self.ingestor = SaltDataIngestor(dataset_path)
        self.scenarios = self.ingestor.get_scenarios()
        
        # NOTE: For large scale test, this should be the CSR graph or Neo4j.
        # This defaults to the mock graph if my_graph isn't passed.
        self.generators = {
            "Graph": CodenamesClueGenerator(), 
            "Semantic": SemanticClueGenerator()
        }

    def precision_at_k(self, predicted_targets, human_targets):
        if not human_targets or not predicted_targets: return 0.0
        # K is the number of human targets
        k = len(human_targets)
        # Take the top K predictions
        top_k_pred = set(predicted_targets[:k])
        human_set = set(human_targets)
        hits = len(top_k_pred.intersection(human_set))
        return hits / k

    def run_comparison(self, max_games=50):
        results = {name: {"yield": 0, "precision": [], "safe": 0, "time": []} for name in self.generators}
        
        for i, s in enumerate(self.scenarios[:max_games]):
            for name, gen in self.generators.items():
                t0 = time.time()
                clues = gen.generate_clues(s["team"], s["opponent"], s["neutral"], s["assassin"])
                results[name]["time"].append(time.time() - t0)
                
                if clues:
                    results[name]["yield"] += 1
                    best_clue = clues[0]
                    # Compute Precision@K against human clues
                    if s["human_clues"]:
                        human_targets = s["human_clues"][0].get('targets', [])
                        p_at_k = self.precision_at_k(best_clue["targets"], human_targets)
                        results[name]["precision"].append(p_at_k)
                    
                    if self.generators["Semantic"].similarity(best_clue["clue"], s["assassin"]) < 0.5: 
                        results[name]["safe"] += 1
                        
        print("\\n--- Evaluation Report ---")
        for name, r in results.items():
            avg_p = sum(r["precision"])/len(r["precision"]) if r["precision"] else 0
            avg_t = sum(r["time"])/len(r["time"]) if r["time"] else 0
            print(f"[{name}] Yield: {r['yield']}/{max_games} | Avg Precision@K: {avg_p:.3f} | Safe: {r['safe']} | Avg Time: {avg_t:.3f}s")

if __name__ == "__main__":
    evaluator = MethodologyEvaluator()
    evaluator.run_comparison()
