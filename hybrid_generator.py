from codenames_generator import CodenamesClueGenerator
from semantic_generator import SemanticClueGenerator

class HybridClueGenerator:
    def __init__(self, graph_gen=None):
        self.graph_gen = graph_gen if graph_gen else CodenamesClueGenerator()
        self.semantic_gen = SemanticClueGenerator()

    async def generate_clues(self, my_team, opponent, neutral, assassin, max_depth=2):
        graph_clues = await self.graph_gen.generate_clues(my_team, opponent, neutral, assassin, max_depth)
        hybrid_clues = []
        for clue_data in graph_clues:
            candidate, graph_score, targets = clue_data["clue"], clue_data["score"], clue_data["targets"]
            semantic_score, valid_targets = 0, 0
            for target in targets:
                sim = self.semantic_gen.similarity(candidate, target)
                if sim > 0.25: semantic_score += (sim * 5); valid_targets += 1
            if valid_targets < 1: continue
            if self.semantic_gen.similarity(candidate, assassin) > 0.4: semantic_score -= 10
            hybrid_clues.append({"clue": candidate, "score": graph_score + semantic_score, "targets": targets, "number": len(targets)})
        return sorted(hybrid_clues, key=lambda x: x["score"], reverse=True)
