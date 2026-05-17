import itertools
import numpy as np
try: import gensim.downloader as api; HAS_GENSIM = True
except ImportError: HAS_GENSIM = False

class SemanticClueGenerator:
    def __init__(self, model_name="glove-wiki-gigaword-50"):
        self.model = api.load(model_name) if HAS_GENSIM else None

    def is_legal_clue(self, clue, board_words):
        return not any(clue.lower() in w.lower() or w.lower() in clue.lower() for w in board_words)

    def similarity(self, word1, word2):
        if not self.model: return 0.5 
        w1, w2 = word1.lower(), word2.lower()
        # OOV handling for multi-word phrases
        v1 = self._get_vector(w1)
        v2 = self._get_vector(w2)
        if v1 is None or v2 is None: return 0.0
        
        # Cosine similarity
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _get_vector(self, word):
        if word in self.model: return self.model[word]
        # Average vectors for multi-word phrases if out of vocabulary
        parts = word.split()
        vectors = [self.model[p] for p in parts if p in self.model]
        if not vectors: return None
        return np.mean(vectors, axis=0)

    def generate_clues(self, my_team, opponent, neutral, assassin, max_depth=2):
        if not self.model: return []
        scored_clues = []
        candidates = set()
        
        # Use single words that exist in model
        team_valid = [w for w in my_team if self._get_vector(w.lower()) is not None]
        
        # Generate candidates from single words as well
        for word in team_valid:
            try:
                candidates.update([w[0] for w in self.model.most_similar(positive=[word.lower()], topn=20)])
            except KeyError: continue
            
        for pair in itertools.combinations(team_valid, 2):
            try: 
                # Gensim most_similar only works for in-vocab words, so we use lower() splits if needed
                p1, p2 = pair[0].lower(), pair[1].lower()
                if p1 in self.model and p2 in self.model:
                    candidates.update([w[0] for w in self.model.most_similar(positive=[p1, p2], topn=20)])
            except KeyError: continue

        for candidate in candidates:
            if not candidate.isalpha() or not self.is_legal_clue(candidate, my_team + opponent + neutral + [assassin] if assassin else my_team + opponent + neutral): continue
            score, targets = 0.0, []
            for team_word in my_team:
                sim = self.similarity(candidate, team_word)
                if sim > 0.6: score += (sim * 10); targets.append(team_word)
            if len(targets) < 1 or (assassin and self.similarity(candidate, assassin) > 0.5): continue
            
            for opp_word in opponent:
                sim = self.similarity(candidate, opp_word)
                if sim > 0.5: score -= (sim * 15)
            for neut_word in neutral:
                sim = self.similarity(candidate, neut_word)
                if sim > 0.5: score -= (sim * 5)
            if score > 0: scored_clues.append({"clue": candidate.upper(), "score": score, "targets": targets, "number": len(targets)})
        return sorted(scored_clues, key=lambda x: x["score"], reverse=True)
