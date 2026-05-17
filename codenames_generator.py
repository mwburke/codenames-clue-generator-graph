import networkx as nx, json
from google import genai

class CodenamesClueGenerator:
    def __init__(self, graph: nx.Graph = None):
        self.graph = graph if graph else self._build_mock_graph()

    def _build_mock_graph(self):
        G = nx.Graph()
        G.add_edges_from([("Apple", "Fruit"), ("Strike", "Baseball"), ("Baseball", "Bat"), ("Fruit", "Tree"), ("Bear", "Forest")])
        return G

    def is_legal_clue(self, clue, board_words):
        return not any(clue.lower() in w.lower() or w.lower() in clue.lower() for w in board_words)

    def generate_clues(self, my_team, opponent, neutral, assassin, max_depth=2):
        all_board_words = my_team + opponent + neutral + [assassin]
        my_team_nodes = [w for w in my_team if w in self.graph]
        
        candidates = set()
        for word in my_team_nodes:
            candidates.update(nx.single_source_shortest_path_length(self.graph, word, cutoff=max_depth).keys())

        scored_clues = []
        for candidate in candidates:
            if not self.is_legal_clue(candidate, all_board_words): continue
            distances = nx.single_source_shortest_path_length(self.graph, candidate, cutoff=max_depth + 1)
            
            if distances.get(assassin, float('inf')) <= 1: continue
            if min([distances.get(opp, float('inf')) for opp in opponent] + [float('inf')]) <= 1: continue

            score, targeted_words = 0.0, []
            for word in my_team:
                d = distances.get(word, float('inf'))
                if d == 1: score += 10.0; targeted_words.append(word)
                elif d == 2: score += 3.0; targeted_words.append(word)
            if len(targeted_words) < 2: continue

            for word in neutral:
                d = distances.get(word, float('inf'))
                if d == 1: score -= 2.0
                elif d == 2: score -= 0.5
            for word in opponent:
                if distances.get(word, float('inf')) == 2: score -= 5.0
            if distances.get(assassin, float('inf')) == 2: score -= 8.0

            scored_clues.append({"clue": candidate, "score": score, "targets": list(set(targeted_words)), "number": len(set(targeted_words))})
        return sorted(scored_clues, key=lambda x: x["score"], reverse=True)

class BoardVisionParser:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def parse_board_image(self, image_path, my_color="red"):
        import PIL.Image
        img = PIL.Image.open(image_path)
        prompt = f"""Read the 25 cards on the board. Categorize into strict JSON based on colors. My team is {my_color}. Categories: "my_team", "opponent", "neutral", "assassin"."""
        try:
            response = self.client.models.generate_content(model='gemini-2.5-pro', contents=[prompt, img])
            return json.loads(response.text.replace('```json', '').replace('```', '').strip())
        except Exception as e: return None
