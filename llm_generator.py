"""
Pure LLM clue generator — no graph, no embeddings.

Sends the full board to Gemini and asks it to reason directly about
clue quality, safety, and target coverage. Used as a baseline to
compare against the graph-hybrid pipeline.
"""
import asyncio
import json
import re

from production_generators import _generate_with_fallback


class LLMDirectGenerator:
    """
    Generates Codenames clues by prompting Gemini with the full board.

    Pros over graph approach:
      - 100% yield (no Wikipedia mapping failures)
      - Handles polysemy, slang, proper nouns, cultural references naturally
      - Creative lateral connections the graph can't find
      - Zero infrastructure (no Neo4j)

    Cons vs graph approach:
      - Cannot guarantee legality — must be post-filtered
      - Hallucination risk: may invent connections that feel right but aren't
      - Higher latency per call (full board in context each time)
      - Non-deterministic: same board can give different clues each run
      - Harder to tune systematically
      - No transparency: can't explain why a connection exists
      - Cost scales with every request; graph is free after setup
    """

    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)

    async def generate_clues(self, my_team, opponent, neutral, assassin):
        return await self._generate(my_team, opponent, neutral, assassin)

    async def _generate(self, my_team, opponent, neutral, assassin):
        assassin_line = f"Assassin (instant loss if guessed): {assassin.upper()}" if assassin else ""
        prompt = f"""You are an expert Codenames spymaster. Generate the best clues for the board below.

YOUR team words (try to connect as many as possible with one clue):
{', '.join(w.upper() for w in my_team)}

Opponent words (avoid leading to these):
{', '.join(w.upper() for w in opponent) if opponent else 'none'}

Neutral words (avoid):
{', '.join(w.upper() for w in neutral) if neutral else 'none'}

{assassin_line}

Rules:
- The clue must be a single word (no hyphens, no proper nouns that contain a board word)
- The clue cannot be a form of any board word (no plurals, conjugations, or substrings)
- Prefer clues connecting 2+ team words

Return ONLY valid JSON — a list of up to 10 clues, best first:
[
  {{"clue": "WORD", "targets": ["team_word1", "team_word2"], "justification": "brief reason"}},
  ...
]"""

        try:
            response = await asyncio.wait_for(
                _generate_with_fallback(self.client, prompt),
                timeout=25.0,
            )
            match = re.search(r"\[.*?\]", response.text, re.DOTALL)
            if not match:
                return []
            raw = json.loads(match.group(0))
            return [
                {
                    "clue": c["clue"].upper(),
                    "raw_clue": c["clue"],
                    "score": (len(c.get("targets", [])) ** 2) * 30.0,
                    "targets": [t.lower() for t in c.get("targets", [])],
                    "number": len(c.get("targets", [])),
                    "justification": c.get("justification", ""),
                }
                for c in raw
                if isinstance(c, dict) and c.get("clue")
            ]
        except Exception as e:
            print(f"DEBUG: LLMDirectGenerator failed: {e}", flush=True)
            return []
