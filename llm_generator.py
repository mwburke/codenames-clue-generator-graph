"""
Pure LLM clue generator — no graph, no embeddings.

Sends the full board to Gemini and asks it to reason directly about
clue quality, safety, and target coverage. Used as a baseline to
compare against the graph-hybrid pipeline.

Every call records what happened in `self.last_call` (status, served model,
token usage, latency, how many clues/targets were dropped) so evaluation can
report real yield and cost instead of assuming them.
"""
import asyncio
import time

from pydantic import BaseModel

from llm_json import LLMJSONError, parse_json_list
from production_generators import _generate_with_fallback, _is_quota_error


class ClueSuggestion(BaseModel):
    clue: str
    targets: list[str]
    justification: str = ""


class LLMDirectGenerator:
    """
    Generates Codenames clues by prompting Gemini with the full board.

    Yield is NOT guaranteed: a call can time out, hit quota, return text that
    doesn't parse, or return clues whose targets aren't team words. Each of
    those is recorded in `last_call["status"]` and counts against yield.

    Args:
        api_key: Gemini API key.
        models: model ids to try in order. Pass a single-item list to pin one
            model (required for evaluation, so cost and quality are attributable).
            None uses the shared quota fallback chain.
        structured_output: ask Gemini for JSON matching ClueSuggestion via
            response_schema. False reproduces the original prompt-only behavior.
        timeout_s: per-call timeout.
    """

    def __init__(self, api_key: str, models: list[str] | None = None,
                 structured_output: bool = True, timeout_s: float = 25.0):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.models = models
        self.structured_output = structured_output
        self.timeout_s = timeout_s
        self.last_call: dict = {}

    async def generate_clues(self, my_team, opponent, neutral, assassin):
        return await self._generate(my_team, opponent, neutral, assassin)

    def _prompt(self, my_team, opponent, neutral, assassin) -> str:
        assassin_line = f"Assassin (instant loss if guessed): {assassin.upper()}" if assassin else ""
        return f"""You are an expert Codenames spymaster. Generate the best clues for the board below.

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

    def _config(self):
        if not self.structured_output:
            return None
        from google.genai import types
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[ClueSuggestion],
        )

    @staticmethod
    def _usage(response) -> dict:
        u = getattr(response, "usage_metadata", None)
        if u is None:
            return {}
        return {
            "prompt_tokens": u.prompt_token_count or 0,
            "cached_tokens": u.cached_content_token_count or 0,
            "output_tokens": u.candidates_token_count or 0,
            "thinking_tokens": u.thoughts_token_count or 0,
        }

    def _raw_items(self, response) -> list:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, list) and parsed:
            return [p.model_dump() if isinstance(p, BaseModel) else p for p in parsed]
        return parse_json_list(response.text)

    async def _generate(self, my_team, opponent, neutral, assassin):
        call = {"status": None, "served_model": None, "latency_s": None,
                "n_raw": 0, "n_kept": 0, "dropped_targets": 0}
        self.last_call = call
        team = {w.lower() for w in my_team}
        prompt = self._prompt(my_team, opponent, neutral, assassin)
        kwargs = {}
        config = self._config()
        if config is not None:
            kwargs["config"] = config

        t0 = time.time()
        try:
            response, served = await asyncio.wait_for(
                _generate_with_fallback(self.client, prompt, models=self.models,
                                        return_model=True, **kwargs),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            call.update(status="timeout", latency_s=time.time() - t0)
            return []
        except Exception as e:
            call.update(status="quota" if _is_quota_error(e) else "api_error",
                        error=str(e)[:300], latency_s=time.time() - t0)
            return []
        call.update(served_model=served, latency_s=time.time() - t0, **self._usage(response))

        try:
            raw = self._raw_items(response)
        except LLMJSONError as e:
            call.update(status="parse_error", error=str(e)[:300])
            return []

        clues = []
        for c in raw:
            if not (isinstance(c, dict) and c.get("clue")):
                continue
            call["n_raw"] += 1
            targets = [t.lower() for t in c.get("targets", []) if isinstance(t, str)]
            valid = [t for t in targets if t in team]
            call["dropped_targets"] += len(targets) - len(valid)
            if not valid:
                continue  # every target was off the team list: not a usable clue
            clues.append({
                "clue": c["clue"].upper(),
                "raw_clue": c["clue"],
                "score": (len(valid) ** 2) * 30.0,
                "targets": valid,
                "number": len(valid),
                "justification": c.get("justification", ""),
            })
        call["n_kept"] = len(clues)
        call["status"] = "ok" if clues else "empty"
        return clues
