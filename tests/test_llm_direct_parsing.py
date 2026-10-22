"""Offline tests for LLM-Direct parsing. No API key or Neo4j needed."""
import asyncio
import json
import re
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_json import LLMJSONError, parse_json_list  # noqa: E402

TYPICAL = """```json
[
  {"clue": "OCEAN", "targets": ["wave", "ship"], "justification": "sea things"},
  {"clue": "BANK", "targets": ["river"], "justification": "river bank"}
]
```"""


def test_old_regex_breaks_on_typical_response():
    """Documents the original bug: non-greedy match stops at the first ']'."""
    m = re.search(r"\[.*?\]", TYPICAL, re.DOTALL)
    with pytest.raises(json.JSONDecodeError):
        json.loads(m.group(0))


@pytest.mark.parametrize("text", [
    TYPICAL,
    TYPICAL.strip("`").removeprefix("json"),
    "Here are my clues:\n" + TYPICAL + "\nGood luck!",
    '{"clues": ' + TYPICAL.strip("`").removeprefix("json") + "}",
    '[{"clue": "A]B", "targets": ["x"], "justification": "bracket ] in a string"}]',
])
def test_parse_json_list_handles_real_shapes(text):
    items = parse_json_list(text)
    assert isinstance(items, list) and items and "clue" in items[0]


def test_parse_json_list_rejects_garbage():
    with pytest.raises(LLMJSONError):
        parse_json_list("Sorry, I can't help with that.")


# --- Generator-level tests with a fake Gemini client ------------------------

class _Usage:
    prompt_token_count, cached_content_token_count = 210, 0
    candidates_token_count, thoughts_token_count = 180, 0


class _Resp:
    def __init__(self, text):
        self.text, self.parsed, self.usage_metadata = text, None, _Usage()


class _FakeModels:
    def __init__(self, text):
        self._text = text

    def generate_content(self, model, contents, **kwargs):
        return _Resp(self._text)


def _gen(text, monkeypatch):
    fake_genai = types.SimpleNamespace(
        Client=lambda api_key: types.SimpleNamespace(models=_FakeModels(text)))
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google", types.SimpleNamespace(genai=fake_genai))
    from llm_generator import LLMDirectGenerator
    return LLMDirectGenerator("k", models=["pinned-model"], structured_output=False)


def test_generator_returns_clues_and_records_call(monkeypatch):
    gen = _gen(TYPICAL, monkeypatch)
    clues = asyncio.run(gen.generate_clues(["wave", "ship", "river"], ["dog"], ["cat"], ""))
    assert [c["clue"] for c in clues] == ["OCEAN", "BANK"]
    assert gen.last_call["status"] == "ok"
    assert gen.last_call["served_model"] == "pinned-model"
    assert gen.last_call["prompt_tokens"] == 210


def test_generator_drops_off_team_targets(monkeypatch):
    gen = _gen('[{"clue": "PET", "targets": ["dog", "cat"], "justification": ""}]', monkeypatch)
    clues = asyncio.run(gen.generate_clues(["wave"], ["dog"], ["cat"], ""))
    assert clues == []
    assert gen.last_call["status"] == "empty"
    assert gen.last_call["dropped_targets"] == 2


def test_generator_reports_parse_error(monkeypatch):
    gen = _gen("no json here", monkeypatch)
    assert asyncio.run(gen.generate_clues(["wave"], [], [], "")) == []
    assert gen.last_call["status"] == "parse_error"
