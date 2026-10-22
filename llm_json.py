"""
Robust JSON extraction for LLM responses.

The previous approach, re.search(r"\\[.*?\\]", text, re.DOTALL), is non-greedy:
it stops at the first "]" in the response. For a list of clue objects that
first "]" closes the first clue's "targets" array, so the captured text is not
valid JSON, json.loads raises, and the caller returns []. That silently turned
almost every LLM-Direct response into "no clues".

This module parses the whole payload instead, and only falls back to scanning
for a balanced JSON array/object when the model wraps it in prose or fences.
"""
import json
import re

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMJSONError(ValueError):
    """Raised when no JSON value can be recovered from a model response."""


def _balanced_spans(text: str, open_ch: str, close_ch: str):
    """Yield (start, end) spans of balanced open/close pairs, string-aware."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            if depth == 0:
                start = i
            depth += 1
        elif ch == close_ch and depth > 0:
            depth -= 1
            if depth == 0:
                yield start, i + 1


def parse_json_payload(text: str):
    """Return the first JSON value (list or dict) found in an LLM response."""
    if text is None:
        raise LLMJSONError("empty response")
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        for s, e in _balanced_spans(cleaned, open_ch, close_ch):
            try:
                return json.loads(cleaned[s:e])
            except json.JSONDecodeError:
                continue
    raise LLMJSONError(f"no JSON value found in response: {text[:120]!r}")


def parse_json_list(text: str, wrapper_keys=("clues", "results", "items", "ratings")) -> list:
    """Like parse_json_payload, but always returns a list.

    Accepts a bare list, or an object that wraps the list under a common key.
    """
    value = parse_json_payload(text)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for k in wrapper_keys:
            if isinstance(value.get(k), list):
                return value[k]
    raise LLMJSONError(f"expected a JSON list, got {type(value).__name__}")
