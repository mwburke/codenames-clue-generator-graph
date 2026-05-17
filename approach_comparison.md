# Clue Generation Approach Comparison

## Approaches

### 1. Graph-Hybrid (current production path)

```
board words
  → UNWIND title lookup (Neo4j, indexed)
  → WikipediaWordMapper fallback (full-text + embeddings + LLM)
  → Cypher query: GoodClue neighbors, scored by base_score + topology
  → GeminiEmbeddingReranker: clue-to-target similarity + cluster tightness
  → LLMClueScorer: naturalness rating (1–10)
  → post-filter: legality, single-word extraction
```

### 2. Pure LLM (new baseline, `LLMDirectGenerator`)

```
board words
  → single Gemini Flash prompt with full board context
  → returns clues with targets + justification
  → post-filter: legality check only
```

### 3. Semantic / GloVe (legacy baseline)

```
board words
  → GloVe-50 most_similar() for each team word
  → cosine similarity scoring
  → no external calls
```

---

## Side-by-Side Comparison

| Dimension | Graph-Hybrid | Pure LLM | GloVe |
|---|---|---|---|
| **Yield** | ~68% (mapping gap) | ~100% | ~86% |
| **Avg latency** | 0.04s (graph only) | 2–5s | 0.01s |
| **Latency with all AI steps** | ~3–5s | 2–5s | 0.01s |
| **Precision@K (SALT)** | 0.941 | TBD | 0.965* |
| **Clue quality** | Medium — can produce Wikipedia-jargon | High — natural English | Low — GloVe artifacts |
| **Connection transparency** | Full — graph path is auditable | None — black box | Partial — cosine sim |
| **Legality guarantee** | Strong — graph edges are real | Weak — LLM hallucinates | Medium |
| **Handles polysemy** | No — all Wikipedia senses merged | Yes | Partially |
| **Handles proper nouns** | Yes (Wikipedia coverage) | Yes | Poorly |
| **Handles slang / neologisms** | No | Yes | Poorly |
| **Infrastructure required** | Neo4j + Docker | API key only | GloVe model file |
| **Cost per request** | ~$0 (graph) + API for AI steps | ~$0.002–0.005 | $0 |
| **Determinism** | Yes (graph) + No (AI steps) | No | Yes |
| **Tunable offline** | Yes — precompute + grid search | No | Limited |
| **Explainability** | High — shows Wikipedia path | Low — shows justification string | Medium |

*GloVe Precision@K is inflated — SALT targets equal team words, so any generator returning team words scores ~1.0.

---

## Where each approach wins

### Graph-Hybrid wins when:
- The connection is **specific and verifiable** — "MERCURY connects PLANET and THERMOMETER" via shared Wikipedia neighbors is a concrete fact, not a guess
- You need **offline tuneability** — scoring weights (team_weight, hub_penalty) can be grid-searched against labeled data without burning API quota
- **Latency matters** — the base graph query runs in 40ms; AI steps are opt-in
- You want **auditable clues** — you can trace exactly which Wikipedia articles link to which board words
- Board words are **common nouns** with strong Wikipedia presence

### Pure LLM wins when:
- Board words **don't map to Wikipedia** — the 32% yield gap disappears entirely
- The connection requires **cultural or contextual reasoning** — "SWIFT connects TAYLOR and BIRD" relies on knowing Taylor Swift is a singer, not a Wikipedia graph path
- Clues need to feel **natural to human players** — LLM produces words people actually use as Codenames clues
- You need **polysemy awareness** — LLM knows BANK means money in a financial context, not river
- The board has **proper nouns, slang, or recent events** outside Wikipedia's graph structure

### Neither wins clearly on:
- **Safety** — both can produce clues that touch the assassin. Graph uses hard exclusion (1-hop) but misses semantic proximity; LLM reasons about it but can be wrong.
- **Multi-word clue handling** — graph returns Wikipedia titles like "Harry Potter" that need extraction heuristics; LLM is prompted for single words but occasionally ignores that.

---

## Potential hybrid strategy

The strongest approach would be **LLM brainstorm → graph verify**, but reversed from the current `SemanticFirstGenerator` design:

```
1. Graph generates top 30 candidates (fast, verifiable connections)
2. LLM scores and selects the best N (naturalness, Codenames intuition)
```

vs. the alternative:

```
1. LLM brainstorms 20 candidate clues (creative, no mapping failures)
2. Graph verifies which ones have real Wikipedia connections (precision filter)
3. Score by verified connection count
```

The second variant is what `SemanticFirstGenerator` tried but failed to execute well (slow toLower query, wrong verification logic). The first variant is what the current pipeline does.

**Key question the eval will answer**: does the LLM's ability to reason about naturalness and safety outweigh the graph's precision advantage? Run `evaluation_framework.py` with `GOOGLE_API_KEY` set to get the LLM-Direct numbers alongside Neo4j and GloVe.

---

## Running the comparison

```bash
export GOOGLE_API_KEY=your_key_here
cd /path/to/project
python3 evaluation_framework.py
```

This runs all three generators (Neo4j, Semantic/GloVe, LLM-Direct) on the same 50 SALT scenarios and prints a side-by-side report. LLM-Direct is rate-limited by the free tier (15 RPM for Flash); use `max_scenarios=15` if hitting quota.
