# Codenames Clue Generator

A spymaster assistant for the board game [Codenames](https://codenames.game/). Given the current board state, it generates ranked clues with target words and scores.

This is v2 of the project — the original embedding-based approach is documented in [this blog post](https://mwburke.github.io/data%20science/2021/12/12/codenames-clue-generator-version-1.html). The v2 graph-based approach and results are written up [here](https://mwburke.github.io).

---

## Approaches

Three generators are implemented, each with different tradeoffs:

| | Graph (Neo4j) | LLM-Direct | GloVe |
|---|---|---|---|
| Yield | ~68% | ~100% | ~86% |
| Avg latency | ~40ms | 2–5s | ~12ms |
| Legality guarantee | Strong | Weak | Medium |
| Infrastructure | Neo4j + Docker | API key only | GloVe model file |

**Graph** (`Neo4jClueGenerator`) — finds clue candidates as shared Wikipedia neighbors of team words. Fast, auditable, and free at runtime after setup. Has a yield gap for words with sparse Wikipedia connections.

**LLM-Direct** (`LLMDirectGenerator`) — sends the full board to Gemini and asks it to reason about clue quality directly. 100% yield, natural-sounding output, no infrastructure required.

**GloVe** (`SemanticClueGenerator`) — cosine similarity over GloVe-50 vectors. Fast and free, no API required. Lower quality than the other two.

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for Neo4j)
- A Google API key for Gemini features (optional but recommended)

### 1. Neo4j + Wikipedia graph

Start Neo4j:

```bash
docker-compose up -d
```

Import the Wikipedia link graph (requires the raw dump — see `import_graph.sh` for details):

```bash
bash import_graph.sh
```

Run the one-time precomputation step (assigns `GoodClue` labels and scores ~950k nodes):

```bash
python precompute_graph.py
```

This takes ~12 minutes on first run. Re-run with `--force` if you change scoring parameters:

```bash
python precompute_graph.py --hub-penalty 25 --force
```

### 2. Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or: pip install fastapi uvicorn neo4j google-genai numpy
```

### 3. Environment variables

```bash
export GOOGLE_API_KEY=your_key_here   # enables WikipediaWordMapper, embedding reranker, LLM scorer, LLM-Direct
```

---

## Running

### Backend API

```bash
uvicorn backend_api:app --reload --port 8000
```

The API exposes:
- `POST /api/generate-clues` — generate clues for a board state
- `GET /api/load-salt-board` — load a random board from the SALT-NLP dataset

### Frontend

```bash
cd frontend && npm install && npm start
```

---

## Evaluation

Run the evaluation framework against 50 SALT-NLP scenarios:

```bash
python evaluation_framework.py
```

With `GOOGLE_API_KEY` set, this also runs the LLM-Direct generator for comparison.

---

## Key files

| File | Purpose |
|---|---|
| `production_generators.py` | `Neo4jClueGenerator`, `GeminiEmbeddingReranker`, `WikipediaWordMapper`, `LLMClueScorer` |
| `llm_generator.py` | `LLMDirectGenerator` — pure LLM baseline |
| `precompute_graph.py` | One-time graph precomputation (run before first use) |
| `evaluation_framework.py` | Benchmarks all generators against SALT-NLP |
| `semantic_generator.py` | GloVe baseline |
| `backend_api.py` | FastAPI server |
| `approach_comparison.md` | Detailed side-by-side comparison of all three approaches |

---

## Data

The SALT-NLP dataset (`data/all.csv`) contains human-labeled Codenames games and is used for evaluation. See the [SALT paper](https://arxiv.org/abs/2112.09169) for details.

Wikipedia dump files are not included in this repo (they are multi-GB). See `import_graph.sh` for download and import instructions.
