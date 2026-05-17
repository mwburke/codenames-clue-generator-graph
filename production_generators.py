import time
import json
import asyncio
import re
import os
import numpy as np
from neo4j import GraphDatabase

# Ordered by free-tier daily quota (highest first). The fallback helper tries
# each model in sequence, skipping on 429/quota errors.
FLASH_FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "quota", "resource_exhausted", "exhausted"))


async def _generate_with_fallback(client, contents, models=None, **kwargs) -> object:
    """Try each model in order, skipping on quota errors. Raises if all exhausted."""
    for model in (models or FLASH_FALLBACK_MODELS):
        try:
            return await asyncio.to_thread(
                client.models.generate_content, model=model, contents=contents, **kwargs
            )
        except Exception as e:
            if _is_quota_error(e):
                print(f"DEBUG: quota exceeded on {model}, trying next...", flush=True)
                continue
            raise
    raise RuntimeError(f"All models exhausted: {models or FLASH_FALLBACK_MODELS}")

class GeminiEmbeddingReranker:
    """
    Re-ranks graph candidates using Gemini gemini-embedding-001.

    Adds two semantic signals on top of the graph topology score:
      1. Clue-to-target similarity: how semantically close is the clue to
         each of its target words? Filters out topological flukes where
         two targets share a Wikipedia hub but aren't meaningfully related.
      2. Target cluster tightness: average pairwise similarity of the target
         words. A clue connecting BANK+VAULT is better than BANK+RIVER+COLD
         because the first cluster is more coherent.

    Embeddings are cached in-process (per instance lifetime) so repeated
    board words across requests don't re-hit the API.
    """

    EMBEDDING_MODEL = "gemini-embedding-001"  # also try gemini-embedding-2 for higher quality
    BATCH_LIMIT = 100  # Gemini API max per request

    def __init__(self, api_key: str, semantic_weight: float = 20.0):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.semantic_weight = semantic_weight
        self._cache: dict[str, np.ndarray] = {}

    async def _embed_batch(self, words: list[str]) -> dict[str, np.ndarray]:
        to_fetch = [w for w in words if w not in self._cache]
        for i in range(0, len(to_fetch), self.BATCH_LIMIT):
            batch = to_fetch[i : i + self.BATCH_LIMIT]
            try:
                response = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=self.EMBEDDING_MODEL,
                    contents=batch,
                )
                for word, emb in zip(batch, response.embeddings):
                    self._cache[word] = np.array(emb.values, dtype=np.float32)
            except Exception as e:
                print(f"DEBUG: Embedding batch failed: {e}", flush=True)
        return {w: self._cache[w] for w in words if w in self._cache}

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    async def rerank(self, candidates: list[dict], assassin: str) -> list[dict]:
        if not candidates:
            return candidates

        # Collect all words to embed in one batch
        all_words = list({
            c["raw_clue"] for c in candidates
        } | {
            t for c in candidates for t in c.get("targets", [])
        } | ({assassin} if assassin else set()))

        t0 = time.time()
        embeddings = await self._embed_batch(all_words)
        print(f"DEBUG: Embedding {len(all_words)} words took {time.time()-t0:.3f}s "
              f"({len(all_words) - sum(1 for w in all_words if w in self._cache)} new, "
              f"{sum(1 for w in all_words if w in self._cache)} cached)", flush=True)

        assassin_vec = embeddings.get(assassin) if assassin else None
        w = self.semantic_weight

        for c in candidates:
            clue_vec = embeddings.get(c["raw_clue"])
            if clue_vec is None:
                c["semantic_score"] = 0.0
                continue

            target_vecs = [embeddings[t] for t in c.get("targets", []) if t in embeddings]

            # Signal 1: avg clue-to-target similarity
            avg_target_sim = (
                float(np.mean([self._cosine(clue_vec, tv) for tv in target_vecs]))
                if target_vecs else 0.0
            )

            # Signal 2: target cluster tightness (avg pairwise)
            cluster_sim = 0.0
            if len(target_vecs) >= 2:
                pairs = [
                    self._cosine(target_vecs[i], target_vecs[j])
                    for i in range(len(target_vecs))
                    for j in range(i + 1, len(target_vecs))
                ]
                cluster_sim = float(np.mean(pairs))

            # Penalty: assassin proximity
            assassin_penalty = 0.0
            if assassin_vec is not None:
                assassin_sim = self._cosine(clue_vec, assassin_vec)
                if assassin_sim > 0.3:
                    assassin_penalty = assassin_sim * w * 1.5

            c["semantic_score"] = avg_target_sim * w + cluster_sim * w * 0.5 - assassin_penalty
            c["score"] = c["score"] + c["semantic_score"]

        return sorted(candidates, key=lambda x: x["score"], reverse=True)


class WikipediaWordMapper:
    """
    Fallback mapper for Codenames words that don't match the exact/capitalized
    Neo4j title lookup.

    For each unmapped word:
      1. Full-text search on Page.title to get candidate Wikipedia titles.
      2. Gemini gemini-embedding-001 to rank candidates by cosine similarity
         to the original word.
      3. Gemini Flash (only when top candidates are too close to call) to
         pick the title a Codenames player would most naturally associate
         with the word.

    Results are cached in-process so repeated board words are free.
    Requires the page_title_fulltext index created by precompute_graph.py.
    """

    EMBEDDING_MODEL = "gemini-embedding-001"  # also try gemini-embedding-2 for higher quality
    AMBIGUITY_GAP = 0.05   # call LLM when gap between top-2 similarities < this
    CANDIDATE_LIMIT = 10

    def __init__(self, driver, api_key: str):
        self.driver = driver
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self._cache: dict[str, str | None] = {}

    async def resolve_batch(self, words: list[str]) -> dict[str, str]:
        """Return {word: wikipedia_title} for words not already in cache."""
        uncached = [w for w in words if w not in self._cache]
        if not uncached:
            return {w: self._cache[w] for w in words if self._cache.get(w)}

        # Step 1: full-text search for all uncached words in parallel
        candidate_map: dict[str, list[str]] = {}
        for word in uncached:
            candidates = await asyncio.to_thread(self._fulltext_search, word)
            candidate_map[word] = candidates

        words_needing_embed = [w for w, c in candidate_map.items() if len(c) > 1]
        all_texts = list({t for w in words_needing_embed for t in [w] + candidate_map[w]})

        # Step 2: embed word + candidates in batches of 100
        embeddings: dict[str, np.ndarray] = {}
        for i in range(0, len(all_texts), 100):
            batch = all_texts[i : i + 100]
            try:
                response = await asyncio.to_thread(
                    self.client.models.embed_content,
                    model=self.EMBEDDING_MODEL,
                    contents=batch,
                )
                for text, emb in zip(batch, response.embeddings):
                    embeddings[text] = np.array(emb.values, dtype=np.float32)
            except Exception as e:
                print(f"DEBUG: WikipediaWordMapper embedding batch failed: {e}", flush=True)

        # Step 3: rank candidates, call LLM when ambiguous
        llm_tasks = []
        for word in uncached:
            candidates = candidate_map.get(word, [])
            if not candidates:
                self._cache[word] = None
                continue
            if len(candidates) == 1:
                self._cache[word] = candidates[0]
                continue

            word_vec = embeddings.get(word)
            if word_vec is None:
                self._cache[word] = candidates[0]
                continue

            scored = sorted(
                [(self._cosine(word_vec, embeddings[t]), t)
                 for t in candidates if t in embeddings],
                reverse=True,
            )
            if not scored:
                self._cache[word] = candidates[0]
                continue

            if len(scored) == 1 or scored[0][0] - scored[1][0] > self.AMBIGUITY_GAP:
                self._cache[word] = scored[0][1]
            else:
                llm_tasks.append((word, [t for _, t in scored[:5]]))

        # Batch LLM disambiguation
        if llm_tasks:
            results = await asyncio.gather(*[
                self._llm_pick(word, candidates)
                for word, candidates in llm_tasks
            ])
            for (word, _), title in zip(llm_tasks, results):
                self._cache[word] = title

        return {w: self._cache[w] for w in words if self._cache.get(w)}

    def _fulltext_search(self, word: str) -> list[str]:
        # Prefer: exact capitalized match > single-word titles > short titles.
        # This ordering acts as a safe fallback when embeddings are unavailable.
        capitalized = word[0].upper() + word[1:] if word else word
        with self.driver.session() as s:
            try:
                res = s.run("""
                    CALL db.index.fulltext.queryNodes('page_title_fulltext', $search_term)
                    YIELD node, score
                    WHERE node:Page
                    WITH node,
                         CASE WHEN node.title = $cap THEN 0
                              WHEN node.title STARTS WITH $cap + ' (' THEN 1
                              WHEN NOT node.title CONTAINS ' ' THEN 2
                              ELSE 3 END AS preference
                    RETURN node.title AS title
                    ORDER BY preference ASC, size(node.title) ASC
                    LIMIT $limit
                """, search_term=word, cap=capitalized, limit=self.CANDIDATE_LIMIT)
                return [r["title"] for r in res]
            except Exception as e:
                print(f"DEBUG: Full-text search failed for {word!r}: {e}", flush=True)
                return []

    @staticmethod
    def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    async def _llm_pick(self, word: str, candidates: list[str]) -> str:
        numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(candidates))
        prompt = f"""A Codenames board has the word "{word.upper()}".
Which Wikipedia article title best represents the common English meaning
a Codenames player would associate with this word?

{numbered}

Reply with ONLY the index number (0, 1, 2, ...)."""
        try:
            response = await asyncio.wait_for(
                _generate_with_fallback(self.client, prompt),
                timeout=15.0,
            )
            idx = int(response.text.strip().split()[0])
            return candidates[idx] if 0 <= idx < len(candidates) else candidates[0]
        except Exception as e:
            print(f"DEBUG: LLM disambiguation failed for {word!r}: {e}", flush=True)
            return candidates[0]


class Neo4jClueGenerator:
    def __init__(self, uri, user, password,
                 team_weight=30.0, opp_penalty=40.0, neut_penalty=15.0):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.sanitizer = None
        self.reranker = None
        self.word_mapper = None
        self.team_weight = team_weight
        self.opp_penalty = opp_penalty
        self.neut_penalty = neut_penalty

    def close(self):
        self.driver.close()

    async def generate_clues(self, my_team, opponent, neutral, assassin):
        start_total = time.time()

        # 1. Map board words to exact Wikipedia titles
        t0 = time.time()
        with self.driver.session() as session:
            # Batch title lookup
            all_words = my_team + opponent + neutral + ([assassin] if assassin else [])
            # Batch title lookup (Index-friendly ONLY)
            res = session.run("""
                UNWIND $words as w
                MATCH (p:Page) 
                WHERE p.title = w 
                   OR p.title = toUpper(substring(w, 0, 1)) + substring(w, 1)
                   OR p.title = toUpper(w)
                RETURN p.title as title, w as lower_title
                LIMIT 100
            """, words=[w.lower() for w in all_words])
            title_map = {r["lower_title"]: r["title"] for r in res}
        
        mapped_titles = list(title_map.values())
        team_set = {title_map.get(w.lower()) for w in my_team if title_map.get(w.lower())}
        opp_set = {title_map.get(w.lower()) for w in opponent if title_map.get(w.lower())}
        neut_set = {title_map.get(w.lower()) for w in neutral if title_map.get(w.lower())}
        assassin_title = title_map.get(assassin.lower(), "")
        
        print(f"DEBUG: Map Discovery took {time.time()-t0:.3f}s (Mapped {len(mapped_titles)} words)", flush=True)

        # Fallback: resolve words that didn't match via full-text search + embeddings
        if self.word_mapper:
            unmapped = [w for w in all_words if w.lower() not in title_map]
            if unmapped:
                t_fb = time.time()
                fallbacks = await self.word_mapper.resolve_batch(unmapped)
                for lower_word, title in fallbacks.items():
                    title_map[lower_word] = title
                print(f"DEBUG: Fallback mapping resolved {len(fallbacks)}/{len(unmapped)} words "
                      f"in {time.time()-t_fb:.3f}s", flush=True)
                # Rebuild sets with newly resolved words
                team_set = {title_map.get(w.lower()) for w in my_team if title_map.get(w.lower())}
                opp_set  = {title_map.get(w.lower()) for w in opponent if title_map.get(w.lower())}
                neut_set = {title_map.get(w.lower()) for w in neutral if title_map.get(w.lower())}
                assassin_title = title_map.get(assassin.lower() if assassin else "", "")

        if not team_set: return []

        # 2. Neo4j Execution
        t1 = time.time()
        with self.driver.session() as session:
            query = self._build_cypher_query(has_assassin=bool(assassin_title))
            result = session.run(query,
                                team_words=list(team_set),
                                opp_words=list(opp_set),
                                neut_words=list(neut_set),
                                assassin=assassin_title,
                                team_words_count=len(team_set),
                                team_weight=self.team_weight,
                                opp_penalty=self.opp_penalty,
                                neut_penalty=self.neut_penalty)
            
            scored_clues = [{"raw_clue": r["clue"], "score": r["score"], "targets": r["targets"], "number": len(r["targets"])} for r in result]
        
        print(f"DEBUG: Neo4j Query took {time.time()-t1:.3f}s (Found {len(scored_clues)} candidates)", flush=True)

        # 3. Semantic re-ranking
        if self.reranker and scored_clues:
            t2 = time.time()
            scored_clues = await self.reranker.rerank(scored_clues, assassin_title)
            print(f"DEBUG: Semantic re-ranking took {time.time()-t2:.3f}s", flush=True)

        # 4. LLM scoring
        if self.sanitizer and scored_clues:
            t3 = time.time()
            scored_clues = await self.sanitizer.score(scored_clues, assassin_title)
            print(f"DEBUG: LLM scoring took {time.time()-t3:.3f}s", flush=True)
        
        print(f"--- TOTAL GENERATION TIME: {time.time()-start_total:.3f}s ---", flush=True)
        return scored_clues

    def _build_cypher_query(self, has_assassin: bool) -> str:
        # Title filters and hub penalty are baked into GoodClue label and base_score
        # by precompute_graph.py — only board-specific signals are computed here.
        #
        # has_assassin controls whether the NOT EXISTS subquery is emitted at all;
        # skipping it when assassin is empty avoids a pointless full-neighborhood
        # scan on every candidate.
        assassin_clause = """
        AND NOT EXISTS {
            MATCH (candidate)--(bad:Page)
            WHERE bad.title = $assassin
        }""" if has_assassin else ""

        return f"""
        MATCH (team:Page) WHERE team.title IN $team_words AND coalesce(team.degree, 0) < 5000
        MATCH (candidate:GoodClue)--(team)
        WHERE NOT candidate.title IN $team_words
        {assassin_clause}

        WITH candidate, collect(DISTINCT team.title) AS targets
        WHERE size(targets) > 1 OR $team_words_count < 3

        WITH candidate, targets
        ORDER BY candidate.base_score DESC
        LIMIT 150

        WITH candidate, targets,
             [(candidate)--(opp:Page) WHERE opp.title IN $opp_words | opp.title] AS opp_targets,
             [(candidate)--(neut:Page) WHERE neut.title IN $neut_words | neut.title] AS neut_targets

        RETURN candidate.title AS clue, targets,
               (coalesce(candidate.base_score, 0.0)
                + (size(targets) * size(targets) * $team_weight)
                - (size(opp_targets) * $opp_penalty)
                - (size(neut_targets) * $neut_penalty)) AS score
        ORDER BY score DESC LIMIT 50
        """

class SemanticFirstGenerator:
    def __init__(self, neo4j_gen, api_key):
        self.neo4j_gen = neo4j_gen
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or "").strip()
        
        if not self.api_key:
            print("DEBUG: SemanticFirstGenerator - NO API KEY FOUND", flush=True)
            self.model = None
            return
            
        print(f"DEBUG: SemanticFirstGenerator - Key Length: {len(self.api_key)} | Prefix: {self.api_key[:10]}...", flush=True)
        from google import generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')

    async def brainstorm_candidates(self, my_team):
        if not self.model: return await self.graph_brainstorm(my_team)
        prompt = f"Brainstorm 50 high-quality Codenames clues for: {', '.join(my_team)}. Return ONLY a JSON list of objects: [{{'clue': 'WORD', 'justification': '...'}}]"
        try:
            # Increase timeout to 15s to allow for cold starts
            response = await asyncio.wait_for(
                asyncio.to_thread(self.model.generate_content, prompt),
                timeout=15.0
            )
            match = re.search(r"\[.*\]", response.text, re.DOTALL)
            if not match: return await self.graph_brainstorm(my_team)
            return json.loads(match.group(0))
        except Exception as e: 
            import traceback
            err_msg = traceback.format_exc()
            print(f"DEBUG: LLM Brainstorming failed or timed out:\n{err_msg}", flush=True)
            return await self.graph_brainstorm(my_team)

    async def graph_brainstorm(self, my_team):
        # Optimized fallback: use indexed title matches instead of slow toLower scans
        query = """
        UNWIND $team_words as t_word
        MATCH (team:Page) 
        WHERE team.title = t_word 
           OR team.title = toUpper(substring(t_word, 0, 1)) + substring(t_word, 1)
        MATCH (team)--(candidate:GoodClue)
        WHERE NOT candidate.title IN $team_words
        AND coalesce(candidate.degree, 0) > 100 AND coalesce(candidate.degree, 0) < 10000
        RETURN DISTINCT candidate.title as clue, "Graph-based association for " + team.title as justification
        LIMIT 50
        """
        try:
            with self.neo4j_gen.driver.session() as s:
                res = s.run(query, team_words=my_team)
                return [{"clue": r["clue"], "justification": r["justification"]} for r in res]
        except Exception as e:
            print(f"DEBUG: Graph fallback failed: {e}", flush=True)
            return []

    async def generate_clues(self, my_team, opponent, neutral, assassin):
        candidates = await self.brainstorm_candidates(my_team)
        if not candidates: return []
        
        clue_list = [c["clue"].lower() for c in candidates]
        print(f"DEBUG: Attempting to verify {len(clue_list)} clues against Graph", flush=True)
        # OPTIMIZED VERIFICATION: UNWIND to hit index for each candidate
        query = """
        UNWIND $clues as c_word
        MATCH (c:Page) WHERE toLower(c.title) = c_word
        WITH c,
             [(c)--(t:Page) WHERE toLower(t.title) IN $team | t.title] as team_hits,
             [(c)--(o:Page) WHERE toLower(o.title) IN $opp | o.title] as opp_hits,
             [(c)--(n:Page) WHERE toLower(n.title) IN $neut | n.title] as neut_hits,
             [(c)--(a:Page) WHERE toLower(a.title) = $assassin | a.title] as kills
        RETURN toLower(c.title) as match, c.title as actual, team_hits, opp_hits, neut_hits, size(kills) > 0 as dead
        """
        
        with self.neo4j_gen.driver.session() as s:
            res = s.run(query, clues=clue_list, team=[w.lower() for w in my_team],
                        opp=[w.lower() for w in opponent], neut=[w.lower() for w in neutral],
                        assassin=assassin.lower())
            lookup = {r["match"]: r for r in res}
            
        print(f"DEBUG: Graph verified {len(lookup)} / {len(clue_list)} candidates", flush=True)

        final = []
        for c in candidates:
            match = c["clue"].lower()
            if match not in lookup: continue
            gr = lookup[match]
            if gr["dead"]: continue
            
            t_count = len(gr["team_hits"])
            if t_count < 1: continue
            
            score = (t_count * t_count * 50.0) - (len(gr["opp_hits"]) * 40.0) - (len(gr["neut_hits"]) * 15.0)
            final.append({
                "clue": c["clue"].upper(),
                "raw_clue": gr["actual"],
                "score": score,
                "targets": gr["team_hits"],
                "number": t_count,
                "justification": c["justification"]
            })
        return sorted(final, key=lambda x: x["score"], reverse=True)

class LLMClueScorer:
    """
    Re-scores graph candidates using Gemini Flash.

    Asks the model to rate each candidate on naturalness (is it a real,
    common English word a player would guess?) and adds that rating as a
    continuous score term rather than binary include/exclude filtering.
    Falls back gracefully to returning the graph-scored list unchanged.
    """

    SCORE_WEIGHT = 8.0  # LLM rating (1-10) multiplied by this

    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)

    async def score(self, clues: list[dict], assassin: str, n: int = 20) -> list[dict]:
        if not clues:
            return clues

        candidates = clues[:n]
        clue_lines = "\n".join(
            f'{i}. {c["raw_clue"]} → targets: {", ".join(c.get("targets", []))}'
            for i, c in enumerate(candidates)
        )
        prompt = f"""You are a Codenames expert. Rate each candidate clue from 1–10 on how
naturally a human player would use it as a single-word clue for its listed target words.

Criteria:
- Is it a common English word (not a Wikipedia article title or jargon)?
- Is the connection to the targets clear and intuitive, not obscure?
- Does it avoid accidentally evoking the assassin word: "{assassin}"?

Return ONLY a JSON array, one object per candidate, in order:
[{{"idx": 0, "score": 7}}, {{"idx": 1, "score": 4}}, ...]

Candidates:
{clue_lines}"""

        try:
            response = await asyncio.wait_for(
                _generate_with_fallback(self.client, prompt),
                timeout=20.0,
            )
            match = re.search(r"\[.*?\]", response.text, re.DOTALL)
            if not match:
                return clues

            ratings = json.loads(match.group(0))
            rating_map = {r["idx"]: r["score"] for r in ratings if "idx" in r and "score" in r}

            for i, c in enumerate(candidates):
                llm_score = rating_map.get(i, 5)  # default to neutral 5 if missing
                c["llm_score"] = llm_score
                c["score"] = c["score"] + llm_score * self.SCORE_WEIGHT

            # Append any candidates beyond n unchanged, then re-sort everything
            remainder = clues[n:]
            return sorted(candidates + remainder, key=lambda x: x["score"], reverse=True)

        except Exception as e:
            print(f"DEBUG: LLM scoring failed ({e}), returning graph scores.", flush=True)
            return clues
