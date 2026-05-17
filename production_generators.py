import time
import json
import asyncio
import re
import os
from neo4j import GraphDatabase

class Neo4jClueGenerator:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.sanitizer = None
        self.penalize_hubs = True
        self.hub_penalty_factor = 25.0

    def close(self):
        self.driver.close()

    async def generate_clues(self, my_team, opponent, neutral, assassin, max_depth=1):
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
        
        if not team_set: return []

        # 2. Neo4j Execution
        t1 = time.time()
        with self.driver.session() as session:
            query = self._build_cypher_query(max_depth)
            result = session.run(query, 
                                team_words=list(team_set), 
                                opp_words=list(opp_set), 
                                neut_words=list(neut_set), 
                                assassin=assassin_title, 
                                team_words_count=len(team_set),
                                penalize_hubs=self.penalize_hubs,
                                hub_penalty_factor=self.hub_penalty_factor)
            
            scored_clues = [{"raw_clue": r["clue"], "score": r["score"], "targets": r["targets"], "number": len(r["targets"])} for r in result]
        
        print(f"DEBUG: Neo4j Query took {time.time()-t1:.3f}s (Found {len(scored_clues)} candidates)", flush=True)

        # 3. AI Sanitization
        if self.sanitizer and scored_clues:
            t2 = time.time()
            all_board_words = my_team + opponent + neutral + ([assassin] if assassin else [])
            scored_clues = await self.sanitizer.sanitize_top_n(scored_clues, all_board_words, n=15)
            print(f"DEBUG: AI Sanitization took {time.time()-t2:.3f}s", flush=True)
        
        print(f"--- TOTAL GENERATION TIME: {time.time()-start_total:.3f}s ---", flush=True)
        return scored_clues

    def _build_cypher_query(self, max_depth: int) -> str:
        return f"""
        MATCH (team:Page) WHERE team.title IN $team_words AND coalesce(team.degree, 0) < 10000
        MATCH (candidate:GoodClue)--(team)
        
        WHERE NOT candidate.title IN $team_words 
        AND NOT candidate.title CONTAINS '(identifier)'
        AND NOT candidate.title CONTAINS 'ISSN'
        AND NOT candidate.title CONTAINS 'ISBN'
        AND NOT candidate.title STARTS WITH 'Category:'
        AND NOT candidate.title STARTS WITH 'List of'
        AND NOT candidate.title STARTS WITH 'Template:'
        AND NOT candidate.title STARTS WITH 'Wikipedia:'
        AND NOT candidate.title STARTS WITH 'Portal:'
        AND NOT candidate.title ENDS WITH 'oides'
        AND NOT candidate.title ENDS WITH 'aceae'
        AND NOT candidate.title ENDS WITH 'idae'
        AND NOT candidate.title CONTAINS 'List'
        AND NOT candidate.title CONTAINS 'list'
        AND NOT candidate.title CONTAINS 'Slang'
        AND NOT candidate.title CONTAINS 'slang'
        AND NOT candidate.title CONTAINS 'Proto-'
        AND NOT candidate.title CONTAINS 'Vocabulary'
        AND NOT candidate.title CONTAINS 'Glossary'
        AND NOT candidate.title CONTAINS 'Etymology'
        AND NOT candidate.title CONTAINS 'Dictionary'
        AND NOT candidate.title CONTAINS 'Dialect'
        AND NOT candidate.title CONTAINS 'Language'
        AND size(candidate.title) > 3
        AND coalesce(candidate.degree, 0) > 200
        
        AND NOT EXISTS {{
            MATCH (candidate)--(bad:Page)
            WHERE bad.title = $assassin
        }}
        
        WITH candidate, collect(DISTINCT team.title) as targets
        WHERE size(targets) > 1 OR $team_words_count < 3
        
        WITH candidate, targets, 
             [(candidate)--(opp:Page) WHERE opp.title IN $opp_words | opp.title] as opp_targets,
             [(candidate)--(neut:Page) WHERE neut.title IN $neut_words | neut.title] as neut_targets
        
        WITH candidate, targets, opp_targets, neut_targets,
             (size(targets) * size(targets) * 30.0) as signal_score,
             (size(opp_targets) * 40.0) as risk_penalty,
             (size(neut_targets) * 15.0) as neut_penalty,
             CASE WHEN $penalize_hubs THEN log10(coalesce(candidate.degree, 1.0)) * $hub_penalty_factor ELSE 0.0 END as hub_penalty,
             rand() * 5.0 as jitter
             
        RETURN candidate.title as clue, targets, (signal_score - risk_penalty - neut_penalty - hub_penalty + jitter) as score
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
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')

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

    async def generate_clues(self, my_team, opponent, neutral, assassin, max_depth=1):
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

class LLMClueSanitizer:
    def __init__(self, api_key):
        from google import generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-flash-latest')

    async def sanitize_top_n(self, clues, all_words, n=15):
        if not self.model or not clues: return clues[:n]
        
        # Take top 20 candidates for the AI to review
        candidates = clues[:20]
        prompt = f"""
        Review these Codenames clues for the board: {', '.join(all_words)}.
        For each clue, decide if it is a GOOD, creative, and legal clue.
        Clues: {json.dumps([c['raw_clue'] for c in candidates])}
        
        Return ONLY a JSON list of indices for the top {n} clues that are most creative and safe.
        Example: [0, 2, 5, ...]
        """
        try:
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            match = re.search(r"\[.*\]", response.text, re.DOTALL)
            if not match: return clues[:n]
            indices = json.loads(match.group(0))
            
            final = []
            for idx in indices:
                if 0 <= idx < len(candidates):
                    c = candidates[idx]
                    final.append({
                        "clue": c["raw_clue"].upper(),
                        "score": c["score"],
                        "targets": c["targets"],
                        "number": c["number"],
                        "justification": f"AI Verified: {c['raw_clue']} connects to {', '.join(c['targets'])}"
                    })
            return final if final else clues[:n]
        except Exception as e:
            print(f"DEBUG: AI Sanitization failed ({e}), returning raw clues.", flush=True)
            return clues[:n]
