import os
import json
import random
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from production_generators import Neo4jClueGenerator, LLMClueScorer, GeminiEmbeddingReranker, WikipediaWordMapper
from hybrid_generator import HybridClueGenerator, SemanticClueGenerator
from codenames_generator import CodenamesClueGenerator
from salt_ingestion import SaltDataIngestor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BoardState(BaseModel):
    my_team: list = []
    myTeam: list = [] # Handle camelCase from frontend
    opponent: list = []
    neutral: list = []
    assassin: str = ""
    method: str = "graph"
    api_key: str = ""

# Keep global but initialize lazily to avoid startup deadlock
_neo4j_gen_instance = None

def get_neo4j_generator(api_key: str):
    global _neo4j_gen_instance
    if _neo4j_gen_instance is None:
        print("--- INITIALIZING NEO4J DRIVER (FIRST RUN) ---")
        _neo4j_gen_instance = Neo4jClueGenerator("bolt://localhost:7687", "neo4j", "password")
    
    if api_key:
        if _neo4j_gen_instance.word_mapper is None:
            print("--- ATTACHING WIKIPEDIA WORD MAPPER ---")
            _neo4j_gen_instance.word_mapper = WikipediaWordMapper(_neo4j_gen_instance.driver, api_key)
        if _neo4j_gen_instance.reranker is None:
            print("--- ATTACHING SEMANTIC RERANKER TO GRAPH ENGINE ---")
            _neo4j_gen_instance.reranker = GeminiEmbeddingReranker(api_key)
        if _neo4j_gen_instance.sanitizer is None:
            print("--- ATTACHING LLM SCORER TO GRAPH ENGINE ---")
            _neo4j_gen_instance.sanitizer = LLMClueScorer(api_key)
    return _neo4j_gen_instance

generators = {
    "graph": CodenamesClueGenerator(),
    "semantic": SemanticClueGenerator(),
}

@app.get("/api/load-salt-board")
async def load_salt_board():
    try:
        ingestor = SaltDataIngestor()
        scenarios = ingestor.get_scenarios()
        if not scenarios: return {"error": "No scenarios found"}
        s = random.choice(scenarios)
        
        # Split neutrals to create an opponent team for UI colors
        all_neutrals = s["neutral"]
        opp = all_neutrals[:len(all_neutrals)//2]
        neut = all_neutrals[len(all_neutrals)//2:]
        
        # Normalize everything to lowercase
        team = [w.lower() for w in s["team"]]
        opp = [w.lower() for w in opp]
        neut = [w.lower() for w in neut]
        assassin = s["assassin"].lower() if s["assassin"] else "air"
        
        # Standard Codenames Distribution: 9 Team, 8 Opponent, 1 Assassin, 7 Neutral
        all_board_words = set(team + opp + neut + [assassin])
        padding_pool = ["apple", "band", "car", "dog", "eagle", "fire", "glass", "heart", "ice", "jump", 
                        "kite", "lion", "moon", "ninja", "octopus", "penguin", "queen", "robot", "star", 
                        "train", "umbrella", "volcano", "water", "xylophone", "yacht", "zebra", "alien", 
                        "bear", "cat", "duck", "elephant", "frog", "ghost", "horse", "igloo", "jacket", 
                        "kangaroo", "lemon", "monkey", "nut", "owl", "pig", "quilt", "rabbit", "snake", 
                        "turtle", "uniform", "vampire", "whale", "yard"]
        
        # Fast Deterministic Padding
        extra_words = [w for w in padding_pool if w not in all_board_words]
        
        # Distribute extra words
        while len(team) < 9 and extra_words:
            w = extra_words.pop(0); team.append(w)
        while len(opp) < 8 and extra_words:
            w = extra_words.pop(0); opp.append(w)
        while len(neut) < 7 and extra_words:
            w = extra_words.pop(0); neut.append(w)
        
        return {
            "my_team": team,
            "opponent": opp,
            "neutral": neut,
            "assassin": assassin,
            "human_clue": s["human_clues"][0]["word"].lower() if s["human_clues"] else "",
            "human_count": len(s["team"])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/api/generate-clues")
async def generate_clues(board: BoardState):
    print(f"--- GENERATE REQUEST RECEIVED (Method: {board.method}) ---")
    
    # Merge team names from different possible keys
    team = board.my_team if board.my_team else board.myTeam
    
    # Default to neo4j if not specified
    method = board.method if board.method else "neo4j"
    
    if method in ["neo4j", "hybrid", "graph", "semantic_first", "semantic-first"]:
        # If method is "graph", we now route it to the high-performance Neo4j engine by default
        generator = get_neo4j_generator(board.api_key)
        if method == "hybrid":
            generator = HybridClueGenerator(graph_gen=generator)
        elif method == "semantic_first" or method == "semantic-first":
            from production_generators import SemanticFirstGenerator
            generator = SemanticFirstGenerator(neo4j_gen=generator, api_key=board.api_key)
    else:
        generator = generators.get(method)
        
    if not generator: raise HTTPException(status_code=400, detail=f"Method '{method}' unsupported.")
    
    try:
        import inspect
        if inspect.iscoroutinefunction(generator.generate_clues):
            clues = await generator.generate_clues(
                my_team=team, opponent=board.opponent,
                neutral=board.neutral, assassin=board.assassin
            )
        else:
            clues = generator.generate_clues(
                my_team=team, opponent=board.opponent,
                neutral=board.neutral, assassin=board.assassin
            )
        
        # Unified safety and normalization loop
        final_clues = []
        all_board_words = team + board.opponent + board.neutral + ([board.assassin] if board.assassin else [])
        all_board_words_up = [bw.upper() for bw in all_board_words]
        
        # Regex for valid human-like words (allows Latin accents, blocks technical noise)
        import re
        human_word_pattern = re.compile(r'^[A-ZÀ-ÖØ-Þ][a-zß-ÿ]+$|^[A-ZÀ-ÖØ-Þ]+$')

        for c in clues:
            raw = c.get("raw_clue", c.get("candidate", ""))
            
            # 1. Extraction (First word if no LLM)
            clue_word = c.get("sanitized_word", raw.split()[0].upper())
            clue_word = clue_word.rstrip('.') # Kill initials noise
            
            # 2. Safety Check: Length & Pattern
            if len(clue_word) < 3: continue # Restore 3-letter words (ART, OIL), block 1-2 letters
            
            # 3. Safety Check: Illegal board overlap (Substring)
            is_illegal = False
            for bw_up in all_board_words_up:
                if bw_up in clue_word or clue_word in bw_up:
                    is_illegal = True
                    break
            if is_illegal: continue
            
            # 4. Success: Format for UI
            c["clue"] = clue_word if board.api_key else f"{clue_word} (auto)"
            c["number"] = len(c["targets"])
            c["targets"] = [t.lower() for t in c["targets"]]
            final_clues.append(c)

        print(f"--- GENERATE SUCCESS: Found {len(final_clues)} legal clues ---")
        return {"clues": final_clues[:15]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
