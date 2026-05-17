"""
One-time (or periodic) script to pre-compute node properties on GoodClue nodes.

Bakes the title-quality filters into the GoodClue label and stores base_score
as a node property, so runtime Cypher queries do indexed lookups + simple
arithmetic instead of repeating the same 13+ string conditions on every request.

Re-run whenever filtering criteria or scoring weights change.

Usage:
    python precompute_graph.py
    python precompute_graph.py --hub-penalty 30 --uri bolt://localhost:7687
"""
import argparse
import time
from neo4j import GraphDatabase


BAD_TITLE_CONDITIONS = [
    "n.title CONTAINS '(identifier)'",
    "n.title CONTAINS 'ISSN'",
    "n.title CONTAINS 'ISBN'",
    "n.title STARTS WITH 'Category:'",
    "n.title STARTS WITH 'List of'",
    "n.title STARTS WITH 'Template:'",
    "n.title STARTS WITH 'Wikipedia:'",
    "n.title STARTS WITH 'Portal:'",
    "n.title ENDS WITH 'oides'",
    "n.title ENDS WITH 'aceae'",
    "n.title ENDS WITH 'idae'",
    "n.title CONTAINS 'List'",
    "n.title CONTAINS 'Slang'",
    "n.title CONTAINS 'Proto-'",
    "n.title CONTAINS 'Vocabulary'",
    "n.title CONTAINS 'Glossary'",
    "n.title CONTAINS 'Etymology'",
    "n.title CONTAINS 'Dictionary'",
    "n.title CONTAINS 'Dialect'",
    "n.title CONTAINS 'Language'",
]


class GraphPrecomputer:
    def __init__(self, uri, user, password, hub_penalty_factor=25.0,
                 min_degree=50, max_degree=10000, force=False):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.hub_penalty_factor = hub_penalty_factor
        self.min_degree = min_degree
        self.max_degree = max_degree
        self.force = force

    def close(self):
        self.driver.close()

    def run(self):
        print("=== Graph Precomputation ===")
        t_start = time.time()
        self._phase0_assign_goodclue_label()
        self._phase1_strip_bad_nodes()
        self._phase2_compute_base_scores()
        self._phase3_create_indexes()
        print(f"=== Done in {time.time()-t_start:.1f}s ===")

    def _phase0_assign_goodclue_label(self):
        """
        Assign GoodClue label to Page nodes that pass degree criteria.
        Skipped if GoodClue nodes already exist unless --force is passed.
        """
        print(f"Phase 0: Assigning GoodClue label (degree {self.min_degree}–{self.max_degree})...")
        t0 = time.time()
        with self.driver.session() as s:
            count = s.run("MATCH (n:GoodClue) RETURN count(n) AS c").single()["c"]
            if count > 0 and not self.force:
                print(f"  {count:,} GoodClue nodes already exist. Use --force to reset.")
                return
            if count > 0:
                print(f"  --force: removing {count:,} existing GoodClue labels...")
                s.run("""
                    MATCH (n:GoodClue)
                    CALL { WITH n REMOVE n:GoodClue } IN TRANSACTIONS OF 50000 ROWS
                """)
            s.run("""
                MATCH (n:Page)
                WHERE coalesce(n.degree, 0) >= $min_degree
                  AND coalesce(n.degree, 0) < $max_degree
                  AND size(n.title) > 3
                CALL { WITH n SET n:GoodClue } IN TRANSACTIONS OF 50000 ROWS
            """, min_degree=self.min_degree, max_degree=self.max_degree)
            added = s.run("MATCH (n:GoodClue) RETURN count(n) AS c").single()["c"]
        print(f"  Labeled {added:,} nodes as GoodClue. ({time.time()-t0:.1f}s)")

    def _phase1_strip_bad_nodes(self):
        """Remove GoodClue label from nodes matching known bad title patterns."""
        print("Phase 1: Stripping bad nodes from GoodClue label...")
        t0 = time.time()
        total_removed = 0
        with self.driver.session() as s:
            for condition in BAD_TITLE_CONDITIONS:
                before = s.run("MATCH (n:GoodClue) RETURN count(n) AS c").single()["c"]
                s.run(f"""
                    MATCH (n:GoodClue)
                    WHERE {condition}
                    CALL {{ WITH n REMOVE n:GoodClue }} IN TRANSACTIONS OF 10000 ROWS
                """)
                after = s.run("MATCH (n:GoodClue) RETURN count(n) AS c").single()["c"]
                removed = before - after
                if removed > 0:
                    print(f"  -{removed:,}: {condition}")
                    total_removed += removed
        print(f"  Removed {total_removed:,} bad nodes total. ({time.time()-t0:.1f}s)")

    def _phase2_compute_base_scores(self):
        """
        Store base_score on all GoodClue nodes.

        Formula:
          - degree penalty: -log10(degree) * hub_penalty_factor
          - single-word bonus: +10 for clues with no spaces (single words are
            better Codenames clues than multi-word phrases)

        Board-specific signals (target count, opponent/neutral hits) are still
        computed at query time since they depend on the current board state.
        """
        print(f"Phase 2: Computing base_score (hub_penalty={self.hub_penalty_factor})...")
        t0 = time.time()
        with self.driver.session() as s:
            s.run("""
                MATCH (n:GoodClue)
                CALL {
                    WITH n
                    SET n.base_score =
                        -(log10(toFloat(coalesce(n.degree, 1))) * $penalty)
                        + CASE WHEN NOT n.title CONTAINS ' ' THEN 10.0 ELSE 0.0 END
                } IN TRANSACTIONS OF 50000 ROWS
            """, penalty=self.hub_penalty_factor)
            scored = s.run(
                "MATCH (n:GoodClue) WHERE n.base_score IS NOT NULL RETURN count(n) AS c"
            ).single()["c"]
        print(f"  Scored {scored:,} nodes. ({time.time()-t0:.1f}s)")

    def _phase3_create_indexes(self):
        """Create indexes used by the runtime query."""
        print("Phase 3: Creating indexes...")
        indexes = [
            "CREATE INDEX goodclue_title IF NOT EXISTS FOR (n:GoodClue) ON (n.title)",
            "CREATE INDEX goodclue_base_score IF NOT EXISTS FOR (n:GoodClue) ON (n.base_score)",
            "CREATE INDEX page_title IF NOT EXISTS FOR (n:Page) ON (n.title)",
            # Full-text index used by WikipediaWordMapper for fallback word resolution
            "CREATE FULLTEXT INDEX page_title_fulltext IF NOT EXISTS FOR (n:Page) ON EACH [n.title]",
        ]
        with self.driver.session() as s:
            for stmt in indexes:
                try:
                    s.run(stmt)
                    print(f"  OK: {stmt[:70]}")
                except Exception as e:
                    print(f"  Skip: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="password")
    parser.add_argument("--hub-penalty", type=float, default=25.0,
                        help="Coefficient for log10(degree) penalty stored in base_score")
    parser.add_argument("--min-degree", type=int, default=50,
                        help="Minimum page degree to qualify as GoodClue (default: 50)")
    parser.add_argument("--max-degree", type=int, default=10000,
                        help="Maximum page degree to qualify as GoodClue (default: 10000)")
    parser.add_argument("--force", action="store_true",
                        help="Reset and recompute all GoodClue labels even if they exist")
    args = parser.parse_args()

    pc = GraphPrecomputer(args.uri, args.user, args.password,
                          hub_penalty_factor=args.hub_penalty,
                          min_degree=args.min_degree,
                          max_degree=args.max_degree,
                          force=args.force)
    try:
        pc.run()
    finally:
        pc.close()


if __name__ == "__main__":
    main()
