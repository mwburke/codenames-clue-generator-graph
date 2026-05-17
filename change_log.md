# Project Experiment & Change Log

This document tracks all major experiments, performance optimizations, and technical resolutions for the Codenames Graph/LLM Clue Generator.

---

## 1. Performance & Latency Experiments

### **Exp 1: Map Discovery Optimization**
*   **Initial Approach**: Used `MATCH (p:Page) WHERE toLower(p.title) IN $words` to map the board to Wikipedia titles.
*   **Problem**: Forced a **Full Table Scan** on 13 million nodes. Latency was **~24.2 seconds**.
*   **Result/Outcome**: Switched to an `UNWIND` pattern with direct equality checks (`p.title = w OR p.title = Capitalized`).
*   **Impact**: Latency dropped to **<0.1 seconds**.

### **Exp 2: LLM Timeout Management**
*   **Initial Approach**: Standard asynchronous calls to the Google Generative AI SDK.
*   **Problem**: In cases of API 404s or network slowdowns, the UI would hang for 30+ seconds before falling back to the graph.
*   **Result/Outcome**: Implemented `asyncio.wait_for` with a 15-second strict timeout.
*   **Impact**: Ensured the "Unbreakable Fallback" triggers quickly, keeping the UX responsive.

---

## 2. Failures, Gotchas & Lessons Learned

### **Failure: The "Plugin Assumption" Crash**
*   **What happened**: Attempted to use `apoc.text.capitalize()` for high-speed title matching.
*   **Why it failed**: The Neo4j environment did not have the APOC plugin installed.
*   **Outcome**: Server crashed with a `CypherSyntaxError`.
*   **Lesson**: Always use **Standard Cypher** unless a plugin is explicitly confirmed. Standardized on `toUpper(substring(...))` for case-handling.

### **Failure: The "Infinity Mapping" Hang**
*   **What happened**: Tried to "catch all" cases by adding `OR toLower(p.title) = w` back into the high-speed query.
*   **Why it failed**: In Neo4j, using `OR` with a function-on-node-property often disables the index for the **entire** query.
*   **Outcome**: Mapping time jumped from milliseconds to **3000+ seconds** (nearly 1 hour).
*   **Lesson**: Never use `toLower()` or `OR` on large node sets if an index hit is required. Stick to explicit `UNWIND` matches.

### **Failure: The "KeyError" Field Mismatch**
*   **What happened**: Renamed the database output field from `candidate` to `clue` (to align with the Hybrid engine) but forgot to update the Python dictionary comprehension.
*   **Why it failed**: Python was looking for `r["candidate"]` in a result set that only had `r["clue"]`.
*   **Outcome**: Generation crashed with a `KeyError: 'candidate'`.
*   **Lesson**: Database schema changes and Python object mapping must be updated in **one single atomic change**.

---

## 3. Git Version Control & Deployment

### **Exp 6: Repository Initialization & Huge Data Protection**
*   **Initial State**: Code was unversioned, mixed with 11GB of Wikipedia compressed SQL dumps (`data/enwiki-...`).
*   **Action**: Created a comprehensive `.gitignore` filtering out virtual environments (`venv/`), dependencies (`node_modules/`), all logs, and major database dumps while retaining small JSON index assets.
*   **Result**: Safely pushed clean codebase to [git@github.com:mwburke/codenames-clue-generator-graph.git](git@github.com:mwburke/codenames-clue-generator-graph.git).
