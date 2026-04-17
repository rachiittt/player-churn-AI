# 🤖 Agent Workflow Documentation — Milestone 2

## Overview

ChurnSense AI extends the Milestone 1 ML pipeline into a full **agentic AI system** using LangGraph. The agent reasons about player behavior, retrieves engagement strategies via RAG, and generates structured retention recommendations.

---

## Architecture

```
Player Input (Streamlit UI)
        │
        ▼
 ┌─────────────────┐
 │   ML Pipeline   │  ← Random Forest (model.pkl) + Scaler (scaler.pkl)
 │ Churn Prob → %  │
 └────────┬────────┘
          │
          ▼
 ┌──────────────────────────────────────────────┐
 │           LangGraph Agent Graph              │
 │                                              │
 │  [analyze_profile]                           │
 │       │ (conditional: error → fallback)      │
 │       ▼                                      │
 │  [retrieve_strategies]  ← FAISS Vector DB    │
 │       │ (conditional: error → fallback)      │
 │       ▼                                      │
 │  [generate_analysis]                         │
 │       │ (conditional: error → fallback)      │
 │       ▼                                      │
 │  [build_retention_plan]                      │
 │       │                                      │
 │       ▼                                      │
 │      END                                     │
 │                                              │
 │  [error_fallback] ← catches any node error   │
 └──────────────────────────────────────────────┘
          │
          ▼
 Structured Output (Streamlit UI)
 Summary | Analysis | Plan | Refs | Disclaimer
```

---

## LangGraph State Schema

```python
class AgentState(TypedDict):
    player_profile:      dict          # Raw player inputs
    churn_probability:   float         # ML model output (0–1)
    risk_level:          str           # LOW / MEDIUM / HIGH
    retrieved_strategies: list         # RAG results from FAISS
    summary:             str           # Node 1 output
    analysis:            str           # Node 3 output
    plan:                str           # Node 4 output
    references:          list          # Cited strategy snippets
    disclaimer:          str           # Ethical notice
    error:               Optional[str] # Captures any node failure
    steps_completed:     list          # Audit trail
```

---

## Node Descriptions

### Node 1: `analyze_profile`
- **Input:** player_profile dict, churn_probability float
- **Output:** risk_level (LOW/MEDIUM/HIGH), human-readable summary
- **Logic:** Computes risk tier from probability score; writes a natural-language player description

### Node 2: `retrieve_strategies`
- **Input:** player_profile, risk_level
- **Output:** retrieved_strategies (list of 4 strategy strings), references
- **Logic:**
  - Primary: FAISS similarity search with query built from profile features
  - Fallback: Keyword-based retrieval from STRATEGY_DOCS list
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (free, runs on CPU)

### Node 3: `generate_analysis`
- **Input:** player_profile, churn_probability
- **Output:** Structured analysis with risk factors and positive signals
- **Logic:** Rule-based reasoning over session frequency, playtime, achievements, level, purchase behaviour

### Node 4: `build_retention_plan`
- **Input:** retrieved_strategies, risk_level, player_profile
- **Output:** Numbered action plan with genre-specific tip and priority action
- **Logic:** Combines RAG results with genre/risk heuristics

### Node 5: `error_fallback`
- **Triggered by:** `should_fallback` conditional edge when `state["error"]` is not None
- **Output:** Safe default values for any missing fields
- **Ensures:** App never crashes — always produces a usable report

---

## RAG System

**Knowledge Base:** 20 expert-curated gaming engagement strategies  
**Vector Store:** FAISS (in-memory, CPU)  
**Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2`  
**Retrieval:** Top-4 semantic matches per query  
**Fallback:** Keyword-matching if embeddings unavailable  

---

## Error Handling Strategy

| Scenario | Handling |
|----------|----------|
| LangGraph not installed | Falls back to sequential Python pipeline (same nodes) |
| FAISS/embeddings fail | Keyword-based retrieval from in-memory list |
| Any node exception | `error_fallback` node populates defaults, app continues |
| Model file missing | Streamlit `cache_resource` error shown cleanly |

---

## Structured Output Format

Every analysis report follows this exact schema (matches rubric requirements):

| Section | Tag | Content |
|---------|-----|---------|
| Summary | `[SUMMARY]` | Player profile in natural language |
| Analysis | `[ANALYSIS]` | Risk factors + positive signals |
| Plan | `[PLAN]` | Numbered RAG-backed action items |
| References | `[REFS]` | Cited knowledge base snippets |
| Disclaimer | `[DISCLAIMER]` | Ethical & UX notice |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph 0.2.x |
| ML Model | Scikit-Learn Random Forest |
| Vector DB | FAISS (CPU) |
| Embeddings | sentence-transformers |
| UI | Streamlit |
| Hosting | Streamlit Community Cloud |
