# Agentic Marketing AI - Technical Architecture Guide

## Executive Summary (Pyramid Principle: Start with the Answer)

```
THE BIG PICTURE
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   "How do we make AI actually useful for marketing teams?"              │
│                                                                         │
│   ANSWER: Build THREE specialized AI systems, each solving              │
│           a different problem with the right architecture               │
│                                                                         │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│   │ BUDGET          │  │ KNOWLEDGE       │  │ TEAM OF         │         │
│   │ OPTIMIZER       │  │ ASSISTANT       │  │ AI EXPERTS      │         │
│   │                 │  │                 │  │                 │         │
│   │ "How should we  │  │ "What happened  │  │ "Analyze this   │         │
│   │  spend our      │  │  in our last    │  │  from every     │         │
│   │  money?"        │  │  campaign?"     │  │  angle"         │         │
│   │                 │  │                 │  │                 │         │
│   │ LangGraph       │  │ LangChain RAG   │  │ Google ADK      │         │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# PART 1: Budget Optimizer (LangGraph)

## 1.1 The Problem We're Solving

```
THE BUSINESS PROBLEM
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   A marketing manager has €1 million to spend across 5 channels:        │
│                                                                         │
│   • Search (Google Ads)                                                 │
│   • Social (Facebook, Instagram)                                        │
│   • Display (Banner ads)                                                │
│   • Video (YouTube)                                                     │
│   • Email                                                               │
│                                                                         │
│   CONSTRAINTS:                                                          │
│   • Search must be 15-40% of budget (company policy)                    │
│   • Social must be 10-30% (brand guidelines)                            │
│   • Each channel has historical performance data                        │
│                                                                         │
│   QUESTION: "What's the optimal split to maximize return?"              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1.2 Why We Chose LangGraph (The Design Decision)

```
WHY NOT JUST USE CHATGPT DIRECTLY?
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   NAIVE APPROACH: "Hey ChatGPT, allocate my €1M budget"                 │
│                                                                         │
│   PROBLEMS:                                                             │
│   ❌ No validation - might suggest 150% allocation                      │
│   ❌ No retry logic - if it fails, you start over                       │
│   ❌ No audit trail - can't explain decisions to CFO                    │
│   ❌ No constraints enforcement - ignores business rules                │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   LANGGRAPH APPROACH: Break it into steps with checkpoints              │
│                                                                         │
│   BENEFITS:                                                             │
│   ✅ Each step validates before moving forward                          │
│   ✅ Automatic retry if optimization fails                              │
│   ✅ Full audit trail of every decision                                 │
│   ✅ Constraints enforced at multiple stages                            │
│   ✅ Fallback to safe defaults if all else fails                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

ANALOGY: Think of it like a car assembly line
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ChatGPT alone = One person trying to build entire car                 │
│                   (might forget wheels, no quality checks)              │
│                                                                         │
│   LangGraph = Assembly line with stations                               │
│               Station 1: Check parts are correct                        │
│               Station 2: Analyze what type of car to build              │
│               Station 3: Assemble the car                               │
│               Station 4: Quality inspection                             │
│               Station 5: Final approval or send back                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1.3 The Workflow Explained (Step by Step)

```
THE 6-STEP ASSEMBLY LINE
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   STEP 1: VALIDATE INPUT                                                │
│   ────────────────────                                                  │
│   Purpose: "Is the request even valid?"                                 │
│                                                                         │
│   What it checks:                                                       │
│   • Is budget > 0? (Can't allocate negative money)                      │
│   • Do constraints add up? (min values can't exceed 100%)               │
│   • Does historical data exist and make sense?                          │
│                                                                         │
│   WHY THIS STEP EXISTS:                                                 │
│   Garbage in = Garbage out. If someone asks to allocate                 │
│   €1M with constraints requiring 150% minimum, we catch                 │
│   it HERE, not after wasting compute on optimization.                   │
│                                                                         │
│   LAYMAN ANALOGY: Checking your ingredients before cooking.             │
│   No point starting a recipe if you're missing flour.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   STEP 2: ANALYZE PERFORMANCE                                           │
│   ──────────────────────────                                            │
│   Purpose: "What does history tell us?"                                 │
│                                                                         │
│   What it calculates:                                                   │
│   • ROAS per channel (Search returned €5 for every €1 spent)            │
│   • Trends (Is Social improving or declining?)                          │
│   • Seasonality (Video does better in Q4)                               │
│   • Channel rankings (Best to worst performers)                         │
│                                                                         │
│   WHY THIS STEP EXISTS:                                                 │
│   You can't optimize without understanding current state.               │
│   This step transforms raw CSV data into actionable insights.           │
│                                                                         │
│   LAYMAN ANALOGY: A doctor reviewing your medical history               │
│   before prescribing treatment. Past data informs decisions.            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   STEP 3: OPTIMIZE ALLOCATION                                           │
│   ──────────────────────────                                            │
│   Purpose: "Calculate the best split"                                   │
│                                                                         │
│   What it does:                                                         │
│   • Runs optimization algorithm (maximize expected ROAS)                │
│   • Respects all constraints (min/max per channel)                      │
│   • Generates rationale for each allocation                             │
│                                                                         │
│   Example output:                                                       │
│   • Search: €320,000 (32%) - "Highest ROAS historically"                │
│   • Social: €250,000 (25%) - "Strong Q4 performance"                    │
│   • Display: €130,000 (13%) - "At minimum threshold"                    │
│   • Video: €200,000 (20%) - "Brand awareness needs"                     │
│   • Email: €100,000 (10%) - "Highest efficiency, limited scale"         │
│                                                                         │
│   WHY THIS STEP EXISTS:                                                 │
│   This is the "brain" of the system. Uses math + AI to find             │
│   the optimal solution within business constraints.                     │
│                                                                         │
│   LAYMAN ANALOGY: A financial advisor creating your portfolio.          │
│   Balances growth potential with risk limits you've set.                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   STEP 4: VALIDATE OUTPUT                                               │
│   ────────────────────────                                              │
│   Purpose: "Did the optimizer respect the rules?"                       │
│                                                                         │
│   What it checks:                                                       │
│   • Sum of allocations = Total budget (not 99% or 101%)                 │
│   • Each channel within min/max bounds                                  │
│   • No negative allocations                                             │
│   • Expected metrics are realistic                                      │
│                                                                         │
│   WHY THIS STEP EXISTS:                                                 │
│   Even good optimizers can produce bad results. AI can                  │
│   "hallucinate" or make math errors. This is the safety net.            │
│                                                                         │
│   LAYMAN ANALOGY: Spell-check before sending an important email.        │
│   You wrote it, but you still verify before sending.                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   STEP 5: THE DECISION POINT (Conditional Routing)                      │
│   ─────────────────────────────────────────────────                     │
│   Purpose: "What do we do if validation fails?"                         │
│                                                                         │
│   THREE POSSIBLE PATHS:                                                 │
│                                                                         │
│   PATH A: VALID ────────────────────────────► Go to EXPLAIN             │
│           "All constraints satisfied"         Generate final report     │
│                                                                         │
│   PATH B: INVALID + RETRIES LEFT ───────────► Go back to OPTIMIZE       │
│           "Search is 45%, max is 40%"         Try again (up to 3x)      │
│           "Iteration 1 of 3"                                            │
│                                                                         │
│   PATH C: INVALID + NO RETRIES LEFT ────────► Go to FALLBACK            │
│           "Still failing after 3 tries"       Use safe defaults         │
│                                                                         │
│   WHY THIS STEP EXISTS:                                                 │
│   Real-world systems need graceful degradation. If optimization         │
│   keeps failing, we don't crash - we use a safe backup plan.            │
│                                                                         │
│   LAYMAN ANALOGY: GPS recalculating route.                              │
│   "Turn left" - missed it - "Recalculating..."                          │
│   After 3 wrong turns: "Here's a simpler route instead"                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│                               │   │                               │
│   STEP 6A: EXPLAIN            │   │   STEP 6B: FALLBACK           │
│   ────────────────            │   │   ─────────────────           │
│   Purpose: "Justify to CFO"   │   │   Purpose: "Safe defaults"    │
│                               │   │                               │
│   Generates:                  │   │   Uses:                       │
│   • Plain English summary     │   │   • Equal split across        │
│   • Why each channel got $X   │   │     channels                  │
│   • Expected outcomes         │   │   • Or last known good        │
│   • Confidence score          │   │     allocation                │
│   • Caveats and warnings      │   │   • With warning message      │
│                               │   │                               │
└───────────────────────────────┘   └───────────────────────────────┘
```

## 1.4 The State Object (Why TypedDict?)

```
WHAT IS "STATE" AND WHY DO WE NEED IT?
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PROBLEM: Each step needs to know what happened in previous steps      │
│                                                                         │
│   EXAMPLE:                                                              │
│   • Step 3 (Optimize) needs the analysis from Step 2                    │
│   • Step 4 (Validate) needs the allocation from Step 3                  │
│   • Step 5 (Router) needs to know how many retries we've done           │
│                                                                         │
│   SOLUTION: A shared "state" object that travels through the pipeline   │
│                                                                         │
│   ANALOGY: A patient's medical chart                                    │
│   ─────────────────────────────────                                     │
│   • Receptionist writes: "Patient arrived, complained of headache"      │
│   • Nurse adds: "Blood pressure 120/80, temperature 98.6°F"             │
│   • Doctor adds: "Diagnosed migraine, prescribed medication"            │
│   • Pharmacist reads entire chart to dispense correct medicine          │
│                                                                         │
│   Each person ADDS to the chart, and everyone can READ everything.      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

THE STATE OBJECT STRUCTURE
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   OptimizationState = {                                                 │
│                                                                         │
│       # SECTION 1: INPUTS (Set at start, never changed)                 │
│       "total_budget": 1000000,                                          │
│       "channels": ["search", "social", "display", "video", "email"],    │
│       "constraints": {...},                                             │
│                                                                         │
│       # SECTION 2: ANALYSIS (Filled by Step 2)                          │
│       "performance_analysis": {...},                                    │
│       "channel_rankings": ["email", "search", "social", ...],           │
│       "seasonality_factors": {"Q4": 1.2, "Q1": 0.8},                     │
│                                                                         │
│       # SECTION 3: OPTIMIZATION (Filled by Step 3, maybe updated)       │
│       "proposed_allocation": {"search": 320000, ...},                   │
│       "iteration_count": 1,  # Tracks retry attempts                    │
│                                                                         │
│       # SECTION 4: VALIDATION (Filled by Step 4)                        │
│       "validation_passed": True,                                        │
│       "validation_errors": [],                                          │
│                                                                         │
│       # SECTION 5: OUTPUTS (Filled by Step 6)                           │
│       "final_allocation": {...},                                        │
│       "explanation": "Search gets 32% because...",                      │
│       "confidence_score": 0.82                                          │
│   }                                                                     │
│                                                                         │
│   WHY TypedDict?                                                        │
│   • Type safety: Python knows what fields exist                         │
│   • IDE autocomplete: Developers see available fields                   │
│   • Documentation: Self-documenting code                                │
│   • Validation: Catches typos like "totla_budget"                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# PART 2: Marketing RAG System (LangChain)

## 2.1 The Problem We're Solving

```
THE BUSINESS PROBLEM
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   A marketing analyst has 500 pages of documents:                       │
│   • Campaign reports (PDF)                                              │
│   • Performance data (CSV)                                              │
│   • Strategy documents (Word/PDF)                                       │
│   • Meeting notes (Text files)                                          │
│                                                                         │
│   QUESTIONS THEY ASK:                                                   │
│   • "What was our ROAS on social in Q4?"                                │
│   • "Why did display performance drop in November?"                     │
│   • "What did we recommend in the last quarterly review?"               │
│                                                                         │
│   CURRENT APPROACH: Ctrl+F through dozens of files                      │
│   TIME WASTED: Hours per question                                       │
│                                                                         │
│   DESIRED STATE: Ask in plain English, get answer with source           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.2 Why We Chose RAG (The Design Decision)

```
WHAT IS RAG AND WHY NOT JUST USE GPT-4?
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   OPTION 1: Fine-tune GPT-4 on your documents                           │
│   ────────────────────────────────────────────                          │
│   ❌ Expensive ($10,000+ for custom training)                           │
│   ❌ Outdated immediately (training takes weeks)                        │
│   ❌ Can't update easily when new reports come in                       │
│   ❌ Model might "hallucinate" facts from training                      │
│                                                                         │
│   OPTION 2: Paste entire document into ChatGPT                          │
│   ─────────────────────────────────────────────                         │
│   ❌ Context limit (can't paste 500 pages)                              │
│   ❌ Expensive (charged per token)                                      │
│   ❌ Slow (processes everything every time)                             │
│   ❌ No source citation                                                 │
│                                                                         │
│   OPTION 3: RAG (Retrieval-Augmented Generation) ✅                     │
│   ───────────────────────────────────────────────                       │
│   ✅ Only retrieve RELEVANT chunks (fast + cheap)                       │
│   ✅ Easy to update (just add new documents)                            │
│   ✅ Source citations built-in                                          │
│   ✅ Grounded answers (can't make up facts)                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

RAG EXPLAINED IN LAYMAN TERMS
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ANALOGY: A research librarian                                         │
│                                                                         │
│   You ask: "What was Shakespeare's view on jealousy?"                   │
│                                                                         │
│   BAD LIBRARIAN (GPT alone):                                            │
│   • Answers from memory                                                 │
│   • Might misremember or make things up                                 │
│   • Can't show you the source                                           │
│                                                                         │
│   GOOD LIBRARIAN (RAG):                                                 │
│   • Goes to the Shakespeare section                                     │
│   • Pulls out Othello (the relevant book)                               │
│   • Opens to Act 3, Scene 3 (the relevant page)                         │
│   • Reads the passage to you                                            │
│   • Shows you exactly where it came from                                │
│                                                                         │
│   RAG = Retrieve first, THEN answer with evidence                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.3 The Two Pipelines Explained

```
PIPELINE 1: INGESTION (One-time setup + updates)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PURPOSE: Turn documents into searchable knowledge                     │
│                                                                         │
│   STEP 1: LOAD DOCUMENTS                                                │
│   ───────────────────────                                               │
│   Input: Raw files (PDF, CSV, TXT)                                      │
│   Output: Text content with metadata                                    │
│                                                                         │
│   "Q4_Report.pdf" → "Our Q4 social campaigns achieved ROAS of 3.2x..."  │
│                      metadata: {source: "Q4_Report.pdf", page: 12}      │
│                                                                         │
│   WHY: Computers can't read PDFs directly. We extract the text.         │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│   STEP 2: CHUNK THE TEXT                                                │
│   ──────────────────────                                                │
│   Input: Full document text (might be 50 pages)                         │
│   Output: Small chunks (1000 characters each)                           │
│                                                                         │
│   WHY WE CHUNK:                                                         │
│   • LLMs have token limits (can't send 50 pages)                        │
│   • Smaller = more precise retrieval                                    │
│   • Overlap ensures we don't cut sentences in half                      │
│                                                                         │
│   ANALOGY: Cutting a book into index cards                              │
│   Each card has one idea, but you note which page it's from             │
│                                                                         │
│   Settings explained:                                                   │
│   • chunk_size=1000: Each chunk is ~1000 characters                     │
│   • chunk_overlap=200: Chunks share 200 characters with neighbors       │
│                        (so sentences aren't split awkwardly)            │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│   STEP 3: CREATE EMBEDDINGS                                             │
│   ─────────────────────────                                             │
│   Input: Text chunk ("Social campaigns achieved ROAS of 3.2x...")       │
│   Output: Vector [0.023, -0.156, 0.089, ... 1536 numbers]               │
│                                                                         │
│   WHAT ARE EMBEDDINGS?                                                  │
│   • A way to represent MEANING as numbers                               │
│   • Similar meanings → similar numbers                                  │
│   • "dog" and "puppy" have similar vectors                              │
│   • "dog" and "airplane" have different vectors                         │
│                                                                         │
│   WHY 1536 NUMBERS?                                                     │
│   • OpenAI's text-embedding-3-small produces 1536-dimensional vectors   │
│   • More dimensions = more nuanced meaning capture                      │
│   • Trade-off: accuracy vs. storage/speed                               │
│                                                                         │
│   ANALOGY: GPS coordinates for meaning                                  │
│   Just like (40.7128, -74.0060) represents New York City,               │
│   [0.023, -0.156, ...] represents the meaning of a text chunk           │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│   STEP 4: STORE IN VECTOR DATABASE                                      │
│   ────────────────────────────────                                      │
│   Input: Vectors + original text + metadata                             │
│   Output: Searchable database (ChromaDB)                                │
│                                                                         │
│   ChromaDB stores:                                                      │
│   {                                                                     │
│       vector: [0.023, -0.156, ...],                                     │
│       text: "Social campaigns achieved ROAS of 3.2x...",                │
│       metadata: {source: "Q4_Report.pdf", page: 12, channel: "social"}  │
│   }                                                                     │
│                                                                         │
│   WHY CHROMA (not regular database)?                                    │
│   • Regular SQL: "Find rows where channel = 'social'"                   │
│   • Vector DB: "Find chunks with SIMILAR MEANING to my question"        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

PIPELINE 2: QUERY (Every time user asks a question)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   USER QUESTION: "What was our social ROAS in Q4?"                      │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│   STEP 1: EMBED THE QUESTION                                            │
│   ──────────────────────────                                            │
│   "What was our social ROAS in Q4?" → [0.045, -0.123, ...]              │
│                                                                         │
│   Same process as documents: convert meaning to numbers                 │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│   STEP 2: FIND SIMILAR CHUNKS (Retrieval)                               │
│   ───────────────────────────────────────                               │
│   Compare question vector to all stored vectors                         │
│   Return top 5 most similar chunks                                      │
│                                                                         │
│   Results:                                                              │
│   1. "Social campaigns achieved ROAS of 3.2x..." (similarity: 0.92)     │
│   2. "Q4 social spend was €250,000..." (similarity: 0.87)               │
│   3. "Facebook outperformed Instagram..." (similarity: 0.81)            │
│   ...                                                                   │
│                                                                         │
│   MMR EXPLAINED (Maximum Marginal Relevance):                           │
│   • Problem: Top 5 results might all say the same thing                 │
│   • Solution: MMR balances relevance WITH diversity                     │
│   • lambda_mult=0.7 means: 70% relevance, 30% diversity                 │
│                                                                         │
│   ANALOGY: Google search results                                        │
│   You don't want 10 results from the same website.                      │
│   You want diverse sources that are all relevant.                       │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│   STEP 3: BUILD PROMPT WITH CONTEXT                                     │
│   ─────────────────────────────────                                     │
│                                                                         │
│   System: "You are a marketing analytics expert.                        │
│            Answer based ONLY on the provided context.                   │
│            Cite your sources using [Source: filename] format."          │
│                                                                         │
│   Context:                                                              │
│   [Source 1: Q4_Report.pdf, Page 12]                                    │
│   Social campaigns achieved ROAS of 3.2x in Q4...                       │
│                                                                         │
│   [Source 2: Budget_Summary.csv]                                        │
│   Q4 social spend was €250,000...                                       │
│                                                                         │
│   User: "What was our social ROAS in Q4?"                               │
│                                                                         │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                         │
│   STEP 4: LLM GENERATES ANSWER                                          │
│   ────────────────────────────                                          │
│                                                                         │
│   GPT-4 Output:                                                         │
│   "Our social campaigns achieved a ROAS of 3.2x in Q4, based on         │
│    a total spend of €250,000. This was above our benchmark of 2.8x.     │
│    [Source: Q4_Report.pdf, Page 12]"                                    │
│                                                                         │
│   KEY INSIGHT: The LLM can ONLY use information from the context.       │
│   It cannot hallucinate facts because it's grounded in documents.       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2.4 Auto-Metadata Extraction (Smart Tagging)

```
WHY EXTRACT METADATA?
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PROBLEM: User asks "Show me only social media reports"                │
│                                                                         │
│   WITHOUT METADATA:                                                     │
│   • Search entire database                                              │
│   • Retrieve irrelevant email/search documents                          │
│   • Waste tokens and get noisy results                                  │
│                                                                         │
│   WITH METADATA:                                                        │
│   • Filter: WHERE channel = "social"                                    │
│   • Only search social-related chunks                                   │
│   • Faster, more accurate results                                       │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   AUTOMATIC DETECTION (No manual tagging needed):                       │
│                                                                         │
│   Document text: "Our Facebook and Instagram campaigns..."              │
│                                                                         │
│   System detects:                                                       │
│   • "Facebook" → channel: social                                        │
│   • "Instagram" → channel: social                                       │
│   • "ROAS" mentioned → metrics: ["roas"]                                │
│   • "performance" in text → report_type: performance                    │
│                                                                         │
│   Stored metadata:                                                      │
│   {                                                                     │
│       channels: ["social"],                                             │
│       metrics_mentioned: ["roas"],                                      │
│       report_type: "performance"                                        │
│   }                                                                     │
│                                                                         │
│   ANALOGY: Gmail auto-categorizing into Primary, Social, Promotions     │
│   The system reads content and tags it automatically.                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# PART 3: Multi-Agent Analyst (Google ADK)

## 3.1 The Problem We're Solving

```
THE BUSINESS PROBLEM
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   CMO asks: "Analyze our Q4 performance and recommend Q1 strategy"      │
│                                                                         │
│   THIS QUESTION REQUIRES EXPERTISE IN:                                  │
│   • Performance metrics (ROAS, CPA, trends)                             │
│   • Audience analysis (Who's buying? Who should we target?)             │
│   • Competitive landscape (How do we compare?)                          │
│                                                                         │
│   IN A REAL COMPANY, THIS INVOLVES:                                     │
│   • Performance analyst pulls the numbers                               │
│   • Audience researcher identifies segments                             │
│   • Strategy lead synthesizes into recommendations                      │
│                                                                         │
│   CHALLENGE: How do we replicate this with AI?                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3.2 Why Multi-Agent (The Design Decision)

```
WHY NOT USE ONE SUPER-AGENT?
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   OPTION 1: One agent that knows everything                             │
│   ──────────────────────────────────────────                            │
│                                                                         │
│   System prompt: "You are an expert in performance analysis,            │
│                   audience segmentation, competitive intelligence,      │
│                   creative optimization, budget allocation,             │
│                   media planning, brand strategy..."                    │
│                                                                         │
│   PROBLEMS:                                                             │
│   ❌ Prompt becomes enormous (token limits)                             │
│   ❌ Jack of all trades, master of none                                 │
│   ❌ Can't parallelize (one agent = sequential)                         │
│   ❌ Hard to update one area without affecting others                   │
│   ❌ Tool overload (50+ tools in one agent = confusion)                 │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   OPTION 2: Specialized agents with coordinator ✅                      │
│   ─────────────────────────────────────────────                         │
│                                                                         │
│   COORDINATOR: "Who should answer this question?"                       │
│       │                                                                 │
│       ├── PERFORMANCE ANALYST: Deep expertise in metrics                │
│       │   • Focused system prompt                                       │
│       │   • 3-5 specialized tools                                       │
│       │                                                                 │
│       ├── AUDIENCE ANALYST: Deep expertise in segments                  │
│       │   • Different system prompt                                     │
│       │   • Different specialized tools                                 │
│       │                                                                 │
│       └── COMPETITOR ANALYST: Deep expertise in market                  │
│           • Another focused prompt                                      │
│           • Market-specific tools                                       │
│                                                                         │
│   BENEFITS:                                                             │
│   ✅ Each agent is an expert in their domain                            │
│   ✅ Can run agents in parallel (faster)                                │
│   ✅ Easy to add new specialists                                        │
│   ✅ Each agent has manageable tool set                                 │
│   ✅ Mirrors how real marketing teams work                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

ANALOGY: Hospital vs. GP
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ONE SUPER-AGENT = General Practitioner                                │
│   • Knows a little about everything                                     │
│   • Good for simple questions                                           │
│   • Struggles with complex cases                                        │
│                                                                         │
│   MULTI-AGENT = Hospital with specialists                               │
│   • Cardiologist for heart issues                                       │
│   • Neurologist for brain issues                                        │
│   • Coordinator (triage nurse) routes to right specialist               │
│   • Specialists consult each other for complex cases                    │
│                                                                         │
│   Complex question? You want the hospital.                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3.3 The Agent Architecture Explained

```
THE COORDINATOR PATTERN
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   USER QUERY: "How are social campaigns performing and                  │
│                who should we target next quarter?"                      │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════       │
│                                                                         │
│   STEP 1: COORDINATOR ANALYZES QUERY                                    │
│   ──────────────────────────────────                                    │
│                                                                         │
│   Query contains:                                                       │
│   • "performing" → needs Performance Analyst                            │
│   • "target" → needs Audience Analyst                                   │
│   • No competitor keywords → skip Competitor Analyst                    │
│                                                                         │
│   Decision: Route to [performance_analyst, audience_analyst]            │
│                                                                         │
│   WHY KEYWORD-BASED ROUTING?                                            │
│   • Fast (no LLM call needed)                                           │
│   • Deterministic (same query = same routing)                           │
│   • Cost-effective (saves API calls)                                    │
│   • In production: Could use LLM for ambiguous queries                  │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════       │
│                                                                         │
│   STEP 2: PARALLEL AGENT EXECUTION                                      │
│   ────────────────────────────────                                      │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    PARALLEL EXECUTION                           │   │
│   │                                                                 │   │
│   │   ┌─────────────────────┐     ┌─────────────────────┐           │   │
│   │   │ PERFORMANCE ANALYST │     │  AUDIENCE ANALYST   │           │   │
│   │   │                     │     │                     │           │   │
│   │   │ Thinks about:       │     │ Thinks about:       │           │   │
│   │   │ • Social ROAS       │     │ • Top segments      │           │   │
│   │   │ • Trend direction   │     │ • Targeting gaps    │           │   │
│   │   │ • Channel health    │     │ • LTV by segment    │           │   │
│   │   │                     │     │                     │           │   │
│   │   │ Uses tools:         │     │ Uses tools:         │           │   │
│   │   │ • get_channel_      │     │ • get_segment_      │           │   │
│   │   │   performance()     │     │   performance()     │           │   │
│   │   │ • analyze_trends()  │     │ • get_top_segments()│           │   │
│   │   │                     │     │                     │           │   │
│   │   │ OUTPUT:             │     │ OUTPUT:             │           │   │
│   │   │ "Social ROAS is     │     │ "Top segment is     │           │   │
│   │   │  3.2x, up 12% MoM.  │     │  health-conscious   │           │   │
│   │   │  Instagram leading, │     │  millennials at     │           │   │
│   │   │  Facebook declining"│     │  4.2x ROAS.         │           │   │
│   │   │                     │     │  Opportunity: Gen Z"│           │   │
│   │   └─────────────────────┘     └─────────────────────┘           │   │
│   │              │                           │                      │   │
│   │              └───────────┬───────────────┘                      │   │
│   │                          ▼                                      │   │
│   │                   BOTH COMPLETE                                 │   │
│   │                                                                 │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   WHY PARALLEL?                                                         │
│   • Faster: 2 agents at once vs. one after another                      │
│   • Agents don't depend on each other's output                          │
│   • In production: Async execution for scalability                      │
│                                                                         │
│   ═══════════════════════════════════════════════════════════════       │
│                                                                         │
│   STEP 3: COORDINATOR SYNTHESIZES                                       │
│   ───────────────────────────────                                       │
│                                                                         │
│   Coordinator receives both responses and creates unified analysis:     │
│                                                                         │
│   "## Analysis Summary                                                  │
│                                                                         │
│    **Performance:** Social achieving 3.2x ROAS (+12% MoM).              │
│    Instagram outperforming Facebook.                                    │
│                                                                         │
│    **Audience:** Health-conscious millennials driving results           │
│    (4.2x ROAS). Untapped opportunity in Gen Z.                          │
│                                                                         │
│    **Recommendations:**                                                 │
│    1. Shift budget from Facebook to Instagram                           │
│    2. Expand Gen Z targeting on TikTok                                  │
│    3. Double down on health-conscious messaging"                        │
│                                                                         │
│   WHY SYNTHESIZE?                                                       │
│   • Raw agent outputs might contradict                                  │
│   • User wants ONE cohesive answer, not 3 separate reports              │
│   • Coordinator adds strategic layer on top of analysis                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 3.4 What Are "Tools" and Why Do Agents Need Them?

```
THE TOOL CONCEPT
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PROBLEM: LLMs can only process text. They can't:                      │
│   • Query databases                                                     │
│   • Call APIs                                                           │
│   • Do real math                                                        │
│   • Access current data                                                 │
│                                                                         │
│   SOLUTION: Give them "tools" (functions they can call)                 │
│                                                                         │
│   ANALOGY: A chef with kitchen tools                                    │
│   ─────────────────────────────────                                     │
│   • Chef (LLM) knows recipes (how to reason)                            │
│   • But needs tools: knife, stove, mixer                                │
│   • Tool = "I need to chop onions" → calls knife function               │
│   • Tool returns result → chef continues cooking                        │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   EXAMPLE: PERFORMANCE ANALYST TOOLS                                    │
│                                                                         │
│   Tool 1: get_channel_performance                                       │
│   ────────────────────────────────                                      │
│   What it does: Fetches metrics for a channel                           │
│                                                                         │
│   Agent thinks: "I need social media performance data"                  │
│   Agent calls: get_channel_performance(channel="social", period="Q4")   │
│   Tool returns: {                                                       │
│       "spend": 250000,                                                  │
│       "revenue": 800000,                                                │
│       "roas": 3.2,                                                      │
│       "conversions": 15000                                              │
│   }                                                                     │
│   Agent uses this data to form its analysis                             │
│                                                                         │
│   Tool 2: analyze_trends                                                │
│   ──────────────────────                                                │
│   What it does: Calculates trend direction                              │
│                                                                         │
│   Agent calls: analyze_trends(metric="roas", channel="social")          │
│   Tool returns: {                                                       │
│       "direction": "improving",                                         │
│       "change_pct": 12,                                                 │
│       "forecast_next_quarter": 3.5                                      │
│   }                                                                     │
│                                                                         │
│   WHY TOOLS MATTER:                                                     │
│   • Grounds the agent in REAL data                                      │
│   • Prevents hallucination                                              │
│   • Makes agents actually useful (not just chatbots)                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# PART 4: Design Principles Across All Projects

## 4.1 The Pyramid Principle (Communication Structure)

```
WHAT IS THE PYRAMID PRINCIPLE?
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   Created by Barbara Minto at McKinsey                                  │
│                                                                         │
│   CORE IDEA: Start with the answer, then support with details           │
│                                                                         │
│   BAD COMMUNICATION (Bottom-up):                                        │
│   "We analyzed the data... looked at trends... considered factors...    │
│    evaluated options... and therefore recommend X"                      │
│   (Audience falls asleep before reaching the point)                     │
│                                                                         │
│   GOOD COMMUNICATION (Pyramid):                                         │
│   "Recommendation: Do X.                                                │
│    Reason 1: Data shows Y                                               │
│    Reason 2: Trends indicate Z                                          │
│    Reason 3: Risk analysis supports this"                               │
│   (Audience gets the point immediately, can dig into details if needed) │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

THE PYRAMID STRUCTURE
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                        ┌─────────────────┐                              │
│                        │   MAIN POINT    │  ← Start here                │
│                        │   (Answer)      │     "Allocate 32% to Search" │
│                        └────────┬────────┘                              │
│                                 │                                       │
│              ┌──────────────────┼──────────────────┐                    │
│              │                  │                  │                    │
│              ▼                  ▼                  ▼                    │
│      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│      │  SUPPORTING  │   │  SUPPORTING  │   │  SUPPORTING  │             │
│      │   POINT 1    │   │   POINT 2    │   │   POINT 3    │             │
│      │              │   │              │   │              │             │
│      │ "Highest     │   │ "Stable      │   │ "Within      │             │
│      │  ROAS at 5x" │   │  performance"│   │  constraints"│             │
│      └──────┬───────┘   └──────┬───────┘   └──────┬───────┘             │
│             │                  │                  │                     │
│             ▼                  ▼                  ▼                     │
│      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐             │
│      │   DETAILS    │   │   DETAILS    │   │   DETAILS    │             │
│      │              │   │              │   │              │             │
│      │ Historical   │   │ MoM trends   │   │ Min 15%,     │             │
│      │ data table   │   │ chart        │   │ Max 40%      │             │
│      └──────────────┘   └──────────────┘   └──────────────┘             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

HOW OUR PROJECTS APPLY THE PYRAMID:
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   BUDGET OPTIMIZER OUTPUT:                                              │
│                                                                         │
│   Level 1 (Answer):                                                     │
│   "Recommended allocation: Search 32%, Social 25%, ..."                 │
│                                                                         │
│   Level 2 (Supporting):                                                 │
│   • "Search gets highest share due to 5.1x ROAS"                        │
│   • "Social increased for Q4 seasonality"                               │
│   • "Display at minimum - lower marginal returns"                       │
│                                                                         │
│   Level 3 (Details):                                                    │
│   • Confidence score: 0.82                                              │
│   • Expected total ROAS: 4.2x                                           │
│   • Full data tables available                                          │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   RAG RESPONSE:                                                         │
│                                                                         │
│   Level 1 (Answer):                                                     │
│   "Q4 social ROAS was 3.2x"                                             │
│                                                                         │
│   Level 2 (Supporting):                                                 │
│   • "Based on €250,000 spend"                                           │
│   • "Above 2.8x benchmark"                                              │
│                                                                         │
│   Level 3 (Details):                                                    │
│   • [Source: Q4_Report.pdf, Page 12]                                    │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   MULTI-AGENT SYNTHESIS:                                                │
│                                                                         │
│   Level 1 (Answer):                                                     │
│   "Shift budget to Instagram and target Gen Z"                          │
│                                                                         │
│   Level 2 (Supporting):                                                 │
│   • Performance: Instagram up 12%, Facebook declining                   │
│   • Audience: Gen Z underserved, 5.1x ROAS potential                    │
│                                                                         │
│   Level 3 (Details):                                                    │
│   • Full agent responses available                                      │
│   • Data tables and sources                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4.2 Why These Three Architectures?

```
MATCHING PROBLEM TO ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PROBLEM TYPE 1: Sequential decisions with validation                  │
│   ─────────────────────────────────────────────────────                 │
│   Example: Budget allocation                                            │
│   Pattern: STATE MACHINE (LangGraph)                                    │
│   Why: Need checkpoints, retries, audit trail                           │
│                                                                         │
│   Characteristics:                                                      │
│   • Steps must happen in order                                          │
│   • Each step might fail                                                │
│   • Need to track state across steps                                    │
│   • Want automatic retry logic                                          │
│                                                                         │
│   Real-world analogy: Assembly line, loan approval process              │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PROBLEM TYPE 2: Finding information in documents                      │
│   ─────────────────────────────────────────────────                     │
│   Example: "What was Q4 ROAS?"                                          │
│   Pattern: RAG PIPELINE (LangChain)                                     │
│   Why: Need to ground answers in source documents                       │
│                                                                         │
│   Characteristics:                                                      │
│   • Answer exists somewhere in documents                                │
│   • Need source citations                                               │
│   • Can't hallucinate facts                                             │
│   • Knowledge base changes over time                                    │
│                                                                         │
│   Real-world analogy: Research librarian, legal discovery               │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PROBLEM TYPE 3: Multi-perspective analysis                            │
│   ──────────────────────────────────────────                            │
│   Example: "Full marketing review"                                      │
│   Pattern: MULTI-AGENT (Google ADK)                                     │
│   Why: Need diverse expertise synthesized                               │
│                                                                         │
│   Characteristics:                                                      │
│   • Question spans multiple domains                                     │
│   • Experts can work in parallel                                        │
│   • Need synthesis of perspectives                                      │
│   • Different tools for different domains                               │
│                                                                         │
│   Real-world analogy: Hospital specialists, consulting team             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4.3 Common Design Patterns

```
PATTERN 1: FAIL GRACEFULLY
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PRINCIPLE: Never crash. Always return something useful.               │
│                                                                         │
│   Budget Optimizer:                                                     │
│   • If optimization fails 3x → Use safe fallback allocation             │
│   • User gets SOMETHING, even if not optimal                            │
│                                                                         │
│   RAG System:                                                           │
│   • If no relevant documents found → "I couldn't find information..."   │
│   • Better than making up an answer                                     │
│                                                                         │
│   Multi-Agent:                                                          │
│   • If one agent fails → Still return other agents' responses           │
│   • Partial answer > no answer                                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

PATTERN 2: CONFIDENCE SCORES
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PRINCIPLE: Always tell the user how sure you are.                     │
│                                                                         │
│   Why?                                                                  │
│   • Humans need to calibrate trust                                      │
│   • 0.95 confidence → Act on it                                         │
│   • 0.60 confidence → Verify before acting                              │
│                                                                         │
│   How we calculate:                                                     │
│   • Budget Optimizer: Based on data quality + constraint satisfaction   │
│   • RAG: Based on retrieval similarity scores                           │
│   • Multi-Agent: Average of specialist confidences                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

PATTERN 3: SOURCE ATTRIBUTION
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PRINCIPLE: Always cite where information came from.                   │
│                                                                         │
│   Why?                                                                  │
│   • Builds trust ("I can verify this")                                  │
│   • Enables debugging ("Why did it say that?")                          │
│   • Compliance/audit requirements                                       │
│                                                                         │
│   Implementation:                                                       │
│   • RAG: [Source: filename.pdf, Page X]                                 │
│   • Multi-Agent: data_used: ["get_channel_performance", ...]            │
│   • Budget Optimizer: allocation_rationale per channel                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

PATTERN 4: STRUCTURED OUTPUT
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   PRINCIPLE: Return structured data, not free-form text.                │
│                                                                         │
│   BAD: "I recommend spending about 32% on search and maybe              │
│         25% on social, something like that..."                          │
│                                                                         │
│   GOOD: {                                                               │
│       "allocation": {"search": 320000, "social": 250000},               │
│       "confidence": 0.82,                                               │
│       "explanation": "..."                                              │
│   }                                                                     │
│                                                                         │
│   Why?                                                                  │
│   • Downstream systems can use the data                                 │
│   • No parsing errors                                                   │
│   • Consistent format every time                                        │
│   • Easy to store/analyze                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# PART 5: Quick Reference Comparison

```
┌──────────────────┬────────────────────┬────────────────────┬────────────────────┐
│     Aspect       │  Budget Optimizer  │   Marketing RAG    │  Multi-Agent ADK   │
├──────────────────┼────────────────────┼────────────────────┼────────────────────┤
│                  │                    │                    │                    │
│ PROBLEM SOLVED   │ "How should we     │ "What does our     │ "Analyze this from │
│                  │  spend money?"     │  data say?"        │  every angle"      │
│                  │                    │                    │                    │
├──────────────────┼────────────────────┼────────────────────┼────────────────────┤
│                  │                    │                    │                    │
│ REAL-WORLD       │ Assembly line      │ Research librarian │ Hospital           │
│ ANALOGY          │ with QA checks     │ finding sources    │ specialists        │
│                  │                    │                    │                    │
├──────────────────┼────────────────────┼────────────────────┼────────────────────┤
│                  │                    │                    │                    │
│ KEY FEATURE      │ Conditional        │ Source citations   │ Parallel expert    │
│                  │ routing + retries  │ + grounding        │ consultation       │
│                  │                    │                    │                    │
├──────────────────┼────────────────────┼────────────────────┼────────────────────┤
│                  │                    │                    │                    │
│ WHEN TO USE      │ Sequential         │ Question-answering │ Multi-domain       │
│                  │ decision process   │ over documents     │ analysis           │
│                  │                    │                    │                    │
├──────────────────┼────────────────────┼────────────────────┼────────────────────┤
│                  │                    │                    │                    │
│ FAILURE MODE     │ Fallback to safe   │ "I don't know"     │ Return partial     │
│                  │ defaults           │ with honesty       │ results            │
│                  │                    │                    │                    │
├──────────────────┼────────────────────┼────────────────────┼────────────────────┤
│                  │                    │                    │                    │
│ FRAMEWORK        │ LangGraph          │ LangChain          │ Google ADK         │
│                  │ (State machines)   │ (LCEL chains)      │ (Agent framework)  │
│                  │                    │                    │                    │
└──────────────────┴────────────────────┴────────────────────┴────────────────────┘
```

---

# PART 6: Glossary for Non-Technical Readers

```
COMMON TERMS EXPLAINED
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│ LLM (Large Language Model)                                              │
│ → AI that understands and generates text (like ChatGPT)                 │
│                                                                         │
│ Token                                                                   │
│ → A piece of text (roughly 4 characters or 0.75 words)                  │
│ → Models charge per token and have limits                               │
│                                                                         │
│ Embedding                                                               │
│ → Converting text to numbers that capture meaning                       │
│ → Similar texts have similar numbers                                    │
│                                                                         │
│ Vector Database                                                         │
│ → Database that finds things by similarity, not exact match             │
│ → "Find documents LIKE this" instead of "Find documents WITH this"      │
│                                                                         │
│ State Machine                                                           │
│ → A system that moves through defined steps                             │
│ → Each step depends on previous steps                                   │
│                                                                         │
│ RAG (Retrieval-Augmented Generation)                                    │
│ → Find relevant info first, then generate answer                        │
│ → Prevents AI from making things up                                     │
│                                                                         │
│ Agent                                                                   │
│ → AI that can take actions, not just chat                               │
│ → Can use tools, make decisions, complete tasks                         │
│                                                                         │
│ Tool (in AI context)                                                    │
│ → A function the AI can call to do something                            │
│ → Like "get_data()" or "send_email()"                                   │
│                                                                         │
│ ROAS (Return on Ad Spend)                                               │
│ → Revenue generated per dollar spent on ads                             │
│ → ROAS of 3x = $3 revenue for every $1 spent                            │
│                                                                         │
│ CPA (Cost per Acquisition)                                              │
│ → How much it costs to get one customer                                 │
│ → Lower is better                                                       │
│                                                                         │
│ Hallucination                                                           │
│ → When AI makes up facts that aren't true                               │
│ → RAG helps prevent this                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Author

**Jaiguru Thevar**
Head of Data Science, VML (WPP Group)

[LinkedIn](https://linkedin.com/in/jaiguru) | [GitHub](https://github.com/jaigurum) | mjguru@gmail.com
