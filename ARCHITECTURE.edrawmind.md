# Agentic Marketing AI - Technical Architecture

## Overview

```
Agentic Marketing AI Portfolio
├── Budget Optimizer (LangGraph)
│   └── State Machine Architecture
├── Marketing RAG (LangChain)
│   └── Retrieval-Augmented Generation
└── Multi-Agent Analyst (Google ADK)
    └── Coordinator + Specialist Pattern
```

---

## 1. Budget Optimizer (LangGraph)

### 1.1 Core Concept

```
Purpose: Automated marketing budget allocation across channels
Framework: LangGraph (State Machine / Directed Graph)
Pattern: Multi-node workflow with conditional routing
```

### 1.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH STATE MACHINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐                                                   │
│   │   START     │                                                   │
│   │   Input     │                                                   │
│   └──────┬──────┘                                                   │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────┐    ┌──────────────────────────────────────┐       │
│   │  VALIDATE   │    │ • Check budget > 0                   │       │
│   │   INPUT     │───►│ • Validate channel constraints       │       │
│   │   NODE      │    │ • Load historical data               │       │
│   └──────┬──────┘    │ • Calculate data quality score       │       │
│          │           └──────────────────────────────────────┘       │
│          ▼                                                          │
│   ┌─────────────┐    ┌──────────────────────────────────────┐       │
│   │  ANALYZE    │    │ • Calculate per-channel ROAS         │       │
│   │ PERFORMANCE │───►│ • Detect seasonality factors         │       │
│   │   NODE      │    │ • Rank channels by efficiency        │       │
│   └──────┬──────┘    │ • Assess market conditions           │       │
│          │           └──────────────────────────────────────┘       │
│          ▼                                                          │
│   ┌─────────────┐    ┌──────────────────────────────────────┐       │
│   │  OPTIMIZE   │    │ • Convex optimization algorithm      │       │
│   │ ALLOCATION  │◄──►│ • Apply channel constraints          │       │
│   │   NODE      │    │ • Maximize ROAS objective            │       │
│   └──────┬──────┘    │ • Generate allocation rationale      │       │
│          │           └──────────────────────────────────────┘       │
│          ▼                                                          │
│   ┌─────────────┐    ┌──────────────────────────────────────┐       │
│   │  VALIDATE   │    │ • Sum(allocations) == total_budget   │       │
│   │   OUTPUT    │───►│ • All constraints satisfied          │       │
│   │   NODE      │    │ • iteration_count < max_iterations   │       │
│   └──────┬──────┘    └──────────────────────────────────────┘       │
│          │                                                          │
│   ┌──────┴──────────────────────────────────┐                       │
│   │         CONDITIONAL ROUTER              │                       │
│   └────┬─────────────┬───────────────┬──────┘                       │
│        │             │               │                              │
│   [Valid]      [Invalid,       [Max Retries]                        │
│        │        Retry<3]             │                              │
│        ▼             │               ▼                              │
│   ┌─────────┐        │         ┌──────────┐                         │
│   │ EXPLAIN │   ┌────┘         │ FALLBACK │                         │
│   │  NODE   │   │              │   NODE   │                         │
│   └────┬────┘   │              └────┬─────┘                         │
│        │        │                   │                               │
│        ▼        │                   │                               │
│   ┌─────────────┴───────────────────┘                               │
│   │                 END                                             │
│   └─────────────────────────────────────────────────────────────────┘
```

### 1.3 State Schema (TypedDict)

```
OptimizationState
├── INPUTS
│   ├── total_budget: float              # Total € to allocate
│   ├── channels: List[str]              # ["search", "social", "display", "video", "email"]
│   ├── constraints: Dict[str, Dict]     # Per-channel min/max bounds
│   ├── objective: str                   # "maximize_roas" | "minimize_cpa" | "balanced"
│   └── historical_data_path: str        # Path to CSV with performance data
│
├── ANALYSIS RESULTS
│   ├── performance_analysis: Dict       # Channel-level metrics
│   ├── channel_rankings: List[str]      # Ordered by ROAS efficiency
│   ├── seasonality_factors: Dict        # Time-based adjustments
│   └── market_conditions: str           # "stable" | "volatile" | "growth"
│
├── OPTIMIZATION STATE
│   ├── proposed_allocation: Dict        # Current iteration's allocation
│   ├── allocation_rationale: Dict       # Reasoning per channel
│   ├── iteration_count: int             # Retry counter (max 3)
│   └── optimization_history: List       # All iteration attempts
│
├── VALIDATION STATE
│   ├── validation_passed: bool          # Output validation result
│   ├── validation_errors: List[str]     # Specific error messages
│   └── constraint_violations: List[str] # Which constraints failed
│
└── OUTPUTS
    ├── final_allocation: Dict[str, float]   # € per channel
    ├── expected_metrics: Dict[str, float]   # Projected ROAS, revenue
    ├── explanation: str                     # Human-readable rationale
    ├── confidence_score: float              # 0.0-1.0
    └── warnings: List[str]                  # Data quality issues, etc.
```

### 1.4 Key Implementation Details

```
File: budget_optimizer/graph.py
├── create_budget_optimizer_graph()
│   ├── StateGraph(OptimizationState)
│   ├── add_node("validate_input", ...)
│   ├── add_node("analyze", ...)
│   ├── add_node("optimize", ...)
│   ├── add_node("validate_output", ...)
│   ├── add_node("explain", ...)
│   ├── add_node("fallback", ...)
│   ├── add_edge() - Linear flow
│   ├── add_conditional_edges() - Router
│   └── workflow.compile()
│
└── BudgetOptimizerGraph (Class)
    ├── __init__(use_llm, llm_provider)
    ├── run(request) → Dict[str, Any]
    ├── arun(request) → async execution
    └── stream(request) → real-time updates
```

### 1.5 Conditional Routing Logic

```python
def should_reoptimize(state: OptimizationState) -> str:
    """
    Decision function for conditional edges

    Returns:
        "optimize"  → Retry optimization (errors + retries < 3)
        "fallback"  → Use safe defaults (max retries exceeded)
        "explain"   → Success, generate explanation
    """
    if state["validation_errors"] and state["iteration_count"] < 3:
        return "optimize"
    elif state["validation_errors"]:
        return "fallback"
    else:
        return "explain"
```

### 1.6 Sample Input/Output

```
INPUT:
{
    "total_budget": 1000000,
    "channels": ["search", "social", "display", "video", "email"],
    "constraints": {
        "search": {"min": 0.15, "max": 0.40},
        "social": {"min": 0.10, "max": 0.30},
        "display": {"min": 0.05, "max": 0.20},
        "video": {"min": 0.10, "max": 0.25},
        "email": {"min": 0.05, "max": 0.15}
    },
    "objective": "maximize_roas"
}

OUTPUT:
{
    "final_allocation": {
        "search": 320000,
        "social": 250000,
        "display": 130000,
        "video": 200000,
        "email": 100000
    },
    "expected_metrics": {
        "total_roas": 4.2,
        "expected_revenue": 4200000,
        "confidence_interval": [3.8, 4.6]
    },
    "confidence_score": 0.82,
    "iterations": 1
}
```

---

## 2. Marketing RAG System (LangChain)

### 2.1 Core Concept

```
Purpose: Natural language Q&A over marketing documents
Framework: LangChain (LCEL - LangChain Expression Language)
Pattern: Retrieval-Augmented Generation with citations
```

### 2.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│   │   DOCUMENTS  │     │   LOADERS    │     │   CHUNKING   │        │
│   │              │     │              │     │              │        │
│   │  • PDF       │────►│ PyPDFLoader  │────►│ Recursive    │        │
│   │  • CSV       │     │ CSVLoader    │     │ Character    │        │
│   │  • TXT/MD    │     │ TextLoader   │     │ TextSplitter │        │
│   └──────────────┘     └──────────────┘     └──────┬───────┘        │
│                                                     │               │
│                                              ┌──────▼───────┐       │
│                                              │   METADATA   │       │
│                                              │  EXTRACTION  │       │
│                                              │              │       │
│                                              │ • Channels   │       │
│                                              │ • Metrics    │       │
│                                              │ • Report type│       │
│                                              └──────┬───────┘       │
│                                                     │               │
│   ┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐        │
│   │   CHROMA DB  │◄────│   VECTORS    │◄────│  EMBEDDING   │        │
│   │              │     │              │     │              │        │
│   │  Persistent  │     │  1536-dim    │     │ text-embed-  │        │
│   │  Vector Store│     │  Vectors     │     │ ding-3-small │        │
│   └──────────────┘     └──────────────┘     └──────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE (LCEL)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐                                                   │
│   │   QUESTION  │                                                   │
│   │             │                                                   │
│   │  "What was  │                                                   │
│   │   Q4 ROAS?" │                                                   │
│   └──────┬──────┘                                                   │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────┐     ┌──────────────────────────────────────┐      │
│   │  RETRIEVER  │     │  MMR (Maximum Marginal Relevance)    │      │
│   │             │────►│  • k=5 (top 5 results)               │      │
│   │   .mmr()    │     │  • fetch_k=20 (candidate pool)       │      │
│   └──────┬──────┘     │  • lambda_mult=0.7 (diversity)       │      │
│          │            └──────────────────────────────────────┘      │
│          ▼                                                          │
│   ┌─────────────┐     ┌──────────────────────────────────────┐      │
│   │  FORMAT     │     │  Document → "[Source: file.pdf]"     │      │
│   │   DOCS      │────►│  Adds source attribution             │      │
│   │             │     │  Preserves page numbers              │      │
│   └──────┬──────┘     └──────────────────────────────────────┘      │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────┐     ┌──────────────────────────────────────┐      │
│   │   PROMPT    │     │  System: "You are a marketing        │      │
│   │  TEMPLATE   │────►│  analytics expert..."                │      │
│   │             │     │  Context: {retrieved_docs}           │      │
│   └──────┬──────┘     │  Question: {user_question}           │      │
│          │            └──────────────────────────────────────┘      │
│          ▼                                                          │
│   ┌─────────────┐                                                   │
│   │    LLM      │     GPT-4 / Claude 3 Sonnet                       │
│   │             │     temperature=0.1 (factual)                     │
│   └──────┬──────┘                                                   │
│          │                                                          │
│          ▼                                                          │
│   ┌─────────────┐                                                   │
│   │   OUTPUT    │     Answer + [Source: filename, page] citations   │
│   │   PARSER    │                                                   │
│   └─────────────┘                                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Class Structure

```
MarketingRAG
├── __init__(persist_directory, embedding_model, llm_provider, ...)
│   ├── self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
│   ├── self.llm = ChatOpenAI(model="gpt-4") | ChatAnthropic(...)
│   ├── self.text_splitter = RecursiveCharacterTextSplitter(
│   │       chunk_size=1000, chunk_overlap=200)
│   ├── self.vectorstore = Chroma(persist_directory, embeddings)
│   └── self.rag_chain = _build_rag_chain()
│
├── INGESTION
│   ├── ingest_file(file_path, metadata) → int (chunks created)
│   ├── ingest_documents(directory, glob) → int (total chunks)
│   └── _extract_marketing_metadata(doc) → Dict
│       ├── channels: ["search", "social", ...]
│       ├── metrics_mentioned: ["roas", "cpa", ...]
│       └── report_type: "performance" | "budget" | "audience"
│
├── QUERY
│   ├── query(question, filters, top_k) → RAGResponse
│   │   ├── content: str (answer)
│   │   ├── sources: List[Dict] (file, page, preview)
│   │   ├── confidence: float
│   │   └── tokens_used: int
│   └── _format_docs(docs) → str (with source citations)
│
└── UTILITIES
    ├── clear() → Reset vector store
    └── get_stats() → Dict (document count, config)
```

### 2.4 LCEL Chain Definition

```python
# LangChain Expression Language (LCEL) Chain
rag_chain = (
    {
        "context": retriever | format_docs,   # Parallel: retrieve + format
        "question": RunnablePassthrough()     # Pass through user question
    }
    | prompt                                   # ChatPromptTemplate
    | llm                                      # ChatOpenAI / ChatAnthropic
    | StrOutputParser()                        # Extract string response
)
```

### 2.5 Metadata Extraction

```
Auto-detected Marketing Metadata
├── CHANNELS (keyword detection)
│   ├── search → ["search", "sem", "ppc", "google ads"]
│   ├── social → ["social", "facebook", "instagram", "meta", "tiktok"]
│   ├── display → ["display", "banner", "programmatic", "dv360"]
│   ├── video → ["video", "youtube", "ctv", "ott"]
│   └── email → ["email", "newsletter", "crm"]
│
├── METRICS MENTIONED
│   └── ["roas", "cpa", "cpc", "ctr", "cvr", "roi", "impressions", ...]
│
└── REPORT TYPE
    ├── performance → ["performance", "results", "analysis"]
    ├── budget → ["budget", "allocation", "spend"]
    └── audience → ["audience", "segment", "targeting"]
```

---

## 3. Multi-Agent Analyst (Google ADK)

### 3.1 Core Concept

```
Purpose: Collaborative AI agents for comprehensive marketing analysis
Framework: Google Agent Development Kit (ADK)
Pattern: Coordinator + Specialist agents with tool use
```

### 3.2 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-AGENT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                      USER QUERY                              │   │
│   │   "How are social campaigns performing and who should       │   │
│   │    we target next quarter?"                                 │   │
│   └───────────────────────────┬─────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    COORDINATOR AGENT                         │   │
│   │                                                              │   │
│   │   Role: Orchestration + Synthesis                           │   │
│   │                                                              │   │
│   │   ┌─────────────────────────────────────────────────────┐   │   │
│   │   │ determine_specialists(query)                         │   │   │
│   │   │   ├── Keyword matching for routing                  │   │   │
│   │   │   └── Returns: ["performance_analyst",              │   │   │
│   │   │                 "audience_analyst"]                 │   │   │
│   │   └─────────────────────────────────────────────────────┘   │   │
│   │                                                              │   │
│   └───────────────────────────┬─────────────────────────────────┘   │
│                               │                                     │
│           ┌───────────────────┼───────────────────┐                 │
│           │                   │                   │                 │
│           ▼                   ▼                   ▼                 │
│   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐         │
│   │  PERFORMANCE  │   │   AUDIENCE    │   │  COMPETITOR   │         │
│   │   ANALYST     │   │   ANALYST     │   │   ANALYST     │         │
│   │               │   │               │   │               │         │
│   │ Expertise:    │   │ Expertise:    │   │ Expertise:    │         │
│   │ • ROAS/CPA    │   │ • Segments    │   │ • Market share│         │
│   │ • Attribution │   │ • Personas    │   │ • SOV         │         │
│   │ • Trends      │   │ • LTV         │   │ • Benchmarks  │         │
│   │               │   │               │   │               │         │
│   │ Tools:        │   │ Tools:        │   │ Tools:        │         │
│   │ • get_channel │   │ • get_segment │   │ • get_market_ │         │
│   │   _performance│   │   _performance│   │   share       │         │
│   │ • analyze_    │   │ • get_top_    │   │ • get_sov     │         │
│   │   trends      │   │   segments    │   │ • get_        │         │
│   │               │   │               │   │   benchmark   │         │
│   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘         │
│           │                   │                   │                 │
│           └───────────────────┼───────────────────┘                 │
│                               │                                     │
│                               ▼                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    COORDINATOR AGENT                         │   │
│   │                                                              │   │
│   │   ┌─────────────────────────────────────────────────────┐   │   │
│   │   │ synthesize(query, agent_responses)                   │   │   │
│   │   │   ├── Combine specialist insights                   │   │   │
│   │   │   ├── Identify agreements/conflicts                 │   │   │
│   │   │   └── Generate prioritized recommendations          │   │   │
│   │   └─────────────────────────────────────────────────────┘   │   │
│   │                                                              │   │
│   └───────────────────────────┬─────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    TEAM RESPONSE                             │   │
│   │                                                              │   │
│   │   • summary: str (synthesized analysis)                     │   │
│   │   • agent_responses: Dict[str, AgentResponse]               │   │
│   │   • recommendations: List[str]                              │   │
│   │   • confidence: float (averaged)                            │   │
│   │                                                              │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Agent Class Hierarchy

```
BaseAgent (Abstract)
├── Properties
│   ├── name: str
│   ├── description: str
│   ├── role: AgentRole (COORDINATOR | SPECIALIST)
│   └── system_prompt: str (abstract)
│
├── Methods
│   ├── analyze(query, context) → AgentResponse
│   ├── _setup_tools() → Configure agent tools
│   └── clear_history() → Reset conversation
│
└── Implementations
    │
    ├── CoordinatorAgent
    │   ├── determine_specialists(query) → List[str]
    │   └── synthesize(query, responses) → (summary, recommendations)
    │
    ├── PerformanceAnalyst
    │   ├── Tools: get_channel_performance, analyze_trends
    │   └── Focus: ROAS, CPA, CPC, CTR, attribution
    │
    ├── AudienceAnalyst
    │   ├── Tools: get_segment_performance, get_top_segments
    │   └── Focus: Segments, personas, LTV, targeting
    │
    └── CompetitorAnalyst
        ├── Tools: get_market_share, get_sov, get_benchmark
        └── Focus: Market share, SOV, competitive positioning
```

### 3.4 Routing Logic

```
Query Keyword → Specialist Mapping
│
├── PERFORMANCE KEYWORDS
│   │ "performance", "roas", "roi", "cpa", "cpc", "ctr"
│   │ "spend", "budget", "efficiency", "attribution"
│   │ "campaign", "channel", "metrics"
│   └──────► performance_analyst
│
├── AUDIENCE KEYWORDS
│   │ "audience", "segment", "target", "persona", "customer"
│   │ "who", "demographic", "behavior", "ltv", "value"
│   └──────► audience_analyst
│
├── COMPETITOR KEYWORDS
│   │ "competitor", "competition", "market share", "benchmark"
│   │ "industry", "sov", "share of voice", "positioning"
│   └──────► competitor_analyst
│
└── DEFAULT (no keyword match)
    └──────► [performance_analyst, audience_analyst]
```

### 3.5 Tool Schema

```
Performance Analyst Tools
├── get_channel_performance(channel, date_range)
│   ├── Parameters:
│   │   ├── channel: "search" | "social" | "display" | "video" | "email"
│   │   └── date_range: "last_7d" | "last_30d" | "last_90d" | "ytd"
│   └── Returns: {spend, revenue, roas, conversions, cpa}
│
└── analyze_trends(metric, channel, period)
    ├── Parameters:
    │   ├── metric: "roas" | "cpa" | "ctr" | ...
    │   ├── channel: str
    │   └── period: "daily" | "weekly" | "monthly"
    └── Returns: {trend_direction, change_pct, forecast}

Audience Analyst Tools
├── get_segment_performance(segment_id)
│   └── Returns: {segment_name, size, roas, cpa, ltv}
│
└── get_top_segments(metric, limit)
    └── Returns: List[{segment, metric_value, share}]

Competitor Analyst Tools
├── get_market_share(category)
│   └── Returns: Dict[competitor, share_pct]
│
├── get_sov(channel, period)
│   └── Returns: {our_sov, competitors: Dict}
│
└── get_benchmark(metric, industry)
    └── Returns: {industry_avg, percentile_25, percentile_75}
```

### 3.6 Team Response Structure

```
TeamResponse
├── query: str                          # Original user query
├── agents_consulted: List[str]         # ["performance_analyst", "audience_analyst"]
├── agent_responses: Dict[str, AgentResponse]
│   └── AgentResponse
│       ├── agent_name: str
│       ├── content: str                # Analysis text
│       ├── confidence: float           # 0.0-1.0
│       ├── data_used: List[str]        # Tools/data sources
│       └── timestamp: str
├── summary: str                        # Synthesized analysis
├── recommendations: List[str]          # Prioritized action items
└── confidence: float                   # Averaged across agents
```

---

## 4. Comparison Matrix

```
┌────────────────────┬───────────────────┬───────────────────┬───────────────────┐
│    Aspect          │ Budget Optimizer  │   Marketing RAG   │  Multi-Agent ADK  │
├────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Framework          │ LangGraph         │ LangChain         │ Google ADK        │
├────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Pattern            │ State Machine     │ RAG Pipeline      │ Agent Coordinator │
│                    │ Directed Graph    │ LCEL Chain        │ + Specialists     │
├────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Primary Use        │ Budget allocation │ Document Q&A      │ Comprehensive     │
│                    │ optimization      │ with citations    │ analysis          │
├────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Data Flow          │ State accumulates │ Docs → Vectors    │ Query → Route →   │
│                    │ through nodes     │ → Retrieval → LLM │ Agents → Synth    │
├────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Key Feature        │ Conditional       │ MMR retrieval     │ Parallel agent    │
│                    │ routing + retry   │ + metadata filter │ execution         │
├────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Output             │ Allocation +      │ Answer +          │ Synthesized       │
│                    │ confidence +      │ sources +         │ report +          │
│                    │ explanation       │ confidence        │ recommendations   │
├────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Extensibility      │ Add nodes/edges   │ Add loaders/      │ Add specialist    │
│                    │ to graph          │ retrievers        │ agents/tools      │
└────────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

---

## 5. Production Scaling Considerations

```
LOCAL DEMO → PRODUCTION
│
├── BUDGET OPTIMIZER
│   ├── Demo: Single market, 5 channels
│   └── Prod: 7 EU markets, 15+ channels, Monte Carlo simulations
│
├── MARKETING RAG
│   ├── Demo: Local ChromaDB, sample docs
│   └── Prod: AWS OpenSearch, 10M+ vectors, real-time API data
│
└── MULTI-AGENT
    ├── Demo: 3 specialists, sequential execution
    └── Prod: 12 agents, parallel async, hierarchical teams
```

---

## 6. Technical Dependencies

```
Core Dependencies
├── LangChain Ecosystem
│   ├── langchain-core ^0.2
│   ├── langchain-openai ^0.1
│   ├── langchain-anthropic ^0.1
│   └── langchain-community ^0.2
│
├── LangGraph
│   └── langgraph ^0.1
│
├── Vector Store
│   └── chromadb ^0.4
│
├── Data Processing
│   ├── pandas ^2.0
│   └── numpy ^1.24
│
└── LLM Providers
    ├── OpenAI (GPT-4, text-embedding-3-small)
    ├── Anthropic (Claude 3 Sonnet)
    └── Google (Gemini - via ADK)
```

---

## 7. API Keys Required

```
Environment Variables
├── OPENAI_API_KEY      # For GPT-4 and embeddings
├── ANTHROPIC_API_KEY   # For Claude models (optional)
└── GOOGLE_API_KEY      # For Gemini/ADK (optional)
```

---

## Author

**Jaiguru Thevar**
Head of Data Science, VML (WPP Group)
19+ years in AI/ML at Amazon, VML/WPP, Citi, HSBC

Production deployments: Unilever, Colgate, Mars, Samsung
