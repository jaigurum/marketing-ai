# Agentic Marketing AI Portfolio

**Production-Inspired Multi-Agent Systems for Marketing Intelligence**

This portfolio demonstrates multi-agent AI architectures for marketing analytics and optimization. Each project implements patterns I've deployed at scale for enterprise clients including Unilever, Colgate, Mars, and Samsung—simplified here to showcase framework proficiency.

---

## Projects

| Project | Framework | Description | Production Parallel |
|---------|-----------|-------------|---------------------|
| [Budget Optimizer](./budget-optimizer-langgraph) | LangGraph | Multi-step agent for marketing budget allocation | |
| [Marketing RAG](./marketing-rag-langchain) | LangChain | RAG pipeline for campaign performance Q&A | 85% satisfaction system  |
| [Multi-Agent Analyst](./multi-agent-adk) | Google ADK | Coordinator + specialist agents for marketing analysis | 12-agent suite serving 4 enterprise clients |

---

## Architecture Philosophy

These projects share common design principles from production deployments:

### 1. **Modular Agent Design**
Each agent has a single responsibility. Orchestration happens at a higher layer. This enables:
- Independent testing and deployment
- Graceful degradation when individual agents fail
- Easy addition of new capabilities

### 2. **Grounded in Real Data**
All agents use RAG or structured data retrieval—never pure generation. This ensures:
- Factual, verifiable outputs
- Client-specific context
- Audit trails for recommendations

### 3. **Measurable Outcomes**
Every agent output includes confidence scores, source citations, or impact estimates. Production systems must justify their existence with ROI.

### 4. **Enterprise-Ready Patterns**
- Structured outputs (not free-form text)
- Error handling and fallbacks
- Logging for debugging and compliance
- Configurable for different clients/markets

---

## Technical Stack

```
Frameworks:     LangChain 0.2+ | LangGraph 0.1+ | Google ADK
Models:         GPT-4 / Claude / Gemini (configurable)
Vector Store:   ChromaDB (local) / Pinecone (production)
Orchestration:  LangGraph state machines / ADK agent routing
Data:           Pandas for structured data, RAG for unstructured
```

---

## Quick Start

Each project has its own setup instructions. General requirements:

```bash
# Clone the repository
git clone https://github.com/jaigurum/agentic-marketing-ai.git
cd agentic-marketing-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies (each project has requirements.txt)
cd budget-optimizer-langgraph
pip install -r requirements.txt
```

Set your API keys:
```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
# or
export GOOGLE_API_KEY="your-key"
```

---

## Production Context

These simplified implementations mirror production systems I've built:

| This Portfolio | Production at Scale |
|----------------|---------------------|
| Single-market budget optimizer | 7 EU markets, €2B+ annual optimization |
| Local ChromaDB vector store | AWS OpenSearch / Pinecone clusters |
| 3 specialist agents | 12-agent suite with 50+ tools |
| Sample marketing data | Real-time AMC, Google Ads, Meta APIs |
| Console output | Executive dashboards, Slack integrations |

The architectural patterns are identical—the difference is scale, data volume, and enterprise integrations.

---

## Author

**Jaiguru Thevar**  

[LinkedIn](https://linkedin.com/in/jaiguru) | [GitHub](https://github.com/jaigurum) | mjguru@gmail.com



---

## License

MIT License - Feel free to use these patterns in your own projects.
