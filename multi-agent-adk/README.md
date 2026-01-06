# Multi-Agent Marketing Analyst (Google ADK)

**Coordinator + specialist agents for comprehensive marketing analysis**

This project demonstrates a multi-agent system using Google's Agent Development Kit (ADK) where a coordinator agent routes queries to specialized agents (audience analyst, performance analyst, competitor analyst) and synthesizes their outputs.

---

## What It Does

```
Input:  Marketing analysis request
Output: Comprehensive analysis from multiple specialist perspectives
```

### Agent Architecture

```
                    ┌─────────────────────┐
                    │   USER QUERY        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   COORDINATOR       │
                    │   AGENT             │
                    │                     │
                    │  • Analyzes query   │
                    │  • Routes to agents │
                    │  • Synthesizes      │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
           ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   AUDIENCE      │ │   PERFORMANCE   │ │   COMPETITOR    │
│   ANALYST       │ │   ANALYST       │ │   ANALYST       │
│                 │ │                 │ │                 │
│ • Segmentation  │ │ • ROAS/CPA/CTR  │ │ • Market share  │
│ • Targeting     │ │ • Attribution   │ │ • Benchmarks    │
│ • Personas      │ │ • Trends        │ │ • Positioning   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   SYNTHESIZED       │
                    │   RESPONSE          │
                    └─────────────────────┘
```

---

## Production Parallel

This is a simplified version of the 12-agent architecture deployed at VML:

| This Demo | Production System |
|-----------|-------------------|
| 3 specialist agents | 12 specialized agents |
| Hardcoded data | Real-time API connections |
| Sequential routing | Parallel execution with async |
| Single coordinator | Hierarchical agent teams |
| Console output | WPP Open platform integration |

**Production Impact:** Adopted by Unilever, Colgate, Mars, Samsung across 7 EU markets

---

## Installation

```bash
cd multi-agent-adk
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from multi_agent import MarketingAnalystTeam

# Initialize the team
team = MarketingAnalystTeam()

# Run analysis
result = team.analyze("How are our social campaigns performing and who should we target?")

print(result.summary)
print(result.agent_responses)
print(result.recommendations)
```

### Command Line

```bash
# Interactive mode
python main.py chat

# Single analysis
python main.py analyze "Analyze our Q4 performance and suggest improvements"

# Specific agent
python main.py analyze "Who are our highest-value customers?" --agent audience
```

---

## Project Structure

```
multi-agent-adk/
├── README.md
├── requirements.txt
├── main.py                      # CLI entry point
├── multi_agent/
│   ├── __init__.py
│   ├── team.py                  # Main team orchestrator
│   ├── coordinator.py           # Coordinator agent
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # Base agent class
│   │   ├── audience.py          # Audience analyst
│   │   ├── performance.py       # Performance analyst
│   │   └── competitor.py        # Competitor analyst
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── data_tools.py        # Data access tools
│   │   └── analysis_tools.py    # Analysis utilities
│   └── prompts/
│       ├── __init__.py
│       └── templates.py         # Prompt templates
├── data/
│   └── sample_data.json         # Sample marketing data
└── tests/
    └── test_agents.py
```

---

## Key ADK Concepts Demonstrated

### 1. Agent Definition

```python
from google.adk import Agent, Tool

class PerformanceAnalyst(Agent):
    """Specialist agent for performance analysis"""
    
    name = "performance_analyst"
    description = "Analyzes campaign performance metrics including ROAS, CPA, and attribution"
    
    tools = [
        Tool(
            name="get_channel_performance",
            description="Get performance metrics for a specific channel",
            function=get_channel_performance
        ),
        Tool(
            name="calculate_attribution",
            description="Calculate attribution across touchpoints",
            function=calculate_attribution
        )
    ]
    
    system_prompt = """You are a senior performance marketing analyst.
    Your role is to analyze campaign data and provide actionable insights.
    
    When analyzing:
    1. Always cite specific metrics
    2. Compare to benchmarks when available
    3. Identify trends and anomalies
    4. Suggest optimization opportunities
    """
```

### 2. Tool Implementation

```python
from google.adk import Tool, ToolResult

@Tool(
    name="get_channel_performance",
    description="Get performance metrics for a marketing channel",
    parameters={
        "channel": {"type": "string", "description": "Channel name (search, social, display, video, email)"},
        "date_range": {"type": "string", "description": "Date range (last_7d, last_30d, last_90d, ytd)"}
    }
)
def get_channel_performance(channel: str, date_range: str) -> ToolResult:
    """Retrieve channel performance data"""
    
    # In production: API calls to Google Ads, Meta, AMC, etc.
    # Here: Sample data for demonstration
    
    data = load_sample_data()
    channel_data = data.get(channel, {}).get(date_range, {})
    
    return ToolResult(
        success=True,
        data={
            "channel": channel,
            "date_range": date_range,
            "spend": channel_data.get("spend", 0),
            "revenue": channel_data.get("revenue", 0),
            "roas": channel_data.get("roas", 0),
            "conversions": channel_data.get("conversions", 0),
            "cpa": channel_data.get("cpa", 0)
        }
    )
```

### 3. Coordinator Agent

```python
from google.adk import Agent, AgentTeam

class CoordinatorAgent(Agent):
    """Routes queries to specialist agents and synthesizes responses"""
    
    name = "coordinator"
    description = "Orchestrates analysis across specialist agents"
    
    def __init__(self, specialists: List[Agent]):
        self.specialists = {agent.name: agent for agent in specialists}
        self.team = AgentTeam(agents=specialists)
    
    async def route_query(self, query: str) -> List[str]:
        """Determine which specialists should handle this query"""
        
        routing_prompt = f"""
        Given this marketing analysis request:
        "{query}"
        
        Which specialists should be consulted?
        Available: {list(self.specialists.keys())}
        
        Return a JSON list of agent names.
        """
        
        response = await self.llm.generate(routing_prompt)
        return json.loads(response)
    
    async def synthesize(self, query: str, agent_responses: Dict[str, str]) -> str:
        """Synthesize responses from multiple agents"""
        
        synthesis_prompt = f"""
        Original query: {query}
        
        Agent responses:
        {json.dumps(agent_responses, indent=2)}
        
        Synthesize these perspectives into a cohesive analysis.
        Highlight agreements, conflicts, and actionable recommendations.
        """
        
        return await self.llm.generate(synthesis_prompt)
```

### 4. Team Orchestration

```python
from google.adk import AgentTeam, ExecutionMode

team = AgentTeam(
    coordinator=CoordinatorAgent(),
    agents=[
        AudienceAnalyst(),
        PerformanceAnalyst(),
        CompetitorAnalyst()
    ],
    execution_mode=ExecutionMode.PARALLEL  # Run agents in parallel
)

# Execute with context
result = await team.execute(
    query="Analyze our Q4 campaign performance",
    context={
        "client": "Colgate",
        "region": "EU",
        "channels": ["search", "social", "display"]
    }
)
```

### 5. MCP Tool Integration

```python
from google.adk import MCPServer, MCPTool

# Define MCP server for secure data access
class MarketingDataServer(MCPServer):
    """MCP server providing access to marketing data"""
    
    name = "marketing_data"
    
    tools = [
        MCPTool(
            name="query_amc",
            description="Query Amazon Marketing Cloud data",
            input_schema={
                "query": {"type": "string"},
                "date_range": {"type": "string"}
            }
        ),
        MCPTool(
            name="get_google_ads_metrics",
            description="Get Google Ads performance metrics",
            input_schema={
                "campaign_id": {"type": "string"},
                "metrics": {"type": "array", "items": {"type": "string"}}
            }
        )
    ]

# Connect to agent
agent.connect_mcp_server(MarketingDataServer())
```

---

## Sample Interactions

### Multi-Agent Analysis

```
User: How are our social campaigns performing and who should we target next quarter?

Coordinator: Routing to performance_analyst and audience_analyst...

Performance Analyst:
  Social campaigns show ROAS of 3.2x (vs 2.8x benchmark).
  Facebook: 3.5x ROAS, declining 5% MoM
  Instagram: 3.8x ROAS, improving 12% MoM
  TikTok: 2.1x ROAS, high volatility but growing
  
  Recommendation: Shift budget from Facebook to Instagram/TikTok

Audience Analyst:
  Top performing segments:
  1. Health-conscious millennials (35% of conversions, 4.2x ROAS)
  2. Young parents (28% of conversions, 3.8x ROAS)
  3. Fitness enthusiasts (22% of conversions, 3.5x ROAS)
  
  Underserved opportunity: Gen Z health seekers (2% of spend, 5.1x ROAS)
  
  Recommendation: Expand Gen Z targeting on TikTok

Synthesized Response:
  Social performance is strong at 3.2x ROAS. Key actions for Q1:
  
  1. BUDGET SHIFT: Move 15% of Facebook budget to Instagram/TikTok
  2. AUDIENCE EXPANSION: Scale Gen Z health seeker targeting
  3. PLATFORM STRATEGY: Prioritize Instagram for millennials, TikTok for Gen Z
  
  Expected impact: +18% social ROAS improvement
```

---

## Extending the System

### Add New Specialist Agent

```python
class CreativeAnalyst(Agent):
    """Analyzes creative performance and provides optimization recommendations"""
    
    name = "creative_analyst"
    description = "Analyzes ad creative performance, fatigue, and optimization opportunities"
    
    tools = [
        Tool(name="get_creative_metrics", ...),
        Tool(name="detect_creative_fatigue", ...),
        Tool(name="analyze_creative_elements", ...)
    ]
    
    system_prompt = """You are a creative analytics specialist..."""

# Add to team
team.add_agent(CreativeAnalyst())
```

### Add Custom MCP Integration

```python
class ClientCRMServer(MCPServer):
    """MCP server for client CRM data access"""
    
    name = "client_crm"
    
    async def query_customers(self, segment: str) -> List[Dict]:
        # Secure access to client CRM
        pass

agent.connect_mcp_server(ClientCRMServer(
    credentials=load_credentials(),
    allowed_operations=["read"]
))
```

---

## License

MIT
