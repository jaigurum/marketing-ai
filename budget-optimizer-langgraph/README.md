# Budget Optimizer Agent (LangGraph)

**Multi-step agent for marketing budget allocation using LangGraph state machines**

This project demonstrates a production-inspired budget optimization system that takes marketing constraints and historical performance data to recommend optimal budget allocation across channels.

---

## What It Does

```
Input:  Total budget + channel constraints + historical ROAS data
Output: Recommended allocation with expected ROI and confidence scores
```

### Agent Workflow (LangGraph State Machine)

```
┌─────────────────┐
│  START: Input   │
│  Budget & Data  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1. VALIDATE     │ ◄── Check constraints, data quality
│    Inputs       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. ANALYZE      │ ◄── Historical performance, seasonality
│    Performance  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. OPTIMIZE     │ ◄── Allocation algorithm + LLM reasoning
│    Allocation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. VALIDATE     │ ◄── Check against constraints
│    Output       │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Valid?  │
    └────┬────┘
     Yes │ No ──► Loop back to OPTIMIZE
         ▼
┌─────────────────┐
│ 5. EXPLAIN      │ ◄── Generate reasoning narrative
│    & Report     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  END: Output    │
│  Recommendation │
└─────────────────┘
```

---

## Production Parallel

This is a simplified version of the budget optimization system I built at VML:

| This Demo | Production System |
|-----------|-------------------|
| 5 channels | 15+ channels including programmatic, affiliate, influencer |
| Single market | 7 EU markets with currency/VAT handling |
| Static CSV data | Real-time API connections to AMC, Google Ads, Meta |
| Console output | Integration with WPP Open platform, Slack alerts |
| Single optimization | Scenario modeling with Monte Carlo simulations |

**Production Impact:** +20% ROI improvement, adopted by Unilever, Colgate, Mars, Samsung

---

## Installation

```bash
cd budget-optimizer-langgraph
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from budget_optimizer import BudgetOptimizerGraph

# Initialize the optimizer
optimizer = BudgetOptimizerGraph()

# Define your optimization request
request = {
    "total_budget": 1000000,  # €1M
    "channels": ["search", "social", "display", "video", "email"],
    "constraints": {
        "search": {"min": 0.15, "max": 0.40},   # 15-40% of budget
        "social": {"min": 0.10, "max": 0.30},
        "display": {"min": 0.05, "max": 0.20},
        "video": {"min": 0.10, "max": 0.25},
        "email": {"min": 0.05, "max": 0.15}
    },
    "objective": "maximize_roas",
    "historical_data_path": "data/sample_performance.csv"
}

# Run optimization
result = optimizer.run(request)

# Output
print(result["allocation"])
print(result["expected_roas"])
print(result["explanation"])
```

### Command Line

```bash
python main.py --budget 1000000 --data data/sample_performance.csv
```

---

## Project Structure

```
budget-optimizer-langgraph/
├── README.md
├── requirements.txt
├── main.py                    # Entry point
├── budget_optimizer/
│   ├── __init__.py
│   ├── graph.py               # LangGraph state machine definition
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── validator.py       # Input validation node
│   │   ├── analyzer.py        # Performance analysis node
│   │   ├── optimizer.py       # Allocation optimization node
│   │   ├── output_validator.py # Constraint checking node
│   │   └── explainer.py       # Narrative generation node
│   ├── state.py               # State schema definition
│   └── tools.py               # Helper functions
├── data/
│   └── sample_performance.csv # Sample historical data
└── tests/
    └── test_optimizer.py
```

---

## Key LangGraph Concepts Demonstrated

### 1. **State Schema**
```python
class OptimizationState(TypedDict):
    # Inputs
    total_budget: float
    channels: List[str]
    constraints: Dict[str, Dict[str, float]]
    historical_data: pd.DataFrame
    
    # Intermediate state
    performance_analysis: Dict[str, Any]
    proposed_allocation: Dict[str, float]
    validation_errors: List[str]
    iteration_count: int
    
    # Outputs
    final_allocation: Dict[str, float]
    expected_metrics: Dict[str, float]
    explanation: str
    confidence_score: float
```

### 2. **Conditional Routing**
```python
def should_reoptimize(state: OptimizationState) -> str:
    """Route based on validation results"""
    if state["validation_errors"] and state["iteration_count"] < 3:
        return "optimize"  # Try again
    elif state["validation_errors"]:
        return "fallback"  # Use rule-based allocation
    else:
        return "explain"   # Proceed to explanation
```

### 3. **Graph Definition**
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(OptimizationState)

# Add nodes
workflow.add_node("validate_input", validate_input_node)
workflow.add_node("analyze", analyze_performance_node)
workflow.add_node("optimize", optimize_allocation_node)
workflow.add_node("validate_output", validate_output_node)
workflow.add_node("explain", generate_explanation_node)
workflow.add_node("fallback", fallback_allocation_node)

# Add edges
workflow.add_edge("validate_input", "analyze")
workflow.add_edge("analyze", "optimize")
workflow.add_edge("optimize", "validate_output")
workflow.add_conditional_edges(
    "validate_output",
    should_reoptimize,
    {
        "optimize": "optimize",
        "fallback": "fallback",
        "explain": "explain"
    }
)
workflow.add_edge("explain", END)
workflow.add_edge("fallback", END)

# Set entry point
workflow.set_entry_point("validate_input")

# Compile
app = workflow.compile()
```

---

## Sample Output

```json
{
  "allocation": {
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
  "explanation": "Recommended allocation prioritizes Search (32%) based on historical ROAS of 5.1x and stable performance. Social increased to 25% given Q4 seasonality patterns. Video allocation at 20% balances brand awareness needs with direct response efficiency. Display and Email maintained at minimum thresholds due to lower marginal returns in current market conditions.",
  "confidence_score": 0.82,
  "iterations": 1,
  "constraints_satisfied": true
}
```

---

## Extending the System

### Add New Channels
```python
# In config.py
SUPPORTED_CHANNELS = [
    "search", "social", "display", "video", "email",
    "affiliate", "influencer", "programmatic", "ctv"  # Add new channels
]
```

### Custom Optimization Objectives
```python
# In optimizer.py
OBJECTIVES = {
    "maximize_roas": maximize_roas_objective,
    "maximize_reach": maximize_reach_objective,
    "minimize_cpa": minimize_cpa_objective,
    "balanced": balanced_objective  # Add custom objectives
}
```

### Multi-Market Support
```python
# Extend state schema
class MultiMarketState(OptimizationState):
    markets: List[str]
    currency_rates: Dict[str, float]
    market_constraints: Dict[str, Dict[str, Any]]
```

---

## License

MIT
