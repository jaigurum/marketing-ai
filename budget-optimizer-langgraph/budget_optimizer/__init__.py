"""
Budget Optimizer with LangGraph

A production-inspired marketing budget optimization system using LangGraph
state machines for workflow orchestration.

Example:
    from budget_optimizer import BudgetOptimizerGraph, optimize_budget
    
    # Quick usage
    result = optimize_budget(
        total_budget=1000000,
        channels=["search", "social", "display", "video", "email"],
        constraints={
            "search": {"min": 0.15, "max": 0.40},
            "social": {"min": 0.10, "max": 0.30},
            "display": {"min": 0.05, "max": 0.20},
            "video": {"min": 0.10, "max": 0.25},
            "email": {"min": 0.05, "max": 0.15}
        }
    )
    
    print(result["final_allocation"])
    print(result["explanation"])
    
    # Advanced usage with streaming
    optimizer = BudgetOptimizerGraph(use_llm=True, llm_provider="anthropic")
    
    for step in optimizer.stream(request):
        print(f"Step: {step}")
"""

from .graph import BudgetOptimizerGraph, optimize_budget, create_budget_optimizer_graph
from .state import OptimizationState, create_initial_state, SAMPLE_CONSTRAINTS, SAMPLE_CHANNELS

__version__ = "0.1.0"
__author__ = "Jaiguru Thevar"

__all__ = [
    "BudgetOptimizerGraph",
    "optimize_budget",
    "create_budget_optimizer_graph",
    "OptimizationState",
    "create_initial_state",
    "SAMPLE_CONSTRAINTS",
    "SAMPLE_CHANNELS"
]
