"""
Budget Optimizer Graph

LangGraph state machine for marketing budget optimization.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from .state import OptimizationState, create_initial_state
from .nodes import (
    validate_input_node,
    analyze_performance_node,
    optimize_allocation_node,
    validate_output_node,
    should_reoptimize,
    fallback_allocation_node,
    generate_explanation_node
)


def create_budget_optimizer_graph(use_llm: bool = False, llm_provider: str = "openai") -> StateGraph:
    """
    Creates the LangGraph workflow for budget optimization.
    
    Args:
        use_llm: Whether to use LLM-enhanced nodes
        llm_provider: Which LLM to use ("openai", "anthropic")
        
    Returns:
        Compiled StateGraph ready for execution
    """
    
    # Initialize the graph with state schema
    workflow = StateGraph(OptimizationState)
    
    # ===== Add Nodes =====
    
    # Input validation
    workflow.add_node("validate_input", validate_input_node)
    
    # Performance analysis
    workflow.add_node("analyze", analyze_performance_node)
    
    # Optimization
    workflow.add_node("optimize", optimize_allocation_node)
    
    # Output validation
    workflow.add_node("validate_output", validate_output_node)
    
    # Explanation generation
    workflow.add_node("explain", generate_explanation_node)
    
    # Fallback for failed optimization
    workflow.add_node("fallback", fallback_allocation_node)
    
    # ===== Add Edges =====
    
    # Linear flow: validate -> analyze -> optimize -> validate_output
    workflow.add_edge("validate_input", "analyze")
    workflow.add_edge("analyze", "optimize")
    workflow.add_edge("optimize", "validate_output")
    
    # Conditional routing after output validation
    workflow.add_conditional_edges(
        "validate_output",
        should_reoptimize,
        {
            "optimize": "optimize",    # Try again
            "fallback": "fallback",    # Use safe defaults
            "explain": "explain"       # Success - generate explanation
        }
    )
    
    # Terminal edges
    workflow.add_edge("explain", END)
    workflow.add_edge("fallback", END)
    
    # Set entry point
    workflow.set_entry_point("validate_input")
    
    return workflow


class BudgetOptimizerGraph:
    """
    High-level interface for the budget optimization graph.
    
    Usage:
        optimizer = BudgetOptimizerGraph()
        result = optimizer.run({
            "total_budget": 1000000,
            "channels": ["search", "social", "display"],
            "constraints": {...}
        })
    """
    
    def __init__(
        self,
        use_llm: bool = False,
        llm_provider: str = "openai",
        model_name: Optional[str] = None
    ):
        """
        Initialize the optimizer.
        
        Args:
            use_llm: Whether to use LLM-enhanced optimization
            llm_provider: "openai" or "anthropic"
            model_name: Specific model to use (default: gpt-4 or claude-3-sonnet)
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        
        # Create graph
        self.workflow = create_budget_optimizer_graph(use_llm, llm_provider)
        self.app = self.workflow.compile()
        
        # Initialize LLM if needed
        if use_llm:
            if llm_provider == "openai":
                self.llm = ChatOpenAI(
                    model=model_name or "gpt-4",
                    temperature=0.1
                )
            elif llm_provider == "anthropic":
                self.llm = ChatAnthropic(
                    model=model_name or "claude-3-sonnet-20240229",
                    temperature=0.1
                )
            else:
                raise ValueError(f"Unknown LLM provider: {llm_provider}")
    
    def run(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run budget optimization synchronously.
        
        Args:
            request: Optimization request with:
                - total_budget: float
                - channels: List[str]
                - constraints: Dict[str, Dict[str, float]]
                - objective: str (optional, default: "maximize_roas")
                - historical_data_path: str (optional)
                
        Returns:
            Optimization result with:
                - final_allocation: Dict[str, float]
                - expected_metrics: Dict[str, float]
                - explanation: str
                - confidence_score: float
                - warnings: List[str]
        """
        
        # Create initial state
        state = create_initial_state(
            total_budget=request["total_budget"],
            channels=request["channels"],
            constraints=request["constraints"],
            objective=request.get("objective", "maximize_roas"),
            historical_data_path=request.get("historical_data_path", "")
        )
        
        # Run the graph
        import time
        start_time = time.time()
        
        result = self.app.invoke(state)
        
        execution_time = int((time.time() - start_time) * 1000)
        result["execution_time_ms"] = execution_time
        
        return result
    
    async def arun(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run budget optimization asynchronously.
        """
        
        state = create_initial_state(
            total_budget=request["total_budget"],
            channels=request["channels"],
            constraints=request["constraints"],
            objective=request.get("objective", "maximize_roas"),
            historical_data_path=request.get("historical_data_path", "")
        )
        
        import time
        start_time = time.time()
        
        result = await self.app.ainvoke(state)
        
        execution_time = int((time.time() - start_time) * 1000)
        result["execution_time_ms"] = execution_time
        
        return result
    
    def stream(self, request: Dict[str, Any]):
        """
        Stream optimization steps for real-time feedback.
        
        Yields:
            Tuple of (node_name, state) for each step
        """
        
        state = create_initial_state(
            total_budget=request["total_budget"],
            channels=request["channels"],
            constraints=request["constraints"],
            objective=request.get("objective", "maximize_roas"),
            historical_data_path=request.get("historical_data_path", "")
        )
        
        for event in self.app.stream(state):
            yield event
    
    def get_graph_visualization(self) -> str:
        """
        Returns a Mermaid diagram of the graph structure.
        """
        return """
        ```mermaid
        graph TD
            A[Start] --> B[Validate Input]
            B --> C[Analyze Performance]
            C --> D[Optimize Allocation]
            D --> E[Validate Output]
            E -->|Valid| F[Generate Explanation]
            E -->|Invalid, Retry| D
            E -->|Failed| G[Fallback Allocation]
            F --> H[End]
            G --> H
        ```
        """


# Convenience function for quick usage
def optimize_budget(
    total_budget: float,
    channels: list,
    constraints: dict,
    objective: str = "maximize_roas",
    historical_data_path: str = ""
) -> Dict[str, Any]:
    """
    Quick function to run budget optimization.
    
    Example:
        result = optimize_budget(
            total_budget=1000000,
            channels=["search", "social", "display"],
            constraints={
                "search": {"min": 0.2, "max": 0.4},
                "social": {"min": 0.1, "max": 0.3},
                "display": {"min": 0.1, "max": 0.3}
            }
        )
    """
    optimizer = BudgetOptimizerGraph()
    return optimizer.run({
        "total_budget": total_budget,
        "channels": channels,
        "constraints": constraints,
        "objective": objective,
        "historical_data_path": historical_data_path
    })
