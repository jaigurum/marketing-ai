"""
State Schema for Budget Optimizer LangGraph

Defines the typed state that flows through the optimization graph.
"""

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
import pandas as pd


class ChannelConstraint(BaseModel):
    """Constraints for a single channel"""
    min_share: float = Field(ge=0, le=1, description="Minimum budget share (0-1)")
    max_share: float = Field(ge=0, le=1, description="Maximum budget share (0-1)")
    min_absolute: Optional[float] = Field(default=None, description="Minimum absolute budget")
    max_absolute: Optional[float] = Field(default=None, description="Maximum absolute budget")


class PerformanceMetrics(BaseModel):
    """Historical performance metrics for a channel"""
    channel: str
    roas: float = Field(description="Return on Ad Spend")
    cpa: float = Field(description="Cost per Acquisition")
    conversion_rate: float = Field(description="Conversion rate")
    impression_share: float = Field(description="Share of voice/impressions")
    trend: str = Field(description="Performance trend: improving, stable, declining")


class AllocationResult(BaseModel):
    """Final allocation recommendation"""
    channel: str
    budget: float
    share: float
    expected_roas: float
    expected_revenue: float
    confidence: float


class OptimizationState(TypedDict):
    """
    Complete state schema for the budget optimization graph.
    
    This state flows through all nodes and accumulates results
    as the optimization progresses.
    """
    
    # ===== INPUTS =====
    total_budget: float
    channels: List[str]
    constraints: Dict[str, Dict[str, float]]
    objective: str  # maximize_roas, maximize_reach, minimize_cpa, balanced
    historical_data_path: str
    
    # ===== LOADED DATA =====
    historical_data: Optional[Dict[str, Any]]  # Serialized DataFrame
    data_quality_score: float
    data_issues: List[str]
    
    # ===== ANALYSIS RESULTS =====
    performance_analysis: Dict[str, Any]
    channel_rankings: List[str]
    seasonality_factors: Dict[str, float]
    market_conditions: str
    
    # ===== OPTIMIZATION STATE =====
    proposed_allocation: Dict[str, float]
    allocation_rationale: Dict[str, str]
    iteration_count: int
    optimization_history: List[Dict[str, Any]]
    
    # ===== VALIDATION STATE =====
    validation_passed: bool
    validation_errors: List[str]
    constraint_violations: List[str]
    
    # ===== FINAL OUTPUTS =====
    final_allocation: Dict[str, float]
    expected_metrics: Dict[str, float]
    explanation: str
    confidence_score: float
    warnings: List[str]
    
    # ===== METADATA =====
    run_id: str
    timestamp: str
    model_used: str
    execution_time_ms: int


def create_initial_state(
    total_budget: float,
    channels: List[str],
    constraints: Dict[str, Dict[str, float]],
    objective: str = "maximize_roas",
    historical_data_path: str = "data/sample_performance.csv"
) -> OptimizationState:
    """
    Factory function to create a properly initialized state.
    
    Args:
        total_budget: Total budget to allocate
        channels: List of channel names
        constraints: Channel-level constraints
        objective: Optimization objective
        historical_data_path: Path to historical data CSV
        
    Returns:
        Initialized OptimizationState
    """
    import uuid
    from datetime import datetime
    
    return OptimizationState(
        # Inputs
        total_budget=total_budget,
        channels=channels,
        constraints=constraints,
        objective=objective,
        historical_data_path=historical_data_path,
        
        # Loaded data (populated by validator node)
        historical_data=None,
        data_quality_score=0.0,
        data_issues=[],
        
        # Analysis results (populated by analyzer node)
        performance_analysis={},
        channel_rankings=[],
        seasonality_factors={},
        market_conditions="",
        
        # Optimization state
        proposed_allocation={},
        allocation_rationale={},
        iteration_count=0,
        optimization_history=[],
        
        # Validation state
        validation_passed=False,
        validation_errors=[],
        constraint_violations=[],
        
        # Final outputs
        final_allocation={},
        expected_metrics={},
        explanation="",
        confidence_score=0.0,
        warnings=[],
        
        # Metadata
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),
        model_used="",
        execution_time_ms=0
    )


# Sample constraints for demonstration
SAMPLE_CONSTRAINTS = {
    "search": {"min": 0.15, "max": 0.40},
    "social": {"min": 0.10, "max": 0.30},
    "display": {"min": 0.05, "max": 0.20},
    "video": {"min": 0.10, "max": 0.25},
    "email": {"min": 0.05, "max": 0.15}
}

SAMPLE_CHANNELS = ["search", "social", "display", "video", "email"]
