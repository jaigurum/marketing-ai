"""
Allocation Optimization Node

Core optimization logic combining algorithmic approaches with LLM reasoning.
"""

from typing import Dict, Any, List, Tuple
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def optimize_allocation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates optimal budget allocation based on performance analysis.
    
    Uses a hybrid approach:
    1. Algorithmic baseline based on historical ROAS and constraints
    2. Adjustments for seasonality and trends
    3. LLM reasoning for edge cases and strategic considerations
    
    Args:
        state: Current optimization state with performance analysis
        
    Returns:
        Updated state with proposed allocation
    """
    
    total_budget = state.get("total_budget", 0)
    channels = state.get("channels", [])
    constraints = state.get("constraints", {})
    objective = state.get("objective", "maximize_roas")
    performance_analysis = state.get("performance_analysis", {})
    seasonality_factors = state.get("seasonality_factors", {})
    channel_rankings = state.get("channel_rankings", channels)
    
    iteration_count = state.get("iteration_count", 0) + 1
    
    # ===== Step 1: Calculate Base Allocation =====
    # Start with ROAS-weighted allocation
    
    total_weighted_score = 0
    channel_scores = {}
    
    for channel in channels:
        analysis = performance_analysis.get(channel, {})
        
        # Base score from ROAS
        roas = analysis.get("avg_roas", 1.0)
        
        # Adjust for seasonality
        seasonality = seasonality_factors.get(channel, 1.0)
        
        # Adjust for trend
        trend = analysis.get("trend", "stable")
        trend_multiplier = {
            "improving": 1.15,
            "stable": 1.0,
            "declining": 0.85,
            "unknown": 0.9
        }.get(trend, 1.0)
        
        # Adjust for diminishing returns
        diminishing = analysis.get("diminishing_factor", 0.9)
        
        # Adjust for data reliability
        reliability = analysis.get("reliability", "low")
        reliability_weight = {"high": 1.0, "medium": 0.9, "low": 0.7}.get(reliability, 0.7)
        
        # Calculate weighted score based on objective
        if objective == "maximize_roas":
            score = roas * trend_multiplier * seasonality * reliability_weight
        elif objective == "maximize_reach":
            # Favor channels with lower CPA (more conversions per dollar)
            cpa = analysis.get("avg_cpa", 100)
            score = (1 / max(1, cpa)) * trend_multiplier * seasonality * reliability_weight * 1000
        elif objective == "minimize_cpa":
            cpa = analysis.get("avg_cpa", 100)
            score = (1 / max(1, cpa)) * reliability_weight * 1000
        else:  # balanced
            cpa = analysis.get("avg_cpa", 100)
            score = (roas * 0.5 + (100 / max(1, cpa)) * 0.5) * trend_multiplier * seasonality
        
        channel_scores[channel] = max(0.1, score)  # Ensure positive score
        total_weighted_score += channel_scores[channel]
    
    # Calculate initial proportional allocation
    base_allocation = {}
    for channel in channels:
        proportion = channel_scores[channel] / total_weighted_score
        base_allocation[channel] = proportion
    
    # ===== Step 2: Apply Constraints =====
    allocation = _apply_constraints(base_allocation, constraints, channels)
    
    # ===== Step 3: Convert to Absolute Values =====
    proposed_allocation = {}
    for channel, share in allocation.items():
        proposed_allocation[channel] = round(total_budget * share, 2)
    
    # ===== Step 4: Generate Rationale =====
    allocation_rationale = {}
    for channel in channels:
        analysis = performance_analysis.get(channel, {})
        constraint = constraints.get(channel, {})
        
        rationale_parts = []
        
        # Explain the allocation
        share = allocation[channel]
        if share == constraint.get("max"):
            rationale_parts.append(f"At maximum allowed ({share:.0%})")
        elif share == constraint.get("min"):
            rationale_parts.append(f"At minimum allowed ({share:.0%})")
        else:
            rationale_parts.append(f"Allocated {share:.0%} based on performance")
        
        # Add performance context
        roas = analysis.get("avg_roas", 0)
        trend = analysis.get("trend", "unknown")
        if roas > 0:
            rationale_parts.append(f"ROAS: {roas:.1f}x")
        if trend != "unknown":
            rationale_parts.append(f"Trend: {trend}")
        
        allocation_rationale[channel] = "; ".join(rationale_parts)
    
    # ===== Step 5: Calculate Expected Metrics =====
    expected_revenue = 0
    for channel, budget in proposed_allocation.items():
        analysis = performance_analysis.get(channel, {})
        roas = analysis.get("avg_roas", 1.0)
        diminishing = analysis.get("diminishing_factor", 0.9)
        
        # Apply diminishing returns for higher-than-historical spend
        historical_spend = analysis.get("total_spend", budget * 12) / 12  # Monthly average
        if budget > historical_spend:
            overspend_ratio = budget / max(1, historical_spend)
            adjusted_roas = roas * (diminishing ** (overspend_ratio - 1))
        else:
            adjusted_roas = roas
        
        expected_revenue += budget * adjusted_roas
    
    expected_roas = expected_revenue / total_budget if total_budget > 0 else 0
    
    # Store in optimization history
    optimization_history = state.get("optimization_history", [])
    optimization_history.append({
        "iteration": iteration_count,
        "allocation": proposed_allocation.copy(),
        "expected_roas": round(expected_roas, 2)
    })
    
    return {
        **state,
        "proposed_allocation": proposed_allocation,
        "allocation_rationale": allocation_rationale,
        "iteration_count": iteration_count,
        "optimization_history": optimization_history,
        "expected_metrics": {
            "expected_roas": round(expected_roas, 2),
            "expected_revenue": round(expected_revenue, 2),
            "total_budget": total_budget
        }
    }


def _apply_constraints(
    base_allocation: Dict[str, float],
    constraints: Dict[str, Dict[str, float]],
    channels: List[str]
) -> Dict[str, float]:
    """
    Adjusts allocation to satisfy min/max constraints while preserving ratios.
    
    Uses iterative adjustment:
    1. Clip values to min/max bounds
    2. Redistribute excess to channels with room
    3. Repeat until stable or max iterations
    """
    
    allocation = base_allocation.copy()
    max_iterations = 10
    
    for _ in range(max_iterations):
        # Calculate how much is allocated
        total_allocated = sum(allocation.values())
        
        # Track excess/deficit
        excess = 0
        channels_with_room_up = []
        channels_with_room_down = []
        
        # Apply bounds and track excess
        for channel in channels:
            constraint = constraints.get(channel, {"min": 0, "max": 1})
            min_share = constraint.get("min", 0)
            max_share = constraint.get("max", 1)
            
            current = allocation[channel]
            
            if current < min_share:
                excess -= (min_share - current)
                allocation[channel] = min_share
            elif current > max_share:
                excess += (current - max_share)
                allocation[channel] = max_share
            else:
                if current < max_share:
                    channels_with_room_up.append(channel)
                if current > min_share:
                    channels_with_room_down.append(channel)
        
        # Redistribute excess
        if abs(excess) < 0.001:
            break
        
        if excess > 0 and channels_with_room_up:
            # Distribute excess proportionally to channels with room
            distribution_per_channel = excess / len(channels_with_room_up)
            for channel in channels_with_room_up:
                constraint = constraints.get(channel, {"max": 1})
                max_allowed = constraint.get("max", 1)
                room = max_allowed - allocation[channel]
                allocation[channel] += min(distribution_per_channel, room)
        
        elif excess < 0 and channels_with_room_down:
            # Take from channels with room to give
            deficit = abs(excess)
            reduction_per_channel = deficit / len(channels_with_room_down)
            for channel in channels_with_room_down:
                constraint = constraints.get(channel, {"min": 0})
                min_allowed = constraint.get("min", 0)
                room = allocation[channel] - min_allowed
                allocation[channel] -= min(reduction_per_channel, room)
    
    # Normalize to ensure sum is exactly 1.0
    total = sum(allocation.values())
    if total > 0:
        allocation = {ch: v / total for ch, v in allocation.items()}
    
    return allocation


async def optimize_with_llm(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    Enhanced optimization using LLM for strategic adjustments.
    
    The LLM reviews the algorithmic allocation and can suggest
    adjustments based on qualitative factors.
    """
    
    # First run algorithmic optimization
    state = optimize_allocation_node(state)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior marketing strategist reviewing a budget allocation.
        
        Review the proposed allocation and suggest any adjustments based on:
        1. Strategic considerations not captured in the data
        2. Risk balancing across channels
        3. Market dynamics and competitive positioning
        4. Testing/learning budget for emerging channels
        
        Return JSON with:
        - approved: boolean (true if allocation looks good)
        - adjustments: dict of channel -> new_share (only if changes needed)
        - reasoning: string explaining your review
        
        Be conservative with adjustments - only change if there's a clear strategic reason."""),
        ("user", """
        Proposed Allocation: {proposed_allocation}
        Total Budget: {total_budget}
        Performance Analysis: {performance_analysis}
        Constraints: {constraints}
        Current Rationale: {allocation_rationale}
        Market Conditions: {market_conditions}
        """)
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        llm_review = await chain.ainvoke({
            "proposed_allocation": state["proposed_allocation"],
            "total_budget": state["total_budget"],
            "performance_analysis": state["performance_analysis"],
            "constraints": state["constraints"],
            "allocation_rationale": state["allocation_rationale"],
            "market_conditions": state.get("market_conditions", "neutral")
        })
        
        if not llm_review.get("approved", True) and llm_review.get("adjustments"):
            # Apply LLM adjustments
            adjustments = llm_review["adjustments"]
            for channel, new_share in adjustments.items():
                if channel in state["proposed_allocation"]:
                    state["proposed_allocation"][channel] = round(
                        state["total_budget"] * new_share, 2
                    )
                    state["allocation_rationale"][channel] += f" | LLM: {llm_review.get('reasoning', '')}"
        
        state["llm_review"] = llm_review
        
    except Exception as e:
        state["warnings"] = state.get("warnings", []) + [f"LLM optimization review failed: {str(e)}"]
    
    return state
