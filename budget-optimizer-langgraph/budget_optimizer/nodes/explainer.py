"""
Explanation Generation Node

Generates human-readable explanations for the optimization results.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def generate_explanation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a clear explanation of the optimization results.
    
    Creates:
    - Executive summary
    - Channel-by-channel rationale
    - Key assumptions and caveats
    - Confidence assessment
    
    Args:
        state: Current optimization state with final allocation
        
    Returns:
        Updated state with explanation and final outputs
    """
    
    proposed_allocation = state.get("proposed_allocation", {})
    total_budget = state.get("total_budget", 0)
    performance_analysis = state.get("performance_analysis", {})
    allocation_rationale = state.get("allocation_rationale", {})
    expected_metrics = state.get("expected_metrics", {})
    channel_rankings = state.get("channel_rankings", [])
    market_conditions = state.get("market_conditions", "neutral")
    data_quality_score = state.get("data_quality_score", 0)
    warnings = state.get("warnings", [])
    
    # ===== Build Explanation =====
    explanation_parts = []
    
    # Executive Summary
    expected_roas = expected_metrics.get("expected_roas", 0)
    expected_revenue = expected_metrics.get("expected_revenue", 0)
    
    explanation_parts.append("## Executive Summary\n")
    explanation_parts.append(
        f"Recommended allocation of €{total_budget:,.0f} across {len(proposed_allocation)} channels, "
        f"with expected ROAS of {expected_roas:.1f}x (€{expected_revenue:,.0f} projected revenue).\n"
    )
    
    # Market context
    market_context = {
        "favorable": "Market conditions are favorable with improving channel performance.",
        "challenging": "Market conditions are challenging with declining performance trends.",
        "neutral": "Market conditions are stable."
    }.get(market_conditions, "Market conditions are uncertain.")
    explanation_parts.append(f"\n{market_context}\n")
    
    # Channel Breakdown
    explanation_parts.append("\n## Channel Allocation\n")
    
    # Sort by allocation amount
    sorted_channels = sorted(
        proposed_allocation.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for channel, amount in sorted_channels:
        share = amount / total_budget if total_budget > 0 else 0
        analysis = performance_analysis.get(channel, {})
        rationale = allocation_rationale.get(channel, "")
        
        roas = analysis.get("avg_roas", 0)
        trend = analysis.get("trend", "unknown")
        
        explanation_parts.append(f"\n### {channel.title()}: €{amount:,.0f} ({share:.0%})\n")
        explanation_parts.append(f"- Historical ROAS: {roas:.1f}x\n")
        explanation_parts.append(f"- Performance trend: {trend}\n")
        explanation_parts.append(f"- Rationale: {rationale}\n")
    
    # Confidence Assessment
    explanation_parts.append("\n## Confidence Assessment\n")
    
    # Calculate confidence score
    confidence_factors = []
    
    # Data quality factor
    confidence_factors.append(("Data quality", data_quality_score))
    
    # Iteration factor (more iterations = less confident)
    iteration_count = state.get("iteration_count", 1)
    iteration_confidence = max(0.5, 1 - (iteration_count - 1) * 0.15)
    confidence_factors.append(("Optimization stability", iteration_confidence))
    
    # Constraint factor (are we at bounds?)
    constraints = state.get("constraints", {})
    at_bounds_count = 0
    for channel, amount in proposed_allocation.items():
        share = amount / total_budget if total_budget > 0 else 0
        constraint = constraints.get(channel, {})
        if abs(share - constraint.get("min", 0)) < 0.01 or abs(share - constraint.get("max", 1)) < 0.01:
            at_bounds_count += 1
    
    bounds_confidence = 1 - (at_bounds_count / max(1, len(proposed_allocation))) * 0.3
    confidence_factors.append(("Constraint flexibility", bounds_confidence))
    
    # Market conditions factor
    market_confidence = {"favorable": 0.9, "neutral": 0.75, "challenging": 0.6}.get(market_conditions, 0.7)
    confidence_factors.append(("Market conditions", market_confidence))
    
    # Calculate overall confidence
    overall_confidence = sum(score for _, score in confidence_factors) / len(confidence_factors)
    
    explanation_parts.append(f"\nOverall confidence: {overall_confidence:.0%}\n")
    for factor, score in confidence_factors:
        explanation_parts.append(f"- {factor}: {score:.0%}\n")
    
    # Warnings and Caveats
    if warnings:
        explanation_parts.append("\n## Warnings & Caveats\n")
        for warning in warnings:
            explanation_parts.append(f"- {warning}\n")
    
    # Assumptions
    explanation_parts.append("\n## Key Assumptions\n")
    explanation_parts.append("- Historical performance is indicative of future results\n")
    explanation_parts.append("- Market conditions remain relatively stable\n")
    explanation_parts.append("- Diminishing returns apply at higher spend levels\n")
    explanation_parts.append("- Channel interactions (halo effects) are not modeled\n")
    
    explanation = "".join(explanation_parts)
    
    return {
        **state,
        "final_allocation": proposed_allocation,
        "explanation": explanation,
        "confidence_score": round(overall_confidence, 2)
    }


async def generate_explanation_with_llm(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    Generates a more sophisticated explanation using LLM.
    """
    
    # First generate structured explanation
    state = generate_explanation_node(state)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior marketing strategist presenting budget recommendations 
        to a CMO. Write a clear, compelling executive summary that:
        
        1. Leads with the key recommendation and expected impact
        2. Explains the strategic rationale (not just the numbers)
        3. Acknowledges uncertainties and risks
        4. Suggests next steps or monitoring approach
        
        Keep it concise (3-4 paragraphs) and avoid jargon."""),
        ("user", """
        Allocation: {final_allocation}
        Total Budget: {total_budget}
        Expected ROAS: {expected_roas}
        Expected Revenue: {expected_revenue}
        Market Conditions: {market_conditions}
        Top Channels by Performance: {channel_rankings}
        Confidence Score: {confidence_score}
        Warnings: {warnings}
        """)
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        llm_summary = await chain.ainvoke({
            "final_allocation": state["final_allocation"],
            "total_budget": state["total_budget"],
            "expected_roas": state["expected_metrics"].get("expected_roas", 0),
            "expected_revenue": state["expected_metrics"].get("expected_revenue", 0),
            "market_conditions": state.get("market_conditions", "neutral"),
            "channel_rankings": state.get("channel_rankings", [])[:3],
            "confidence_score": state.get("confidence_score", 0),
            "warnings": state.get("warnings", [])
        })
        
        # Prepend LLM summary to structured explanation
        state["explanation"] = f"# Executive Summary\n\n{llm_summary}\n\n---\n\n{state['explanation']}"
        
    except Exception as e:
        state["warnings"] = state.get("warnings", []) + [f"LLM explanation generation failed: {str(e)}"]
    
    return state
