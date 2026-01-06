"""
Performance Analysis Node

Analyzes historical data to inform optimization decisions.
"""

import pandas as pd
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


def analyze_performance_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes historical performance data to identify patterns and inform optimization.
    
    Analysis includes:
    - Channel performance metrics (ROAS, CPA, trends)
    - Seasonality patterns
    - Diminishing returns estimation
    - Market condition assessment
    
    Args:
        state: Current optimization state with loaded historical data
        
    Returns:
        Updated state with performance analysis results
    """
    
    historical_data = state.get("historical_data", [])
    channels = state.get("channels", [])
    
    if not historical_data:
        return {
            **state,
            "performance_analysis": {},
            "channel_rankings": channels,
            "seasonality_factors": {ch: 1.0 for ch in channels},
            "market_conditions": "insufficient_data",
            "warnings": state.get("warnings", []) + ["No historical data for analysis"]
        }
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(historical_data)
    
    # ===== Channel Performance Analysis =====
    performance_analysis = {}
    
    for channel in channels:
        channel_data = df[df["channel"] == channel]
        
        if len(channel_data) == 0:
            performance_analysis[channel] = {
                "status": "no_data",
                "avg_roas": 0,
                "trend": "unknown"
            }
            continue
        
        # Calculate key metrics
        avg_roas = channel_data["roas"].mean() if "roas" in channel_data.columns else \
                   (channel_data["revenue"].sum() / channel_data["spend"].sum())
        
        avg_cpa = channel_data["cpa"].mean() if "cpa" in channel_data.columns else \
                  (channel_data["spend"].sum() / max(1, channel_data["conversions"].sum()))
        
        total_spend = channel_data["spend"].sum()
        total_revenue = channel_data["revenue"].sum()
        
        # Calculate trend (compare last 30 days vs previous 30 days)
        if "date" in channel_data.columns:
            channel_data = channel_data.copy()
            channel_data["date"] = pd.to_datetime(channel_data["date"])
            channel_data = channel_data.sort_values("date")
            
            if len(channel_data) >= 60:
                recent = channel_data.tail(30)
                previous = channel_data.iloc[-60:-30]
                
                recent_roas = recent["revenue"].sum() / max(1, recent["spend"].sum())
                previous_roas = previous["revenue"].sum() / max(1, previous["spend"].sum())
                
                if recent_roas > previous_roas * 1.1:
                    trend = "improving"
                elif recent_roas < previous_roas * 0.9:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_history"
        else:
            trend = "no_dates"
        
        # Estimate diminishing returns
        # Simple approach: higher spend periods tend to have lower marginal ROAS
        if len(channel_data) >= 30:
            high_spend = channel_data.nlargest(int(len(channel_data) * 0.25), "spend")
            low_spend = channel_data.nsmallest(int(len(channel_data) * 0.25), "spend")
            
            high_roas = high_spend["revenue"].sum() / max(1, high_spend["spend"].sum())
            low_roas = low_spend["revenue"].sum() / max(1, low_spend["spend"].sum())
            
            diminishing_factor = high_roas / max(0.01, low_roas)
        else:
            diminishing_factor = 0.9  # Default assumption
        
        performance_analysis[channel] = {
            "avg_roas": round(avg_roas, 2),
            "avg_cpa": round(avg_cpa, 2),
            "total_spend": round(total_spend, 2),
            "total_revenue": round(total_revenue, 2),
            "trend": trend,
            "diminishing_factor": round(diminishing_factor, 2),
            "data_points": len(channel_data),
            "reliability": "high" if len(channel_data) >= 90 else "medium" if len(channel_data) >= 30 else "low"
        }
    
    # ===== Channel Rankings =====
    # Rank by efficiency (ROAS) weighted by reliability
    def channel_score(ch):
        analysis = performance_analysis.get(ch, {})
        roas = analysis.get("avg_roas", 0)
        reliability_weight = {"high": 1.0, "medium": 0.8, "low": 0.6}.get(
            analysis.get("reliability", "low"), 0.5
        )
        trend_weight = {"improving": 1.1, "stable": 1.0, "declining": 0.9}.get(
            analysis.get("trend", "stable"), 1.0
        )
        return roas * reliability_weight * trend_weight
    
    channel_rankings = sorted(channels, key=channel_score, reverse=True)
    
    # ===== Seasonality Analysis =====
    seasonality_factors = {}
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        
        # Calculate monthly indices relative to annual average
        for channel in channels:
            channel_data = df[df["channel"] == channel]
            if len(channel_data) < 30:
                seasonality_factors[channel] = 1.0
                continue
            
            monthly_roas = channel_data.groupby("month").apply(
                lambda x: x["revenue"].sum() / max(1, x["spend"].sum())
            )
            
            if len(monthly_roas) > 0:
                avg_roas = monthly_roas.mean()
                current_month = pd.Timestamp.now().month
                current_factor = monthly_roas.get(current_month, avg_roas) / max(0.01, avg_roas)
                seasonality_factors[channel] = round(current_factor, 2)
            else:
                seasonality_factors[channel] = 1.0
    else:
        seasonality_factors = {ch: 1.0 for ch in channels}
    
    # ===== Market Conditions Assessment =====
    # Simple heuristic based on overall trends
    trend_counts = {"improving": 0, "stable": 0, "declining": 0}
    for ch, analysis in performance_analysis.items():
        trend = analysis.get("trend", "stable")
        if trend in trend_counts:
            trend_counts[trend] += 1
    
    if trend_counts["improving"] > trend_counts["declining"]:
        market_conditions = "favorable"
    elif trend_counts["declining"] > trend_counts["improving"]:
        market_conditions = "challenging"
    else:
        market_conditions = "neutral"
    
    return {
        **state,
        "performance_analysis": performance_analysis,
        "channel_rankings": channel_rankings,
        "seasonality_factors": seasonality_factors,
        "market_conditions": market_conditions
    }


async def analyze_with_llm(state: Dict[str, Any], llm) -> Dict[str, Any]:
    """
    Enhanced analysis using LLM for deeper insights.
    
    This version uses an LLM to interpret the numerical analysis
    and provide strategic recommendations.
    """
    
    # First run numerical analysis
    state = analyze_performance_node(state)
    
    # Then enhance with LLM interpretation
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a marketing analytics expert. Analyze the following 
        performance data and provide strategic insights for budget optimization.
        
        Focus on:
        1. Which channels show the best efficiency and growth potential
        2. Any concerning trends that should influence allocation
        3. Seasonality considerations for the current period
        4. Risk factors in the data quality or market conditions
        
        Return your analysis as JSON with keys: key_insights, recommendations, risks"""),
        ("user", """
        Performance Analysis: {performance_analysis}
        Channel Rankings: {channel_rankings}
        Seasonality Factors: {seasonality_factors}
        Market Conditions: {market_conditions}
        Objective: {objective}
        """)
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        llm_analysis = await chain.ainvoke({
            "performance_analysis": state["performance_analysis"],
            "channel_rankings": state["channel_rankings"],
            "seasonality_factors": state["seasonality_factors"],
            "market_conditions": state["market_conditions"],
            "objective": state["objective"]
        })
        
        state["llm_insights"] = llm_analysis
    except Exception as e:
        state["warnings"] = state.get("warnings", []) + [f"LLM analysis failed: {str(e)}"]
    
    return state
