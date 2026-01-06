"""
Multi-Agent Marketing Analyst

A multi-agent system using Google ADK patterns for comprehensive marketing analysis.

Example:
    from multi_agent import MarketingAnalystTeam
    
    # Initialize team
    team = MarketingAnalystTeam()
    
    # Run analysis
    result = team.analyze_sync("How are our campaigns performing?")
    
    print(result.summary)
    print(result.recommendations)
    
    # Or async
    result = await team.analyze("Analyze Q4 performance and suggest improvements")
"""

from .team import MarketingAnalystTeam, CoordinatorAgent, TeamResponse
from .agents.base import BaseAgent, AgentRole, AgentResponse, Tool
from .agents.specialists import PerformanceAnalyst, AudienceAnalyst, CompetitorAnalyst

__version__ = "0.1.0"
__author__ = "Jaiguru Thevar"

__all__ = [
    "MarketingAnalystTeam",
    "CoordinatorAgent",
    "TeamResponse",
    "BaseAgent",
    "AgentRole",
    "AgentResponse",
    "Tool",
    "PerformanceAnalyst",
    "AudienceAnalyst",
    "CompetitorAnalyst"
]
