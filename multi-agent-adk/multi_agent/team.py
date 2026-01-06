"""
Marketing Analyst Team

Coordinator and team orchestration for multi-agent marketing analysis.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
import asyncio

from .agents.base import BaseAgent, AgentRole, AgentResponse
from .agents.specialists import PerformanceAnalyst, AudienceAnalyst, CompetitorAnalyst


@dataclass
class TeamResponse:
    """Response from the full analyst team"""
    query: str
    agents_consulted: List[str]
    agent_responses: Dict[str, AgentResponse]
    summary: str
    recommendations: List[str]
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "agents_consulted": self.agents_consulted,
            "agent_responses": {
                k: v.to_dict() for k, v in self.agent_responses.items()
            },
            "summary": self.summary,
            "recommendations": self.recommendations,
            "confidence": self.confidence
        }


class CoordinatorAgent(BaseAgent):
    """
    Coordinator agent that routes queries and synthesizes responses.
    
    The coordinator:
    1. Analyzes incoming queries to determine which specialists to consult
    2. Routes queries to appropriate specialist agents
    3. Synthesizes responses into a cohesive analysis
    """
    
    name = "coordinator"
    description = "Orchestrates analysis across specialist agents"
    role = AgentRole.COORDINATOR
    
    @property
    def system_prompt(self) -> str:
        return """You are a senior marketing strategy director coordinating a team of specialist analysts.

Your role:
1. Analyze incoming requests to determine which specialists to consult
2. Synthesize multiple perspectives into cohesive recommendations
3. Identify when specialists agree or disagree
4. Prioritize actionable recommendations

Available specialists:
- performance_analyst: Campaign metrics, ROAS, CPA, attribution
- audience_analyst: Segments, targeting, personas
- competitor_analyst: Market share, SOV, positioning

When synthesizing:
1. Lead with the key insight or recommendation
2. Support with specific data from specialists
3. Note any conflicting perspectives
4. Provide clear, prioritized action items"""
    
    def _setup_tools(self):
        """Coordinator doesn't have data tools - it orchestrates"""
        self.tools = []
    
    def determine_specialists(self, query: str) -> List[str]:
        """
        Determine which specialists should handle this query.
        
        Uses keyword matching (in production: LLM-based routing)
        """
        query_lower = query.lower()
        specialists = []
        
        # Performance keywords
        performance_keywords = [
            "performance", "roas", "roi", "cpa", "cpc", "ctr",
            "spend", "budget", "efficiency", "attribution",
            "campaign", "channel", "metrics"
        ]
        if any(kw in query_lower for kw in performance_keywords):
            specialists.append("performance_analyst")
        
        # Audience keywords
        audience_keywords = [
            "audience", "segment", "target", "persona", "customer",
            "who", "demographic", "behavior", "ltv", "value"
        ]
        if any(kw in query_lower for kw in audience_keywords):
            specialists.append("audience_analyst")
        
        # Competitor keywords
        competitor_keywords = [
            "competitor", "competition", "market share", "benchmark",
            "industry", "sov", "share of voice", "positioning"
        ]
        if any(kw in query_lower for kw in competitor_keywords):
            specialists.append("competitor_analyst")
        
        # Default: consult all if unclear
        if not specialists:
            specialists = ["performance_analyst", "audience_analyst"]
        
        return specialists
    
    def synthesize(
        self,
        query: str,
        agent_responses: Dict[str, AgentResponse]
    ) -> tuple[str, List[str]]:
        """
        Synthesize responses from multiple agents.
        
        Returns:
            Tuple of (summary, recommendations)
        """
        # In production: Use LLM to synthesize
        # Here: Rule-based synthesis for demo
        
        summary_parts = [f"## Analysis Summary\n\n**Query:** {query}\n"]
        recommendations = []
        
        # Add each agent's key points
        for agent_name, response in agent_responses.items():
            summary_parts.append(f"\n### {agent_name.replace('_', ' ').title()}\n")
            summary_parts.append(response.content[:500] + "..." if len(response.content) > 500 else response.content)
        
        # Extract recommendations (simplified)
        recommendations = [
            "Review and implement performance optimization recommendations",
            "Adjust audience targeting based on segment analysis",
            "Monitor competitive positioning and adjust SOV strategy"
        ]
        
        summary = "\n".join(summary_parts)
        
        return summary, recommendations


class MarketingAnalystTeam:
    """
    Multi-agent team for comprehensive marketing analysis.
    
    Usage:
        team = MarketingAnalystTeam()
        result = await team.analyze("How is our marketing performing?")
    """
    
    def __init__(self, llm=None):
        """
        Initialize the analyst team.
        
        Args:
            llm: Language model to use (shared across agents)
        """
        self.llm = llm
        
        # Initialize specialist agents
        self.specialists: Dict[str, BaseAgent] = {
            "performance_analyst": PerformanceAnalyst(llm=llm),
            "audience_analyst": AudienceAnalyst(llm=llm),
            "competitor_analyst": CompetitorAnalyst(llm=llm)
        }
        
        # Initialize coordinator
        self.coordinator = CoordinatorAgent(llm=llm)
    
    async def analyze(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        agents: Optional[List[str]] = None
    ) -> TeamResponse:
        """
        Run analysis across the specialist team.
        
        Args:
            query: Analysis request
            context: Additional context (client, region, etc.)
            agents: Specific agents to consult (default: auto-route)
            
        Returns:
            TeamResponse with synthesized analysis
        """
        # Determine which specialists to consult
        if agents:
            specialists_to_consult = agents
        else:
            specialists_to_consult = self.coordinator.determine_specialists(query)
        
        # Consult each specialist (in parallel in production)
        agent_responses: Dict[str, AgentResponse] = {}
        
        for agent_name in specialists_to_consult:
            if agent_name in self.specialists:
                agent = self.specialists[agent_name]
                response = await agent.analyze(query, context)
                agent_responses[agent_name] = response
        
        # Synthesize responses
        summary, recommendations = self.coordinator.synthesize(query, agent_responses)
        
        # Calculate overall confidence
        if agent_responses:
            avg_confidence = sum(r.confidence for r in agent_responses.values()) / len(agent_responses)
        else:
            avg_confidence = 0.5
        
        return TeamResponse(
            query=query,
            agents_consulted=specialists_to_consult,
            agent_responses=agent_responses,
            summary=summary,
            recommendations=recommendations,
            confidence=avg_confidence
        )
    
    def analyze_sync(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        agents: Optional[List[str]] = None
    ) -> TeamResponse:
        """Synchronous wrapper for analyze"""
        return asyncio.run(self.analyze(query, context, agents))
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get a specific agent by name"""
        return self.specialists.get(name)
    
    def list_agents(self) -> List[str]:
        """List available agents"""
        return list(self.specialists.keys())
    
    def clear_history(self):
        """Clear conversation history for all agents"""
        self.coordinator.clear_history()
        for agent in self.specialists.values():
            agent.clear_history()
