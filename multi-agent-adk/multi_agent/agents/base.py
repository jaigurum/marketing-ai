"""
Base Agent Class

Foundation for all specialist agents in the marketing analyst team.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class AgentRole(Enum):
    """Agent specialization roles"""
    COORDINATOR = "coordinator"
    AUDIENCE = "audience_analyst"
    PERFORMANCE = "performance_analyst"
    COMPETITOR = "competitor_analyst"
    CREATIVE = "creative_analyst"
    BUDGET = "budget_analyst"


@dataclass
class Tool:
    """Tool definition for agents"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: callable
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to function calling schema"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys())
                }
            }
        }
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool function"""
        return self.function(**kwargs)


@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_name: str
    role: AgentRole
    content: str
    tools_used: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "role": self.role.value,
            "content": self.content,
            "tools_used": self.tools_used,
            "data": self.data,
            "confidence": self.confidence
        }


class BaseAgent(ABC):
    """
    Abstract base class for marketing analyst agents.
    
    Each specialist agent inherits from this and implements
    domain-specific analysis capabilities.
    """
    
    name: str = "base_agent"
    description: str = "Base agent class"
    role: AgentRole = AgentRole.COORDINATOR
    
    def __init__(self, llm=None):
        """
        Initialize the agent.
        
        Args:
            llm: Language model to use (default: Gemini)
        """
        self.llm = llm
        self.tools: List[Tool] = []
        self.conversation_history: List[Dict[str, str]] = []
        self._setup_tools()
    
    @abstractmethod
    def _setup_tools(self):
        """Set up agent-specific tools. Override in subclasses."""
        pass
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining agent behavior. Override in subclasses."""
        pass
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get function calling schemas for all tools"""
        return [tool.to_schema() for tool in self.tools]
    
    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
    
    async def analyze(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Analyze a query using this agent's expertise.
        
        Args:
            query: The analysis request
            context: Additional context (client, region, etc.)
            
        Returns:
            AgentResponse with analysis results
        """
        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add context if provided
        if context:
            context_str = f"\nContext:\n{json.dumps(context, indent=2)}"
            messages[0]["content"] += context_str
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Get tool schemas
        tools = self.get_tool_schemas()
        
        # Call LLM (simplified - in production would use actual ADK)
        response_content, tools_used, data = await self._execute_with_tools(
            messages, tools
        )
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response_content})
        
        return AgentResponse(
            agent_name=self.name,
            role=self.role,
            content=response_content,
            tools_used=tools_used,
            data=data
        )
    
    async def _execute_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]]
    ) -> tuple[str, List[str], Dict[str, Any]]:
        """
        Execute LLM call with tool use.
        
        This is a simplified implementation. In production,
        this would use Google ADK's actual tool calling.
        """
        tools_used = []
        collected_data = {}
        
        if self.llm is None:
            # Fallback: Generate response without LLM
            return self._generate_fallback_response(messages[-1]["content"]), [], {}
        
        # In production: Use ADK's tool calling
        # response = await self.llm.generate(
        #     messages=messages,
        #     tools=tools,
        #     tool_choice="auto"
        # )
        
        # Simplified: Direct generation
        try:
            # Use the LLM to generate a response
            full_prompt = "\n".join([
                f"{m['role'].upper()}: {m['content']}"
                for m in messages
            ])
            
            response = await self.llm.ainvoke(full_prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            response_content = self._generate_fallback_response(messages[-1]["content"])
        
        return response_content, tools_used, collected_data
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate a response without LLM (for testing/demo)"""
        return f"[{self.name}] Analysis of: {query}\n\nThis is a fallback response. Configure an LLM for full functionality."
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, tools={len(self.tools)})>"
