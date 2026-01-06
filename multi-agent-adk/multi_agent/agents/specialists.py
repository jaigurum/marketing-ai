"""
Specialist Agents

Domain-specific agents for marketing analysis.
"""

from typing import Dict, Any, List, Optional
from .base import BaseAgent, AgentRole, Tool, AgentResponse


# ============================================================
# SAMPLE DATA (In production: API calls)
# ============================================================

SAMPLE_PERFORMANCE_DATA = {
    "search": {
        "spend": 450000,
        "revenue": 2025000,
        "roas": 4.5,
        "conversions": 18500,
        "cpa": 24.32,
        "trend": "stable",
        "benchmark_roas": 4.0
    },
    "social": {
        "spend": 320000,
        "revenue": 1024000,
        "roas": 3.2,
        "conversions": 12800,
        "cpa": 25.00,
        "trend": "improving",
        "benchmark_roas": 2.8,
        "platforms": {
            "facebook": {"roas": 2.9, "trend": "declining"},
            "instagram": {"roas": 3.8, "trend": "improving"},
            "tiktok": {"roas": 2.1, "trend": "growing"}
        }
    },
    "display": {
        "spend": 180000,
        "revenue": 378000,
        "roas": 2.1,
        "conversions": 4200,
        "cpa": 42.86,
        "trend": "declining",
        "benchmark_roas": 2.5
    },
    "video": {
        "spend": 220000,
        "revenue": 616000,
        "roas": 2.8,
        "conversions": 5500,
        "cpa": 40.00,
        "trend": "stable",
        "benchmark_roas": 2.5
    },
    "email": {
        "spend": 45000,
        "revenue": 315000,
        "roas": 7.0,
        "conversions": 8400,
        "cpa": 5.36,
        "trend": "stable",
        "benchmark_roas": 6.0
    }
}

SAMPLE_AUDIENCE_DATA = {
    "segments": [
        {
            "name": "Health-conscious millennials",
            "size": 2500000,
            "conversion_share": 0.35,
            "roas": 4.2,
            "avg_order_value": 85,
            "ltv": 420
        },
        {
            "name": "Young parents",
            "size": 1800000,
            "conversion_share": 0.28,
            "roas": 3.8,
            "avg_order_value": 72,
            "ltv": 380
        },
        {
            "name": "Fitness enthusiasts",
            "size": 1200000,
            "conversion_share": 0.22,
            "roas": 3.5,
            "avg_order_value": 95,
            "ltv": 350
        },
        {
            "name": "Gen Z health seekers",
            "size": 3200000,
            "conversion_share": 0.08,
            "roas": 5.1,
            "avg_order_value": 45,
            "ltv": 280,
            "opportunity": "underserved"
        }
    ],
    "lookalikes": {
        "available": True,
        "seed_size": 150000,
        "expansion_potential": "3-5x"
    }
}

SAMPLE_COMPETITOR_DATA = {
    "market_share": {
        "our_brand": 0.24,
        "competitor_a": 0.31,
        "competitor_b": 0.22,
        "competitor_c": 0.15,
        "others": 0.08
    },
    "share_of_voice": {
        "search": {"our_brand": 0.28, "competitor_a": 0.35},
        "social": {"our_brand": 0.22, "competitor_a": 0.30},
        "display": {"our_brand": 0.18, "competitor_a": 0.25}
    },
    "positioning": {
        "our_brand": ["quality", "innovation", "sustainability"],
        "competitor_a": ["value", "variety", "convenience"],
        "competitor_b": ["premium", "expertise", "results"]
    }
}


# ============================================================
# PERFORMANCE ANALYST
# ============================================================

class PerformanceAnalyst(BaseAgent):
    """
    Specialist agent for campaign performance analysis.
    
    Analyzes ROAS, CPA, attribution, and identifies optimization opportunities.
    """
    
    name = "performance_analyst"
    description = "Analyzes campaign performance metrics including ROAS, CPA, and attribution"
    role = AgentRole.PERFORMANCE
    
    @property
    def system_prompt(self) -> str:
        return """You are a senior performance marketing analyst with expertise in:
- Campaign performance optimization
- ROAS and CPA analysis
- Multi-touch attribution
- Budget allocation efficiency
- Trend analysis and forecasting

When analyzing performance:
1. Always cite specific metrics with numbers
2. Compare to benchmarks and historical performance
3. Identify trends (improving, stable, declining)
4. Highlight anomalies or concerns
5. Provide actionable optimization recommendations

Focus on data-driven insights. Be specific about what's working and what needs improvement."""
    
    def _setup_tools(self):
        """Set up performance analysis tools"""
        self.tools = [
            Tool(
                name="get_channel_performance",
                description="Get performance metrics for a specific marketing channel",
                parameters={
                    "channel": {
                        "type": "string",
                        "description": "Channel name: search, social, display, video, email"
                    }
                },
                function=self._get_channel_performance
            ),
            Tool(
                name="compare_channels",
                description="Compare performance across multiple channels",
                parameters={
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of channels to compare"
                    }
                },
                function=self._compare_channels
            ),
            Tool(
                name="get_performance_summary",
                description="Get overall performance summary across all channels",
                parameters={},
                function=self._get_performance_summary
            )
        ]
    
    def _get_channel_performance(self, channel: str) -> Dict[str, Any]:
        """Get performance data for a channel"""
        data = SAMPLE_PERFORMANCE_DATA.get(channel.lower(), {})
        if not data:
            return {"error": f"Unknown channel: {channel}"}
        return {"channel": channel, **data}
    
    def _compare_channels(self, channels: List[str]) -> Dict[str, Any]:
        """Compare performance across channels"""
        comparison = {}
        for channel in channels:
            data = SAMPLE_PERFORMANCE_DATA.get(channel.lower(), {})
            if data:
                comparison[channel] = {
                    "roas": data.get("roas"),
                    "cpa": data.get("cpa"),
                    "trend": data.get("trend")
                }
        return comparison
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        total_spend = sum(d["spend"] for d in SAMPLE_PERFORMANCE_DATA.values())
        total_revenue = sum(d["revenue"] for d in SAMPLE_PERFORMANCE_DATA.values())
        
        return {
            "total_spend": total_spend,
            "total_revenue": total_revenue,
            "blended_roas": round(total_revenue / total_spend, 2),
            "top_performer": "email",
            "needs_attention": "display",
            "channels": list(SAMPLE_PERFORMANCE_DATA.keys())
        }
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate performance analysis without LLM"""
        summary = self._get_performance_summary()
        
        return f"""## Performance Analysis

**Overall Performance:**
- Total Spend: ${summary['total_spend']:,.0f}
- Total Revenue: ${summary['total_revenue']:,.0f}
- Blended ROAS: {summary['blended_roas']}x

**Channel Breakdown:**

| Channel | ROAS | CPA | Trend | vs Benchmark |
|---------|------|-----|-------|--------------|
| Search | 4.5x | $24.32 | Stable | +12% |
| Social | 3.2x | $25.00 | Improving | +14% |
| Display | 2.1x | $42.86 | Declining | -16% |
| Video | 2.8x | $40.00 | Stable | +12% |
| Email | 7.0x | $5.36 | Stable | +17% |

**Key Insights:**
1. Email shows highest efficiency (7.0x ROAS) but limited scale
2. Social is improving (+14% vs benchmark), driven by Instagram
3. Display underperforming (-16% vs benchmark) - needs optimization

**Recommendations:**
1. Shift 10% of Display budget to Social
2. Test new creative on Display to improve performance
3. Scale Email where possible within deliverability limits"""


# ============================================================
# AUDIENCE ANALYST
# ============================================================

class AudienceAnalyst(BaseAgent):
    """
    Specialist agent for audience analysis.
    
    Analyzes segments, targeting opportunities, and persona insights.
    """
    
    name = "audience_analyst"
    description = "Analyzes audience segments, targeting strategies, and persona development"
    role = AgentRole.AUDIENCE
    
    @property
    def system_prompt(self) -> str:
        return """You are a senior audience strategist with expertise in:
- Customer segmentation
- Targeting optimization
- Persona development
- Audience expansion strategies
- LTV and value-based targeting

When analyzing audiences:
1. Identify high-value segments with specific metrics
2. Highlight underserved opportunities
3. Consider segment overlap and exclusion
4. Recommend targeting refinements
5. Suggest expansion paths (lookalikes, broader targeting)

Focus on actionable audience insights that drive performance."""
    
    def _setup_tools(self):
        """Set up audience analysis tools"""
        self.tools = [
            Tool(
                name="get_segment_analysis",
                description="Get analysis of audience segments",
                parameters={},
                function=self._get_segment_analysis
            ),
            Tool(
                name="get_segment_details",
                description="Get detailed information about a specific segment",
                parameters={
                    "segment_name": {
                        "type": "string",
                        "description": "Name of the segment"
                    }
                },
                function=self._get_segment_details
            ),
            Tool(
                name="get_expansion_opportunities",
                description="Get audience expansion opportunities",
                parameters={},
                function=self._get_expansion_opportunities
            )
        ]
    
    def _get_segment_analysis(self) -> Dict[str, Any]:
        """Get segment overview"""
        return {
            "total_segments": len(SAMPLE_AUDIENCE_DATA["segments"]),
            "segments": SAMPLE_AUDIENCE_DATA["segments"],
            "lookalike_available": SAMPLE_AUDIENCE_DATA["lookalikes"]["available"]
        }
    
    def _get_segment_details(self, segment_name: str) -> Dict[str, Any]:
        """Get details for a specific segment"""
        for segment in SAMPLE_AUDIENCE_DATA["segments"]:
            if segment_name.lower() in segment["name"].lower():
                return segment
        return {"error": f"Segment not found: {segment_name}"}
    
    def _get_expansion_opportunities(self) -> Dict[str, Any]:
        """Get expansion opportunities"""
        underserved = [
            s for s in SAMPLE_AUDIENCE_DATA["segments"]
            if s.get("opportunity") == "underserved"
        ]
        return {
            "underserved_segments": underserved,
            "lookalike_potential": SAMPLE_AUDIENCE_DATA["lookalikes"]
        }
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate audience analysis without LLM"""
        segments = SAMPLE_AUDIENCE_DATA["segments"]
        
        return f"""## Audience Analysis

**Current Segment Performance:**

| Segment | Size | Conv Share | ROAS | LTV |
|---------|------|------------|------|-----|
| Health-conscious millennials | 2.5M | 35% | 4.2x | $420 |
| Young parents | 1.8M | 28% | 3.8x | $380 |
| Fitness enthusiasts | 1.2M | 22% | 3.5x | $350 |
| Gen Z health seekers | 3.2M | 8% | 5.1x | $280 |

**Key Insights:**

1. **Top Performer:** Health-conscious millennials drive 35% of conversions with 4.2x ROAS
2. **Opportunity:** Gen Z health seekers show highest ROAS (5.1x) but only 8% of spend
3. **Expansion:** Lookalike audiences available with 3-5x expansion potential

**Recommendations:**

1. **Scale Gen Z:** Increase Gen Z targeting by 3x - highest efficiency segment
2. **Platform Match:** Target Gen Z primarily on TikTok and Instagram
3. **Lookalike Expansion:** Test 1% lookalikes from top converters
4. **Suppress Low LTV:** Exclude segments with LTV < $200 from prospecting"""


# ============================================================
# COMPETITOR ANALYST
# ============================================================

class CompetitorAnalyst(BaseAgent):
    """
    Specialist agent for competitive analysis.
    
    Analyzes market share, share of voice, and competitive positioning.
    """
    
    name = "competitor_analyst"
    description = "Analyzes competitive landscape, market share, and positioning"
    role = AgentRole.COMPETITOR
    
    @property
    def system_prompt(self) -> str:
        return """You are a competitive intelligence analyst with expertise in:
- Market share analysis
- Share of voice tracking
- Competitive positioning
- Industry benchmarking
- Strategic recommendations

When analyzing competition:
1. Provide specific market share numbers
2. Compare share of voice across channels
3. Identify competitive threats and opportunities
4. Analyze positioning differences
5. Recommend differentiation strategies

Focus on actionable competitive insights."""
    
    def _setup_tools(self):
        """Set up competitive analysis tools"""
        self.tools = [
            Tool(
                name="get_market_share",
                description="Get market share data",
                parameters={},
                function=self._get_market_share
            ),
            Tool(
                name="get_share_of_voice",
                description="Get share of voice by channel",
                parameters={
                    "channel": {
                        "type": "string",
                        "description": "Channel to analyze (search, social, display)"
                    }
                },
                function=self._get_share_of_voice
            ),
            Tool(
                name="get_positioning_analysis",
                description="Get competitive positioning analysis",
                parameters={},
                function=self._get_positioning_analysis
            )
        ]
    
    def _get_market_share(self) -> Dict[str, Any]:
        """Get market share data"""
        return SAMPLE_COMPETITOR_DATA["market_share"]
    
    def _get_share_of_voice(self, channel: str) -> Dict[str, Any]:
        """Get share of voice for a channel"""
        sov = SAMPLE_COMPETITOR_DATA["share_of_voice"].get(channel.lower(), {})
        return {"channel": channel, "share_of_voice": sov}
    
    def _get_positioning_analysis(self) -> Dict[str, Any]:
        """Get positioning analysis"""
        return SAMPLE_COMPETITOR_DATA["positioning"]
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate competitive analysis without LLM"""
        market = SAMPLE_COMPETITOR_DATA["market_share"]
        
        return f"""## Competitive Analysis

**Market Share:**

| Brand | Share |
|-------|-------|
| Competitor A | 31% |
| Our Brand | 24% |
| Competitor B | 22% |
| Competitor C | 15% |
| Others | 8% |

**Share of Voice by Channel:**

| Channel | Our Brand | Competitor A | Gap |
|---------|-----------|--------------|-----|
| Search | 28% | 35% | -7% |
| Social | 22% | 30% | -8% |
| Display | 18% | 25% | -7% |

**Positioning Comparison:**
- **Our Brand:** Quality, Innovation, Sustainability
- **Competitor A:** Value, Variety, Convenience
- **Competitor B:** Premium, Expertise, Results

**Key Insights:**

1. Competitor A leads with 7-8% SOV advantage across channels
2. Our sustainability positioning is differentiated
3. Gap is largest in Social (-8%) - opportunity area

**Recommendations:**

1. **Close SOV Gap:** Increase Social spend to close 8% gap
2. **Leverage Differentiation:** Emphasize sustainability in messaging
3. **Defend Position:** Monitor Competitor B's premium push"""
