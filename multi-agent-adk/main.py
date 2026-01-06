#!/usr/bin/env python3
"""
Multi-Agent Marketing Analyst CLI

Command-line interface for the multi-agent marketing analysis system.

Usage:
    python main.py analyze "How are our campaigns performing?"
    python main.py analyze "Who should we target?" --agent audience
    python main.py chat
"""

import argparse
import asyncio
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from multi_agent import MarketingAnalystTeam


console = Console()


def cmd_analyze(args):
    """Run analysis on a query"""
    team = MarketingAnalystTeam()
    
    # Determine agents
    agents = [args.agent] if args.agent else None
    
    console.print(f"\n[bold]Query:[/bold] {args.query}\n")
    
    with console.status("Consulting specialists..."):
        result = team.analyze_sync(args.query, agents=agents)
    
    # Show which agents were consulted
    console.print(f"[dim]Agents consulted: {', '.join(result.agents_consulted)}[/dim]\n")
    
    # Show individual agent responses if verbose
    if args.verbose:
        for agent_name, response in result.agent_responses.items():
            console.print(Panel(
                Markdown(response.content),
                title=f"[blue]{agent_name.replace('_', ' ').title()}[/blue]",
                border_style="blue"
            ))
    
    # Show synthesized summary
    console.print(Panel(
        Markdown(result.summary),
        title="[green]Synthesized Analysis[/green]",
        border_style="green"
    ))
    
    # Show recommendations
    if result.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, rec in enumerate(result.recommendations, 1):
            console.print(f"  {i}. {rec}")
    
    # Show confidence
    console.print(f"\n[dim]Confidence: {result.confidence:.0%}[/dim]")


def cmd_chat(args):
    """Interactive chat mode"""
    team = MarketingAnalystTeam()
    
    console.print(Panel(
        f"[bold]Marketing Analyst Team[/bold]\n\n"
        f"Available specialists:\n"
        f"  • performance_analyst - ROAS, CPA, attribution\n"
        f"  • audience_analyst - Segments, targeting\n"
        f"  • competitor_analyst - Market share, SOV\n\n"
        f"Commands: 'quit', 'agents', 'clear'\n",
        border_style="blue"
    ))
    
    while True:
        try:
            query = console.input("\n[bold blue]You:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            break
        
        if not query.strip():
            continue
        
        if query.lower() == "quit":
            break
        
        if query.lower() == "agents":
            console.print(f"Available agents: {', '.join(team.list_agents())}")
            continue
        
        if query.lower() == "clear":
            team.clear_history()
            console.print("[yellow]Conversation history cleared[/yellow]")
            continue
        
        with console.status("Consulting specialists..."):
            result = team.analyze_sync(query)
        
        console.print(f"\n[dim]Consulted: {', '.join(result.agents_consulted)}[/dim]")
        console.print(f"\n[bold green]Team Analysis:[/bold green]")
        console.print(Markdown(result.summary))
        
        if result.recommendations:
            console.print("\n[bold]Quick Actions:[/bold]")
            for rec in result.recommendations[:3]:
                console.print(f"  → {rec}")


def cmd_agents(args):
    """Show available agents"""
    team = MarketingAnalystTeam()
    
    table = Table(title="Available Marketing Analysts")
    table.add_column("Agent", style="cyan")
    table.add_column("Role", style="green")
    table.add_column("Description")
    
    for name, agent in team.specialists.items():
        table.add_row(
            name,
            agent.role.value,
            agent.description
        )
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Marketing Analyst System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis")
    analyze_parser.add_argument("query", type=str, help="Analysis query")
    analyze_parser.add_argument(
        "--agent",
        type=str,
        choices=["performance_analyst", "audience_analyst", "competitor_analyst"],
        help="Specific agent to consult"
    )
    analyze_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show individual agent responses"
    )
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    
    # Agents command
    agents_parser = subparsers.add_parser("agents", help="List available agents")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "agents":
        cmd_agents(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
