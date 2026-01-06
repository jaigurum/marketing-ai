#!/usr/bin/env python3
"""
Marketing RAG CLI

Command-line interface for the Marketing RAG system.

Usage:
    python main.py ingest --path data/reports/
    python main.py query "What was our Q4 ROAS?"
    python main.py chat
"""

import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from marketing_rag import MarketingRAG


console = Console()


def cmd_ingest(args):
    """Ingest documents into the RAG system."""
    rag = MarketingRAG(persist_directory=args.db)
    
    if args.file:
        chunks = rag.ingest_file(args.file)
        console.print(f"[green]✓ Ingested {chunks} chunks from {args.file}[/green]")
    elif args.path:
        chunks = rag.ingest_documents(args.path)
        console.print(f"[green]✓ Ingested {chunks} total chunks from {args.path}[/green]")
    else:
        console.print("[red]Error: Specify --file or --path[/red]")


def cmd_query(args):
    """Run a single query."""
    rag = MarketingRAG(
        persist_directory=args.db,
        llm_provider=args.llm
    )
    
    stats = rag.get_stats()
    if stats["total_documents"] == 0:
        console.print("[yellow]Warning: No documents in database. Run 'ingest' first.[/yellow]\n")
    
    console.print(f"\n[bold]Question:[/bold] {args.question}\n")
    
    with console.status("Thinking..."):
        response = rag.query(args.question, top_k=args.top_k)
    
    # Display answer
    console.print(Panel(
        Markdown(response.content),
        title="Answer",
        border_style="green"
    ))
    
    # Display sources
    if args.sources:
        console.print("\n[bold]Sources:[/bold]")
        for i, source in enumerate(response.sources, 1):
            console.print(f"  {i}. {source['file']}")
            if source.get('channels'):
                console.print(f"     Channels: {', '.join(source['channels'])}")
    
    # Display confidence
    console.print(f"\n[dim]Confidence: {response.confidence:.0%}[/dim]")


def cmd_chat(args):
    """Interactive chat mode."""
    rag = MarketingRAG(
        persist_directory=args.db,
        llm_provider=args.llm
    )
    
    stats = rag.get_stats()
    
    console.print(Panel(
        f"[bold]Marketing RAG Chat[/bold]\n\n"
        f"Documents: {stats['total_documents']} chunks\n"
        f"Type 'quit' to exit, 'stats' for info, 'clear' to reset\n",
        border_style="blue"
    ))
    
    while True:
        try:
            question = console.input("\n[bold blue]You:[/bold blue] ")
        except (KeyboardInterrupt, EOFError):
            break
        
        if not question.strip():
            continue
        
        if question.lower() == "quit":
            break
        
        if question.lower() == "stats":
            stats = rag.get_stats()
            table = Table(title="RAG Statistics")
            table.add_column("Property")
            table.add_column("Value")
            for key, value in stats.items():
                table.add_row(key, str(value))
            console.print(table)
            continue
        
        if question.lower() == "clear":
            if console.input("Clear all documents? [y/N] ").lower() == "y":
                rag.clear()
                console.print("[yellow]Database cleared[/yellow]")
            continue
        
        with console.status("Thinking..."):
            response = rag.query(question)
        
        console.print(f"\n[bold green]Assistant:[/bold green]")
        console.print(Markdown(response.content))
        console.print(f"\n[dim]({response.confidence:.0%} confidence, {len(response.sources)} sources)[/dim]")


def cmd_stats(args):
    """Show database statistics."""
    rag = MarketingRAG(persist_directory=args.db)
    stats = rag.get_stats()
    
    table = Table(title="Marketing RAG Statistics")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        table.add_row(key, str(value))
    
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Marketing RAG System - Q&A over marketing documents"
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default="./chroma_db",
        help="Path to vector database"
    )
    
    parser.add_argument(
        "--llm",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--path", type=str, help="Directory to ingest")
    ingest_parser.add_argument("--file", type=str, help="Single file to ingest")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a query")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    query_parser.add_argument("--sources", action="store_true", help="Show sources")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
