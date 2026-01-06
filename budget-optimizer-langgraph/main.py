#!/usr/bin/env python3
"""
Budget Optimizer CLI

Command-line interface for running budget optimization.

Usage:
    python main.py --budget 1000000
    python main.py --budget 500000 --objective maximize_reach
    python main.py --budget 1000000 --data data/my_performance.csv
"""

import argparse
import json
from budget_optimizer import BudgetOptimizerGraph, SAMPLE_CONSTRAINTS, SAMPLE_CHANNELS


def main():
    parser = argparse.ArgumentParser(
        description="Optimize marketing budget allocation across channels"
    )
    
    parser.add_argument(
        "--budget",
        type=float,
        required=True,
        help="Total budget to allocate"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="",
        help="Path to historical performance CSV"
    )
    
    parser.add_argument(
        "--objective",
        type=str,
        default="maximize_roas",
        choices=["maximize_roas", "maximize_reach", "minimize_cpa", "balanced"],
        help="Optimization objective"
    )
    
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=SAMPLE_CHANNELS,
        help="Channels to optimize (default: search social display video email)"
    )
    
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM-enhanced optimization"
    )
    
    parser.add_argument(
        "--llm-provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider to use"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output file for results (JSON)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    # Build constraints for specified channels
    constraints = {}
    for channel in args.channels:
        if channel in SAMPLE_CONSTRAINTS:
            constraints[channel] = SAMPLE_CONSTRAINTS[channel]
        else:
            # Default constraints for unknown channels
            constraints[channel] = {"min": 0.05, "max": 0.30}
    
    print(f"\n{'='*60}")
    print("BUDGET OPTIMIZER")
    print(f"{'='*60}")
    print(f"Total Budget: €{args.budget:,.0f}")
    print(f"Channels: {', '.join(args.channels)}")
    print(f"Objective: {args.objective}")
    print(f"LLM Enhanced: {args.use_llm}")
    print(f"{'='*60}\n")
    
    # Initialize optimizer
    optimizer = BudgetOptimizerGraph(
        use_llm=args.use_llm,
        llm_provider=args.llm_provider
    )
    
    # Build request
    request = {
        "total_budget": args.budget,
        "channels": args.channels,
        "constraints": constraints,
        "objective": args.objective,
        "historical_data_path": args.data
    }
    
    # Run optimization with streaming output
    print("Running optimization...\n")
    
    if args.verbose:
        # Stream mode - show each step
        for event in optimizer.stream(request):
            for node_name, state in event.items():
                print(f"✓ {node_name}")
                if node_name == "validate_input":
                    if state.get("validation_errors"):
                        print(f"  Errors: {state['validation_errors']}")
                    else:
                        print(f"  Data quality: {state.get('data_quality_score', 0):.0%}")
                elif node_name == "analyze":
                    rankings = state.get("channel_rankings", [])
                    print(f"  Top channels: {', '.join(rankings[:3])}")
                elif node_name == "optimize":
                    print(f"  Iteration: {state.get('iteration_count', 0)}")
                elif node_name == "validate_output":
                    if state.get("validation_passed"):
                        print(f"  Validation passed")
                    else:
                        print(f"  Issues: {state.get('constraint_violations', [])}")
        print()
        result = state  # Last state is the result
    else:
        # Simple mode
        result = optimizer.run(request)
    
    # Display results
    print(f"{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")
    
    # Allocation table
    print("RECOMMENDED ALLOCATION:")
    print("-" * 40)
    
    final_allocation = result.get("final_allocation", {})
    for channel, amount in sorted(final_allocation.items(), key=lambda x: -x[1]):
        share = amount / args.budget * 100
        bar = "█" * int(share / 2)
        print(f"{channel:12} €{amount:>12,.0f}  ({share:>5.1f}%) {bar}")
    
    print("-" * 40)
    print(f"{'TOTAL':12} €{sum(final_allocation.values()):>12,.0f}")
    
    # Expected metrics
    print(f"\nEXPECTED METRICS:")
    metrics = result.get("expected_metrics", {})
    print(f"  ROAS: {metrics.get('expected_roas', 0):.1f}x")
    print(f"  Revenue: €{metrics.get('expected_revenue', 0):,.0f}")
    
    # Confidence
    print(f"\nCONFIDENCE: {result.get('confidence_score', 0):.0%}")
    
    # Warnings
    warnings = result.get("warnings", [])
    if warnings:
        print(f"\nWARNINGS:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    # Execution time
    print(f"\nExecution time: {result.get('execution_time_ms', 0)}ms")
    
    # Full explanation if verbose
    if args.verbose:
        print(f"\n{'='*60}")
        print("DETAILED EXPLANATION")
        print(f"{'='*60}")
        print(result.get("explanation", ""))
    
    # Save to file if requested
    if args.output:
        output_data = {
            "request": request,
            "allocation": final_allocation,
            "expected_metrics": metrics,
            "confidence_score": result.get("confidence_score", 0),
            "explanation": result.get("explanation", ""),
            "warnings": warnings,
            "execution_time_ms": result.get("execution_time_ms", 0)
        }
        
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
    
    print()


if __name__ == "__main__":
    main()
