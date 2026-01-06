"""
Output Validation Node

Validates the proposed allocation against constraints and business rules.
"""

from typing import Dict, Any, List


def validate_output_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the proposed allocation before finalizing.
    
    Checks:
    - All constraints are satisfied
    - Total allocation equals budget
    - No negative allocations
    - Business rule validation
    
    Args:
        state: Current optimization state with proposed allocation
        
    Returns:
        Updated state with validation results
    """
    
    proposed_allocation = state.get("proposed_allocation", {})
    total_budget = state.get("total_budget", 0)
    constraints = state.get("constraints", {})
    channels = state.get("channels", [])
    
    validation_errors = []
    constraint_violations = []
    warnings = state.get("warnings", [])
    
    # ===== Check Total Allocation =====
    total_allocated = sum(proposed_allocation.values())
    tolerance = total_budget * 0.001  # 0.1% tolerance
    
    if abs(total_allocated - total_budget) > tolerance:
        validation_errors.append(
            f"Total allocation ({total_allocated:,.2f}) does not match budget ({total_budget:,.2f})"
        )
    
    # ===== Check All Channels Present =====
    missing_channels = set(channels) - set(proposed_allocation.keys())
    if missing_channels:
        validation_errors.append(f"Missing allocations for channels: {missing_channels}")
    
    # ===== Check Constraints =====
    for channel, amount in proposed_allocation.items():
        # Negative check
        if amount < 0:
            validation_errors.append(f"Negative allocation for {channel}: {amount}")
            continue
        
        # Calculate share
        share = amount / total_budget if total_budget > 0 else 0
        
        # Get constraints
        constraint = constraints.get(channel, {})
        min_share = constraint.get("min", 0)
        max_share = constraint.get("max", 1)
        
        # Check bounds (with small tolerance for floating point)
        tolerance_pct = 0.001
        
        if share < min_share - tolerance_pct:
            constraint_violations.append(
                f"{channel}: {share:.1%} below minimum {min_share:.1%}"
            )
        
        if share > max_share + tolerance_pct:
            constraint_violations.append(
                f"{channel}: {share:.1%} above maximum {max_share:.1%}"
            )
    
    # ===== Business Rule Validation =====
    # Example: Ensure diversification (no single channel > 50%)
    max_allocation = max(proposed_allocation.values()) if proposed_allocation else 0
    if max_allocation > total_budget * 0.5:
        max_channel = max(proposed_allocation, key=proposed_allocation.get)
        warnings.append(
            f"High concentration risk: {max_channel} has {max_allocation/total_budget:.1%} of budget"
        )
    
    # Example: Ensure minimum testing budget for low-data channels
    performance_analysis = state.get("performance_analysis", {})
    for channel in channels:
        analysis = performance_analysis.get(channel, {})
        reliability = analysis.get("reliability", "low")
        allocation = proposed_allocation.get(channel, 0)
        share = allocation / total_budget if total_budget > 0 else 0
        
        # If low reliability but no test budget
        if reliability == "low" and share < 0.05:
            warnings.append(
                f"Consider allocating more to {channel} for learning (currently {share:.1%})"
            )
    
    # ===== Determine Validation Status =====
    # Hard failures
    validation_passed = len(validation_errors) == 0
    
    # Constraint violations are warnings if iteration count is low, errors if we've tried many times
    iteration_count = state.get("iteration_count", 0)
    if constraint_violations:
        if iteration_count < 3:
            # Treat as recoverable - will trigger reoptimization
            validation_passed = False
            warnings.extend([f"Constraint violation (attempt {iteration_count}): {v}" for v in constraint_violations])
        else:
            # After multiple attempts, accept with warnings
            validation_passed = True
            warnings.extend([f"Constraint violation (accepted): {v}" for v in constraint_violations])
    
    return {
        **state,
        "validation_passed": validation_passed,
        "validation_errors": validation_errors,
        "constraint_violations": constraint_violations,
        "warnings": warnings
    }


def should_reoptimize(state: Dict[str, Any]) -> str:
    """
    Routing function to determine next step based on validation results.
    
    Returns:
        "optimize" - Try optimization again
        "fallback" - Use rule-based fallback
        "explain" - Proceed to explanation (success)
    """
    
    validation_passed = state.get("validation_passed", False)
    validation_errors = state.get("validation_errors", [])
    iteration_count = state.get("iteration_count", 0)
    
    # Hard errors - need to fallback
    if validation_errors:
        return "fallback"
    
    # Validation passed - proceed to explanation
    if validation_passed:
        return "explain"
    
    # Soft failures - try again if we haven't too many times
    if iteration_count < 3:
        return "optimize"
    
    # Too many iterations - use fallback
    return "fallback"


def fallback_allocation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a safe, rule-based allocation when optimization fails.
    
    Uses midpoint of constraints as a conservative default.
    """
    
    total_budget = state.get("total_budget", 0)
    channels = state.get("channels", [])
    constraints = state.get("constraints", {})
    
    # Calculate midpoint allocation
    allocation = {}
    for channel in channels:
        constraint = constraints.get(channel, {"min": 0, "max": 1})
        min_share = constraint.get("min", 0)
        max_share = constraint.get("max", 1)
        midpoint = (min_share + max_share) / 2
        allocation[channel] = midpoint
    
    # Normalize to sum to 1
    total_share = sum(allocation.values())
    if total_share > 0:
        allocation = {ch: v / total_share for ch, v in allocation.items()}
    
    # Convert to absolute values
    final_allocation = {
        ch: round(total_budget * share, 2)
        for ch, share in allocation.items()
    }
    
    return {
        **state,
        "final_allocation": final_allocation,
        "expected_metrics": {
            "expected_roas": 0,  # Unknown for fallback
            "expected_revenue": 0,
            "total_budget": total_budget,
            "method": "fallback_midpoint"
        },
        "explanation": "Optimization could not find a valid solution within constraints. "
                       "Using conservative midpoint allocation as fallback. "
                       "Consider reviewing constraints for feasibility.",
        "confidence_score": 0.3,
        "warnings": state.get("warnings", []) + ["Using fallback allocation due to optimization failure"]
    }
