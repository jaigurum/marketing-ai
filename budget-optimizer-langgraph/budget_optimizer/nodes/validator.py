"""
Input Validation Node

Validates inputs and loads historical data before optimization begins.
"""

import pandas as pd
from typing import Dict, Any, List
import os


def validate_input_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates optimization inputs and loads historical data.
    
    Checks:
    - Budget is positive
    - Channels list is not empty
    - Constraints are valid (min <= max, sum of mins <= 1, sum of maxs >= 1)
    - Historical data file exists and has required columns
    
    Args:
        state: Current optimization state
        
    Returns:
        Updated state with loaded data and validation results
    """
    
    errors = []
    warnings = []
    
    # ===== Validate Budget =====
    total_budget = state.get("total_budget", 0)
    if total_budget <= 0:
        errors.append(f"Total budget must be positive, got {total_budget}")
    elif total_budget < 10000:
        warnings.append(f"Budget of {total_budget} is quite small for meaningful optimization")
    
    # ===== Validate Channels =====
    channels = state.get("channels", [])
    if not channels:
        errors.append("At least one channel must be specified")
    
    # ===== Validate Constraints =====
    constraints = state.get("constraints", {})
    
    # Check all channels have constraints
    for channel in channels:
        if channel not in constraints:
            errors.append(f"Missing constraints for channel: {channel}")
    
    # Check constraint validity
    min_sum = 0
    max_sum = 0
    
    for channel, constraint in constraints.items():
        min_share = constraint.get("min", 0)
        max_share = constraint.get("max", 1)
        
        if min_share < 0 or min_share > 1:
            errors.append(f"Invalid min constraint for {channel}: {min_share}")
        if max_share < 0 or max_share > 1:
            errors.append(f"Invalid max constraint for {channel}: {max_share}")
        if min_share > max_share:
            errors.append(f"Min > max for {channel}: {min_share} > {max_share}")
        
        min_sum += min_share
        max_sum += max_share
    
    if min_sum > 1.0:
        errors.append(f"Sum of minimum constraints ({min_sum:.2f}) exceeds 100%")
    if max_sum < 1.0:
        errors.append(f"Sum of maximum constraints ({max_sum:.2f}) is less than 100%")
    
    # ===== Load Historical Data =====
    historical_data = None
    data_quality_score = 0.0
    data_issues = []
    
    data_path = state.get("historical_data_path", "")
    
    if data_path and os.path.exists(data_path):
        try:
            df = pd.read_csv(data_path)
            
            # Check required columns
            required_columns = ["channel", "spend", "revenue", "conversions"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                errors.append(f"Missing required columns in data: {missing_columns}")
            else:
                # Validate data quality
                quality_checks = []
                
                # Check for nulls
                null_pct = df[required_columns].isnull().sum().sum() / (len(df) * len(required_columns))
                if null_pct > 0.1:
                    data_issues.append(f"High null rate: {null_pct:.1%}")
                quality_checks.append(1 - null_pct)
                
                # Check for negative values
                negative_spend = (df["spend"] < 0).sum()
                if negative_spend > 0:
                    data_issues.append(f"{negative_spend} rows with negative spend")
                quality_checks.append(1 - (negative_spend / len(df)))
                
                # Check channel coverage
                data_channels = set(df["channel"].unique())
                missing_channels = set(channels) - data_channels
                if missing_channels:
                    data_issues.append(f"No data for channels: {missing_channels}")
                quality_checks.append(len(data_channels & set(channels)) / len(channels))
                
                # Check data recency (assuming date column exists)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    latest_date = df["date"].max()
                    days_old = (pd.Timestamp.now() - latest_date).days
                    if days_old > 90:
                        data_issues.append(f"Data is {days_old} days old")
                    quality_checks.append(max(0, 1 - (days_old / 365)))
                
                data_quality_score = sum(quality_checks) / len(quality_checks)
                
                # Convert to serializable format
                historical_data = df.to_dict(orient="records")
                
        except Exception as e:
            errors.append(f"Error loading historical data: {str(e)}")
    else:
        # Generate sample data if no file provided
        warnings.append("No historical data file found, using sample data")
        historical_data = _generate_sample_data(channels)
        data_quality_score = 0.7
        data_issues.append("Using synthetic sample data")
    
    # ===== Return Updated State =====
    return {
        **state,
        "historical_data": historical_data,
        "data_quality_score": data_quality_score,
        "data_issues": data_issues,
        "validation_errors": errors,
        "warnings": state.get("warnings", []) + warnings,
        "validation_passed": len(errors) == 0
    }


def _generate_sample_data(channels: List[str]) -> List[Dict[str, Any]]:
    """Generate sample historical data for demonstration"""
    import random
    from datetime import datetime, timedelta
    
    data = []
    base_date = datetime.now() - timedelta(days=365)
    
    # Channel-specific performance characteristics
    channel_profiles = {
        "search": {"base_roas": 4.5, "volatility": 0.2},
        "social": {"base_roas": 3.2, "volatility": 0.3},
        "display": {"base_roas": 2.1, "volatility": 0.25},
        "video": {"base_roas": 2.8, "volatility": 0.35},
        "email": {"base_roas": 6.0, "volatility": 0.15}
    }
    
    for i in range(365):
        date = base_date + timedelta(days=i)
        
        for channel in channels:
            profile = channel_profiles.get(channel, {"base_roas": 3.0, "volatility": 0.25})
            
            # Add seasonality
            seasonality = 1 + 0.3 * (1 if date.month in [11, 12] else 0)  # Q4 bump
            
            spend = random.uniform(5000, 50000) * seasonality
            roas = profile["base_roas"] * (1 + random.uniform(-1, 1) * profile["volatility"])
            revenue = spend * roas
            cpa = spend / max(1, revenue / 100)  # Assume $100 avg order value
            
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "channel": channel,
                "spend": round(spend, 2),
                "revenue": round(revenue, 2),
                "conversions": int(revenue / 100),
                "impressions": int(spend * random.uniform(50, 200)),
                "clicks": int(spend * random.uniform(0.5, 2)),
                "roas": round(roas, 2),
                "cpa": round(cpa, 2)
            })
    
    return data
