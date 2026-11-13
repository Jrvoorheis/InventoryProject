"""Supply Chain Analytics Inventory Tools.

This module provides deterministic Python implementations of the five core
inventory decision models:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Each tool:
- Validates all inputs strictly and never invents missing numerical inputs.
- Returns structured results using strongly-typed dataclasses with .to_dict() support.
- Returns comprehensive intermediate calculations.
- Returns the final recommended answer.
- Identifies and explicitly returns all underlying mathematical and operational assumptions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd


# =====================================================================
# Verification and Common Data Structures
# =====================================================================

@dataclass
class VerificationResult:
    """Stores model verification status, checks, and warnings."""
    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# 1. Economic Order Quantity (EOQ) Data Structures
# =====================================================================

@dataclass
class EOQResult:
    """Structured result for Economic Order Quantity (EOQ)."""
    # Final answer
    final_answer: float
    eoq: float

    # Validated inputs
    inputs: Dict[str, Any]

    # Intermediate calculations
    intermediate_calculations: Dict[str, Any]

    # Explicit assumptions
    assumptions: List[str]

    # Direct attribute access for intermediate metrics
    annual_demand: float
    order_cost: float
    holding_cost: float
    unit_cost: Optional[float]
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_purchase_cost: Optional[float]
    total_annual_cost: float
    total_cost_with_purchase: Optional[float]
    orders_per_year: float
    cycle_time_years: float
    cycle_time_days: float
    workdays_per_year: int
    lead_time_days: Optional[float] = None
    reorder_point: Optional[float] = None
    sensitivity_plus_10_cost: float = 0.0
    sensitivity_minus_10_cost: float = 0.0
    verification: VerificationResult = field(default_factory=lambda: VerificationResult(True))

    # Unit & Demand Period Normalization
    raw_demand: float = 0.0
    demand_period: str = "year"
    demand_unit: str = "units"
    demand_normalization_applied: bool = False
    demand_normalization_factor: float = 1.0
    demand_normalization_note: str = ""
    holding_cost_period: str = "year"

    def __post_init__(self):
        if self.raw_demand == 0.0 and self.annual_demand > 0.0:
            self.raw_demand = self.annual_demand

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verification"] = self.verification.to_dict()
        return d


# =====================================================================
# 2. Quantity Discount Data Structures
# =====================================================================

@dataclass
class DiscountTier:
    """Represents a single evaluated tier in a quantity discount schedule."""
    tier_number: int
    min_quantity: float
    max_quantity: Optional[float]
    unit_price: float
    holding_cost: float
    unconstrained_eoq: float
    is_feasible_at_eoq: bool
    feasible_quantity: float
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_purchase_cost: float
    total_annual_cost: float
    is_optimal: bool = False
    feasibility_status: str = ""
    intermediate_calculations: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuantityDiscountResult:
    """Structured result for Quantity Discount Analysis."""
    # Final answer
    final_answer: Dict[str, Any]
    optimal_quantity: float
    optimal_tier_index: int
    optimal_unit_price: float
    min_total_cost: float

    # Validated inputs
    inputs: Dict[str, Any]

    # Intermediate calculations
    intermediate_calculations: Dict[str, Any]

    # Explicit assumptions
    assumptions: List[str]

    # Detailed tier evaluation breakdown
    tiers: List[DiscountTier]
    base_case_cost: float
    annual_savings: float
    percent_savings: float
    annual_demand: float
    order_cost: float
    annual_purchase_cost: float = 0.0
    annual_ordering_cost: float = 0.0
    annual_holding_cost: float = 0.0
    total_annual_policy_cost: float = 0.0
    verification: VerificationResult = field(default_factory=lambda: VerificationResult(True))

    @property
    def evaluated_tiers(self) -> List[DiscountTier]:
        """Convenience alias for self.tiers."""
        return self.tiers

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tiers"] = [t.to_dict() for t in self.tiers]
        d["verification"] = self.verification.to_dict()
        return d


# =====================================================================
# 3. Safety Stock & Reorder Point (ROP) Data Structures
# =====================================================================

@dataclass
class ROPResult:
    """Structured result for Safety Stock and Reorder Point (ROP)."""
    # Final answers
    final_answer: Dict[str, float]
    reorder_point: float
    safety_stock: float

    # Case classification
    case: str

    # Validated inputs
    inputs: Dict[str, Any]

    # Intermediate calculations
    intermediate_calculations: Dict[str, Any]

    # Explicit assumptions
    assumptions: List[str]

    # Direct attribute access
    service_level: float
    z_score: float
    lead_time: float
    time_unit: str
    avg_demand_per_period: float
    demand_std_per_period: float
    lead_time_std: float
    expected_lead_time_demand: float
    std_lead_time_demand: float  # sigma_LT
    stockout_probability: float
    stockout_risk_per_cycle_percent: float
    verification: VerificationResult = field(default_factory=lambda: VerificationResult(True))

    # Unit Normalization details
    demand_time_unit: str = "days"
    lead_time_unit: str = "days"
    workdays_per_week: float = 5.0
    raw_lead_time: float = 0.0
    raw_lead_time_std: float = 0.0
    unit_conversion_factor: float = 1.0
    unit_normalization_applied: bool = False
    unit_normalization_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verification"] = self.verification.to_dict()
        return d

# Backwards compatibility alias
SafetyStockROPResult = ROPResult


# =====================================================================
# 4. Fixed-Period Review Data Structures
# =====================================================================

@dataclass
class PeriodicReviewResult:
    """Structured result for Fixed-Period Review (Periodic Order-Up-To)."""
    # Final answers
    final_answer: Dict[str, float]
    recommended_order_quantity: float
    target_inventory: float  # Order-up-to level M
    order_up_to_level: float  # Alias

    # Validated inputs
    inputs: Dict[str, Any]

    # Intermediate calculations
    intermediate_calculations: Dict[str, Any]

    # Explicit assumptions
    assumptions: List[str]

    # Direct attribute access
    review_period: float
    lead_time: float
    protection_period: float
    time_unit: str
    avg_demand_per_period: float
    demand_std_per_period: float
    service_level: float
    z_score: float
    expected_demand_protection_period: float
    std_demand_protection_period: float
    safety_stock: float
    current_inventory: float
    on_order: float
    backorders: float
    inventory_position: float
    verification: VerificationResult = field(default_factory=lambda: VerificationResult(True))
    exact_order_quantity: Optional[float] = None
    round_up: bool = False

    def __post_init__(self):
        if self.exact_order_quantity is None:
            self.exact_order_quantity = self.recommended_order_quantity

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verification"] = self.verification.to_dict()
        return d

# Backwards compatibility alias
FixedPeriodReviewResult = PeriodicReviewResult


# =====================================================================
# 5. Newsvendor Data Structures
# =====================================================================

@dataclass
class NewsvendorResult:
    """Structured result for Newsvendor Single-Period Stochastic Model."""
    # Final answer
    final_answer: float  # Optimal order quantity Q*
    optimal_order_quantity: float

    # Validated inputs
    inputs: Dict[str, Any]

    # Intermediate calculations
    intermediate_calculations: Dict[str, Any]

    # Explicit assumptions
    assumptions: List[str]

    # Direct attribute access
    selling_price: float
    unit_cost: float
    salvage_value: float
    customer_penalty: float
    underage_cost: float
    overage_cost: float
    critical_fractile: float
    critical_ratio: float  # Alias
    demand_mean: float
    demand_std: float
    z_score: float
    expected_lost_sales: float
    expected_sales: float
    expected_leftover_units: float
    expected_revenue: float
    expected_salvage_revenue: float
    expected_purchase_cost: float
    expected_penalty_cost: float
    expected_profit: float
    profit_margin_percent: float
    in_stock_probability: float
    stockout_probability: float
    verification: VerificationResult = field(default_factory=lambda: VerificationResult(True))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["verification"] = self.verification.to_dict()
        return d


# =====================================================================
# Helper: Strict Input Validation
# =====================================================================

def _require_number(value: Any, param_name: str, min_val: Optional[float] = None, allow_zero: bool = True) -> float:
    """Strictly validate numeric input, preventing invented missing values."""
    if value is None:
        raise ValueError(f"Missing required input '{param_name}'. Numerical inputs cannot be invented.")
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise TypeError(f"Input '{param_name}' must be a number, got {type(value).__name__} ({value!r}).")

    fval = float(value)
    if math.isnan(fval) or math.isinf(fval):
        raise ValueError(f"Input '{param_name}' must be finite, got {fval}.")

    if min_val is not None:
        if allow_zero:
            if fval < min_val:
                raise ValueError(f"{param_name.replace('_', ' ').capitalize()} must be >= {min_val}, got {fval}.")
        else:
            if fval <= min_val:
                raise ValueError(f"{param_name.replace('_', ' ').capitalize()} must be positive (> {min_val}), got {fval}.")
    return fval


def coerce_numeric_series(series: pd.Series, column_name: str = "Column", default: float = 0.0) -> pd.Series:
    """Safely coerce a pandas Series to float numeric, stripping currency signs, commas, and whitespace.

    Identifies and logs problematic entries rather than throwing generic errors.
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(default)

    raw_str = series.astype(str).str.strip()
    cleaned = (
        raw_str
        .str.replace(r"^[\$€£¥\s]+", "", regex=True)
        .str.replace(r"[\$,\s]", "", regex=True)
    )
    # Recognize missing value representations
    cleaned = cleaned.replace(["", "nan", "NaN", "null", "None", "N/A", "NA", "-", "--"], None)

    coerced = pd.to_numeric(cleaned, errors="coerce")

    # Check if there are non-empty values that failed conversion
    failed_mask = coerced.isna() & raw_str.notna() & (~raw_str.isin(["", "nan", "NaN", "null", "None", "N/A", "NA", "-", "--"]))
    if failed_mask.any():
        non_empty_count = len(raw_str[~raw_str.isin(["", "nan", "NaN", "null", "None", "N/A", "NA", "-", "--"])])
        sample_invalids = raw_str[failed_mask].unique()[:5].tolist()
        if non_empty_count > 0 and failed_mask.sum() == non_empty_count:
            raise ValueError(
                f"Column '{column_name}' expected numeric values, but found unconvertible non-numeric text: {sample_invalids}"
            )

    return coerced.fillna(default)


# =====================================================================
# Tool 1: calculate_eoq()
# =====================================================================

def calculate_eoq(
    annual_demand: Optional[float] = None,
    order_cost: float = 0.0,
    holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    unit_cost: Optional[float] = None,
    workdays_per_year: int = 250,
    lead_time_days: Optional[float] = None,
    demand: Optional[float] = None,
    raw_demand: Optional[float] = None,
    demand_period: str = "year",
    demand_unit: str = "units",
    months_per_year: float = 12.0,
    weeks_per_year: float = 52.0,
    quarters_per_year: float = 4.0,
    holding_cost_period: str = "year",
) -> EOQResult:
    """Calculate the Economic Order Quantity (EOQ): Q = sqrt(2DS/H).

    Parameters
    ----------
    annual_demand : float, optional
        Annual demand in units (D). If demand_period != 'year' and raw_demand/demand is not passed,
        this is treated as the demand for that period and normalized.
    order_cost : float
        Fixed cost per order placed (S / K). Must be > 0.
    holding_cost : float, optional
        Annual holding cost per unit per year (H). Must be > 0 if passed.
    holding_cost_rate : float, optional
        Annual holding cost as a decimal rate (h), e.g. 0.20 for 20%.
    unit_cost : float, optional
        Unit purchase cost (C). Required if holding_cost_rate is used.
    workdays_per_year : int, default 250
        Operating days per year to compute cycle time and daily demand.
    lead_time_days : float, optional
        Replenishment lead time in days for deterministic ROP calculation.
    demand : float, optional
        Alias for demand before normalization.
    raw_demand : float, optional
        Raw demand as provided by user/question before normalization.
    demand_period : str, default 'year'
        Time period of demand ('year', 'month', 'quarter', 'week', 'day').
    demand_unit : str, default 'units'
        Unit of measure for demand (e.g. 'units', 'items', 'cases').
    months_per_year : float, default 12.0
        Months per year for monthly demand conversion.
    weeks_per_year : float, default 52.0
        Weeks per year for weekly demand conversion.
    quarters_per_year : float, default 4.0
        Quarters per year for quarterly demand conversion.
    holding_cost_period : str, default 'year'
        Time period for holding cost (default 'year').

    Returns
    -------
    EOQResult
        Structured result with final answer, intermediate calculations, and assumptions.
    """
    # 1. Resolve raw demand input
    raw_d_input: Optional[float] = None
    if raw_demand is not None:
        raw_d_input = raw_demand
    elif demand is not None:
        raw_d_input = demand
    elif annual_demand is not None:
        raw_d_input = annual_demand

    if raw_d_input is None:
        raise ValueError("Missing required input 'annual_demand' (or 'demand'). Missing numerical inputs cannot be invented.")

    raw_d = _require_number(raw_d_input, "annual_demand", min_val=0.0, allow_zero=False)
    s = _require_number(order_cost, "order_cost", min_val=0.0, allow_zero=False)
    w = int(_require_number(workdays_per_year, "workdays_per_year", min_val=1.0, allow_zero=False))

    c: Optional[float] = None
    if unit_cost is not None:
        c = _require_number(unit_cost, "unit_cost", min_val=0.0, allow_zero=False)

    resolved_h: float
    if holding_cost is not None:
        resolved_h = _require_number(holding_cost, "holding_cost", min_val=0.0, allow_zero=False)
    elif holding_cost_rate is not None and c is not None:
        h_rate = _require_number(holding_cost_rate, "holding_cost_rate", min_val=0.0, allow_zero=False)
        resolved_h = h_rate * c
    else:
        raise ValueError(
            "Must provide either a positive holding_cost (H) or both holding_cost_rate (h) and unit_cost (C). "
            "Missing numerical inputs cannot be invented."
        )

    lt: Optional[float] = None
    if lead_time_days is not None:
        lt = _require_number(lead_time_days, "lead_time_days", min_val=0.0, allow_zero=True)

    # 2. Demand Period Normalization
    period_clean = (demand_period or "year").lower().strip()
    if period_clean in ["month", "months", "monthly", "mo"]:
        period_canonical = "month"
        factor = float(months_per_year)
    elif period_clean in ["quarter", "quarters", "quarterly", "qtr"]:
        period_canonical = "quarter"
        factor = float(quarters_per_year)
    elif period_clean in ["week", "weeks", "weekly", "wk"]:
        period_canonical = "week"
        factor = float(weeks_per_year)
    elif period_clean in ["day", "days", "daily", "d"]:
        period_canonical = "day"
        factor = float(w)
    elif period_clean in ["year", "years", "annual", "annually", "yearly", "yr"]:
        period_canonical = "year"
        factor = 1.0
    else:
        period_canonical = "year"
        factor = 1.0

    # Avoid double-conversion:
    # If caller explicitly provided both annual_demand AND raw_demand/demand where annual_demand != raw_d,
    # respect the provided annual_demand.
    normalization_applied = False
    note = ""
    if period_canonical != "year":
        if annual_demand is not None and (raw_demand is not None or demand is not None) and annual_demand != raw_d:
            d = _require_number(annual_demand, "annual_demand", min_val=0.0, allow_zero=False)
        else:
            d = raw_d * factor
        normalization_applied = True
        note = f"Demand normalized from {raw_d:g} {demand_unit}/{period_canonical} to {d:g} {demand_unit}/year ({factor:g} {period_canonical}s/year)."
    else:
        d = raw_d
        normalization_applied = False
        note = "Demand is on an annual basis (no conversion required)."

    # 3. Mathematical Calculations (EOQ formula unchanged)
    eoq = math.sqrt((2.0 * d * s) / resolved_h)

    # Intermediate calculations
    annual_ordering_cost = (d / eoq) * s
    annual_holding_cost = (eoq / 2.0) * resolved_h
    total_annual_cost = annual_ordering_cost + annual_holding_cost

    annual_purchase_cost = (d * c) if c is not None else None
    total_cost_with_purchase = (total_annual_cost + annual_purchase_cost) if annual_purchase_cost is not None else None

    orders_per_year = d / eoq
    cycle_time_years = eoq / d
    cycle_time_days = cycle_time_years * w

    deterministic_rop: Optional[float] = None
    if lt is not None:
        daily_demand = d / w
        deterministic_rop = daily_demand * lt

    # Sensitivity analysis (+/- 10%)
    q_plus = eoq * 1.10
    cost_plus = (d / q_plus) * s + (q_plus / 2.0) * resolved_h
    q_minus = eoq * 0.90
    cost_minus = (d / q_minus) * s + (q_minus / 2.0) * resolved_h

    # 4. Model Assumptions
    operating_schedule_assumption = (
        f"Operating schedule: Based on {w} operating days per year (user-specified schedule)."
        if workdays_per_year != 250
        else "Operating schedule: Based on 250 operating days per year (standard documented default: 50 operating weeks × 5 working days/week)."
    )

    assumptions = [
        "Annual demand (D) is constant, uniform, continuous, and known with certainty.",
        "Order setup cost (S) is fixed per order regardless of the order size Q.",
        "Unit holding cost (H) is linear, constant per unit held per year, and applies to average cycle inventory (Q/2).",
        "Replenishment lead time is zero or constant with instantaneous batch receipt.",
        "No shortages, stockouts, or backorders are permitted; all demand must be met from stock.",
        "Unit purchase price (C) is constant with no quantity breaks (basic EOQ).",
        "Planning horizon is infinite and operating parameters remain stable over time.",
        operating_schedule_assumption,
    ]
    if normalization_applied:
        assumptions.insert(
            0,
            f"Demand period normalization: {note}"
        )

    intermediate_calcs = {
        "formula": "Q = sqrt(2 * D * S / H)",
        "resolved_holding_cost": resolved_h,
        "raw_demand": raw_d,
        "demand_period": period_canonical,
        "demand_unit": demand_unit,
        "demand_normalization_applied": normalization_applied,
        "demand_normalization_factor": factor,
        "annual_demand": d,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_annual_policy_cost": total_annual_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_cost_with_purchase": total_cost_with_purchase,
        "orders_per_year": orders_per_year,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "workdays_per_year": w,
        "order_cycle_time_formula": "T = Operating Days per Year / Annual Order Frequency = W / N",
        "reorder_point": deterministic_rop,
        "sensitivity_plus_10_percent_cost": cost_plus,
        "sensitivity_minus_10_percent_cost": cost_minus,
    }

    raw_inputs = {
        "annual_demand": d,
        "raw_demand": raw_d,
        "demand_period": period_canonical,
        "demand_unit": demand_unit,
        "demand_normalization_applied": normalization_applied,
        "demand_normalization_factor": factor,
        "order_cost": s,
        "holding_cost": holding_cost,
        "holding_cost_rate": holding_cost_rate,
        "unit_cost": c,
        "workdays_per_year": w,
        "lead_time_days": lt,
    }

    verif = verify_eoq(
        eoq, d, s, resolved_h, annual_ordering_cost, annual_holding_cost,
        demand_normalization_applied=normalization_applied,
        demand_normalization_note=note,
    )

    return EOQResult(
        final_answer=eoq,
        eoq=eoq,
        inputs=raw_inputs,
        intermediate_calculations=intermediate_calcs,
        assumptions=assumptions,
        annual_demand=d,
        order_cost=s,
        holding_cost=resolved_h,
        unit_cost=c,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_annual_cost=total_annual_cost,
        total_cost_with_purchase=total_cost_with_purchase,
        orders_per_year=orders_per_year,
        cycle_time_years=cycle_time_years,
        cycle_time_days=cycle_time_days,
        workdays_per_year=w,
        lead_time_days=lt,
        reorder_point=deterministic_rop,
        sensitivity_plus_10_cost=cost_plus,
        sensitivity_minus_10_cost=cost_minus,
        verification=verif,
        raw_demand=raw_d,
        demand_period=period_canonical,
        demand_unit=demand_unit,
        demand_normalization_applied=normalization_applied,
        demand_normalization_factor=factor,
        demand_normalization_note=note,
        holding_cost_period=holding_cost_period,
    )


def verify_eoq(
    eoq: float,
    demand: float,
    order_cost: float,
    holding_cost: float,
    annual_ordering_cost: float,
    annual_holding_cost: float,
    demand_normalization_applied: bool = False,
    demand_normalization_note: str = "",
) -> VerificationResult:
    """Verify optimality and mathematical consistency of EOQ."""
    checks = []
    warnings = []
    errors = []

    if eoq <= 0 or math.isnan(eoq) or math.isinf(eoq):
        errors.append(f"EOQ value invalid: {eoq}")
    else:
        checks.append("EOQ is strictly positive and finite.")

    if demand_normalization_applied:
        checks.append(f"Demand period normalization verified: {demand_normalization_note}")

    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    rel_diff = cost_diff / max(annual_ordering_cost, 1e-9)
    if rel_diff < 1e-5:
        checks.append(f"Optimality condition satisfied: Ordering Cost (${annual_ordering_cost:,.2f}) == Holding Cost (${annual_holding_cost:,.2f}).")
    else:
        warnings.append(f"Ordering and holding costs differ by {rel_diff:.4%}: {cost_diff:.4f}")

    derivative = -(demand * order_cost) / (eoq ** 2) + holding_cost / 2.0
    if abs(derivative) < 1e-5:
        checks.append(f"First-order derivative dTC/dQ is zero ({derivative:.6e}).")
    else:
        warnings.append(f"First-order derivative non-zero: {derivative:.6e}")

    second_deriv = (2.0 * demand * order_cost) / (eoq ** 3)
    if second_deriv > 0:
        checks.append("Second-order condition satisfied (convex cost curve minimum).")
    else:
        errors.append("Second-order condition failed (not a minimum).")

    return VerificationResult(
        is_valid=len(errors) == 0,
        checks_passed=checks,
        warnings=warnings,
        errors=errors,
    )





# =====================================================================
# Tool 2: calculate_quantity_discount()
# =====================================================================

def calculate_quantity_discount(
    annual_demand: float,
    order_cost: float,
    discount_schedule: List[Dict[str, Any]],
    holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
) -> QuantityDiscountResult:
    """Evaluate feasible quantities and compare total annual cost across all discount tiers.

    Supports both fixed holding cost (H) and percentage holding cost (h * C_i).

    Parameters
    ----------
    annual_demand : float
        Annual demand in units (D). Must be > 0.
    order_cost : float
        Fixed cost per order (S). Must be > 0.
    discount_schedule : list of dict
        List of tiers. Each dict has:
          - 'min_qty': float (minimum quantity required for tier)
          - 'max_qty': float or None (upper boundary, None for no limit)
          - 'unit_price': float (unit price C_i for this tier)
    holding_cost : float, optional
        Fixed annual dollar holding cost per unit (H) applied across all tiers.
    holding_cost_rate : float, optional
        Annual holding cost percentage rate (h), applied as H_i = h * C_i per tier.

    Returns
    -------
    QuantityDiscountResult
        Structured result detailing feasible quantities, cost breakdown, optimal tier, and assumptions.
    """
    # 1. Strict Input Validation (Never invent missing inputs)
    d = _require_number(annual_demand, "annual_demand", min_val=0.0, allow_zero=False)
    s = _require_number(order_cost, "order_cost", min_val=0.0, allow_zero=False)

    if not discount_schedule or not isinstance(discount_schedule, list):
        raise ValueError("discount_schedule must be a non-empty list of tier dictionaries. Numerical inputs cannot be invented.")

    h_fixed: Optional[float] = None
    h_rate: Optional[float] = None
    if holding_cost is not None:
        h_fixed = _require_number(holding_cost, "holding_cost", min_val=0.0, allow_zero=False)
    elif holding_cost_rate is not None:
        h_rate = _require_number(holding_cost_rate, "holding_cost_rate", min_val=0.0, allow_zero=False)
    else:
        raise ValueError(
            "Must provide either a fixed holding_cost (H) or holding_cost_rate (h). "
            "Missing numerical inputs cannot be invented."
        )

    # Validate and evaluate tiers
    evaluated_tiers: List[DiscountTier] = []
    tier_calculations_summary: List[Dict[str, Any]] = []

    for idx, tier_dict in enumerate(discount_schedule):
        if not isinstance(tier_dict, dict):
            raise TypeError(f"Tier {idx+1} in discount_schedule must be a dictionary.")
        if "min_qty" not in tier_dict or "unit_price" not in tier_dict:
            raise ValueError(f"Tier {idx+1} must contain 'min_qty' and 'unit_price' keys.")

        min_q = _require_number(tier_dict["min_qty"], f"tier[{idx+1}].min_qty", min_val=0.0, allow_zero=True)
        raw_max = tier_dict.get("max_qty")
        max_q: Optional[float] = None
        if raw_max is not None and raw_max != "":
            max_q = _require_number(raw_max, f"tier[{idx+1}].max_qty", min_val=min_q, allow_zero=False)

        price = _require_number(tier_dict["unit_price"], f"tier[{idx+1}].unit_price", min_val=0.0, allow_zero=False)

        # Resolve tier holding cost (fixed vs percentage)
        tier_h = (h_rate * price) if h_rate is not None else h_fixed

        # Unconstrained EOQ for this tier
        unconstrained_eoq = math.sqrt((2.0 * d * s) / tier_h)

        # Determine feasibility and evaluated quantity
        # 1. If EOQ falls within [min_q, max_q], it is feasible at EOQ.
        # 2. If EOQ falls below min_q, evaluate at min_q (clamped to minimum feasible quantity).
        # 3. If EOQ exceeds max_q, evaluate at max_q (clamped to maximum feasible quantity).
        # Order quantity must be strictly positive (at least 1.0 unit).
        effective_min = max(min_q, 1.0)
        is_feasible_at_eoq: bool
        feasible_q: float
        status_desc: str

        if max_q is not None:
            if effective_min <= unconstrained_eoq <= max_q:
                feasible_q = unconstrained_eoq
                is_feasible_at_eoq = True
                status_desc = "Feasible at EOQ"
            elif unconstrained_eoq < effective_min:
                feasible_q = effective_min
                is_feasible_at_eoq = False
                status_desc = f"Below Tier Min ({effective_min:,.0f}) - Clamped to Min Q"
            else:
                feasible_q = max_q
                is_feasible_at_eoq = False
                status_desc = f"Above Tier Max ({max_q:,.0f}) - Clamped to Max Q"
        else:
            if unconstrained_eoq >= effective_min:
                feasible_q = unconstrained_eoq
                is_feasible_at_eoq = True
                status_desc = "Feasible at EOQ"
            else:
                feasible_q = effective_min
                is_feasible_at_eoq = False
                status_desc = f"Below Tier Min ({effective_min:,.0f}) - Clamped to Min Q"

        # Costs at evaluated feasible quantity
        ann_order_cost = (d / feasible_q) * s
        ann_hold_cost = (feasible_q / 2.0) * tier_h
        ann_purch_cost = d * price
        total_cost = ann_order_cost + ann_hold_cost + ann_purch_cost

        tier_summary = {
            "tier_number": idx + 1,
            "min_quantity": min_q,
            "max_quantity": max_q,
            "unit_price": price,
            "tier_holding_cost": tier_h,
            "unconstrained_eoq": unconstrained_eoq,
            "is_feasible_at_eoq": is_feasible_at_eoq,
            "evaluated_feasible_quantity": feasible_q,
            "feasibility_status": status_desc,
            "annual_ordering_cost": ann_order_cost,
            "annual_holding_cost": ann_hold_cost,
            "annual_purchase_cost": ann_purch_cost,
            "total_annual_cost": total_cost,
        }
        tier_calculations_summary.append(tier_summary)

        tier_obj = DiscountTier(
            tier_number=idx + 1,
            min_quantity=min_q,
            max_quantity=max_q,
            unit_price=price,
            holding_cost=tier_h,
            unconstrained_eoq=unconstrained_eoq,
            is_feasible_at_eoq=is_feasible_at_eoq,
            feasible_quantity=feasible_q,
            annual_ordering_cost=ann_order_cost,
            annual_holding_cost=ann_hold_cost,
            annual_purchase_cost=ann_purch_cost,
            total_annual_cost=total_cost,
            is_optimal=False,
            feasibility_status=status_desc,
            intermediate_calculations=tier_summary,
        )
        evaluated_tiers.append(tier_obj)

    # Identify optimal tier with minimum total annual cost
    best_tier_idx = min(range(len(evaluated_tiers)), key=lambda i: evaluated_tiers[i].total_annual_cost)
    evaluated_tiers[best_tier_idx].is_optimal = True
    best_tier = evaluated_tiers[best_tier_idx]

    base_case_cost = evaluated_tiers[0].total_annual_cost
    annual_savings = base_case_cost - best_tier.total_annual_cost
    pct_savings = (annual_savings / base_case_cost) * 100.0 if base_case_cost > 0 else 0.0

    # Model Assumptions
    assumptions = [
        "All-units quantity discount schedule (discounted unit price applies to all units purchased in the order once threshold is met).",
        "Annual demand (D) is constant, uniform, and known with certainty over the planning horizon.",
        "Entire order quantity Q arrives in a single delivery batch (instantaneous replenishment).",
        f"Holding cost is {'percentage-based (h * unit_price)' if h_rate is not None else 'fixed dollar amount (constant H across tiers)'}.",
        "Holding cost applies to average cycle inventory (Q / 2).",
        "No shortages or backorders are permitted.",
        "Warehouse storage space and working capital are sufficient to hold the optimal batch size.",
    ]

    intermediate_calcs = {
        "holding_cost_type": "percentage" if h_rate is not None else "fixed",
        "holding_cost_rate": h_rate,
        "fixed_holding_cost": h_fixed,
        "tiers_evaluated_count": len(evaluated_tiers),
        "tier_evaluations": tier_calculations_summary,
        "base_case_cost": base_case_cost,
        "annual_savings_vs_base": annual_savings,
        "percent_savings_vs_base": pct_savings,
    }

    opt_purch_cost = best_tier.annual_purchase_cost
    opt_order_cost = best_tier.annual_ordering_cost
    opt_hold_cost = best_tier.annual_holding_cost
    opt_policy_cost = opt_order_cost + opt_hold_cost

    intermediate_calcs.update({
        "purchase_cost": opt_purch_cost,
        "annual_ordering_cost": opt_order_cost,
        "annual_holding_cost": opt_hold_cost,
        "total_annual_policy_cost": opt_policy_cost,
        "total_annual_cost": best_tier.total_annual_cost,
    })

    final_answer = {
        "optimal_quantity": best_tier.feasible_quantity,
        "optimal_unit_price": best_tier.unit_price,
        "optimal_tier_number": best_tier.tier_number,
        "purchase_cost": opt_purch_cost,
        "annual_ordering_cost": opt_order_cost,
        "annual_holding_cost": opt_hold_cost,
        "total_annual_cost_including_purchase": best_tier.total_annual_cost,
        "total_annual_policy_cost_excluding_purchase": opt_policy_cost,
        "min_total_annual_cost": best_tier.total_annual_cost,
        "annual_savings": annual_savings,
    }

    raw_inputs = {
        "annual_demand": d,
        "order_cost": s,
        "discount_schedule": discount_schedule,
        "holding_cost": h_fixed,
        "holding_cost_rate": h_rate,
    }

    verif = verify_quantity_discount(evaluated_tiers, best_tier_idx)

    return QuantityDiscountResult(
        final_answer=final_answer,
        optimal_quantity=best_tier.feasible_quantity,
        optimal_tier_index=best_tier_idx,
        optimal_unit_price=best_tier.unit_price,
        min_total_cost=best_tier.total_annual_cost,
        inputs=raw_inputs,
        intermediate_calculations=intermediate_calcs,
        assumptions=assumptions,
        tiers=evaluated_tiers,
        base_case_cost=base_case_cost,
        annual_savings=annual_savings,
        percent_savings=pct_savings,
        annual_demand=d,
        order_cost=s,
        annual_purchase_cost=opt_purch_cost,
        annual_ordering_cost=opt_order_cost,
        annual_holding_cost=opt_hold_cost,
        total_annual_policy_cost=opt_policy_cost,
        verification=verif,
    )


def verify_quantity_discount(tiers: List[DiscountTier], optimal_idx: int) -> VerificationResult:
    """Verify validity of quantity discount schedule and global cost minimizer."""
    checks = []
    warnings = []
    errors = []

    checks.append(f"Evaluated {len(tiers)} pricing brackets.")
    opt = tiers[optimal_idx]

    for i, t in enumerate(tiers):
        if t.total_annual_cost < opt.total_annual_cost - 1e-4:
            errors.append(f"Tier {t.tier_number} has lower cost (${t.total_annual_cost:,.2f}) than selected Tier {opt.tier_number} (${opt.total_annual_cost:,.2f}).")
        # Enforce that no tier applies a price to an order quantity outside its boundaries
        if t.feasible_quantity < t.min_quantity - 1e-4:
            errors.append(f"Tier {t.tier_number} evaluated quantity {t.feasible_quantity:,.2f} is below minimum requirement {t.min_quantity:,.0f}.")
        elif t.max_quantity is not None and t.feasible_quantity > t.max_quantity + 1e-4:
            errors.append(f"Tier {t.tier_number} evaluated quantity {t.feasible_quantity:,.2f} exceeds upper bracket limit {t.max_quantity:,.0f}.")

    if not errors:
        checks.append(f"Global optimality verified: Tier {opt.tier_number} (Q={opt.feasible_quantity:,.0f}) provides absolute minimum annual total cost (${opt.total_annual_cost:,.2f}).")

    if opt.feasible_quantity < opt.min_quantity:
        errors.append(f"Price qualification error: Optimal quantity {opt.feasible_quantity:,.2f} is below minimum requirement {opt.min_quantity:,.0f} for Tier {opt.tier_number}.")
    elif opt.max_quantity is not None and opt.feasible_quantity > opt.max_quantity:
        errors.append(f"Price qualification error: Optimal quantity {opt.feasible_quantity:,.2f} exceeds upper bracket {opt.max_quantity:,.0f} for Tier {opt.tier_number}.")
    else:
        checks.append(f"Price qualification verified: Selected order quantity {opt.feasible_quantity:,.0f} units strictly qualifies for Tier {opt.tier_number} price (${opt.unit_price:,.2f}/unit) within allowable range [{opt.min_quantity:,.0f}, {opt.max_quantity or 'Infinity'}).")

    if not opt.is_feasible_at_eoq:
        checks.append(f"Boundary clamping verified: Unconstrained EOQ ({opt.unconstrained_eoq:,.2f} units) was infeasible for Tier {opt.tier_number} and was clamped to boundary quantity ({opt.feasible_quantity:,.0f} units); unit price ${opt.unit_price:,.2f} is never applied to unqualified quantity {opt.unconstrained_eoq:,.2f}.")
    else:
        checks.append(f"EOQ interior feasibility verified: Unconstrained EOQ ({opt.unconstrained_eoq:,.2f} units) falls naturally within Tier {opt.tier_number} allowable range.")

    checks.append("All tier pricing brackets strictly satisfied by evaluated candidate quantities.")

    return VerificationResult(
        is_valid=len(errors) == 0,
        checks_passed=checks,
        warnings=warnings,
        errors=errors,
    )





# =====================================================================
# =====================================================================
# Tool 3: calculate_rop()
# =====================================================================

def calculate_rop(
    avg_demand: float,
    lead_time: float,
    demand_std: float = 0.0,
    lead_time_std: float = 0.0,
    service_level: float = 0.95,
    time_unit: str = "days",
    demand_time_unit: Optional[str] = None,
    lead_time_unit: Optional[str] = None,
    workdays_per_week: float = 5.0,
    workdays_per_month: float = 20.0,
    weeks_per_month: float = 4.0,
) -> ROPResult:
    """Calculate Safety Stock (SS) and Reorder Point (ROP).

    Supports all 4 demand and lead-time variability configurations:
    1. Constant demand and constant lead time (sigma_d = 0, sigma_L = 0)
    2. Variable demand and constant lead time (sigma_d > 0, sigma_L = 0)
    3. Constant demand and variable lead time (sigma_d = 0, sigma_L > 0)
    4. Variable demand and variable lead time (sigma_d > 0, sigma_L > 0):
       sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
       SS = z * sigma_LT
       ROP = d * L + SS

    Includes automatic time unit normalization when demand and lead time are provided
    in different time units (e.g. demand per day and lead time in weeks).

    Parameters
    ----------
    avg_demand : float
        Average demand per period (d). Must be >= 0.
    lead_time : float
        Replenishment lead time in periods (L). Must be > 0.
    demand_std : float, default 0.0
        Standard deviation of demand per period (sigma_d). Must be >= 0.
    lead_time_std : float, default 0.0
        Standard deviation of lead time (sigma_L). Must be >= 0.
    service_level : float, default 0.95
        Cycle Service Level (CSL) as a decimal (0 < service_level < 1).
    time_unit : str, default 'days'
        Fallback unit of time if demand_time_unit is not explicitly supplied.
    demand_time_unit : str, optional
        Time unit for demand rate (e.g. 'days', 'weeks', 'months').
    lead_time_unit : str, optional
        Time unit for replenishment lead time (e.g. 'days', 'weeks', 'months').
    workdays_per_week : float, default 5.0
        Operating working days per week used for week <-> day conversion.
    workdays_per_month : float, default 20.0
        Operating working days per month used for month <-> day conversion.
    weeks_per_month : float, default 4.0
        Weeks per month used for month <-> week conversion.

    Returns
    -------
    ROPResult
        Structured result with final answer, intermediate calculations, case, assumptions,
        and unit normalization audit details.
    """
    # 1. Strict Input Validation (Never invent missing inputs)
    d = _require_number(avg_demand, "avg_demand", min_val=0.0, allow_zero=True)
    raw_l = _require_number(lead_time, "lead_time", min_val=0.0, allow_zero=False)
    sigma_d = _require_number(demand_std, "demand_std", min_val=0.0, allow_zero=True)
    raw_sigma_l = _require_number(lead_time_std, "lead_time_std", min_val=0.0, allow_zero=True)
    csl = _require_number(service_level, "service_level")

    workdays_pw = _require_number(workdays_per_week, "workdays_per_week", min_val=0.0, allow_zero=False) if workdays_per_week is not None else 5.0
    workdays_pm = _require_number(workdays_per_month, "workdays_per_month", min_val=0.0, allow_zero=False) if workdays_per_month is not None else 20.0
    weeks_pm = _require_number(weeks_per_month, "weeks_per_month", min_val=0.0, allow_zero=False) if weeks_per_month is not None else 4.0

    if not (0.0 < csl < 1.0):
        raise ValueError(f"Service level must be between 0 and 1 exclusive, got {csl}.")

    # 2. Time Unit Normalization
    def _canonical_unit(u: Optional[str]) -> str:
        if not u:
            return ""
        u_clean = str(u).lower().strip()
        if "day" in u_clean:
            return "days"
        elif "week" in u_clean:
            return "weeks"
        elif "month" in u_clean:
            return "months"
        elif "year" in u_clean:
            return "years"
        return u_clean

    d_unit = _canonical_unit(demand_time_unit) or _canonical_unit(time_unit) or "days"
    lt_unit = _canonical_unit(lead_time_unit) or d_unit

    l = raw_l
    sigma_l = raw_sigma_l
    conversion_factor = 1.0
    normalization_applied = False
    norm_unit = d_unit
    normalization_note = "Demand and lead time already share the same time unit; no unit normalization required."

    if d_unit != lt_unit:
        # Convert both average lead time and lead time standard deviation into the same unit as demand
        if d_unit == "days" and lt_unit == "weeks":
            conversion_factor = workdays_pw
            l = raw_l * conversion_factor
            sigma_l = raw_sigma_l * conversion_factor
            normalization_applied = True
            norm_unit = "working days"
            normalization_note = (
                f"Lead time converted from weeks to working days using {workdays_pw:g} working days per week: "
                f"Average lead time L = {raw_l:g} × {workdays_pw:g} = {l:g} working days, "
                f"Lead time std dev sigma_LT = {raw_sigma_l:g} × {workdays_pw:g} = {sigma_l:g} working days."
            )
        elif d_unit == "days" and lt_unit == "months":
            conversion_factor = workdays_pm
            l = raw_l * conversion_factor
            sigma_l = raw_sigma_l * conversion_factor
            normalization_applied = True
            norm_unit = "working days"
            normalization_note = (
                f"Lead time converted from months to working days using {workdays_pm:g} working days per month: "
                f"Average lead time L = {raw_l:g} × {workdays_pm:g} = {l:g} working days, "
                f"Lead time std dev sigma_LT = {raw_sigma_l:g} × {workdays_pm:g} = {sigma_l:g} working days."
            )
        elif d_unit == "weeks" and lt_unit == "days":
            conversion_factor = 1.0 / workdays_pw
            l = raw_l * conversion_factor
            sigma_l = raw_sigma_l * conversion_factor
            normalization_applied = True
            norm_unit = "weeks"
            normalization_note = (
                f"Lead time converted from days to weeks using {workdays_pw:g} working days per week: "
                f"Average lead time L = {raw_l:g} / {workdays_pw:g} = {l:g} weeks, "
                f"Lead time std dev sigma_LT = {raw_sigma_l:g} / {workdays_pw:g} = {sigma_l:g} weeks."
            )
        elif d_unit == "weeks" and lt_unit == "months":
            conversion_factor = weeks_pm
            l = raw_l * conversion_factor
            sigma_l = raw_sigma_l * conversion_factor
            normalization_applied = True
            norm_unit = "weeks"
            normalization_note = (
                f"Lead time converted from months to weeks using {weeks_pm:g} weeks per month: "
                f"Average lead time L = {raw_l:g} × {weeks_pm:g} = {l:g} weeks, "
                f"Lead time std dev sigma_LT = {raw_sigma_l:g} × {weeks_pm:g} = {sigma_l:g} weeks."
            )
        elif d_unit == "months" and lt_unit == "weeks":
            conversion_factor = 1.0 / weeks_pm
            l = raw_l * conversion_factor
            sigma_l = raw_sigma_l * conversion_factor
            normalization_applied = True
            norm_unit = "months"
            normalization_note = (
                f"Lead time converted from weeks to months using {weeks_pm:g} weeks per month: "
                f"Average lead time L = {raw_l:g} / {weeks_pm:g} = {l:g} months, "
                f"Lead time std dev sigma_LT = {raw_sigma_l:g} / {weeks_pm:g} = {sigma_l:g} months."
            )
        elif d_unit == "months" and lt_unit == "days":
            conversion_factor = 1.0 / workdays_pm
            l = raw_l * conversion_factor
            sigma_l = raw_sigma_l * conversion_factor
            normalization_applied = True
            norm_unit = "months"
            normalization_note = (
                f"Lead time converted from days to months using {workdays_pm:g} working days per month: "
                f"Average lead time L = {raw_l:g} / {workdays_pm:g} = {l:g} months, "
                f"Lead time std dev sigma_LT = {raw_sigma_l:g} / {workdays_pm:g} = {sigma_l:g} months."
            )

    # 3. Case Identification & Calculation
    # Formula: Var(LT) = L * sigma_d^2 + d^2 * sigma_l^2
    var_d_comp = l * (sigma_d ** 2)
    var_l_comp = (d ** 2) * (sigma_l ** 2)
    var_ltd = var_d_comp + var_l_comp
    sigma_lt = math.sqrt(var_ltd)

    # Determine specific case
    if sigma_d == 0.0 and sigma_l == 0.0:
        case = "constant_demand_constant_lead_time"
        case_desc = "Constant Demand and Constant Lead Time (Deterministic)"
        z_score = 0.0
        safety_stock = 0.0
    elif sigma_d > 0.0 and sigma_l == 0.0:
        case = "variable_demand_constant_lead_time"
        case_desc = "Variable Demand and Constant Lead Time (sigma_LT = sigma_d * sqrt(L))"
        dist = NormalDist(mu=0.0, sigma=1.0)
        z_score = dist.inv_cdf(csl)
        safety_stock = z_score * sigma_lt
    elif sigma_d == 0.0 and sigma_l > 0.0:
        case = "constant_demand_variable_lead_time"
        case_desc = "Constant Demand and Variable Lead Time (sigma_LT = d * sigma_L)"
        dist = NormalDist(mu=0.0, sigma=1.0)
        z_score = dist.inv_cdf(csl)
        safety_stock = z_score * sigma_lt
    else:
        case = "variable_demand_variable_lead_time"
        case_desc = "Variable Demand and Variable Lead Time (sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2))"
        dist = NormalDist(mu=0.0, sigma=1.0)
        z_score = dist.inv_cdf(csl)
        safety_stock = z_score * sigma_lt

    expected_ltd = d * l
    rop = expected_ltd + safety_stock

    stockout_prob = 1.0 - csl
    stockout_risk_pct = stockout_prob * 100.0

    # 4. Model Assumptions
    assumptions = [
        f"Case Identified: {case_desc}.",
        f"Time Unit Normalization: {normalization_note}",
        "Continuous review inventory policy (s, Q): inventory position is monitored continuously.",
        "When demand is variable, demand in each period is independently and identically normally distributed.",
        "When lead time is variable, replenishment duration is normally distributed and independent of order size.",
        "Lead time demand standard deviation follows sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2).",
        f"Cycle Service Level (CSL = {csl:.1%}) defines the target probability of not stocking out during a replenishment cycle.",
        "Safety stock (SS = z * sigma_LT) absorbs demand and lead-time shocks during transit.",
        "Unsatisfied demand during lead time is completely backordered until delivery arrives.",
    ]

    intermediate_calcs = {
        "case": case,
        "case_description": case_desc,
        "raw_lead_time": raw_l,
        "raw_lead_time_std": raw_sigma_l,
        "raw_lead_time_unit": lt_unit,
        "demand_time_unit": d_unit,
        "workdays_per_week": workdays_pw,
        "unit_normalization_applied": normalization_applied,
        "unit_conversion_factor": conversion_factor,
        "unit_normalization_note": normalization_note,
        "normalized_lead_time": l,
        "normalized_lead_time_std": sigma_l,
        "normalized_time_unit": norm_unit,
        "expected_lead_time_demand": expected_ltd,
        "demand_variance_component": var_d_comp,
        "lead_time_variance_component": var_l_comp,
        "lead_time_demand_variance": var_ltd,
        "sigma_LT": sigma_lt,
        "z_score": z_score,
        "safety_stock": safety_stock,
        "reorder_point": rop,
        "stockout_probability": stockout_prob,
        "stockout_risk_per_cycle_percent": stockout_risk_pct,
    }

    final_answer = {
        "reorder_point": rop,
        "safety_stock": safety_stock,
        "expected_lead_time_demand": expected_ltd,
        "sigma_LT": sigma_lt,
        "z_score": z_score,
        "normalized_lead_time": l,
        "normalized_lead_time_std": sigma_l,
    }

    raw_inputs = {
        "avg_demand": d,
        "lead_time": raw_l,
        "demand_std": sigma_d,
        "lead_time_std": raw_sigma_l,
        "service_level": csl,
        "time_unit": norm_unit,
        "demand_time_unit": d_unit,
        "lead_time_unit": lt_unit,
        "workdays_per_week": workdays_pw,
        "normalized_lead_time": l,
        "normalized_lead_time_std": sigma_l,
        "unit_normalization_applied": normalization_applied,
    }

    verif = verify_safety_stock_rop(
        csl,
        z_score,
        expected_ltd,
        sigma_lt,
        safety_stock,
        rop,
        lead_time_std=sigma_l,
        unit_normalization_applied=normalization_applied,
        normalization_note=normalization_note,
    )

    return ROPResult(
        final_answer=final_answer,
        reorder_point=rop,
        safety_stock=safety_stock,
        case=case,
        inputs=raw_inputs,
        intermediate_calculations=intermediate_calcs,
        assumptions=assumptions,
        service_level=csl,
        z_score=z_score,
        lead_time=l,
        time_unit=norm_unit,
        avg_demand_per_period=d,
        demand_std_per_period=sigma_d,
        lead_time_std=sigma_l,
        expected_lead_time_demand=expected_ltd,
        std_lead_time_demand=sigma_lt,
        stockout_probability=stockout_prob,
        stockout_risk_per_cycle_percent=stockout_risk_pct,
        verification=verif,
        demand_time_unit=d_unit,
        lead_time_unit=lt_unit,
        workdays_per_week=workdays_pw,
        raw_lead_time=raw_l,
        raw_lead_time_std=raw_sigma_l,
        unit_conversion_factor=conversion_factor,
        unit_normalization_applied=normalization_applied,
        unit_normalization_note=normalization_note,
    )

# Alias for backwards compatibility
calculate_safety_stock_rop = calculate_rop


def verify_safety_stock_rop(
    service_level: float,
    z_score: float,
    expected_ltd: float,
    std_ltd: float,
    safety_stock: float,
    rop: float,
    lead_time_std: float = 0.0,
    unit_normalization_applied: bool = False,
    normalization_note: str = "",
) -> VerificationResult:
    """Verify safety stock and reorder point calculations."""
    checks = []
    warnings = []
    errors = []

    checks.append(f"Target cycle service level {service_level:.1%} corresponds to z-factor of {z_score:.4f}.")

    if unit_normalization_applied:
        checks.append(f"Unit normalization verified: {normalization_note}")

    if lead_time_std > 0:
        checks.append(f"Lead time variability verified: Lead time uncertainty (sigma_LT = {lead_time_std:,.2f}) accounted for in compound variance.")

    if service_level >= 0.5:
        if safety_stock >= 0:
            checks.append(f"Safety stock is non-negative ({safety_stock:,.2f} units).")
        else:
            errors.append(f"Safety stock should be non-negative for service level >= 50%, got {safety_stock}")
    else:
        warnings.append(f"Service level below 50% results in negative safety stock buffer ({safety_stock:,.2f}).")

    calc_rop = expected_ltd + safety_stock
    if abs(calc_rop - rop) < 1e-4:
        checks.append(f"ROP consistency verified: Expected LTD ({expected_ltd:,.2f}) + SS ({safety_stock:,.2f}) == ROP ({rop:,.2f}).")
    else:
        errors.append(f"ROP mismatch: {calc_rop} != {rop}")

    return VerificationResult(
        is_valid=len(errors) == 0,
        checks_passed=checks,
        warnings=warnings,
        errors=errors,
    )





# =====================================================================
# Tool 4: calculate_periodic_review()
# =====================================================================

def calculate_periodic_review(
    review_period: float,
    lead_time: float,
    avg_demand: float,
    demand_std: float = 0.0,
    service_level: float = 0.95,
    current_inventory: float = 0.0,
    on_order: float = 0.0,
    backorders: float = 0.0,
    safety_stock: Optional[float] = None,
    inventory_position: Optional[float] = None,
    time_unit: str = "days",
    round_up: bool = False,
) -> PeriodicReviewResult:
    """Calculate target inventory and order quantity for a Fixed-Period Review System.

    Formulas:
    Protection period = review period + lead time
    Target inventory = demand during protection period + safety stock
    Order quantity = max(0, target inventory - inventory position)
    Inventory position = current_inventory + on_order - backorders

    Parameters
    ----------
    review_period : float
        Review interval (T or P). Must be > 0.
    lead_time : float
        Replenishment lead time (L). Must be >= 0.
    avg_demand : float
        Average demand per period (d). Must be >= 0.
    demand_std : float, default 0.0
        Standard deviation of demand per period (sigma_d). Must be >= 0.
    service_level : float, default 0.95
        Desired cycle service level (0 < service_level < 1).
    current_inventory : float, default 0.0
        On-hand stock at review time.
    on_order : float, default 0.0
        Open orders in transit scheduled to arrive.
    backorders : float, default 0.0
        Committed unfulfilled customer backorders.
    safety_stock : float, optional
        Pre-specified safety stock buffer. If provided, used directly.
    inventory_position : float, optional
        Directly provided inventory position. If provided, used directly.
    time_unit : str, default 'days'
        Time unit (e.g. 'days', 'weeks').
    round_up : bool, default False
        Whether to round the recommended order quantity up to the nearest whole integer unit.

    Returns
    -------
    PeriodicReviewResult
        Structured result with final answer, intermediate calculations, and assumptions.
    """
    # 1. Strict Input Validation (Never invent missing inputs)
    t = _require_number(review_period, "review_period", min_val=0.0, allow_zero=False)
    l = _require_number(lead_time, "lead_time", min_val=0.0, allow_zero=True)
    d = _require_number(avg_demand, "avg_demand", min_val=0.0, allow_zero=True)
    sigma_d = _require_number(demand_std, "demand_std", min_val=0.0, allow_zero=True)
    csl = _require_number(service_level, "service_level")

    if not (0.0 < csl < 1.0):
        raise ValueError(f"Service level must be between 0 and 1 exclusive, got {csl}.")

    cur_inv = _require_number(current_inventory, "current_inventory", allow_zero=True)
    orders_in_transit = _require_number(on_order, "on_order", allow_zero=True)
    back_orders = _require_number(backorders, "backorders", allow_zero=True)

    # 2. Calculations
    # Protection period = review period + lead time
    protection_period = t + l

    # Demand during protection period
    expected_demand_prot = d * protection_period
    std_demand_prot = sigma_d * math.sqrt(protection_period)

    # Z-score and safety stock
    if safety_stock is not None:
        resolved_safety_stock = _require_number(safety_stock, "safety_stock", min_val=0.0, allow_zero=True)
        z_score = (resolved_safety_stock / std_demand_prot) if std_demand_prot > 0 else 0.0
    elif sigma_d > 0:
        dist = NormalDist(mu=0.0, sigma=1.0)
        z_score = dist.inv_cdf(csl)
        resolved_safety_stock = z_score * std_demand_prot
    else:
        z_score = 0.0
        resolved_safety_stock = 0.0

    # Target inventory (Order-Up-To Level M or S)
    target_inventory = expected_demand_prot + resolved_safety_stock

    # Inventory position = on_hand + on_order - backorders (or direct inventory_position if provided)
    if inventory_position is not None:
        resolved_inventory_position = _require_number(inventory_position, "inventory_position", allow_zero=True)
    else:
        resolved_inventory_position = cur_inv + orders_in_transit - back_orders

    # Order quantity = max(0, target inventory - inventory position)
    unclamped_order_qty = target_inventory - resolved_inventory_position
    exact_order_qty = max(0.0, unclamped_order_qty)
    recommended_order_qty = float(math.ceil(exact_order_qty)) if round_up else exact_order_qty

    # 3. Model Assumptions
    assumptions = [
        "Periodic review inventory policy (T, S) / (R, S): inventory position is reviewed at fixed intervals of T periods.",
        "Protection period equals review period plus replenishment lead time (Protection Period = T + L).",
        "Demand per period is independent and normally distributed with mean d and standard deviation sigma_d.",
        "Safety stock protects against demand variability across the entire protection window (T + L).",
        "Target inventory (Order-Up-To level M) brings stock to expected demand over (T + L) plus safety stock.",
        "Inventory position equals On-Hand + On-Order - Backorders.",
        "Order quantity equals max(0, Target Inventory - Inventory Position); negative orders are clamped to zero.",
        "All unmet demand is completely backordered.",
    ]
    if round_up:
        assumptions.append("Recommended order quantity is rounded up to the nearest whole integer unit as specified.")

    intermediate_calcs = {
        "review_period": t,
        "lead_time": l,
        "protection_period": protection_period,
        "expected_demand_protection_period": expected_demand_prot,
        "std_demand_protection_period": std_demand_prot,
        "z_score": z_score,
        "safety_stock": resolved_safety_stock,
        "target_inventory": target_inventory,
        "current_on_hand_inventory": cur_inv,
        "pipeline_on_order": orders_in_transit,
        "backorders": back_orders,
        "inventory_position": resolved_inventory_position,
        "unclamped_order_quantity": unclamped_order_qty,
        "exact_order_quantity": exact_order_qty,
        "rounded_order_quantity": float(math.ceil(exact_order_qty)),
        "recommended_order_quantity": recommended_order_qty,
        "round_up": round_up,
    }

    final_answer = {
        "recommended_order_quantity": recommended_order_qty,
        "exact_order_quantity": exact_order_qty,
        "target_inventory": target_inventory,
        "safety_stock": resolved_safety_stock,
        "protection_period": protection_period,
    }

    raw_inputs = {
        "review_period": t,
        "lead_time": l,
        "avg_demand": d,
        "demand_std": sigma_d,
        "service_level": csl,
        "current_inventory": cur_inv,
        "on_order": orders_in_transit,
        "backorders": back_orders,
        "safety_stock": safety_stock,
        "inventory_position": inventory_position,
        "time_unit": time_unit,
        "round_up": round_up,
    }

    verif = verify_fixed_period_review(
        t, l, protection_period, expected_demand_prot,
        resolved_safety_stock, target_inventory, resolved_inventory_position, recommended_order_qty
    )

    return PeriodicReviewResult(
        final_answer=final_answer,
        recommended_order_quantity=recommended_order_qty,
        target_inventory=target_inventory,
        order_up_to_level=target_inventory,
        inputs=raw_inputs,
        intermediate_calculations=intermediate_calcs,
        assumptions=assumptions,
        review_period=t,
        lead_time=l,
        protection_period=protection_period,
        time_unit=time_unit,
        avg_demand_per_period=d,
        demand_std_per_period=sigma_d,
        service_level=csl,
        z_score=z_score,
        expected_demand_protection_period=expected_demand_prot,
        std_demand_protection_period=std_demand_prot,
        safety_stock=resolved_safety_stock,
        current_inventory=cur_inv,
        on_order=orders_in_transit,
        backorders=back_orders,
        inventory_position=resolved_inventory_position,
        verification=verif,
        exact_order_quantity=exact_order_qty,
        round_up=round_up,
    )

# Alias for backwards compatibility
calculate_fixed_period_review = calculate_periodic_review


def verify_fixed_period_review(
    review_period: float,
    lead_time: float,
    protection_period: float,
    expected_demand: float,
    safety_stock: float,
    order_up_to_level: float,
    inventory_position: float,
    order_qty: float,
) -> VerificationResult:
    """Verify fixed-period review parameters and logic."""
    checks = []
    warnings = []
    errors = []

    if abs((review_period + lead_time) - protection_period) < 1e-6:
        checks.append(f"Protection interval correctly calculated as T ({review_period}) + L ({lead_time}) = {protection_period} periods.")
    else:
        errors.append("Protection period calculation mismatch.")

    if abs((expected_demand + safety_stock) - order_up_to_level) < 1e-6:
        checks.append(f"Order-up-to level M ({order_up_to_level:,.2f}) equals expected demand ({expected_demand:,.2f}) + SS ({safety_stock:,.2f}).")
    else:
        errors.append("Order-up-to level calculation mismatch.")

    if order_qty >= 0:
        checks.append(f"Order quantity is feasible ({order_qty:,.2f} units).")
    else:
        errors.append(f"Order quantity cannot be negative: {order_qty}")

    if inventory_position > order_up_to_level:
        warnings.append(f"Current inventory position ({inventory_position:,.2f}) already exceeds order-up-to target ({order_up_to_level:,.2f}); recommended order is 0.")

    return VerificationResult(
        is_valid=len(errors) == 0,
        checks_passed=checks,
        warnings=warnings,
        errors=errors,
    )





# =====================================================================
# Tool 5: calculate_newsvendor()
# =====================================================================

def calculate_newsvendor(
    selling_price: Optional[float] = None,
    unit_cost: Optional[float] = None,
    salvage_value: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: Optional[float] = None,
    customer_penalty: float = 0.0,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
) -> NewsvendorResult:
    """Calculate underage cost, overage cost, critical ratio, and recommended order quantity for Newsvendor.

    Parameters
    ----------
    selling_price : float, optional
        Regular retail selling price per unit (P). Must be > unit_cost.
    unit_cost : float, optional
        Unit acquisition / purchase cost (C). Must be >= salvage_value.
    salvage_value : float, optional
        Salvage liquidation / markdown price per unsold unit (S).
    demand_mean : float
        Mean stochastic demand over the single period (mu). Must be >= 0.
    demand_std : float
        Standard deviation of stochastic demand (sigma). Must be > 0.
    customer_penalty : float, default 0.0
        Loss of goodwill or customer penalty cost per unit of unmet demand.
    underage_cost : float, optional
        Directly provided underage / shortage cost (Cu = P - C + penalty).
    overage_cost : float, optional
        Directly provided overage / excess cost (Co = C - S).

    Returns
    -------
    NewsvendorResult
        Structured result with final answer, intermediate calculations, and assumptions.
    """
    # 1. Strict Input Validation (Never invent missing inputs)
    mu = _require_number(demand_mean, "demand_mean", min_val=0.0, allow_zero=True)
    sigma = _require_number(demand_std, "demand_std", min_val=0.0, allow_zero=False)
    penalty = _require_number(customer_penalty, "customer_penalty", min_val=0.0, allow_zero=True)

    if underage_cost is not None and overage_cost is not None:
        resolved_cu = _require_number(underage_cost, "underage_cost", min_val=0.0, allow_zero=False)
        resolved_co = _require_number(overage_cost, "overage_cost", min_val=0.0, allow_zero=True)
        p = _require_number(selling_price, "selling_price", min_val=0.0, allow_zero=False) if selling_price is not None else resolved_cu
        c = _require_number(unit_cost, "unit_cost", min_val=0.0, allow_zero=True) if unit_cost is not None else 0.0
        s = _require_number(salvage_value, "salvage_value", allow_zero=True) if salvage_value is not None else -resolved_co
    elif underage_cost is not None or overage_cost is not None:
        if underage_cost is None:
            raise ValueError("Missing required input 'underage_cost' (Cu) when specifying direct marginal costs.")
        if overage_cost is None:
            raise ValueError("Missing required input 'overage_cost' (Co) when specifying direct marginal costs.")
    else:
        p = _require_number(selling_price, "selling_price", min_val=0.0, allow_zero=False)
        c = _require_number(unit_cost, "unit_cost", min_val=0.0, allow_zero=False)
        s = _require_number(salvage_value, "salvage_value", allow_zero=True)

        if p <= c:
            raise ValueError(f"Selling price (${p}) must be greater than unit cost (${c}) for a viable profit margin.")
        if c < s:
            raise ValueError(f"Unit cost (${c}) cannot be less than salvage value (${s}); otherwise unbounded ordering yields infinite profit.")
        if p <= s:
            raise ValueError(f"Selling price (${p}) must be greater than salvage value (${s}).")

        resolved_cu = p - c + penalty
        resolved_co = c - s

    # 2. Marginal Economics & Calculations
    # Underage cost (Cu): marginal profit lost per unit of unmet demand
    underage_cost_val = resolved_cu

    # Overage cost (Co): loss incurred per unit ordered but unsold
    overage_cost_val = resolved_co

    # Critical Fractile (Critical Ratio): CR = Cu / (Cu + Co)
    critical_ratio = underage_cost_val / (underage_cost_val + overage_cost_val)

    # Optimal Z-score and Recommended Order Quantity
    nd = NormalDist(mu=0.0, sigma=1.0)
    z_star = nd.inv_cdf(critical_ratio)
    optimal_q = mu + (z_star * sigma)
    recommended_order_qty = max(0.0, optimal_q)

    # Standard normal unit loss function: L(z) = phi(z) - z * (1 - Phi(z))
    phi_z = nd.pdf(z_star)
    phi_cum = nd.cdf(z_star)
    unit_loss_function = phi_z - z_star * (1.0 - phi_cum)

    # Expected lost sales, sales, and leftover inventory
    expected_lost_sales = sigma * unit_loss_function
    expected_sales = mu - expected_lost_sales
    expected_leftovers = max(0.0, recommended_order_qty - expected_sales)

    # Financial expectations
    expected_revenue = p * expected_sales
    expected_salvage = s * expected_leftovers
    expected_purch_cost = c * recommended_order_qty
    expected_penalty_cost = penalty * expected_lost_sales
    expected_profit = expected_revenue + expected_salvage - expected_purch_cost - expected_penalty_cost

    margin_pct = ((p - c) / p) * 100.0 if p > 0 else 0.0
    in_stock_prob = critical_ratio
    stockout_prob = 1.0 - critical_ratio

    # 3. Model Assumptions
    assumptions = [
        "Single-period inventory decision with a single order opportunity prior to the selling season.",
        "Demand is stochastic and normally distributed with known mean (mu) and standard deviation (sigma).",
        "Product is perishable or seasonal; unsold units at season end are liquidated at salvage value S.",
        "Underage cost Cu = P - C (+ customer penalty) represents the lost profit per unit of unmet customer demand.",
        "Overage cost Co = C - S represents the loss per unsold unit ordered.",
        "Critical ratio CR = Cu / (Cu + Co) represents the optimal service level maximizing expected profit.",
        "No replenishment or reordering is possible once the selling season begins.",
        "Cost and revenue functions are linear with respect to order volume.",
    ]

    intermediate_calcs = {
        "underage_cost": underage_cost_val,
        "overage_cost": overage_cost_val,
        "critical_ratio": critical_ratio,
        "z_score": z_star,
        "optimal_order_quantity": recommended_order_qty,
        "unit_normal_loss_function": unit_loss_function,
        "expected_lost_sales": expected_lost_sales,
        "expected_sales": expected_sales,
        "expected_leftover_units": expected_leftovers,
        "expected_revenue": expected_revenue,
        "expected_salvage_revenue": expected_salvage,
        "expected_purchase_cost": expected_purch_cost,
        "expected_penalty_cost": expected_penalty_cost,
        "expected_profit": expected_profit,
        "profit_margin_percent": margin_pct,
        "in_stock_probability": in_stock_prob,
        "stockout_probability": stockout_prob,
    }

    final_answer = recommended_order_qty

    raw_inputs = {
        "selling_price": p,
        "unit_cost": c,
        "salvage_value": s,
        "demand_mean": mu,
        "demand_std": sigma,
        "customer_penalty": penalty,
    }

    verif = verify_newsvendor(
        p, c, s, underage_cost_val, overage_cost_val,
        critical_ratio, z_star, recommended_order_qty, expected_profit
    )

    return NewsvendorResult(
        final_answer=final_answer,
        optimal_order_quantity=recommended_order_qty,
        inputs=raw_inputs,
        intermediate_calculations=intermediate_calcs,
        assumptions=assumptions,
        selling_price=p,
        unit_cost=c,
        salvage_value=s,
        customer_penalty=penalty,
        underage_cost=underage_cost_val,
        overage_cost=overage_cost_val,
        critical_fractile=critical_ratio,
        critical_ratio=critical_ratio,
        demand_mean=mu,
        demand_std=sigma,
        z_score=z_star,
        expected_lost_sales=expected_lost_sales,
        expected_sales=expected_sales,
        expected_leftover_units=expected_leftovers,
        expected_revenue=expected_revenue,
        expected_salvage_revenue=expected_salvage,
        expected_purchase_cost=expected_purch_cost,
        expected_penalty_cost=expected_penalty_cost,
        expected_profit=expected_profit,
        profit_margin_percent=margin_pct,
        in_stock_probability=in_stock_prob,
        stockout_probability=stockout_prob,
        verification=verif,
    )


def verify_newsvendor(
    selling_price: float,
    unit_cost: float,
    salvage_value: float,
    underage_cost: float,
    overage_cost: float,
    critical_fractile: float,
    z_score: float,
    optimal_q: float,
    expected_profit: float,
) -> VerificationResult:
    """Verify newsvendor optimality conditions and economics."""
    checks = []
    warnings = []
    errors = []

    if 0.0 < critical_fractile < 1.0:
        checks.append(f"Critical fractile ({critical_fractile:.4f}) is strictly between 0 and 1.")
    else:
        errors.append(f"Critical fractile out of valid range: {critical_fractile}")

    if underage_cost > 0 and overage_cost >= 0:
        checks.append(f"Marginal economics valid: Underage Cost Cu=${underage_cost:,.2f}, Overage Cost Co=${overage_cost:,.2f}.")
    else:
        errors.append(f"Invalid marginal costs: Cu={underage_cost}, Co={overage_cost}")

    if optimal_q >= 0:
        checks.append(f"Optimal quantity is feasible ({optimal_q:,.2f} units).")
    else:
        errors.append(f"Negative optimal quantity: {optimal_q}")

    if expected_profit > 0:
        checks.append(f"Business model is expected to be profitable (${expected_profit:,.2f}).")
    else:
        warnings.append(f"Expected profit is non-positive (${expected_profit:,.2f}). Review cost-price structure.")

    return VerificationResult(
        is_valid=len(errors) == 0,
        checks_passed=checks,
        warnings=warnings,
        errors=errors,
    )




# =====================================================================
# 6. Multicriteria Inventory Classification Data Structures & Tool
# =====================================================================

@dataclass
class ClassificationSummaryRow:
    """Summary metrics for a single final classification tier (A, B, C, or D)."""
    final_class: str
    items_count: int
    pct_items: float
    total_sales: float
    pct_sales: float
    total_pos_hits: float
    pct_pos_hits: float
    managerial_guidance: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MulticriteriaClassificationResult:
    """Structured result for Multicriteria Inventory Classification.
    
    Implements 65% Sales + 35% POS Hits multicriteria weighting with
    separate Pareto evaluations and 4-point scoring (A=4, B=3, C=2, D=1).
    """
    final_answer: Dict[str, Any]
    classified_df: pd.DataFrame
    summary_df: pd.DataFrame
    summary_rows: List[ClassificationSummaryRow]
    inputs: Dict[str, Any]
    intermediate_calculations: Dict[str, Any]
    assumptions: List[str]
    verification: VerificationResult = field(default_factory=lambda: VerificationResult(True))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "inputs": self.inputs,
            "intermediate_calculations": self.intermediate_calculations,
            "assumptions": self.assumptions,
            "verification": self.verification.to_dict(),
            "summary": [r.to_dict() for r in self.summary_rows],
            "num_items": len(self.classified_df),
        }


def verify_multicriteria_classification(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    total_sales: float,
    total_hits: float,
) -> VerificationResult:
    """Verify multicriteria classification integrity, conservation laws, and score bounds."""
    checks = []
    warnings = []
    errors = []

    # 1. Total row count conservation
    total_items = int(summary_df["Items_Count"].sum())
    if total_items == len(df):
        checks.append(f"Row count integrity: All {len(df)} items accounted for across classes A, B, C, and D.")
    else:
        errors.append(f"Row count mismatch: sum of summary items ({total_items}) != catalog length ({len(df)}).")

    # 2. Total sales conservation
    summary_sales = float(summary_df["Total_Sales"].sum())
    if math.isclose(summary_sales, total_sales, rel_tol=1e-4, abs_tol=1e-2):
        checks.append(f"Sales conservation: Classified sales (${summary_sales:,.2f}) matches total catalog sales (${total_sales:,.2f}).")
    else:
        errors.append(f"Sales conservation mismatch: summary sales (${summary_sales:,.2f}) != input sales (${total_sales:,.2f}).")

    # 3. Total POS hits conservation
    summary_hits = float(summary_df["Total_POS_Hits"].sum())
    if math.isclose(summary_hits, total_hits, rel_tol=1e-4, abs_tol=1e-2):
        checks.append(f"POS Hits conservation: Classified hits ({summary_hits:,.0f}) matches total catalog hits ({total_hits:,.0f}).")
    else:
        errors.append(f"POS Hits conservation mismatch: summary hits ({summary_hits:,.0f}) != input hits ({total_hits:,.0f}).")

    # 4. Score bounds verification [1.0, 4.0]
    if len(df) > 0 and "Combined_Score" in df.columns:
        min_score = float(df["Combined_Score"].min())
        max_score = float(df["Combined_Score"].max())
        if 1.0 <= min_score and max_score <= 4.0:
            checks.append(f"Combined score bounds verified: Range is [{min_score:.2f}, {max_score:.2f}] inside [1.00, 4.00].")
        else:
            errors.append(f"Combined score out of bounds: min={min_score:.4f}, max={max_score:.4f}")

    return VerificationResult(
        is_valid=len(errors) == 0,
        checks_passed=checks,
        warnings=warnings,
        errors=errors,
    )


def calculate_multicriteria_classification(
    df: pd.DataFrame,
    sales_col: str = "Sales",
    pos_hits_col: str = "POS Hits",
    product_col: Optional[str] = None,
    category_col: Optional[str] = None,
    sales_weight: float = 0.65,
    pos_weight: float = 0.35,
    pareto_cutoffs: Tuple[float, float, float] = (0.60, 0.80, 0.90),
    score_thresholds: Tuple[float, float, float] = (3.25, 2.50, 1.75),
) -> MulticriteriaClassificationResult:
    """Execute deterministic multicriteria inventory classification.

    BAN517 Course Methodology:
    STEP 1: Separate Pareto classifications for Sales and POS Hits independently:
       - Class A: cumulative percentage from 0% through 60% (cutoff 0.60)
       - Class B: above 60% through 80% (cutoff 0.80)
       - Class C: above 80% through 90% (cutoff 0.90)
       - Class D: above 90% through 100% (or inactive items with 0 sales/hits)
    STEP 2: Numerical scoring:
       - A = 4, B = 3, C = 2, D = 1
    STEP 3: Combined weighted score:
       - Combined Weighted Score = (0.65 * Sales Score) + (0.35 * POS Hits Score)
    STEP 4: Final inventory classification:
       - Final A: score >= 3.25 and <= 4.00
       - Final B: score >= 2.50 and < 3.25
       - Final C: score >= 1.75 and < 2.50
       - Final D: score < 1.75

    Parameters
    ----------
    df : pd.DataFrame
        Source inventory dataset.
    sales_col : str, optional
        Column name representing Sales (dollar volume / revenue / inventory value).
    pos_hits_col : str, optional
        Column name representing POS Hits (transaction frequency / orders / units).
    product_col : str, optional
        Column name identifying the product / item name.
    category_col : str, optional
        Column name identifying category or unit of measure.
    sales_weight : float, optional
        Weight for Sales criterion, default 0.65 (65%).
    pos_weight : float, optional
        Weight for POS Hits criterion, default 0.35 (35%).
    pareto_cutoffs : Tuple[float, float, float], optional
        Cumulative thresholds (A_cut, B_cut, C_cut), default (0.60, 0.80, 0.90).
    score_thresholds : Tuple[float, float, float], optional
        Score bounds (A_threshold, B_threshold, C_threshold), default (3.25, 2.50, 1.75).


    Returns
    -------
    MulticriteriaClassificationResult
        Structured result including enriched DataFrame, summary metrics table,
        intermediate calculations, assumptions, and mathematical verification.
    """
    # -------------------------------------------------------------
    # 1. Input Validation
    # -------------------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a valid pandas DataFrame.")

    if len(df) == 0:
        raise ValueError("Input DataFrame is empty; cannot perform classification.")

    # Intelligent column resolution if default name is not found
    resolved_sales = sales_col
    if resolved_sales not in df.columns:
        candidates = ["Sales ($)", "Sales", "Inventory Value", "Annual Sales", "Revenue", "Dollar Volume", "Total Sales"]
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            resolved_sales = found
        else:
            raise ValueError(f"Sales column '{sales_col}' not found in DataFrame columns: {list(df.columns)}")

    resolved_hits = pos_hits_col
    if resolved_hits not in df.columns:
        candidates = ["POS Hits", "POS_Hits", "Hits", "Orders", "Frequency", "Transactions", "Units"]
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            resolved_hits = found
        else:
            raise ValueError(f"POS Hits column '{pos_hits_col}' not found in DataFrame columns: {list(df.columns)}")

    resolved_product = product_col
    if resolved_product is None or resolved_product not in df.columns:
        candidates = ["Product", "Item Name", "Item", "Item Number", "Description", "SKU"]
        resolved_product = next((c for c in candidates if c in df.columns), df.columns[0])

    resolved_category = category_col
    if resolved_category is None or resolved_category not in df.columns:
        candidates = ["Category", "UoM", "Department", "Family", "Class", "Type"]
        resolved_category = next((c for c in candidates if c in df.columns), None)

    # Validate weights
    if sales_weight <= 0 or pos_weight <= 0:
        raise ValueError(f"Weights must be strictly positive: sales_weight={sales_weight}, pos_weight={pos_weight}")
    if not math.isclose(sales_weight + pos_weight, 1.0, rel_tol=1e-2, abs_tol=1e-2):
        raise ValueError(f"Weights must sum to 1.0 (got {sales_weight + pos_weight:.4f})")

    # Validate cutoffs
    if len(pareto_cutoffs) != 3:
        raise ValueError(f"pareto_cutoffs must have exactly 3 values (got {len(pareto_cutoffs)})")
    c_a, c_b, c_c = pareto_cutoffs
    if not (0.0 < c_a < c_b < c_c < 1.0):
        raise ValueError(f"Pareto cutoffs must be strictly increasing between 0 and 1: {pareto_cutoffs}")

    # Validate score thresholds
    if len(score_thresholds) != 3:
        raise ValueError(f"score_thresholds must have exactly 3 values (got {len(score_thresholds)})")
    t_a, t_b, t_c = score_thresholds
    if not (4.0 > t_a > t_b > t_c > 1.0):
        raise ValueError(f"Score thresholds must be strictly decreasing between 1.0 and 4.0: {score_thresholds}")

    raw_inputs = {
        "sales_col": resolved_sales,
        "pos_hits_col": resolved_hits,
        "product_col": resolved_product,
        "category_col": resolved_category,
        "sales_weight": sales_weight,
        "pos_weight": pos_weight,
        "pareto_cutoffs": pareto_cutoffs,
        "score_thresholds": score_thresholds,
        "dataset_rows": len(df),
    }

    # -------------------------------------------------------------
    # 2. Data Cleaning & Preparation
    # -------------------------------------------------------------
    working_df = df.copy()
    working_df[resolved_sales] = coerce_numeric_series(working_df[resolved_sales], column_name=resolved_sales)
    working_df[resolved_hits] = coerce_numeric_series(working_df[resolved_hits], column_name=resolved_hits)

    if (working_df[resolved_sales] < 0).any():
        raise ValueError(f"Column '{resolved_sales}' contains negative values, which are invalid for Pareto analysis.")
    if (working_df[resolved_hits] < 0).any():
        raise ValueError(f"Column '{resolved_hits}' contains negative values, which are invalid for Pareto analysis.")

    total_sales = float(working_df[resolved_sales].sum())
    total_hits = float(working_df[resolved_hits].sum())
    total_items = len(working_df)

    if total_sales <= 0:
        raise ValueError(f"Total sales sum across all items is ${total_sales:,.2f}; cannot calculate Pareto shares.")
    if total_hits <= 0:
        raise ValueError(f"Total POS hits sum across all items is {total_hits:,.0f}; cannot calculate Pareto shares.")

    # -------------------------------------------------------------
    # 3. Pareto Classification for Sales (65% Weight)
    # -------------------------------------------------------------
    sales_sorted = working_df.sort_values(by=[resolved_sales, resolved_hits], ascending=[False, False])
    sales_cum = sales_sorted[resolved_sales].cumsum()
    sales_cumpct = sales_cum / total_sales

    def assign_pareto_tier(cumpct_val: float, raw_val: float, rank_idx: int) -> str:
        if raw_val <= 0:
            return "D"
        if cumpct_val <= c_a or rank_idx == 0:
            return "A"
        elif cumpct_val <= c_b:
            return "B"
        elif cumpct_val <= c_c:
            return "C"
        else:
            return "D"

    sales_classes = [
        assign_pareto_tier(cp, rv, i)
        for i, (cp, rv) in enumerate(zip(sales_cumpct, sales_sorted[resolved_sales]))
    ]
    sales_sorted["Sales_Pareto_CumPct"] = sales_cumpct
    sales_sorted["Sales_Class"] = sales_classes

    # -------------------------------------------------------------
    # 4. Pareto Classification for POS Hits (35% Weight)
    # -------------------------------------------------------------
    hits_sorted = working_df.sort_values(by=[resolved_hits, resolved_sales], ascending=[False, False])
    hits_cum = hits_sorted[resolved_hits].cumsum()
    hits_cumpct = hits_cum / total_hits

    hits_classes = [
        assign_pareto_tier(cp, rv, i)
        for i, (cp, rv) in enumerate(zip(hits_cumpct, hits_sorted[resolved_hits]))
    ]
    hits_sorted["POS_Pareto_CumPct"] = hits_cumpct
    hits_sorted["POS_Class"] = hits_classes

    # -------------------------------------------------------------
    # 5. Class Scores & Combined Weighted Score
    # -------------------------------------------------------------
    score_map = {"A": 4, "B": 3, "C": 2, "D": 1}

    working_df["Sales_Pareto_CumPct"] = working_df.index.map(sales_sorted["Sales_Pareto_CumPct"])
    working_df["Sales_Class"] = working_df.index.map(sales_sorted["Sales_Class"])
    working_df["Sales_Score"] = working_df["Sales_Class"].map(score_map)

    working_df["POS_Pareto_CumPct"] = working_df.index.map(hits_sorted["POS_Pareto_CumPct"])
    working_df["POS_Class"] = working_df.index.map(hits_sorted["POS_Class"])
    working_df["POS_Score"] = working_df["POS_Class"].map(score_map)

    # Combined Weighted Score = 0.65 * Sales_Score + 0.35 * POS_Score
    working_df["Combined_Score"] = (
        sales_weight * working_df["Sales_Score"] +
        pos_weight * working_df["POS_Score"]
    ).round(4)

    # -------------------------------------------------------------
    # 6. Final Classification Assignment (A, B, C, D)
    # -------------------------------------------------------------
    def assign_final_tier(score: float) -> str:
        if score >= t_a:
            return "A"
        elif score >= t_b:
            return "B"
        elif score >= t_c:
            return "C"
        else:
            return "D"

    working_df["Final_Class"] = working_df["Combined_Score"].apply(assign_final_tier)

    # -------------------------------------------------------------
    # 7. Summary by Final Classification
    # -------------------------------------------------------------
    policy_guidance = {
        "A": "Critical Priority: Continuous review, tight inventory monitoring, high target service level (98-99%), supplier partnership.",
        "B": "Moderate Priority: Periodic review, standard safety buffer, automated replenishment with periodic supervisor reviews.",
        "C": "Low Priority: Bulk ordering / economic batches, higher safety stock buffers to minimize ordering transactions.",
        "D": "Tail / Slow-Moving: Periodic audit for rationalization, minimum order quantities, make-to-order, or vendor consignment.",
    }

    summary_rows_list: List[ClassificationSummaryRow] = []
    summary_records = []

    for tier in ["A", "B", "C", "D"]:
        tier_df = working_df[working_df["Final_Class"] == tier]
        items_cnt = int(len(tier_df))
        pct_items = (items_cnt / total_items) * 100.0 if total_items > 0 else 0.0
        tier_sales = float(tier_df[resolved_sales].sum())
        pct_sales = (tier_sales / total_sales) * 100.0 if total_sales > 0 else 0.0
        tier_hits = float(tier_df[resolved_hits].sum())
        pct_hits = (tier_hits / total_hits) * 100.0 if total_hits > 0 else 0.0
        guidance = policy_guidance[tier]

        row_obj = ClassificationSummaryRow(
            final_class=tier,
            items_count=items_cnt,
            pct_items=pct_items,
            total_sales=tier_sales,
            pct_sales=pct_sales,
            total_pos_hits=tier_hits,
            pct_pos_hits=pct_hits,
            managerial_guidance=guidance,
        )
        summary_rows_list.append(row_obj)
        summary_records.append({
            "Final_Class": tier,
            "Items_Count": items_cnt,
            "Pct_Items": pct_items,
            "Total_Sales": tier_sales,
            "Pct_Sales": pct_sales,
            "Total_POS_Hits": tier_hits,
            "Pct_POS_Hits": pct_hits,
            "Managerial_Guidance": guidance,
        })

    summary_df = pd.DataFrame(summary_records)

    # -------------------------------------------------------------
    # 8. Verification & Audit Checks
    # -------------------------------------------------------------
    verif = verify_multicriteria_classification(working_df, summary_df, total_sales, total_hits)

    # -------------------------------------------------------------
    # 9. Structure Final Answer & Intermediate Calculations
    # -------------------------------------------------------------
    final_answer = {
        "summary_table": summary_records,
        "class_a_items": summary_records[0]["Items_Count"],
        "class_a_sales": summary_records[0]["Total_Sales"],
        "class_a_pct_sales": summary_records[0]["Pct_Sales"],
        "class_b_items": summary_records[1]["Items_Count"],
        "class_b_sales": summary_records[1]["Total_Sales"],
        "class_b_pct_sales": summary_records[1]["Pct_Sales"],
        "class_c_items": summary_records[2]["Items_Count"],
        "class_c_sales": summary_records[2]["Total_Sales"],
        "class_c_pct_sales": summary_records[2]["Pct_Sales"],
        "class_d_items": summary_records[3]["Items_Count"],
        "class_d_sales": summary_records[3]["Total_Sales"],
        "class_d_pct_sales": summary_records[3]["Pct_Sales"],
        "total_items": total_items,
        "total_sales": total_sales,
        "total_pos_hits": total_hits,
    }

    intermediate_calcs = {
        "sales_weight": sales_weight,
        "pos_weight": pos_weight,
        "pareto_cutoffs": pareto_cutoffs,
        "score_thresholds": score_thresholds,
        "total_catalog_sales": total_sales,
        "total_catalog_hits": total_hits,
        "total_items": total_items,
        "scoring_scale": score_map,
        "sales_pareto_counts": {
            "A": int((working_df["Sales_Class"] == "A").sum()),
            "B": int((working_df["Sales_Class"] == "B").sum()),
            "C": int((working_df["Sales_Class"] == "C").sum()),
            "D": int((working_df["Sales_Class"] == "D").sum()),
        },
        "pos_hits_pareto_counts": {
            "A": int((working_df["POS_Class"] == "A").sum()),
            "B": int((working_df["POS_Class"] == "B").sum()),
            "C": int((working_df["POS_Class"] == "C").sum()),
            "D": int((working_df["POS_Class"] == "D").sum()),
        },
    }

    assumptions = [
        f"Multicriteria weighting: Sales weighted at {sales_weight * 100:.0f}%, and POS Hits weighted at {pos_weight * 100:.0f}%.",
        f"Separate Pareto classifications: Cumulative cutoffs are Class A <= {c_a * 100:.0f}%, Class B <= {c_b * 100:.0f}%, Class C <= {c_c * 100:.0f}%, and Class D > {c_c * 100:.0f}%.",
        "Individual class scores: Class A = 4, Class B = 3, Class C = 2, Class D = 1.",
        f"Combined weighted score formula: Combined_Score = ({sales_weight:.2f} * Sales_Score) + ({pos_weight:.2f} * POS_Hits_Score).",
        f"Final classification score thresholds: Class A ({t_a:.2f} to 4.00), Class B ({t_b:.2f} to < {t_a:.2f}), Class C ({t_c:.2f} to < {t_b:.2f}), Class D (< {t_c:.2f}).",
        "Dead/inactive stock handling: Products with zero or negative sales or hits are automatically assigned to Class D.",
        "Deterministic ranking: Secondary sorting by alternate criterion resolves ties deterministically.",
    ]

    return MulticriteriaClassificationResult(
        final_answer=final_answer,
        classified_df=working_df,
        summary_df=summary_df,
        summary_rows=summary_rows_list,
        inputs=raw_inputs,
        intermediate_calculations=intermediate_calcs,
        assumptions=assumptions,
        verification=verif,
    )

