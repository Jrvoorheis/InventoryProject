"""Supply Chain Analytics Inventory Decision Agent.

Implements the 7-stage architectural decision pipeline:
    Customer Question
    -> 1. Identify Model
    -> 2. Extract Inputs
    -> 3. Select Python Tool
    -> 4. Calculate
    -> 5. Verify
    -> 6. Explain

Supports the 5 core inventory models:
    - Economic Order Quantity (EOQ)
    - Quantity Discount
    - Safety Stock and Reorder Point (ROP)
    - Fixed-Period Review (Periodic Review / Order-Up-To)
    - Newsvendor (Single-Period Stochastic Demand)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from inventory_tools import (
    EOQResult,
    FixedPeriodReviewResult,
    NewsvendorResult,
    PeriodicReviewResult,
    QuantityDiscountResult,
    ROPResult,
    SafetyStockROPResult,
    VerificationResult,
    calculate_eoq,
    calculate_fixed_period_review,
    calculate_newsvendor,
    calculate_periodic_review,
    calculate_quantity_discount,
    calculate_rop,
    calculate_safety_stock_rop,
)


class InventoryModelType(str, Enum):
    """Supported inventory decision models."""
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point (Continuous Review)"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review (Periodic Order-Up-To)"
    NEWSVENDOR = "Newsvendor Model (Single-Period Stochastic)"
    UNKNOWN = "Unknown / General Inventory Inquiry"


@dataclass
class AgentStepLog:
    """Record of execution at each architectural stage."""
    step_name: str
    status: str  # 'SUCCESS', 'SKIPPED', 'WARNING', 'FAILED'
    details: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


@dataclass
class AgentResponse:
    """Complete response emitted by the Inventory Decision Agent."""
    question: str
    identified_model: InventoryModelType
    confidence: float
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    selected_tool_name: Optional[str]
    calculation_result: Optional[Union[EOQResult, QuantityDiscountResult, SafetyStockROPResult, FixedPeriodReviewResult, NewsvendorResult]]
    verification_result: Optional[VerificationResult]
    explanation: str
    steps_log: List[AgentStepLog] = field(default_factory=list)
    needs_clarification: bool = False
    clarification_message: Optional[str] = None


class SpanTracker:
    """Helper to track matched character spans in queries to prevent cross-contamination."""

    def __init__(self, text: str):
        self.text = text
        self.claimed_spans: List[Tuple[int, int]] = []

    def extract_number(self, patterns: List[str], claim: bool = True) -> Optional[float]:
        """Search regex patterns and extract first number not overlapping claimed spans."""
        for pattern in patterns:
            for match in re.finditer(pattern, self.text, re.IGNORECASE):
                if not match.groups():
                    continue
                val_start, val_end = match.span(1)
                # Check overlap of captured group with previously claimed spans
                if any(not (val_end <= s or val_start >= e) for s, e in self.claimed_spans):
                    continue
                raw = match.group(1).replace(",", "").strip()
                try:
                    val = float(raw)
                    if claim:
                        self.claimed_spans.append(match.span(0))
                    return val
                except ValueError:
                    continue
        return None


class InventoryDecisionAgent:
    """Intelligent agent for supply chain inventory analytics and decision optimization."""

    def __init__(self, api_key: Optional[str] = None, use_llm_if_available: bool = True):
        """Initialize the decision agent.

        Parameters
        ----------
        api_key : str, optional
            Optional API key for external LLM parsing (e.g. Gemini).
        use_llm_if_available : bool, default True
            Whether to leverage LLM parsing if configured. Defaults to fallback rule-based NLP engine.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.use_llm = use_llm_if_available and bool(self.api_key)

    # -----------------------------------------------------------------
    # Primary Pipeline Method
    # -----------------------------------------------------------------

    def run(self, question: str) -> AgentResponse:
        """Execute the end-to-end 6-stage architecture pipeline on a user query.

        Parameters
        ----------
        question : str
            Natural language inquiry from supply chain manager or analyst.

        Returns
        -------
        AgentResponse
            Structured response containing inputs, calculations, verification, and explanation.
        """
        steps_log: List[AgentStepLog] = []

        # Stage 1: Identify Model
        model_type, confidence, model_reasoning = self._identify_model(question)
        steps_log.append(AgentStepLog(
            step_name="1. Identify Model",
            status="SUCCESS",
            details={"model": model_type.value, "confidence": confidence, "reasoning": model_reasoning},
            summary=f"Identified {model_type.value} with confidence {confidence:.0%}.",
        ))

        if model_type == InventoryModelType.UNKNOWN:
            explanation = self._explain_unknown_query(question)
            return AgentResponse(
                question=question,
                identified_model=model_type,
                confidence=confidence,
                extracted_inputs={},
                missing_inputs=[],
                selected_tool_name=None,
                calculation_result=None,
                verification_result=None,
                explanation=explanation,
                steps_log=steps_log,
                needs_clarification=True,
                clarification_message=explanation,
            )

        # Stage 2: Extract Inputs
        extracted_inputs, missing_inputs = self._extract_inputs(question, model_type)
        steps_log.append(AgentStepLog(
            step_name="2. Extract Inputs",
            status="WARNING" if missing_inputs else "SUCCESS",
            details={"extracted": extracted_inputs, "missing": missing_inputs},
            summary=f"Extracted {len(extracted_inputs)} parameters; {len(missing_inputs)} required fields missing.",
        ))

        # Check if missing required parameters require clarification
        if missing_inputs:
            clarification_msg = self._generate_clarification_prompt(model_type, extracted_inputs, missing_inputs)
            steps_log.append(AgentStepLog(
                step_name="Clarification Needed",
                status="WARNING",
                details={"missing": missing_inputs},
                summary="Query lacks critical inputs; requesting user clarification.",
            ))
            return AgentResponse(
                question=question,
                identified_model=model_type,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing_inputs,
                selected_tool_name=None,
                calculation_result=None,
                verification_result=None,
                explanation=clarification_msg,
                steps_log=steps_log,
                needs_clarification=True,
                clarification_message=clarification_msg,
            )

        # Stage 3: Select Python Tool
        tool_name, tool_callable = self._select_tool(model_type)
        steps_log.append(AgentStepLog(
            step_name="3. Select Python Tool",
            status="SUCCESS",
            details={"tool_name": tool_name},
            summary=f"Selected tool '{tool_name}' for execution.",
        ))

        # Stage 4: Calculate
        try:
            calc_result = self._execute_calculation(tool_callable, model_type, extracted_inputs)
            steps_log.append(AgentStepLog(
                step_name="4. Calculate",
                status="SUCCESS",
                details={"tool": tool_name},
                summary="Numerical calculation executed successfully.",
            ))
        except Exception as e:
            err_msg = f"Calculation failed in {tool_name}: {str(e)}"
            steps_log.append(AgentStepLog(
                step_name="4. Calculate",
                status="FAILED",
                details={"error": str(e)},
                summary=err_msg,
            ))
            return AgentResponse(
                question=question,
                identified_model=model_type,
                confidence=confidence,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                selected_tool_name=tool_name,
                calculation_result=None,
                verification_result=None,
                explanation=f"**Calculation Error**: {err_msg}. Please review the provided numbers.",
                steps_log=steps_log,
                needs_clarification=False,
            )

        # Stage 5: Verify
        verif_result = getattr(calc_result, "verification", None)
        steps_log.append(AgentStepLog(
            step_name="5. Verify",
            status="SUCCESS" if (verif_result and verif_result.is_valid) else "WARNING",
            details={
                "is_valid": verif_result.is_valid if verif_result else True,
                "checks": verif_result.checks_passed if verif_result else [],
                "warnings": verif_result.warnings if verif_result else [],
                "errors": verif_result.errors if verif_result else [],
            },
            summary="Optimality and feasibility conditions verified." if (verif_result and verif_result.is_valid) else "Verification flagged warnings.",
        ))

        # Stage 6: Explain
        explanation = self._generate_explanation(question, model_type, extracted_inputs, calc_result, verif_result)
        steps_log.append(AgentStepLog(
            step_name="6. Explain",
            status="SUCCESS",
            details={},
            summary="Executive explanation and managerial synthesis generated.",
        ))

        return AgentResponse(
            question=question,
            identified_model=model_type,
            confidence=confidence,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            selected_tool_name=tool_name,
            calculation_result=calc_result,
            verification_result=verif_result,
            explanation=explanation,
            steps_log=steps_log,
            needs_clarification=False,
        )

    # -----------------------------------------------------------------
    # Stage 1: Identify Model
    # -----------------------------------------------------------------

    def _identify_model(self, question: str) -> Tuple[InventoryModelType, float, str]:
        """Classify the user inquiry into one of the 5 inventory models."""
        q_lower = question.lower()

        scores: Dict[InventoryModelType, float] = {
            InventoryModelType.EOQ: 0.0,
            InventoryModelType.QUANTITY_DISCOUNT: 0.0,
            InventoryModelType.SAFETY_STOCK_ROP: 0.0,
            InventoryModelType.FIXED_PERIOD_REVIEW: 0.0,
            InventoryModelType.NEWSVENDOR: 0.0,
        }

        # Model 1: EOQ indicators
        if re.search(r"\beoq\b|economic order quantity", q_lower):
            scores[InventoryModelType.EOQ] += 4.5
        for term in [
            "ordering cost", "order cost", "setup cost", "setup expense",
            "cost per order", "cost of placing an order", "ordering/setup cost",
            "ordering / setup cost", "order / setup cost", "ordering and setup cost"
        ]:
            if term in q_lower:
                scores[InventoryModelType.EOQ] += 1.8
        for term in ["holding cost", "carrying cost", "storage cost", "holding expense", "carrying expense"]:
            if term in q_lower:
                scores[InventoryModelType.EOQ] += 1.2
        if "annual demand" in q_lower or "yearly demand" in q_lower or "annual usage" in q_lower:
            scores[InventoryModelType.EOQ] += 1.2
        if "optimal order quantity" in q_lower or "how many units should we order each time" in q_lower:
            scores[InventoryModelType.EOQ] += 1.0

        # Model 2: Quantity Discount indicators
        for term in [
            "quantity discount", "price break", "volume discount", "bulk discount",
            "discount schedule", "quantity break", "quantity breaks", "all-units discount",
            "all units discount", "price schedule", "pricing tiers", "discount tiers",
            "price tiers", "pricing brackets", "discount brackets", "price brackets",
            "price break schedule", "purchase discounts"
        ]:
            if term in q_lower:
                scores[InventoryModelType.QUANTITY_DISCOUNT] += 5.0

        tier_mentions = len(re.findall(r"\b(?:tier|bracket|level)\s*\d+\b", q_lower))
        if tier_mentions >= 2:
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 6.0
            scores[InventoryModelType.EOQ] -= 2.0
        elif tier_mentions == 1 or "tier" in q_lower or "bracket" in q_lower:
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 3.0

        tier_bracket_matches = len(re.findall(
            r"(?:(?:[qQ]|qty|quantity)?\s*[<>]=?\s*[0-9,]+|[0-9,]+\s*<=?\s*(?:[qQ]|qty|quantity)?\s*<=?\s*[0-9,]+|\b[0-9,]+\s*(?:-|to|\.\.\.|through|and)\s*[0-9,]+\b|\b[0-9,]+\+(?!\d)|\b[0-9,]+\s*(?:or\s*more|and\s*above|units\s*or\s*more)\b)",
            q_lower,
        ))
        if tier_bracket_matches >= 2 and ("$" in q_lower or "price" in q_lower or "@" in q_lower or "cost" in q_lower):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 5.0
            scores[InventoryModelType.EOQ] -= 1.0
        elif tier_bracket_matches >= 1 and ("$" in q_lower or "price" in q_lower or "@" in q_lower):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 3.0
        if "unit purchase price" in q_lower or "unit price" in q_lower:
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 1.0
        if "percentage holding cost" in q_lower or "fixed holding cost" in q_lower:
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 1.0

        # Model 3: Safety Stock & ROP indicators
        if "safety stock" in q_lower:
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 3.5
        if "reorder point" in q_lower or re.search(r"\brop\b", q_lower):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 4.0
        if "continuous review" in q_lower or "(s, q)" in q_lower or "(r, q)" in q_lower:
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 3.5
        if "lead-time standard deviation" in q_lower or "lead time standard deviation" in q_lower:
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 3.0
        if "demand standard deviation" in q_lower and ("lead time" in q_lower or "csl" in q_lower or "service level" in q_lower):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 2.5
        if "service level" in q_lower and ("lead time" in q_lower or "std" in q_lower or "standard deviation" in q_lower):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 1.5

        # Model 4: Fixed-Period Review indicators
        for term in [
            "fixed period", "fixed-period", "periodic review", "review period",
            "review interval", "review cycle", "order-up-to", "order up to",
            "target inventory", "base stock", "protection interval", "protection period"
        ]:
            if term in q_lower:
                scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 5.0
        if "reviewed every" in q_lower or "review every" in q_lower or "orders placed every" in q_lower:
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 3.5
        if "inventory position" in q_lower:
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 3.0
        if ("on hand" in q_lower or "on-hand" in q_lower or "in stock" in q_lower) and ("on order" in q_lower or "backorder" in q_lower or "review" in q_lower):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 2.5

        # Model 5: Newsvendor indicators
        if "newsvendor" in q_lower or "newsboy" in q_lower:
            scores[InventoryModelType.NEWSVENDOR] += 6.0
        for term in [
            "underage cost", "shortage cost", "overage cost", "excess cost",
            "cost of underage", "cost of shortage", "cost of overage", "cost of excess",
            "critical ratio", "critical fractile"
        ]:
            if term in q_lower:
                scores[InventoryModelType.NEWSVENDOR] += 4.5
        for term in [
            "salvage value", "salvage price", "scrap value", "markdown price",
            "liquidation value", "outlet price", "salvage", "salvaged", "liquidated"
        ]:
            if term in q_lower:
                scores[InventoryModelType.NEWSVENDOR] += 3.5
        if "single period" in q_lower or "single-period" in q_lower or "perishable" in q_lower or "seasonal" in q_lower:
            scores[InventoryModelType.NEWSVENDOR] += 3.0
        if ("selling price" in q_lower or "retail price" in q_lower) and ("purchase cost" in q_lower or "unit cost" in q_lower or "wholesale cost" in q_lower):
            scores[InventoryModelType.NEWSVENDOR] += 2.5

        # Determine winner
        best_model = max(scores, key=lambda m: scores[m])
        max_score = scores[best_model]

        if max_score < 2.0:
            return InventoryModelType.UNKNOWN, 0.2, "Insufficient distinct inventory model keywords found."

        # Compute relative confidence
        total_score = sum(scores.values())
        confidence = min(0.99, max_score / (total_score + 1e-6) + 0.35)
        reasoning = f"Matched key indicators for {best_model.value} with aggregate score {max_score:.1f}."

        return best_model, confidence, reasoning

    # -----------------------------------------------------------------
    # Stage 2: Extract Inputs
    # -----------------------------------------------------------------

    def _extract_inputs(self, question: str, model_type: InventoryModelType) -> Tuple[Dict[str, Any], List[str]]:
        """Extract numeric parameters corresponding to the identified model."""
        if model_type == InventoryModelType.EOQ:
            return self._extract_eoq_inputs(question)
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            return self._extract_quantity_discount_inputs(question)
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            return self._extract_safety_stock_rop_inputs(question)
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            return self._extract_fixed_period_inputs(question)
        elif model_type == InventoryModelType.NEWSVENDOR:
            return self._extract_newsvendor_inputs(question)
        return {}, []

    def _extract_number(self, text: str, patterns: List[str]) -> Optional[float]:
        """Helper to extract first floating-point number matching regex patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.groups():
                raw = match.group(1).replace(",", "").strip()
                try:
                    return float(raw)
                except ValueError:
                    continue
        return None

    def _extract_eoq_inputs(self, question: str) -> Tuple[Dict[str, Any], List[str]]:
        """Extract EOQ parameters: demand, order cost, holding cost, unit cost, lead time.

        Strictly treats all specified ordering cost synonyms as order_cost:
        - ordering cost
        - order cost
        - setup cost
        - setup expense
        - cost per order
        - cost of placing an order
        - ordering/setup cost
        """
        tracker = SpanTracker(question)
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        # 1. Ordering cost
        order_cost = tracker.extract_number([
            r"(?:ordering\s*/\s*setup|order\s*/\s*setup|ordering\s*and\s*setup|order\s*and\s*setup)\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"setup\s*expense(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"cost\s*(?:of|to)\s*placing?\s*an?\s*order(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"cost\s*per\s*order(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"(?:ordering|order|setup)\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"(?:ordering|order|setup)\s*expense(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"(?:each|an?)\s*order\s*costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*order|to\s*place(?:\s*an?\s*order)?|to\s*order|each\s*order)",
            r"\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*order|each\s*order|to\s*place\s*an?\s*order)",
            r"\b[sSkK]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        ])
        if order_cost is not None:
            inputs["order_cost"] = order_cost
        else:
            missing.append("order_cost (S)")

        # 2. Optional Operating / Working days per year (default 250)
        operating_days = tracker.extract_number([
            r"(?:annual\s*)?(?:operating|working|business)\s*days\s*(?:per\s*year|a\s*year)?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:days)?",
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:operating|working|business)\s*days\s*(?:per\s*year|a\s*year|\/year)?",
            r"(?:operates?|open)\s*(?:for\s*)?([0-9,]+(?:\.[0-9]+)?)\s*(?:operating\s*|working\s*|business\s*)?days\s*(?:per\s*year|a\s*year|\/year)?",
            r"(?:workdays|work\s*days)\s*(?:per\s*year)?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:workdays|work\s*days)\s*(?:per\s*year|a\s*year|\/year)?",
            r"(?:operating\s*|working\s*|business\s*)?days\s*(?:per\s*year|a\s*year)\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:operating\s*|working\s*|business\s*)?days\s*(?:per\s*year|a\s*year)",
        ])
        workdays = int(operating_days) if operating_days is not None else 250
        if operating_days is not None:
            inputs["workdays_per_year"] = workdays

        # 3. Demand and Demand Period Normalization
        # Detect whether demand is specified monthly, quarterly, weekly, daily, or annually
        demand_patterns: List[Tuple[str, str]] = [
            # Monthly patterns
            (
                r"(?:demand|usage|consumption|sales|requirement|sells?|selling)\s*(?:\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*month|a\s*month|\/month|\/mo\b|monthly|each\s*month)",
                "month",
            ),
            (
                r"\bmonthly\s*(?:demand|usage|consumption|sales|requirement)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?",
                "month",
            ),
            (
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*month|a\s*month|\/month|\/mo\b|each\s*month)(?!\s*%)",
                "month",
            ),
            # Quarterly patterns
            (
                r"(?:demand|usage|consumption|sales|requirement|sells?|selling)\s*(?:\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*quarter|a\s*quarter|\/quarter|\/qtr\b|quarterly|each\s*quarter)",
                "quarter",
            ),
            (
                r"\bquarterly\s*(?:demand|usage|consumption|sales|requirement)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?",
                "quarter",
            ),
            (
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*quarter|a\s*quarter|\/quarter|\/qtr\b|each\s*quarter)(?!\s*%)",
                "quarter",
            ),
            # Weekly patterns
            (
                r"(?:demand|usage|consumption|sales|requirement|sells?|selling)\s*(?:\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*week|a\s*week|\/week|\/wk\b|weekly|each\s*week)",
                "week",
            ),
            (
                r"\bweekly\s*(?:demand|usage|consumption|sales|requirement)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?",
                "week",
            ),
            (
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*week|a\s*week|\/week|\/wk\b|each\s*week)(?!\s*%)",
                "week",
            ),
            # Daily patterns
            (
                r"(?:demand|usage|consumption|sales|requirement|sells?|selling)\s*(?:\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*day|a\s*day|\/day|\/d\b|daily|each\s*day)",
                "day",
            ),
            (
                r"\bdaily\s*(?:demand|usage|consumption|sales|requirement)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?",
                "day",
            ),
            (
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*day|a\s*day|\/day|\/d\b|each\s*day)(?!\s*%)",
                "day",
            ),
            # Annual patterns
            (
                r"(?:annual|yearly)\s*(?:demand|usage|consumption|sales|requirement)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?",
                "year",
            ),
            (
                r"(?:demand|usage|consumption|sales|requirement|sells?|selling)\s*(?:\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*year|annually|a\s*year|\/year|\/yr\b|yearly|each\s*year)",
                "year",
            ),
            (
                r"sells?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*year|annually|a\s*year|\/year|\/yr\b)",
                "year",
            ),
            (
                r"([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?\s*(?:per\s*year|annually|a\s*year|\/year|\/yr\b)",
                "year",
            ),
            (
                r"\b[dD]\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
                "year",
            ),
            (
                r"demand(?:\s*\([a-zA-Z_]\))?\s*(?:is|=|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:(units?|items?|pieces?|cases?|boxes?|pallets?|lbs?|kg|gallons?)\b)?",
                "year",
            ),
        ]

        demand_val: Optional[float] = None
        demand_period: str = "year"
        demand_unit: str = "units"

        for pattern, period in demand_patterns:
            for match in re.finditer(pattern, question, re.IGNORECASE):
                if not match.groups():
                    continue
                val_start, val_end = match.span(1)
                # Check overlap with previously claimed spans (e.g. order cost, operating days)
                if any(not (val_end <= s or val_start >= e) for s, e in tracker.claimed_spans):
                    continue
                raw_str = match.group(1).replace(",", "").strip()
                try:
                    val = float(raw_str)
                    demand_val = val
                    demand_period = period
                    if match.lastindex >= 2 and match.group(2):
                        demand_unit = match.group(2).strip().lower()
                    tracker.claimed_spans.append(match.span(0))
                    break
                except ValueError:
                    continue
            if demand_val is not None:
                break

        if demand_val is not None:
            if demand_period == "month":
                factor = 12.0
                norm_demand = demand_val * factor
            elif demand_period == "quarter":
                factor = 4.0
                norm_demand = demand_val * factor
            elif demand_period == "week":
                factor = 52.0
                norm_demand = demand_val * factor
            elif demand_period == "day":
                factor = float(workdays)
                norm_demand = demand_val * factor
            else:
                factor = 1.0
                norm_demand = demand_val

            inputs["raw_demand"] = demand_val
            inputs["demand"] = demand_val
            inputs["demand_period"] = demand_period
            inputs["demand_unit"] = demand_unit
            inputs["annual_demand"] = norm_demand
        else:
            missing.append("annual_demand (D)")

        # 4. Holding cost (percentage rate vs fixed dollar cost)
        holding_rate = tracker.extract_number([
            r"(?:annual\s*)?percentage\s*(?:holding|carrying)\s*(?:cost|rate)?(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s*)?(?:holding|carrying)\s*(?:cost\s*)?(?:rate|percentage)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s*)?holding\s*(?:rate|percentage)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s*)?holding\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s*)?carrying\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s*(?:unit|item|purchase)\s*(?:cost|price)|holding|carrying|annual\s*carrying)",
            r"\b[iIhH]\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        ])

        holding_cost = tracker.extract_number([
            r"fixed\s*holding\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"(?:annual\s*)?holding\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit)?(?!\s*%)",
            r"(?:annual\s*)?carrying\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit)?(?!\s*%)",
            r"storage\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit)?(?!\s*%)",
            r"holding\s*expense(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"carrying\s*expense(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit\s*per\s*year|per\s*unit\s*annually|\/unit\/year)",
            r"costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit\s*to\s*hold|to\s*hold|to\s*store|to\s*carry)",
            r"\b[hH]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*%)",
        ])

        # 5. Unit purchase cost / price
        unit_cost = tracker.extract_number([
            r"unit\s*purchase\s*price(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"unit\s*purchase\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"purchase\s*(?:price|cost)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"unit\s*(?:cost|price)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"\$?([0-9,]+(?:\.[0-9]+)?)\s*unit\s*(?:purchase\s*)?(?:price|cost)",
            r"\$?([0-9,]+(?:\.[0-9]+)?)\s*purchase\s*(?:price|cost)",
            r"costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each|to\s*buy|per\s*item)",
            r"\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)",
            r"\b[cC]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        ])
        if unit_cost is not None:
            inputs["unit_cost"] = unit_cost

        if holding_cost is not None:
            inputs["holding_cost"] = holding_cost
        elif holding_rate is not None:
            rate_decimal = holding_rate / 100.0 if holding_rate > 1.0 else holding_rate
            inputs["holding_cost_rate"] = rate_decimal
            if unit_cost is None:
                missing.append("unit_cost (C) [required when holding cost is given as a %]")
        else:
            missing.append("holding_cost (H) or holding_cost_rate (%)")

        # 6. Optional lead time
        lead_time = tracker.extract_number([
            r"lead\s*time\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:business\s*|working\s*)?days?",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:business\s*|working\s*)?days?\s*lead\s*time",
            r"\b[lL]\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ])
        if lead_time is not None:
            inputs["lead_time_days"] = lead_time

        return inputs, missing

        return inputs, missing

    def _extract_quantity_discount_inputs(self, question: str) -> Tuple[Dict[str, Any], List[str]]:
        """Extract Quantity Discount parameters with strict distinction of breaks and costs."""
        tracker = SpanTracker(question)
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        # 1. Ordering cost (all synonyms)
        order_cost = tracker.extract_number([
            r"(?:ordering\s*/\s*setup|order\s*/\s*setup|ordering\s*and\s*setup|order\s*and\s*setup)\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"setup\s*expense(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"cost\s*(?:of|to)\s*placing?\s*an?\s*order(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"cost\s*per\s*order(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"(?:ordering|order|setup)\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"(?:ordering|order|setup)\s*expense(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"(?:each|an?)\s*order\s*costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*order|to\s*place(?:\s*an?\s*order)?|to\s*order|each\s*order)",
            r"\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*order|each\s*order|to\s*place\s*an?\s*order)",
            r"\b[sSkK]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        ])
        if order_cost is not None:
            inputs["order_cost"] = order_cost
        else:
            missing.append("order_cost (S)")

        # 2. Annual demand
        demand = tracker.extract_number([
            r"annual\s*(?:demand|usage|consumption|sales|requirement)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"yearly\s*(?:demand|usage|consumption|sales|requirement)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"(?:demand|usage|consumption)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s*year|annually|a\s*year|\/year)",
            r"sells?\s*([0-9,]+(?:\.[0-9]+)?)\s*units\s*(?:per\s*year|annually|a\s*year|\/year)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*units\s*(?:per\s*year|annually|a\s*year|\/year)",
            r"\b[dD]\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand(?:\s*\([a-zA-Z_]\))?\s*(?:is|=|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?",
        ])
        if demand is not None:
            inputs["annual_demand"] = demand
        else:
            missing.append("annual_demand (D)")

        # 3. Holding cost: distinguish percentage holding cost vs fixed holding cost
        holding_rate = tracker.extract_number([
            r"(?:annual\s*)?percentage\s*(?:holding|carrying)\s*(?:cost|rate)?(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s*)?(?:holding|carrying)\s*(?:cost\s*)?(?:rate|percentage)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s*)?holding\s*(?:rate|percentage)(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s*)?holding\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:annual\s*)?carrying\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s*(?:unit|item|purchase)\s*(?:price|cost)|holding|carrying|annual\s*carrying)",
            r"\b[iIhH]\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        ])
        holding_cost = tracker.extract_number([
            r"fixed\s*holding\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"(?:annual\s*)?holding\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit)?(?!\s*%)",
            r"(?:annual\s*)?carrying\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit)?(?!\s*%)",
            r"storage\s*cost(?:\s*\([a-zA-Z_]\))?\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit)?(?!\s*%)",
            r"\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:per\s*unit\s*per\s*year|per\s*unit\s*annually|\/unit\/year)",
            r"\b[hH]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)(?!\s*%)",
        ])

        if holding_rate is not None:
            inputs["holding_cost_rate"] = holding_rate / 100.0 if holding_rate > 1.0 else holding_rate
        elif holding_cost is not None:
            inputs["holding_cost"] = holding_cost
        else:
            missing.append("holding_cost_rate (%) or constant holding_cost (H)")

        # 4. Quantity breaks & discount schedule
        schedule = self._extract_quantity_discount_schedule(question, tracker.claimed_spans)
        if schedule:
            inputs["discount_schedule"] = schedule
        else:
            missing.append("discount_schedule (quantity breaks and unit prices)")

        return inputs, missing

    def _extract_quantity_discount_schedule(
        self, question: str, existing_spans: Optional[List[Tuple[int, int]]] = None
    ) -> List[Dict[str, Any]]:
        """Parse quantity discount schedule, supporting range tiers, open tiers, and break points."""
        schedule: List[Dict[str, Any]] = []
        claimed_spans: List[Tuple[int, int]] = list(existing_spans or [])

        def overlaps(start: int, end: int) -> bool:
            return any(not (end <= s or start >= e) for s, e in claimed_spans)

        tier_prefix = (
            r"(?:(?:tier|bracket|level)?\s*\d+\s*(?:\([^)]*\)|[:\.\-])\s*)?"
            r"(?:(?:orders?|quantity|qty)\s*(?:of|for|from)?\s*)?"
            r"(?:for\s*)?"
            r"(?:between\s*|from\s*)?"
            r"\(?\s*"
        )
        connector = (
            r"\s*(?:\))?\s*(?:units?)?\s*(?:\))?\s*(?:,|;)?\s*"
            r"(?:for|at|@|costs?|unit\s*(?:purchase\s*)?price\s*(?:is|=|:)?|purchase\s*price\s*(?:is|=|:)?|price\s*(?:is|=|:)?|gets?|pays?|receives?|is|=|:|,|\|)?\s*"
            r"(?:is|=|:)?\s*(?:\()?\s*\$?\s*"
        )
        price_suffix = r"(?:\s*(?:\/|\s*per|\s*each|\s*ea)?\s*(?:unit|item|piece|each|ea))?(?:\))?"

        # 1. Price-first compound inequality: e.g. "price = $11.50 for 500 <= Q < 1,000"
        p_comp_pat = (
            r"(?:unit\s*)?price\s*(?:is|=|:)?\s*\$?([0-9]+(?:\.[0-9]+)?)"
            r"(?:\s*(?:\/|\s*per|\s*each|\s*ea)?\s*(?:unit|item|piece|each|ea))?\s*"
            r"(?:for|if|when|at|with)?\s*"
            r"([0-9,]+)\s*(<=|<)\s*(?:[qQ]|qty|quantity|units?)?\s*(<=|<)\s*([0-9,]+)"
            r"(?:\s*units?)?"
        )
        for m in re.finditer(p_comp_pat, question, re.IGNORECASE):
            start, end = m.span()
            if overlaps(start, end):
                continue
            price = float(m.group(1))
            low = float(m.group(2).replace(",", ""))
            rel1 = m.group(3)
            rel2 = m.group(4)
            high = float(m.group(5).replace(",", ""))
            min_q = low + 1.0 if rel1 == "<" else low
            max_q = high - 1.0 if rel2 == "<" else high
            if min_q <= max_q and price > 0:
                schedule.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})
                claimed_spans.append((start, end))

        # 2. Range-first compound inequality: e.g. "500 <= Q < 1,000, price = $11.50"
        r_comp_pat = (
            tier_prefix
            + r"([0-9,]+)\s*(<=|<)\s*(?:[qQ]|qty|quantity|units?)?\s*(<=|<)\s*([0-9,]+)"
            + connector
            + r"([0-9]+(?:\.[0-9]+)?)"
            + price_suffix
        )
        for m in re.finditer(r_comp_pat, question, re.IGNORECASE):
            start, end = m.span()
            if overlaps(start, end):
                continue
            low = float(m.group(1).replace(",", ""))
            rel1 = m.group(2)
            rel2 = m.group(3)
            high = float(m.group(4).replace(",", ""))
            price = float(m.group(5))
            min_q = low + 1.0 if rel1 == "<" else low
            max_q = high - 1.0 if rel2 == "<" else high
            if min_q <= max_q and price > 0:
                schedule.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})
                claimed_spans.append((start, end))

        # 3. Standard range syntax: e.g. "1-49 units for $9.00", "50 to 99 units @ $8.50", "between 500 and 999 units"
        range_pat = (
            tier_prefix
            + r"([0-9,]+)\b\s*(?:-|to|\.\.\.|through|and)\s*([0-9,]+)\b"
            + connector
            + r"([0-9]+(?:\.[0-9]+)?)"
            + price_suffix
        )
        for m in re.finditer(range_pat, question, re.IGNORECASE):
            start, end = m.span()
            if overlaps(start, end):
                continue
            min_q = float(m.group(1).replace(",", ""))
            max_q = float(m.group(2).replace(",", ""))
            price = float(m.group(3))
            min_q = max(min_q, 1.0)
            if min_q <= max_q and price > 0:
                schedule.append({"min_qty": min_q, "max_qty": max_q, "unit_price": price})
                claimed_spans.append((start, end))

        # 4. Price-first single inequalities: e.g. "price = $12.00 for Q < 500", "price = $11.00 for Q >= 1000"
        p_single_pat = (
            r"(?:unit\s*)?price\s*(?:is|=|:)?\s*\$?([0-9]+(?:\.[0-9]+)?)"
            r"(?:\s*(?:\/|\s*per|\s*each|\s*ea)?\s*(?:unit|item|piece|each|ea))?\s*"
            r"(?:for|if|when|at|with)?\s*"
            r"(?:[qQ]|qty|quantity|orders?)?\s*(>=|>|<=|<)\s*([0-9,]+)"
            r"(?:\s*units?)?"
        )
        for m in re.finditer(p_single_pat, question, re.IGNORECASE):
            start, end = m.span()
            if overlaps(start, end):
                continue
            price = float(m.group(1))
            rel = m.group(2)
            val = float(m.group(3).replace(",", ""))
            if price > 0:
                if rel == "<":
                    schedule.append({"min_qty": 1.0, "max_qty": max(val - 1.0, 1.0), "unit_price": price})
                    claimed_spans.append((start, end))
                elif rel == "<=":
                    schedule.append({"min_qty": 1.0, "max_qty": val, "unit_price": price})
                    claimed_spans.append((start, end))
                elif rel == ">=":
                    schedule.append({"min_qty": val, "max_qty": None, "unit_price": price})
                    claimed_spans.append((start, end))
                elif rel == ">":
                    schedule.append({"min_qty": val + 1.0, "max_qty": None, "unit_price": price})
                    claimed_spans.append((start, end))

        # 5. Quantity-first "Under / below / <" syntax: e.g. "Q < 500, price = $12.00", "Quantity below 500 units = $12.00"
        under_pat = (
            tier_prefix
            + r"(?:(?:[qQ]|qty|quantity)\s*)?(?:under|below|strictly\s*below|less\s*than|fewer\s*than|<)\s*([0-9,]+)\b"
            + connector
            + r"([0-9]+(?:\.[0-9]+)?)"
            + price_suffix
        )
        for m in re.finditer(under_pat, question, re.IGNORECASE):
            start, end = m.span()
            if overlaps(start, end):
                continue
            cutoff = float(m.group(1).replace(",", ""))
            price = float(m.group(2))
            if cutoff > 1 and price > 0:
                schedule.append({"min_qty": 1.0, "max_qty": cutoff - 1.0, "unit_price": price})
                claimed_spans.append((start, end))

        # 6. Quantity-first "Up to / <=" syntax: e.g. "Q <= 499: $12.00", "up to 49 units: $9.00"
        upto_pat = (
            tier_prefix
            + r"(?:(?:[qQ]|qty|quantity)\s*)?(?:up\s*to|at\s*most|less\s*than\s*or\s*equal\s*to|<=)\s*([0-9,]+)\b"
            + connector
            + r"([0-9]+(?:\.[0-9]+)?)"
            + price_suffix
        )
        for m in re.finditer(upto_pat, question, re.IGNORECASE):
            start, end = m.span()
            if overlaps(start, end):
                continue
            max_q = float(m.group(1).replace(",", ""))
            price = float(m.group(2))
            if max_q >= 1 and price > 0:
                schedule.append({"min_qty": 1.0, "max_qty": max_q, "unit_price": price})
                claimed_spans.append((start, end))

        # 7. Quantity-first open-ended syntax: e.g. "Q >= 1,000, price = $11.00", "1,000 units or more = $11.00"
        open_pats = [
            tier_prefix
            + r"([0-9,]+)\b\s*(?:\+|(?:\s*units?)?\s*(?:\+|and\s*(?:above|over|up)|or\s*(?:more|higher)|\+)(?:\s*units?)?)"
            + connector
            + r"([0-9]+(?:\.[0-9]+)?)"
            + price_suffix,
            tier_prefix
            + r"(?:(?:[qQ]|qty|quantity\s*(?:of)?)\s*)?(?:>=|at\s*least)\s*([0-9,]+)\b"
            + connector
            + r"([0-9]+(?:\.[0-9]+)?)"
            + price_suffix,
            tier_prefix
            + r"(?:(?:[qQ]|qty|quantity\s*(?:of)?)\s*)?(?:>|over|more\s*than|exceeding)\s*([0-9,]+)\b"
            + connector
            + r"([0-9]+(?:\.[0-9]+)?)"
            + price_suffix,
        ]
        for pat_idx, pat in enumerate(open_pats):
            for m in re.finditer(pat, question, re.IGNORECASE):
                start, end = m.span()
                if overlaps(start, end):
                    continue
                val = float(m.group(1).replace(",", ""))
                price = float(m.group(2))
                min_q = val + 1.0 if pat_idx == 2 else val
                if price > 0:
                    schedule.append({"min_qty": min_q, "max_qty": None, "unit_price": price})
                    claimed_spans.append((start, end))
                min_q = val + 1.0 if pat_idx == 2 else val
                if price > 0:
                    schedule.append({"min_qty": min_q, "max_qty": None, "unit_price": price})
                    claimed_spans.append((start, end))

        # 8. Explicit Quantity Break syntax with base price
        if not schedule:
            break_matches = re.findall(
                r"quantity\s*breaks?\s*(?:at|of)?\s*([0-9,]+)\s*(?:units?)?\s*(?:gives?|for|at|@|yields?)?\s*(?:unit\s*(?:purchase\s*)?price\s*(?:of|is|=)?\s*)?\(?\$?([0-9]+(?:\.[0-9]+)?)\)?",
                question,
                re.IGNORECASE,
            )
            base_match = re.search(
                r"(?:base(?:line)?|regular|standard)\s*(?:unit\s*)?(?:purchase\s*)?price\s*(?:of|is|=|:)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                question,
                re.IGNORECASE,
            )
            if break_matches and base_match:
                base_price = float(base_match.group(1))
                breaks = [(float(b[0].replace(",", "")), float(b[1])) for b in break_matches]
                breaks.sort(key=lambda x: x[0])
                schedule.append({"min_qty": 1.0, "max_qty": breaks[0][0] - 1.0, "unit_price": base_price})
                for i in range(len(breaks)):
                    min_q = breaks[i][0]
                    max_q = breaks[i + 1][0] - 1.0 if i + 1 < len(breaks) else None
                    schedule.append({"min_qty": min_q, "max_qty": max_q, "unit_price": breaks[i][1]})

        # 9. Fallback: pairs of quantities and prices if nothing else matched
        if not schedule:
            tier_pairs = re.findall(
                r"(?:tier\s*\d+|bracket\s*\d+|qty|quantity)\s*:?\s*([0-9,]+)\s*(?:units?)?\s*(?:price|cost|\:|\@)\s*\$?([0-9]+(?:\.[0-9]+)?)",
                question,
                re.IGNORECASE,
            )
            for pair in tier_pairs:
                q = float(pair[0].replace(",", ""))
                p = float(pair[1])
                schedule.append({"min_qty": q, "max_qty": None, "unit_price": p})

        if schedule:
            schedule.sort(key=lambda t: t["min_qty"])
            if schedule[0]["min_qty"] <= 0:
                schedule[0]["min_qty"] = 1.0

            # Deduplicate tiers that share the same min_qty
            seen_min = set()
            deduped = []
            for t in schedule:
                if t["min_qty"] not in seen_min:
                    seen_min.add(t["min_qty"])
                    deduped.append(t)
            schedule = deduped

            # Resolve adjacent overlapping boundaries
            for i in range(len(schedule) - 1):
                next_min = schedule[i + 1]["min_qty"]
                if schedule[i]["max_qty"] is None or schedule[i]["max_qty"] >= next_min:
                    schedule[i]["max_qty"] = next_min - 1.0

            return schedule

        return []

    def _extract_safety_stock_rop_inputs(self, question: str) -> Tuple[Dict[str, Any], List[str]]:
        """Extract Safety Stock & ROP parameters with strict unit and variability disambiguation."""
        tracker = SpanTracker(question)
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        # 1. Configurable working days per week (default 5.0)
        workdays_pw = tracker.extract_number([
            r"(?:assuming|with|using)?\s*([0-9]+(?:\.[0-9]+)?)\s*[- ]?(?:operating|working|business)?\s*days?\s*(?:per|\/|a)\s*week",
            r"([0-9]+(?:\.[0-9]+)?)\s*[- ]day\s*(?:operating|working|business)\s*week",
            r"workdays?\s*per\s*week\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"operating\s*days?\s*per\s*week\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
        ])
        inputs["workdays_per_week"] = workdays_pw if workdays_pw is not None else 5.0

        lt_unit: Optional[str] = None
        d_unit: Optional[str] = None

        # 2. Check compound lead-time pattern first (e.g. "average lead time is 3 weeks with a standard deviation of 0.5 weeks")
        lt_compound_pat = (
            r"(?:average|mean|expected|replenishment)?\s*lead[- ]time\s*(?:averages?|is|=|:|\bof\b)?\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?)?(?:\s*(?:business|working)\s*days?)?\s*(?:,|;)?\s*"
            r"(?:with\s*(?:a\s*)?|and\s*(?:a\s*)?|\(\s*)?(?:standard\s*deviation|std(?: dev)?)\s*(?:of|is|=|:)?\s*"
            r"([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?|years?)?"
        )
        m_lt = None
        for match in re.finditer(lt_compound_pat, question, re.IGNORECASE):
            s, e = match.span()
            if not any(not (e <= cs or s >= ce) for cs, ce in tracker.claimed_spans):
                m_lt = match
                break

        if m_lt is not None:
            inputs["lead_time"] = float(m_lt.group(1))
            raw_u = m_lt.group(2) or m_lt.group(4)
            if raw_u:
                lt_unit = "weeks" if "week" in raw_u.lower() else ("months" if "month" in raw_u.lower() else "days")
            inputs["lead_time_std"] = float(m_lt.group(3))
            tracker.claimed_spans.append(m_lt.span())
        else:
            # Individual Lead-time standard deviation
            lead_time_std = tracker.extract_number([
                r"(?:lead[- ]time\s*standard\s*deviation|standard\s*deviation\s*of\s*lead[- ]time)\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:standard\s*deviation|std(?: dev)?)\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)?\s*(?:in|for|of)?\s*lead[- ]time",
                r"lead[- ]time\s*(?:has\s*(?:a\s*)?)?(?:standard\s*deviation|std(?: dev)?)\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"lead[- ]time\s*std(?: dev)?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"std(?: dev)?\s*of\s*lead[- ]time\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:sigma[ _](?:l|lt))\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                r"lead[- ]time\s*varies\s*(?:with\s*(?:a\s*)?std(?: dev)?\s*(?:of)?)?\s*([0-9]+(?:\.[0-9]+)?)",
            ])
            inputs["lead_time_std"] = lead_time_std if lead_time_std is not None else 0.0

            # Individual Average lead time
            lead_time = tracker.extract_number([
                r"(?:average|mean|expected|replenishment)\s*lead[- ]time\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"lead[- ]time\s*averages?\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"lead[- ]time\s*(?:of|is|=|:)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?|years?)?",
                r"([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?|years?)\s*(?:average\s*)?lead[- ]time",
                r"\b[lL]\s*=\s*([0-9]+(?:\.[0-9]+)?)",
            ])
            if lead_time is not None:
                inputs["lead_time"] = lead_time
            else:
                missing.append("lead_time (L)")

        # 3. Check compound demand pattern (e.g. "average demand of 40 units per day with a standard deviation of 8 units per day")
        d_compound_pat = (
            r"(?:average|mean|expected|daily|weekly|monthly)?\s*demand\s*(?:averages?|is|=|:|\bof\b)?\s*"
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(?:per\s*|/)?(days?|daily|weeks?|weekly|months?|monthly)?\s*(?:,|;)?\s*"
            r"(?:with\s*(?:a\s*)?|and\s*(?:a\s*)?|\(\s*)?(?:standard\s*deviation|std(?: dev)?)\s*(?:of|is|=|:)?\s*"
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(?:per\s*|/)?(days?|daily|weeks?|weekly|months?|monthly)?"
        )
        m_d = None
        for match in re.finditer(d_compound_pat, question, re.IGNORECASE):
            s, e = match.span()
            if not any(not (e <= cs or s >= ce) for cs, ce in tracker.claimed_spans):
                m_d = match
                break

        if m_d is not None:
            inputs["avg_demand"] = float(m_d.group(1).replace(",", ""))
            raw_du = m_d.group(2) or m_d.group(4)
            if raw_du:
                d_unit = "weeks" if "week" in raw_du.lower() else ("months" if "month" in raw_du.lower() else "days")
            inputs["demand_std"] = float(m_d.group(3).replace(",", ""))
            tracker.claimed_spans.append(m_d.span())
        else:
            # Individual Demand standard deviation
            demand_std = tracker.extract_number([
                r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand)\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*std(?: dev)?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"std(?: dev)?\s*of\s*demand\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:sigma[ _]d)\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*varies\s*(?:with\s*(?:a\s*)?std(?: dev)?\s*(?:of)?)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"(?:with\s*(?:a\s*)?)?(?:standard\s*deviation|std(?: dev)?)\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"\+/-\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?",
            ])
            inputs["demand_std"] = demand_std if demand_std is not None else 0.0

            # Individual Average demand
            avg_demand = tracker.extract_number([
                r"(?:average|mean|expected|daily|weekly|monthly)\s*demand\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*averages?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is|=|of|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?\s*per\s*(?:day|week|month)",
                r"([0-9,]+(?:\.[0-9]+)?)\s*units\s*per\s*(?:day|week|month)",
                r"\b[dD]\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
                r"demand\s*(?:is|=|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?",
            ])
            if avg_demand is not None:
                inputs["avg_demand"] = avg_demand
            else:
                missing.append("avg_demand (d)")

        # 4. Service level
        service_level = tracker.extract_number([
            r"(?:cycle\s*)?service\s*level\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*)?service\s*level",
            r"\b[cC][sS][lL]\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:cycle\s*)?service\s*level\s*(?:of|is|=|:)?\s*(0\.[0-9]+)",
            r"\b[cC][sS][lL]\s*(?:of|is|=|:)?\s*(0\.[0-9]+)",
            r"in-stock\s*probability\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        ])
        if service_level is not None:
            inputs["service_level"] = service_level / 100.0 if service_level > 1.0 else service_level
        else:
            inputs["service_level"] = 0.95

        # 5. Resolve time units for demand and lead time
        if not d_unit:
            if re.search(r"\b(?:per\s*week|weekly|\/week|a\s*week)\b", question, re.I):
                d_unit = "weeks"
            elif re.search(r"\b(?:per\s*month|monthly|\/month|a\s*month)\b", question, re.I):
                d_unit = "months"
            else:
                d_unit = "days"

        if not lt_unit:
            lt_m2 = re.search(r"lead[- ]time.*?([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?)", question, re.I)
            if not lt_m2:
                lt_m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(days?|weeks?|months?).*?lead[- ]time", question, re.I)
            if lt_m2:
                u2 = lt_m2.group(2).lower()
                lt_unit = "weeks" if "week" in u2 else ("months" if "month" in u2 else "days")
            else:
                lt_unit = d_unit

        inputs["demand_time_unit"] = d_unit
        inputs["lead_time_unit"] = lt_unit
        inputs["time_unit"] = d_unit

        return inputs, missing

        return inputs, missing

    def _extract_fixed_period_inputs(self, question: str) -> Tuple[Dict[str, Any], List[str]]:
        """Extract Fixed-Period Review parameters with strict distinction of buffer and inventory states."""
        tracker = SpanTracker(question)
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        # 1. Review period
        review_period = tracker.extract_number([
            r"(?:review\s*period|review\s*interval|review\s*cycle)\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"reviewed\s*every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
            r"orders?\s*placed\s*every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
            r"every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)\s*(?:review)?",
            r"\b[tT]\s*=\s*([0-9]+(?:\.[0-9]+)?)",
            r"\b[pP]\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ])
        if review_period is not None:
            inputs["review_period"] = review_period
        else:
            missing.append("review_period (T or P)")

        # 2. Lead time
        lead_time = tracker.extract_number([
            r"(?:replenishment\s*|order\s*)?lead\s*time\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)\s*lead\s*time",
            r"\b[lL]\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        ])
        if lead_time is not None:
            inputs["lead_time"] = lead_time
        else:
            missing.append("lead_time (L)")

        # 3. Demand standard deviation (check before average demand)
        demand_std = tracker.extract_number([
            r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand)\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand\s*std(?: dev)?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"std(?: dev)?\s*of\s*demand\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"(?:sigma[ _]d)\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
            r"(?:with\s*(?:a\s*)?)?(?:standard\s*deviation|std(?: dev)?)\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\+/-\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?",
        ])
        inputs["demand_std"] = demand_std if demand_std is not None else 0.0

        # Demand / average demand
        avg_demand = tracker.extract_number([
            r"(?:average|mean|expected|daily|weekly|monthly)\s*demand\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand\s*averages?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand\s*(?:is|=|of|:)\s*([0-9,]+(?:\.[0-9]+)?)\s*(?:units)?",
            r"([0-9,]+(?:\.[0-9]+)?)\s*units\s*per\s*(?:day|week|month)",
            r"\b[dD]\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
        ])
        if avg_demand is not None:
            inputs["avg_demand"] = avg_demand
        else:
            missing.append("avg_demand (d)")

        # 4. Safety stock (explicit/direct buffer)
        safety_stock = tracker.extract_number([
            r"(?:prescribed\s*|target\s*|direct\s*|buffer\s*)?safety\s*stock\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"buffer\s*stock\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\b[sS][sS]\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
        ])
        if safety_stock is not None:
            inputs["safety_stock"] = safety_stock

        # Service level
        service_level = tracker.extract_number([
            r"(?:cycle\s*)?service\s*level\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*)?service\s*level",
            r"\b[cC][sS][lL]\s*(?:of|is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"(?:cycle\s*)?service\s*level\s*(?:of|is|=|:)?\s*(0\.[0-9]+)",
        ])
        inputs["service_level"] = (service_level / 100.0 if service_level > 1.0 else service_level) if service_level is not None else 0.95

        # 5. Inventory position (explicit/direct)
        inv_position = tracker.extract_number([
            r"(?:direct\s*|current\s*)?inventory\s*position\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\b[iI][pP]\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
        ])
        if inv_position is not None:
            inputs["inventory_position"] = inv_position

        # 6. On-hand inventory
        on_hand = tracker.extract_number([
            r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*units?\s*(?:in\s*stock|on\s*hand|currently\s*on\s*hand)",
            r"(?:on-hand\s*inventory|on-hand\s*stock|current\s*inventory|stock\s*on\s*hand)\s*(?:of|is|=|:)?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
            r"(?:currently\s*have|currently\s*on\s*hand|in\s*stock)\s*(?:of|is|=|:)?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
            r"\b[iI][hH]\s*=\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
            r"\b[oO][hH]\s*=\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
        ])
        inputs["current_inventory"] = on_hand if on_hand is not None else 0.0

        # 7. On-order inventory
        on_order = tracker.extract_number([
            r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*units?\s*(?:already\s*|currently\s*|are\s*|now\s*)?(?:on\s*order|in\s*transit|scheduled\s*to\s*arrive|ordered|due\s*in|in\s*the\s*pipeline)",
            r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*units?\s*of\s*(?:open\s*orders?|scheduled\s*receipts?|pipeline\s*inventory)",
            r"(?:on-order\s*inventory|on-order\s*stock|units?\s*on\s*order|units?\s*in\s*transit)\s*(?:of|is|=|:)?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
            r"(?:already\s*on\s*order|currently\s*on\s*order|on\s*order|in\s*transit|scheduled\s*receipts?|open\s*orders?|pipeline\s*inventory)\s*(?:of|is|are|=|:)?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
            r"\b(?:[oO][oO]|[sS][rR])\s*=\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
        ])
        inputs["on_order"] = on_order if on_order is not None else 0.0

        # 8. Backorders
        backorders = tracker.extract_number([
            r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*units?\s*(?:of\s*|on\s*|in\s*)?(?:customer\s*)?(?:backorders?|backordered|back\s*orders?|backlog)",
            r"([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)\s*(?:customer\s*)?(?:backorders?|backordered|back\s*orders?|in\s*backlog)",
            r"(?:customer\s*)?(?:backorders?|backordered|back\s*orders?|backlog)\s*(?:of|is|are|=|:)?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
            r"\b[bB][oO]\s*=\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)",
        ])
        inputs["backorders"] = backorders if backorders is not None else 0.0

        # Check for rounding up requests
        round_up = bool(re.search(
            r"round\s*(?:[a-zA-Z0-9_\- ]+?\s+)?up\b|ceiling|\bround\s*up\b|nearest\s*(?:whole\s*)?unit\s*up",
            question,
            re.IGNORECASE,
        ))
        if round_up:
            inputs["round_up"] = True

        # Time unit
        q_low = question.lower()
        if "week" in q_low:
            inputs["time_unit"] = "weeks"
        elif "month" in q_low:
            inputs["time_unit"] = "months"
        else:
            inputs["time_unit"] = "days"

        return inputs, missing

    def _extract_newsvendor_inputs(self, question: str) -> Tuple[Dict[str, Any], List[str]]:
        """Extract Newsvendor single-period parameters with strict distinction of marginal costs vs prices."""
        tracker = SpanTracker(question)
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        # 1. Underage / Shortage cost
        underage_cost = tracker.extract_number([
            r"(?:underage\s*cost|shortage\s*cost|cost\s*of\s*underage|cost\s*of\s*shortage)\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"\b[cC][uU]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        ])

        # 2. Overage / Excess cost
        overage_cost = tracker.extract_number([
            r"(?:overage\s*cost|excess\s*cost|cost\s*of\s*overage|cost\s*of\s*excess)\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"\b[cC][oO]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        ])

        # 3. Selling price
        selling_price = tracker.extract_number([
            r"\$?([0-9,]+(?:\.[0-9]+)?)\s*retail",
            r"sell(?:s|ing)?\s+(?:[a-zA-Z0-9_\- ]+?\s+)?for\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"selling\s*price\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"retail\s*price\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"sales?\s*price\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"retails?\s*(?:for|at)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"sells?\s*(?:them\s*)?(?:for|at)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"regular\s*price\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"\b[pP]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        ])

        # 4. Purchase cost
        unit_cost = tracker.extract_number([
            r"purchase\s*cost\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"unit\s*purchase\s*cost\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"wholesale\s*cost\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"acquisition\s*cost\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"manufacturing\s*cost\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"unit\s*cost\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"(?:buy|purchase)(?:s|ing)?\s*(?:them\s*)?(?:from\s*[^$0-9]+)?(?:for|at)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"costs?\s*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:each|per\s*unit|to\s*buy|to\s*manufacture|to\s*produce)",
            r"\b[cC]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        ])

        # 5. Salvage value
        salvage_value = tracker.extract_number([
            r"salvage\s*value\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"salvage\s*price\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"scrap\s*value\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"markdown\s*price\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"liquidation\s*value\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"outlet\s*price\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"salvaged?\s*(?:them\s*)?(?:to\s*[^$0-9]+)?\s*(?:for|at)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"liquidated?\s*(?:them\s*)?(?:to\s*[^$0-9]+)?\s*(?:for|at)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"sold\s*to\s*(?:an?\s*)?outlet\s*(?:for|at)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"unsold\s*(?:[a-zA-Z]+\s*)*(?:are\s*)?(?:sold|salvaged|liquidated)?\s*(?:to\s*[^$0-9]+)?(?:for|at)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
            r"\b[sS]\s*=\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        ])

        # 6. Mean demand
        demand_mean = tracker.extract_number([
            r"(?:mean|expected|average)\s*demand\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand\s*is\s*(?:normally\s*distributed\s*with\s*)?(?:a\s*)?mean\s*(?:of)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"mean\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\bmu\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
        ])

        # 7. Demand standard deviation
        demand_std = tracker.extract_number([
            r"(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand)\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"standard\s*deviation\s*(?:of\s*demand)?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"std(?: dev)?\s*(?:of\s*demand)?\s*(?:of|is|=|:)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"\bsigma\s*=\s*([0-9,]+(?:\.[0-9]+)?)",
        ])

        # 8. Customer penalty (optional)
        penalty = tracker.extract_number([
            r"(?:goodwill\s*penalty|loss\s*of\s*goodwill|customer\s*penalty)\s*(?:of|is|=|:)?\s*\$?([0-9,]+(?:\.[0-9]+)?)",
        ])
        inputs["customer_penalty"] = penalty if penalty is not None else 0.0

        # Validate completeness - examine entire question before declaring information missing
        if demand_mean is not None:
            inputs["demand_mean"] = demand_mean
        else:
            missing.append("demand_mean (mu)")

        if demand_std is not None:
            inputs["demand_std"] = demand_std
        else:
            missing.append("demand_std (sigma)")

        # Economic parameters: either (Cu, Co) or (P, C, S)
        if underage_cost is not None and overage_cost is not None:
            inputs["underage_cost"] = underage_cost
            inputs["overage_cost"] = overage_cost
            if selling_price is not None:
                inputs["selling_price"] = selling_price
            if unit_cost is not None:
                inputs["unit_cost"] = unit_cost
            if salvage_value is not None:
                inputs["salvage_value"] = salvage_value
        elif selling_price is not None and unit_cost is not None:
            inputs["selling_price"] = selling_price
            inputs["unit_cost"] = unit_cost
            inputs["salvage_value"] = salvage_value if salvage_value is not None else 0.0
        else:
            # Neither complete set is provided
            if selling_price is None and underage_cost is None:
                missing.append("selling_price (P) or underage_cost (Cu)")
            if unit_cost is None and overage_cost is None:
                missing.append("unit_cost (C) or overage_cost (Co)")

        return inputs, missing

    # -----------------------------------------------------------------
    # Stage 3: Select Python Tool
    # -----------------------------------------------------------------

    def _select_tool(self, model_type: InventoryModelType) -> Tuple[str, Any]:
        """Map identified model to execution function."""
        mapping = {
            InventoryModelType.EOQ: ("calculate_eoq", calculate_eoq),
            InventoryModelType.QUANTITY_DISCOUNT: ("calculate_quantity_discount", calculate_quantity_discount),
            InventoryModelType.SAFETY_STOCK_ROP: ("calculate_safety_stock_rop", calculate_safety_stock_rop),
            InventoryModelType.FIXED_PERIOD_REVIEW: ("calculate_fixed_period_review", calculate_fixed_period_review),
            InventoryModelType.NEWSVENDOR: ("calculate_newsvendor", calculate_newsvendor),
        }
        return mapping[model_type]

    # -----------------------------------------------------------------
    # Stage 4: Execute Calculation
    # -----------------------------------------------------------------

    def _execute_calculation(self, tool_callable: Any, model_type: InventoryModelType, inputs: Dict[str, Any]) -> Any:
        """Call the Python tool with unpacked parameters, filtering None values."""
        cleaned = {k: v for k, v in inputs.items() if v is not None}
        return tool_callable(**cleaned)

    # -----------------------------------------------------------------
    # Stage 5 & 6: Verification and Explanation Synthesis
    # -----------------------------------------------------------------

    def _generate_clarification_prompt(self, model_type: InventoryModelType, extracted: Dict[str, Any], missing: List[str]) -> str:
        """Construct user-friendly clarification prompt when required data is missing."""
        lines = [
            f"### 📋 Model Identified: **{model_type.value}**",
            "",
            "I've successfully identified the appropriate supply chain model for your question, but need a few additional details to calculate the exact optimal decision:",
            "",
            "**Missing Parameters Required:**",
        ]
        for m in missing:
            lines.append(f"- ❓ **`{m}`**")

        if extracted:
            lines.append("")
            lines.append("**Parameters Already Captured:**")
            for k, v in extracted.items():
                lines.append(f"- ✅ **`{k}`**: `{v}`")

        lines.extend([
            "",
            "Please provide the missing figures so I can run the optimization and provide your recommended order quantity and cost analysis.",
        ])
        return "\n".join(lines)

    def _explain_unknown_query(self, question: str) -> str:
        """Helpful response when the question does not match any of the 5 models."""
        return (
            "### 🤖 Supply Chain Analytics Inventory Decision Agent\n\n"
            "I couldn't confidently identify which inventory decision model to apply to your question.\n\n"
            "I support the following **5 core inventory optimization models**:\n\n"
            "1. **Economic Order Quantity (EOQ)**: Constant demand tradeoff between order setup cost and holding cost.\n"
            "   *Example: 'Annual demand is 10,000 units, order cost is $50, holding cost is $2. What is our EOQ?'*\n\n"
            "2. **Quantity Discount**: All-unit price breaks and volume discounts.\n"
            "   *Example: 'Annual demand is 3,000, order cost $40, holding cost 20%. Tier 1: 1-499 @ $20, Tier 2: 500-999 @ $18, Tier 3: 1000+ @ $16. What quantity should we order?'*\n\n"
            "3. **Safety Stock & Reorder Point (ROP)**: Continuous review $(s, Q)$ with stochastic demand and lead time.\n"
            "   *Example: 'Daily demand is 50 units (std 8), lead time is 5 days, target service level is 95%. What is our safety stock and ROP?'*\n\n"
            "4. **Fixed-Period Review**: Periodic order-up-to base stock system $(T, S)$ over protection interval $(T + L)$.\n"
            "   *Example: 'We review inventory every 14 days with lead time of 7 days. Daily demand is 30 (std 6). Service level is 98%. Current stock is 120 and 50 on order. How much to order?'*\n\n"
            "5. **Newsvendor Model**: Single-period stochastic demand balancing overage and underage costs.\n"
            "   *Example: 'Ski jackets retail for $250, cost $100, and salvage for $40. Demand is normal with mean 400 and std 80. How many should we order?'*\n\n"
            "Please rephrase your query with your specific operational parameters!"
        )

    def _generate_explanation(
        self,
        question: str,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        res: Any,
        verif: Optional[VerificationResult],
    ) -> str:
        """Generate executive, mathematically rigorous, and operationally actionable explanation."""
        if model_type == InventoryModelType.EOQ:
            return self._format_eoq_explanation(res, verif)
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            return self._format_quantity_discount_explanation(res, verif)
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            return self._format_safety_stock_rop_explanation(res, verif)
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            return self._format_fixed_period_explanation(res, verif)
        elif model_type == InventoryModelType.NEWSVENDOR:
            return self._format_newsvendor_explanation(res, verif)
        return ""

    # -----------------------------------------------------------------
    # Formatted Explanations for Each Model
    # -----------------------------------------------------------------

    def _format_eoq_explanation(self, r: EOQResult, v: Optional[VerificationResult]) -> str:
        purchase_line = f"- **Annual Purchase Cost**: `${r.annual_purchase_cost:,.2f}`\n- **Total Annual Cost (inc. purchase)**: `${r.total_cost_with_purchase:,.2f}`" if r.annual_purchase_cost else ""
        rop_line = f"- **Deterministic Reorder Point**: **`{r.reorder_point:,.1f} {r.demand_unit}`** (based on {r.lead_time_days} days lead time)" if r.reorder_point is not None else ""

        checks_md = "\n".join([f"- ✅ {c}" for c in (v.checks_passed if v else [])])
        warnings_md = "\n".join([f"- ⚠️ {w}" for w in (v.warnings if v else [])])

        operating_schedule_label = (
            f"{r.workdays_per_year} working days (user-specified)"
            if r.workdays_per_year != 250
            else f"{r.workdays_per_year} working days (standard documented default: 50 weeks × 5 days/week)"
        )

        unit_norm_callout = ""
        if r.demand_normalization_applied:
            unit_norm_callout = (
                f"> [!NOTE]\n"
                f"> **Demand Period Normalization Applied**:\n"
                f"> - **Original Demand**: **`{r.raw_demand:,.2f} {r.demand_unit} per {r.demand_period}`**\n"
                f"> - **Annual Normalization Factor**: **`{r.demand_normalization_factor:g} {r.demand_period}s per year`**\n"
                f"> - **Normalized Annual Demand ($D$)**: **`{r.annual_demand:,.2f} {r.demand_unit} per year`** (${r.raw_demand:,.2f} \\times {r.demand_normalization_factor:g}$)\n\n---\n\n"
            )

        original_demand_row = ""
        if r.demand_normalization_applied:
            original_demand_row = (
                f"| **Original Demand** | **`{r.raw_demand:,.2f}` {r.demand_unit}/{r.demand_period}** | Operational demand input per {r.demand_period} |\n"
            )

        annual_demand_meaning = (
            f"Normalized annual demand ($D = {r.raw_demand:,.2f} \\times {r.demand_normalization_factor:g}$)"
            if r.demand_normalization_applied
            else "Baseline annual demand requirement"
        )

        # Build step-by-step mathematical derivation
        d_str = f"{r.annual_demand:,.2f}" if abs(r.annual_demand - round(r.annual_demand)) > 1e-4 else f"{r.annual_demand:,.0f}"
        step_idx = 1
        math_steps = []

        # Pre-format LaTeX bold answer values to keep f-strings clean and robust
        ans_d_norm_str = f"{r.annual_demand:,.2f} \\text{{ {r.demand_unit}/year}}"
        ans_eoq_str = f"{r.eoq:,.2f} \\text{{ {r.demand_unit}}}"
        ans_order_cost_str = f"\\${r.annual_ordering_cost:,.2f}"
        ans_holding_cost_str = f"\\${r.annual_holding_cost:,.2f}"
        ans_total_cost_str = f"\\${r.total_annual_cost:,.2f}"
        ans_orders_str = f"{r.orders_per_year:,.2f} \\text{{ orders/year}}"
        ans_cycle_str = f"{r.cycle_time_days:,.2f} \\text{{ working days}}"

        if r.demand_normalization_applied:
            math_steps.append(
                f"{step_idx}. **Demand Normalization to Annual Horizon**:\n"
                f"   $$D = \\text{{Demand}}_{{\\text{{{r.demand_period}}}}} \\times {r.demand_normalization_factor:g} = {r.raw_demand:,.2f} \\times {r.demand_normalization_factor:g} = \\mathbf{{{ans_d_norm_str}}}$$\n"
            )
            step_idx += 1

        math_steps.append(
            f"{step_idx}. **EOQ Formula**:\n"
            f"   $$EOQ = \\sqrt{{\\frac{{2 \\cdot D \\cdot S}}{{H}}}} = \\sqrt{{\\frac{{2 \\cdot {d_str} \\cdot {r.order_cost:,.2f}}}{{{r.holding_cost:,.2f}}}}} = \\mathbf{{{ans_eoq_str}}}$$\n"
        )
        step_idx += 1

        math_steps.append(
            f"{step_idx}. **Annual Ordering Cost**:\n"
            f"   $$\\text{{Ordering Cost}} = \\frac{{D}}{{Q^*}} \\cdot S = \\frac{{{d_str}}}{{{r.eoq:,.2f}}} \\cdot {r.order_cost:,.2f} = \\mathbf{{{ans_order_cost_str}}}$$\n"
        )
        step_idx += 1

        math_steps.append(
            f"{step_idx}. **Annual Holding Cost**:\n"
            f"   $$\\text{{Holding Cost}} = \\frac{{Q^*}}{{2}} \\cdot H = \\frac{{{r.eoq:,.2f}}}{{2}} \\cdot {r.holding_cost:,.2f} = \\mathbf{{{ans_holding_cost_str}}}$$\n"
        )
        step_idx += 1

        math_steps.append(
            f"{step_idx}. **Total Annual Inventory Policy Cost**:\n"
            f"   $$\\text{{Total Annual Cost}} = \\text{{Ordering Cost}} + \\text{{Holding Cost}} = \\${r.annual_ordering_cost:,.2f} + \\${r.annual_holding_cost:,.2f} = \\mathbf{{{ans_total_cost_str}}}$$\n"
        )
        step_idx += 1

        math_steps.append(
            f"{step_idx}. **Annual Order Frequency**:\n"
            f"   $$\\text{{Order Frequency }} (N) = \\frac{{D}}{{Q^*}} = \\frac{{{d_str}}}{{{r.eoq:,.2f}}} = \\mathbf{{{ans_orders_str}}}$$\n"
        )
        step_idx += 1

        math_steps.append(
            f"{step_idx}. **Order Cycle Time**:\n"
            f"   $$\\text{{Order Cycle Time }} (T) = \\frac{{\\text{{Operating Days per Year}}}}{{\\text{{Annual Order Frequency}}}} = \\frac{{{r.workdays_per_year}}}{{{r.orders_per_year:,.2f}}} = \\mathbf{{{ans_cycle_str}}}$$\n"
        )

        derivation_md = "\n".join(math_steps)

        return f"""### 🎯 Executive Decision Summary: Economic Order Quantity (EOQ)

{unit_norm_callout}> **Recommended Order Quantity**: **`{r.eoq:,.0f} {r.demand_unit}`** per order (exact: `{r.eoq:,.2f}`)
> **Total Annual Inventory Policy Cost**: **`${r.total_annual_cost:,.2f}`**

---

#### 📊 Operational Schedule & Cost Breakdown
| Operational Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Optimal Order Quantity ($Q^*$)** | **`{r.eoq:,.2f}` {r.demand_unit}** | Balance point where ordering cost equals holding cost |
{original_demand_row}| **Annual Demand ($D$)** | **`{r.annual_demand:,.2f}` {r.demand_unit}/year** | {annual_demand_meaning} |
| **Annual Ordering Cost** | **`${r.annual_ordering_cost:,.2f}`** | $(D / Q^*) \\times S$ |
| **Annual Holding Cost** | **`${r.annual_holding_cost:,.2f}`** | $(Q^* / 2) \\times H$ |
| **Total Inventory Policy Cost** | **`${r.total_annual_cost:,.2f}`** | Ordering Cost + Holding Cost |
| **Order Frequency ($N$)** | **`{r.orders_per_year:,.2f}` orders/year** | Number of replenishment cycles per year ($D / Q^*$) |
| **Operating Days per Year ($W$)** | **`{r.workdays_per_year}` working days** | Operating schedule baseline ({operating_schedule_label}) |
| **Order Cycle Time ($T$)** | **`{r.cycle_time_days:,.2f}` working days** | Working days between successive orders ($W / N$) |
{purchase_line}
{rop_line}

---

#### 📐 Mathematical Derivation & Calculations
{derivation_md}

---

#### 🔍 Verification & Optimality Audit
{checks_md}
{warnings_md}

---

#### 💡 Sensitivity & Strategic Recommendations
- **Cost Robustness**: The EOQ total cost curve is relatively flat around the minimum. Ordering **+10%** (`{r.eoq * 1.1:,.0f}` {r.demand_unit}) increases policy cost by only **`${r.sensitivity_plus_10_cost - r.total_annual_cost:,.2f}`** (+{((r.sensitivity_plus_10_cost / r.total_annual_cost) - 1.0):.2%}).
- **Batch / Pallet Sizing**: If your supplier requires round packaging (e.g. pallets of 50 or 100), rounding to the nearest batch constraint causes minimal cost penalty.
- **Cycle Time**: Replenishment should be triggered approximately every **`{r.cycle_time_days:,.2f} working days`** (based on {r.workdays_per_year} operating days/year)."""

    def _format_quantity_discount_explanation(self, r: QuantityDiscountResult, v: Optional[VerificationResult]) -> str:
        table_rows = []
        tier_steps = []
        for t in r.tiers:
            max_str = f"{t.max_quantity:,.0f}" if t.max_quantity is not None else "∞"
            bracket_str = f"{t.min_quantity:,.0f} - {max_str}" if t.max_quantity is not None else f"{t.min_quantity:,.0f}+"
            opt_badge = " **(OPTIMAL)**" if t.is_optimal else ""
            q_val_str = f"{t.feasible_quantity:,.2f}" if abs(t.feasible_quantity - round(t.feasible_quantity)) > 1e-4 else f"{t.feasible_quantity:,.0f}"
            eoq_str = f"{t.unconstrained_eoq:,.2f}" if abs(t.unconstrained_eoq - round(t.unconstrained_eoq)) > 1e-4 else f"{t.unconstrained_eoq:,.1f}"

            status_label = t.feasibility_status or (
                "Feasible at EOQ" if t.is_feasible_at_eoq
                else (f"Clamped to Max ({t.max_quantity:,.0f})" if (t.max_quantity is not None and t.unconstrained_eoq > t.max_quantity)
                      else f"Clamped to Min ({t.min_quantity:,.0f})")
            )

            table_rows.append(
                f"| Tier {t.tier_number} | {bracket_str} | ${t.unit_price:,.2f} | "
                f"{eoq_str} | **{q_val_str}** | {status_label} | "
                f"${t.annual_ordering_cost:,.2f} | ${t.annual_holding_cost:,.2f} | "
                f"${t.annual_purchase_cost:,.2f} | **${t.total_annual_cost:,.2f}**{opt_badge} |"
            )

            if t.is_feasible_at_eoq:
                tier_note = (
                    f"Calculated unconstrained EOQ ({eoq_str} units) falls within tier boundaries [{bracket_str}]. "
                    f"Feasible candidate evaluated at **{q_val_str} units**."
                )
            elif t.max_quantity is not None and t.unconstrained_eoq > t.max_quantity:
                tier_note = (
                    f"Calculated unconstrained EOQ ({eoq_str} units) exceeds tier upper limit ({max_str} units). "
                    f"Feasible candidate evaluated at tier upper bound **{q_val_str} units**."
                )
            else:
                tier_note = (
                    f"Calculated unconstrained EOQ ({eoq_str} units) falls below minimum qualifying threshold ({t.min_quantity:,.0f} units). "
                    f"Unit price ${t.unit_price:,.2f} requires at least {t.min_quantity:,.0f} units. Feasible candidate evaluated at tier minimum **{q_val_str} units**."
                )

            tier_steps.append(
                f"- **Tier {t.tier_number} ({bracket_str} @ ${t.unit_price:,.2f}/unit)**: "
                f"Holding cost $H = {t.holding_cost:,.2f}/unit/yr. {tier_note} "
                f"Costs: Ordering = `${t.annual_ordering_cost:,.2f}`, Holding = `${t.annual_holding_cost:,.2f}`, Purchase = `${t.annual_purchase_cost:,.2f}` ➔ **Total Annual Cost = `${t.total_annual_cost:,.2f}`**."
            )

        tiers_table = "\n".join(table_rows)
        tiers_derivation = "\n".join(tier_steps)

        checks_md = "\n".join([f"- ✅ {c}" for c in (v.checks_passed if v else [])])
        warnings_md = "\n".join([f"- ⚠️ {w}" for w in (v.warnings if v else [])])

        opt_q_str = f"{r.optimal_quantity:,.2f}" if abs(r.optimal_quantity - round(r.optimal_quantity)) > 1e-4 else f"{r.optimal_quantity:,.0f}"

        opt = r.tiers[r.optimal_tier_index]
        purch_cost = getattr(r, "annual_purchase_cost", 0.0) or opt.annual_purchase_cost
        order_cost = getattr(r, "annual_ordering_cost", 0.0) or opt.annual_ordering_cost
        hold_cost = getattr(r, "annual_holding_cost", 0.0) or opt.annual_holding_cost
        policy_cost = getattr(r, "total_annual_policy_cost", 0.0) or (order_cost + hold_cost)
        base_tier = r.tiers[0]
        base_price_str = f"${base_tier.unit_price:,.2f}"

        return f"""### 🎯 Executive Decision Summary: Quantity Discount Optimization

> **Recommended Order Quantity**: **`{opt_q_str} units`** (qualifies for Tier {r.optimal_tier_index + 1} @ **`${r.optimal_unit_price:,.2f}/unit`**)
> **Purchase Cost**: **`${purch_cost:,.2f}`**
> **Annual Ordering Cost**: **`${order_cost:,.2f}`**
> **Annual Holding Cost**: **`${hold_cost:,.2f}`**
> **Total Annual Inventory Policy Cost (excluding purchase cost)**: **`${policy_cost:,.2f}`**
> **Total Annual Cost (including purchase cost)**: **`${r.min_total_cost:,.2f}`**
> **Annual Net Savings vs. Baseline Price ({base_price_str}/unit)**: **`${r.annual_savings:,.2f}`** (`{r.percent_savings:.2f}%` total savings)

---

#### 📊 Tier-by-Tier Economic Comparison
| Tier | Bracket Range | Price/Unit | Unconstrained EOQ | Evaluated Candidate $Q$ | Feasibility Status | Ordering Cost | Holding Cost | Purchase Cost | Total Annual Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{tiers_table}

---

#### 📐 Quantity Discount Optimization Logic & Step-by-Step Derivation
1. **Tier-by-Tier Boundary & Feasibility Evaluation**:
{tiers_derivation}

2. **Feasibility Rule**:
   - A tier's unit price applies **only** to order quantities that qualify for that tier. Discounted prices are never applied to order quantities outside their respective tier boundaries.
   - If unconstrained EOQ is within tier boundaries, evaluate at EOQ.
   - If unconstrained EOQ is below the tier minimum, evaluate at the tier minimum quantity.
   - If unconstrained EOQ exceeds the tier maximum, evaluate at the tier maximum quantity.

3. **Global Economic Minimization**:
   - Comparing all feasible candidate order quantities across all evaluated brackets, **Tier {r.optimal_tier_index + 1} at `{opt_q_str} units`** achieves the absolute minimum total annual cost of **`${r.min_total_cost:,.2f}`**, delivering **`${r.annual_savings:,.2f}` in net annual savings**.

---

#### 🔍 Verification & Audit
{checks_md}
{warnings_md}

---

#### 💡 Managerial & Working Capital Insights
- **Warehouse Capacity Check**: Ensure warehouse or facility storage can accommodate peak physical receipts of **`{opt_q_str} units`**.
- **Working Capital Commitment**: An order of `{opt_q_str}` units requires an upfront capital outlay of **`${r.optimal_quantity * r.optimal_unit_price:,.2f}`** per batch."""

    def _format_safety_stock_rop_explanation(self, r: SafetyStockROPResult, v: Optional[VerificationResult]) -> str:
        checks_md = "\n".join([f"- ✅ {c}" for c in (v.checks_passed if v else [])])
        warnings_md = "\n".join([f"- ⚠️ {w}" for w in (v.warnings if v else [])])

        stoch_type = "Demand Variable, Lead Time Constant"
        if r.demand_std_per_period > 0 and r.lead_time_std > 0:
            stoch_type = "Both Demand and Lead Time Variable (Compound Stochastic Risk)"
        elif r.demand_std_per_period == 0 and r.lead_time_std > 0:
            stoch_type = "Demand Constant, Lead Time Variable"
        elif r.demand_std_per_period == 0 and r.lead_time_std == 0:
            stoch_type = "Constant Demand, Constant Lead Time (Deterministic)"

        unit_norm_md = ""
        if r.unit_normalization_applied:
            unit_norm_md = (
                f"> ⏱️ **Time Unit Normalization Applied**:\n"
                f"> - **Demand Time Unit**: **`{r.demand_time_unit}`**\n"
                f"> - **Original Lead Time**: **`{r.raw_lead_time:g} {r.lead_time_unit}`** (std dev: **`{r.raw_lead_time_std:g} {r.lead_time_unit}`**)\n"
                f"> - **Operating Schedule**: **`{r.workdays_per_week:g} working days per week`**\n"
                f"> - **Normalized Lead Time ($L$)**: **`{r.lead_time:,.2f}` working days** (${r.raw_lead_time:g} \\times {r.workdays_per_week:g}$)\n"
                f"> - **Normalized Lead Time Std Dev ($\\sigma_{{LT}}$)**: **`{r.lead_time_std:,.2f}` working days** (${r.raw_lead_time_std:g} \\times {r.workdays_per_week:g}$)\n\n---\n\n"
            )

        d_unit_singular = r.demand_time_unit[:-1] if r.demand_time_unit.endswith('s') else r.demand_time_unit

        step1_formula = (
            f"1. **Expected Lead Time Demand**:\n"
            f"   $$\\mu_L = d \\cdot L = {r.avg_demand_per_period:,.2f} \\times {r.lead_time:,.2f} = \\mathbf{{{r.expected_lead_time_demand:,.2f} \\text{{ units}}}}$$\n"
        )
        if r.demand_std_per_period > 0 and r.lead_time_std > 0:
            d_var_val = r.lead_time * (r.demand_std_per_period**2)
            lt_var_val = (r.avg_demand_per_period**2) * (r.lead_time_std**2)
            tot_var_val = d_var_val + lt_var_val
            step2_formula = (
                f"2. **Standard Deviation of Lead Time Demand (Compound Uncertainty)**:\n"
                f"   $$\\begin{{aligned}}\n"
                f"   \\sigma_{{LTD}} &= \\sqrt{{L \\cdot \\sigma_d^2 + d^2 \\cdot \\sigma_{{LT}}^2}}\\\\\n"
                f"   &= \\sqrt{{{r.lead_time:,.2f} \\cdot ({r.demand_std_per_period:,.2f})^2 + ({r.avg_demand_per_period:,.2f})^2 \\cdot ({r.lead_time_std:,.2f})^2}}\\\\\n"
                f"   &= \\sqrt{{{d_var_val:,.2f} + {lt_var_val:,.2f}}}\\\\\n"
                f"   &= \\sqrt{{{tot_var_val:,.2f}}} = \\mathbf{{{r.std_lead_time_demand:,.2f} \\text{{ units}}}}\n"
                f"   \\end{{aligned}}$$\n"
            )
        elif r.demand_std_per_period > 0 and r.lead_time_std == 0:
            step2_formula = (
                f"2. **Standard Deviation of Lead Time Demand (Constant Lead Time)**:\n"
                f"   $$\\sigma_{{LTD}} = \\sigma_d \\cdot \\sqrt{{L}} = {r.demand_std_per_period:,.2f} \\cdot \\sqrt{{{r.lead_time:,.2f}}} = \\mathbf{{{r.std_lead_time_demand:,.2f} \\text{{ units}}}}$$\n"
            )
        elif r.demand_std_per_period == 0 and r.lead_time_std > 0:
            step2_formula = (
                f"2. **Standard Deviation of Lead Time Demand (Constant Demand)**:\n"
                f"   $$\\sigma_{{LTD}} = d \\cdot \\sigma_{{LT}} = {r.avg_demand_per_period:,.2f} \\cdot {r.lead_time_std:,.2f} = \\mathbf{{{r.std_lead_time_demand:,.2f} \\text{{ units}}}}$$\n"
            )
        else:
            step2_formula = (
                f"2. **Standard Deviation of Lead Time Demand (Deterministic)**:\n"
                f"   $$\\sigma_{{LTD}} = \\mathbf{{0.00 \\text{{ units}}}}$$\n"
            )

        step3_formula = (
            f"3. **Safety Stock Buffer**:\n"
            f"   $$SS = z \\cdot \\sigma_{{LTD}} = {r.z_score:.4f} \\times {r.std_lead_time_demand:,.2f} = \\mathbf{{{r.safety_stock:,.2f} \\text{{ units}}}}$$\n"
        )
        step4_formula = (
            f"4. **Reorder Point ($ROP$)**:\n"
            f"   $$ROP = \\mu_L + SS = {r.expected_lead_time_demand:,.2f} + {r.safety_stock:,.2f} = \\mathbf{{{r.reorder_point:,.2f} \\text{{ units}}}}$$\n"
        )

        lt_interpretation = (
            f"Normalized from {r.raw_lead_time:g} {r.lead_time_unit} using {r.workdays_per_week:g} working days/week"
            if r.unit_normalization_applied
            else "Replenishment order transit duration"
        )
        sigma_lt_interpretation = (
            f"Normalized from {r.raw_lead_time_std:g} {r.lead_time_unit} using {r.workdays_per_week:g} working days/week"
            if r.unit_normalization_applied
            else "Replenishment delivery variability"
        )

        return f"""### 🎯 Executive Decision Summary: Safety Stock & Reorder Point (ROP)

> **Reorder Point (ROP)**: **`{r.reorder_point:,.0f} units`** (exact: `{r.reorder_point:,.2f}`)
> **Safety Stock Buffer**: **`{r.safety_stock:,.0f} units`** (exact: `{r.safety_stock:,.2f}`)
> **Target Cycle Service Level**: **`{r.service_level:.1%}`** ($z = {r.z_score:.4f}$)

---

{unit_norm_md}#### 📊 Policy Parameters & Buffer Metrics
| Inventory Parameter | Value | Interpretation |
| :--- | :--- | :--- |
| **System Classification** | **{stoch_type}** | Continuous Review $(s, Q)$ system |
| **Average Demand ($d$)** | **`{r.avg_demand_per_period:,.2f}` units/{d_unit_singular}** | Mean sales velocity per period |
| **Demand Std Dev ($\\sigma_d$)** | **`{r.demand_std_per_period:,.2f}` units/{d_unit_singular}** | Period demand volatility |
| **Average Replenishment Lead Time ($L$)** | **`{r.lead_time:,.2f}` {r.time_unit}** | {lt_interpretation} |
| **Lead Time Std Dev ($\\sigma_{{LT}}$)** | **`{r.lead_time_std:,.2f}` {r.time_unit}** | {sigma_lt_interpretation} |
| **Expected Lead Time Demand ($\\mu_L$)** | **`{r.expected_lead_time_demand:,.2f}` units** | $d \\times L$ expected sales during transit |
| **Std Dev of Lead Time Demand ($\\sigma_{{LTD}}$)** | **`{r.std_lead_time_demand:,.2f}` units** | Compound lead time demand uncertainty |
| **Safety Stock ($SS$)** | **`{r.safety_stock:,.2f}` units** | Buffer to absorb demand & lead time volatility |
| **Reorder Point ($ROP$)** | **`{r.reorder_point:,.2f}` units** | Trigger replenishment when inventory drops to this level |
| **Stockout Risk Per Cycle** | **`{r.stockout_risk_per_cycle_percent:.2f}%`** | Probability of stocking out before delivery arrives |

---

#### 📐 Mathematical Formulas & Step-by-Step Execution
{step1_formula}
{step2_formula}
{step3_formula}
{step4_formula}
---

#### 🔍 Verification & Audit
{checks_md}
{warnings_md}

---

#### 💡 Managerial & Operational Guidance
- **Trigger Rule**: Whenever inventory on-hand plus open orders drops to or below **`{r.reorder_point:,.0f} units`**, immediately generate a replenishment order.
- **Lead Time Variability Impact**: When supplier lead time is stochastic ($\\sigma_{{LT}} = {r.lead_time_std:,.2f}$ {r.time_unit}), lead time uncertainty often accounts for the dominant portion of safety stock ($d^2 \\cdot \\sigma_{{LT}}^2 = {((r.avg_demand_per_period**2)*(r.lead_time_std**2)):,.0f}$). Standardizing supplier delivery windows directly compresses working capital.
- **Service Level Buffer**: Operating at **`{r.service_level:.1%}`** Cycle Service Level requires holding **`{r.safety_stock:,.0f} units`** in reserve to guarantee an in-stock replenishment cycle probability of **`{r.service_level:.1%}`**."""

    def _format_fixed_period_explanation(self, r: FixedPeriodReviewResult, v: Optional[VerificationResult]) -> str:
        checks_md = "\n".join([f"- ✅ {c}" for c in (v.checks_passed if v else [])])
        warnings_md = "\n".join([f"- ⚠️ {w}" for w in (v.warnings if v else [])])

        exact_q = getattr(r, "exact_order_quantity", r.recommended_order_quantity)
        round_up = getattr(r, "round_up", False) or r.inputs.get("round_up", False)

        if round_up:
            order_qty_header = f"**`{r.recommended_order_quantity:,.0f} units`** (exact: `{exact_q:,.2f}`, rounded up to nearest whole unit)"
            order_qty_table = f"**`{r.recommended_order_quantity:,.0f}` units** (exact: `{exact_q:,.2f}`)"
            step3_text = (
                f"$$\\begin{{aligned}}\n"
                f"   \\text{{Inventory Position ($IP$)}} &= \\text{{On-Hand}} + \\text{{On-Order}} - \\text{{Backorders}}\\\\\n"
                f"   &= {r.current_inventory:,.0f} + {r.on_order:,.0f} - {r.backorders:,.0f} = \\mathbf{{{r.inventory_position:,.2f} \\text{{ units}}}}\\\\\n"
                f"   Q &= \\max(0, M - IP) = \\max(0, {r.order_up_to_level:,.2f} - {r.inventory_position:,.2f}) = {exact_q:,.2f} \\text{{ units}}\\\\\n"
                f"   Q_{{\\text{{rounded up}}}} &= \\lceil {exact_q:,.2f} \\rceil = \\mathbf{{{r.recommended_order_quantity:,.0f} \\text{{ units}}}}\n"
                f"   \\end{{aligned}}$$"
            )
        else:
            order_qty_header = f"**`{r.recommended_order_quantity:,.0f} units`** (exact: `{r.recommended_order_quantity:,.2f}`)"
            order_qty_table = f"**`{r.recommended_order_quantity:,.2f}` units**"
            step3_text = (
                f"$$\\begin{{aligned}}\n"
                f"   \\text{{Inventory Position ($IP$)}} &= \\text{{On-Hand}} + \\text{{On-Order}} - \\text{{Backorders}}\\\\\n"
                f"   &= {r.current_inventory:,.0f} + {r.on_order:,.0f} - {r.backorders:,.0f} = \\mathbf{{{r.inventory_position:,.2f} \\text{{ units}}}}\\\\\n"
                f"   Q &= \\max(0, M - IP) = \\max(0, {r.order_up_to_level:,.2f} - {r.inventory_position:,.2f}) = \\mathbf{{{r.recommended_order_quantity:,.2f} \\text{{ units}}}}\n"
                f"   \\end{{aligned}}$$"
            )

        return f"""### 🎯 Executive Decision Summary: Fixed-Period Review Model (Periodic Review)

> **Recommended Order Quantity**: {order_qty_header}
> **Target Order-Up-To Level ($M$)**: **`{r.order_up_to_level:,.0f} units`** (exact: `{r.order_up_to_level:,.2f}`)
> **Safety Stock over Protection Interval**: **`{r.safety_stock:,.2f} units`**

---

#### 📊 Review System Status & Protection Parameters
| Parameter | Value | Interpretation |
| :--- | :--- | :--- |
| **Review Interval ($T$)** | **`{r.review_period:,.1f}` {r.time_unit}** | Time between consecutive inventory reviews |
| **Replenishment Lead Time ($L$)** | **`{r.lead_time:,.1f}` {r.time_unit}** | Order transit and processing duration |
| **Protection Interval ($T + L$)** | **`{r.protection_period:,.1f}` {r.time_unit}** | Total risk exposure duration between review and next delivery |
| **Expected Demand over $(T + L)$** | **`{r.expected_demand_protection_period:,.2f}` units** | $d \\times (T + L)$ |
| **Std Dev over $(T + L)$** | **`{r.std_demand_protection_period:,.2f}` units** | $\\sigma_d \\times \\sqrt{{T + L}}$ |
| **Current On-Hand Inventory** | **`{r.current_inventory:,.0f}` units** | Stock physically available |
| **On Order (In-Transit)** | **`{r.on_order:,.0f}` units** | Scheduled replenishment arriving soon |
| **Backorders** | **`{r.backorders:,.0f}` units** | Committed unfulfilled customer demand |
| **Current Inventory Position ($IP$)** | **`{r.inventory_position:,.0f}` units** | $\\text{{On-Hand}} + \\text{{On-Order}} - \\text{{Backorders}}$ |
| **Order-Up-To Level ($M$)** | **`{r.order_up_to_level:,.2f}` units** | Target base stock level |
| **Order Quantity ($Q = M - IP$)** | {order_qty_table} | Order required to restore inventory position to target |

---

#### 📐 Mathematical Derivation
1. **Protection Interval**:
   $$\\text{{Risk Window}} = T + L = {r.review_period} + {r.lead_time} = \\mathbf{{{r.protection_period} \\text{{ {r.time_unit}}}}}$$

2. **Order-Up-To Level ($M$)**:
   $$M = d \\cdot (T + L) + z \\cdot \\sigma_d \\sqrt{{T + L}} = {r.expected_demand_protection_period:,.2f} + ({r.z_score:.4f} \\times {r.std_demand_protection_period:,.2f}) = \\mathbf{{{r.order_up_to_level:,.2f}}}$$

3. **Replenishment Order Size ($Q$)**:
   {step3_text}

---

#### 🔍 Verification & Audit
{checks_md}
{warnings_md}

---

#### 💡 Periodic Review vs Continuous Review Comparison
- **Protection Exposure**: Periodic review systems require higher safety stock than continuous $(s, Q)$ systems because they must protect against demand variation across $T + L$ rather than just $L$.
- **Coordination Advantage**: Fixed-period reviews allow joint replenishment of multiple SKUs from the same vendor to take advantage of freight truckload consolidation."""

    def _format_newsvendor_explanation(self, r: NewsvendorResult, v: Optional[VerificationResult]) -> str:
        checks_md = "\n".join([f"- ✅ {c}" for c in (v.checks_passed if v else [])])
        warnings_md = "\n".join([f"- ⚠️ {w}" for w in (v.warnings if v else [])])

        bias_text = (
            "Because profit margin exceeds salvage loss ($C_u > C_o$), the critical ratio is > 50%, "
            f"making it optimal to order above expected demand ({r.optimal_order_quantity:,.0f} vs mean {r.demand_mean:,.0f})."
            if r.critical_fractile > 0.5
            else "Because salvage markdown loss exceeds profit margin ($C_o > C_u$), it is optimal to order conservatively below mean demand."
        )

        return f"""### 🎯 Executive Decision Summary: Newsvendor Model (Single-Period Stochastic)

> **Recommended Order Quantity ($Q^*$)**: **`{r.optimal_order_quantity:,.0f} units`** (exact: `{r.optimal_order_quantity:,.2f}`)
> **Critical Fractile ($CR$)**: **`{r.critical_fractile:.4f}`** (`{r.critical_fractile * 100:.2f}%` target in-stock probability)
> **Expected Profit**: **`${r.expected_profit:,.2f}`**

---

#### 📊 Marginal Economics & Distributional Metrics
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Retail Selling Price ($P$)** | **`${r.selling_price:,.2f}`** | Standard selling price |
| **Unit Purchase Cost ($C$)** | **`${r.unit_cost:,.2f}`** | Wholesale acquisition cost |
| **Salvage / Outlet Value ($S$)** | **`${r.salvage_value:,.2f}`** | Liquidation value for unsold units |
| **Underage Cost ($C_u$)** | **`${r.underage_cost:,.2f}`** | $P - C$: Profit lost per unit of unmet demand |
| **Overage Cost ($C_o$)** | **`${r.overage_cost:,.2f}`** | $C - S$: Loss incurred per unsold leftover unit |
| **Critical Ratio ($CR$)** | **`{r.critical_fractile:.4f}`** | $C_u / (C_u + C_o)$: Optimal cumulative service level |
| **Optimal Z-Score ($z^*$)** | **`{r.z_score:.4f}`** | Standard normal inverse CDF at $CR$ |
| **Expected Sales** | **`{r.expected_sales:,.1f}` units** | Expected volume sold at full retail price |
| **Expected Leftover Inventory** | **`{r.expected_leftover_units:,.1f}` units** | Expected units liquidated at salvage price |
| **Expected Lost Sales** | **`{r.expected_lost_sales:,.1f}` units** | Expected unsatisfied customer demand |
| **Expected Net Profit** | **`${r.expected_profit:,.2f}`** | Expected Revenue + Salvage - Purchase Cost |

---

#### 📐 Step-by-Step Optimization Derivation
1. **Marginal Analysis**:
   $$C_u = P - C = {r.selling_price:,.2f} - {r.unit_cost:,.2f} = \\mathbf{{\\${r.underage_cost:,.2f}}}$$
   $$C_o = C - S = {r.unit_cost:,.2f} - {r.salvage_value:,.2f} = \\mathbf{{\\${r.overage_cost:,.2f}}}$$

2. **Critical Fractile (Critical Ratio)**:
   $$CR = \\frac{{C_u}}{{C_u + C_o}} = \\frac{{{r.underage_cost:,.2f}}}{{{r.underage_cost:,.2f} + {r.overage_cost:,.2f}}} = \\mathbf{{{r.critical_fractile:.4f}}}$$

3. **Optimal Order Quantity ($Q^*$)**:
   $$Q^* = \\mu + z^* \\cdot \\sigma = {r.demand_mean:,.0f} + ({r.z_score:.4f} \\times {r.demand_std:,.0f}) = \\mathbf{{{r.optimal_order_quantity:,.2f} \\text{{ units}}}}$$

4. **Expected Profit Calculation**:
   $$\\mathbb{{E}}[\\Pi] = P \\cdot \\mathbb{{E}}[\\text{{Sales}}] + S \\cdot \\mathbb{{E}}[\\text{{Leftovers}}] - C \\cdot Q^* = \\mathbf{{\\${r.expected_profit:,.2f}}}$$

---

#### 🔍 Verification & Audit
{checks_md}
{warnings_md}

---

#### 💡 Strategic Business Takeaways
- **Risk Asymmetry**: {bias_text}
- **Salvage Negotiation**: Every $1 increase in salvage value reduces $C_o$, shifting the critical ratio upward and enabling you to capture more profitable sales with less downside risk."""
