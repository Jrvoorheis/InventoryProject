"""Comprehensive Test Suite for Supply Chain Analytics Inventory Tools and Agent.

Tests cover:
- Numerical accuracy across all 5 inventory models against standard supply chain benchmarks.
- Edge cases and mathematical boundary conditions.
- Verification audits (convexity, derivative conditions, bracket compliance).
- End-to-end natural language agent classification, extraction, calculation, verification, and explanation.
"""

import math
import pandas as pd
import pytest

from app import process_inventory_dataframe

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_safety_stock_rop,
    calculate_periodic_review,
    calculate_fixed_period_review,
    calculate_newsvendor,
    calculate_multicriteria_classification,
    EOQResult,
    QuantityDiscountResult,
    ROPResult,
    PeriodicReviewResult,
    NewsvendorResult,
    MulticriteriaClassificationResult,
)
from inventory_agent import InventoryDecisionAgent, InventoryModelType


# =====================================================================
# Model 1 Tests: Economic Order Quantity (EOQ)
# =====================================================================

class TestEOQ:
    """Test suite for Economic Order Quantity calculations."""

    def test_classic_eoq_benchmark(self):
        """Classic textbook problem: D=10,000, S=$50, H=$2."""
        res = calculate_eoq(annual_demand=10000, order_cost=50, holding_cost=2.0)
        # Expected EOQ = sqrt(2 * 10000 * 50 / 2) = sqrt(500,000) ~= 707.10678
        assert pytest.approx(res.eoq, rel=1e-4) == 707.1068
        assert pytest.approx(res.annual_ordering_cost, rel=1e-4) == 707.1068
        assert pytest.approx(res.annual_holding_cost, rel=1e-4) == 707.1068
        assert pytest.approx(res.total_annual_cost, rel=1e-4) == 1414.2136
        assert pytest.approx(res.orders_per_year, rel=1e-4) == 14.1421
        assert res.verification.is_valid
        assert len(res.verification.errors) == 0

    def test_eoq_with_holding_cost_rate_and_unit_cost(self):
        """EOQ where H is specified as h * C (e.g. 20% of $25 = $5.00)."""
        res = calculate_eoq(annual_demand=4000, order_cost=25, holding_cost_rate=0.20, unit_cost=25.0)
        # H = 0.20 * 25 = 5.0
        # EOQ = sqrt(2 * 4000 * 25 / 5) = sqrt(40000) = 200
        assert pytest.approx(res.eoq, abs=1e-3) == 200.0
        assert res.holding_cost == 5.0
        assert res.annual_ordering_cost == 500.0
        assert res.annual_holding_cost == 500.0
        assert res.annual_purchase_cost == 100000.0
        assert res.total_cost_with_purchase == 101000.0

    def test_eoq_with_lead_time_deterministic_rop(self):
        """EOQ with lead time computes deterministic ROP."""
        res = calculate_eoq(annual_demand=2500, order_cost=40, holding_cost=2.0, lead_time_days=10, workdays_per_year=250)
        # daily demand = 2500 / 250 = 10 units/day
        # deterministic ROP = 10 * 10 = 100 units
        assert res.reorder_point == 100.0

    def test_eoq_order_cycle_time_benchmark_with_operating_days(self):
        """User benchmark test: D=12,000, S=$75, H=$3/yr, Operating Days=300 days/yr.
        EOQ = sqrt(2 * 12,000 * 75 / 3) = 774.60 units
        Order frequency N = 12,000 / 774.60 = 15.49 orders/year
        Order Cycle Time T = 300 / 15.49 = 19.36 working days.
        """
        res = calculate_eoq(annual_demand=12000, order_cost=75, holding_cost=3.0, workdays_per_year=300)
        assert pytest.approx(res.eoq, abs=1e-2) == 774.60
        assert pytest.approx(res.orders_per_year, abs=1e-2) == 15.49
        assert pytest.approx(res.cycle_time_days, abs=1e-2) == 19.36
        assert res.workdays_per_year == 300
        assert any("300 operating days per year" in a for a in res.assumptions)
        assert res.intermediate_calculations["workdays_per_year"] == 300

    def test_eoq_order_cycle_time_documented_default_250_days(self):
        """Default operating days: D=12,000, S=$75, H=$3/yr without operating days passed.
        Defaults to 250 working days per year (50 weeks * 5 days).
        Order Cycle Time T = 250 / 15.49 = 16.14 working days.
        """
        res = calculate_eoq(annual_demand=12000, order_cost=75, holding_cost=3.0)
        assert pytest.approx(res.eoq, abs=1e-2) == 774.60
        assert pytest.approx(res.orders_per_year, abs=1e-2) == 15.49
        assert pytest.approx(res.cycle_time_days, abs=1e-2) == 16.14
        assert res.workdays_per_year == 250
        assert any("250 operating days per year" in a for a in res.assumptions)

    def test_eoq_monthly_demand_normalization_benchmark(self):
        """Exact user regression test for monthly demand:
        Demand: 1,000 units per month -> Normalized annual demand: 12,000 units/year
        Ordering cost: $25 per order
        Holding cost: $0.50 per unit per year
        Default operating days: 250 days/year
        Expected:
        - EOQ = sqrt(2 * 12,000 * 25 / 0.50) = 1,095.45 units
        - Annual ordering cost = (12,000 / 1,095.45) * 25 = $273.86
        - Annual holding cost = (1,095.45 / 2) * 0.50 = $273.86
        - Total annual inventory policy cost = $547.72
        - Order frequency = 12,000 / 1,095.45 = 10.95 orders/year
        - Order cycle time = 250 / 10.95 = 22.82 working days
        """
        res = calculate_eoq(
            demand=1000,
            demand_period="month",
            order_cost=25,
            holding_cost=0.50,
            workdays_per_year=250,
        )
        assert res.raw_demand == 1000.0
        assert res.demand_period == "month"
        assert res.annual_demand == 12000.0
        assert pytest.approx(res.eoq, abs=1e-2) == 1095.45
        assert pytest.approx(res.annual_ordering_cost, abs=1e-2) == 273.86
        assert pytest.approx(res.annual_holding_cost, abs=1e-2) == 273.86
        assert pytest.approx(res.total_annual_cost, abs=1e-2) == 547.72
        assert pytest.approx(res.orders_per_year, abs=1e-2) == 10.95
        assert pytest.approx(res.cycle_time_days, abs=1e-2) == 22.82
        assert res.demand_normalization_applied is True
        assert res.demand_normalization_factor == 12.0
        assert res.verification.is_valid

    def test_eoq_other_demand_periods_weekly_quarterly_daily(self):
        """Test demand period normalization for weekly, quarterly, and daily demand."""
        # Weekly: 250 units/week * 52 = 13,000 units/year
        res_wk = calculate_eoq(demand=250, demand_period="week", order_cost=50, holding_cost=2.0)
        assert res_wk.raw_demand == 250.0
        assert res_wk.annual_demand == 13000.0
        assert pytest.approx(res_wk.eoq, abs=1e-2) == 806.23
        assert res_wk.demand_normalization_applied is True

        # Quarterly: 3,000 units/quarter * 4 = 12,000 units/year
        res_qtr = calculate_eoq(demand=3000, demand_period="quarter", order_cost=60, holding_cost=2.5)
        assert res_qtr.raw_demand == 3000.0
        assert res_qtr.annual_demand == 12000.0
        assert pytest.approx(res_qtr.eoq, abs=1e-2) == 758.95
        assert res_qtr.demand_normalization_applied is True

        # Daily: 40 units/day * 300 operating days = 12,000 units/year
        res_day = calculate_eoq(demand=40, demand_period="day", order_cost=75, holding_cost=3.0, workdays_per_year=300)
        assert res_day.raw_demand == 40.0
        assert res_day.annual_demand == 12000.0
        assert pytest.approx(res_day.eoq, abs=1e-2) == 774.60
        assert res_day.demand_normalization_applied is True

    def test_eoq_avoid_double_converting_annualized_demand(self):
        """Ensure demand that is already annualized is never double-converted."""
        # Case A: demand_period = 'year'
        res_ann = calculate_eoq(annual_demand=12000, demand_period="year", order_cost=75, holding_cost=3.0)
        assert res_ann.annual_demand == 12000.0
        assert res_ann.demand_normalization_applied is False
        assert pytest.approx(res_ann.eoq, abs=1e-2) == 774.60

        # Case B: annual_demand already converted and passed with raw_demand and demand_period='month'
        res_both = calculate_eoq(annual_demand=12000, raw_demand=1000, demand_period="month", order_cost=25, holding_cost=0.50)
        assert res_both.raw_demand == 1000.0
        assert res_both.annual_demand == 12000.0  # NOT 144,000
        assert pytest.approx(res_both.eoq, abs=1e-2) == 1095.45

    def test_eoq_invalid_inputs_raise(self):
        """Invalid inputs must raise ValueError."""
        with pytest.raises(ValueError, match="Annual demand must be positive"):
            calculate_eoq(annual_demand=-500, order_cost=50, holding_cost=2)

        with pytest.raises(ValueError, match="Order cost must be positive"):
            calculate_eoq(annual_demand=1000, order_cost=-10, holding_cost=2)

        with pytest.raises(ValueError, match="Must provide either a positive holding_cost"):
            calculate_eoq(annual_demand=1000, order_cost=50)


# =====================================================================
# Model 2 Tests: Quantity Discount Model
# =====================================================================

class TestQuantityDiscount:
    """Test suite for All-Units Quantity Discount."""

    def test_quantity_discount_tier_optimization(self):
        """Standard quantity discount with 3 brackets:
        D=1,000, S=$100, holding_cost_rate=20%
        Tier 1: 1-499 @ $10.00
        Tier 2: 500-999 @ $9.00
        Tier 3: 1000+ @ $8.00
        """
        schedule = [
            {"min_qty": 1, "max_qty": 499, "unit_price": 10.0},
            {"min_qty": 500, "max_qty": 999, "unit_price": 9.0},
            {"min_qty": 1000, "max_qty": None, "unit_price": 8.0},
        ]
        res = calculate_quantity_discount(
            annual_demand=1000,
            order_cost=100,
            discount_schedule=schedule,
            holding_cost_rate=0.20,
        )

        assert len(res.tiers) == 3
        # Tier 1 unconstrained EOQ = sqrt(2 * 1000 * 100 / 2.0) = 316.23 (feasible)
        # Tier 2 unconstrained EOQ = sqrt(2 * 1000 * 100 / 1.8) = 333.33 -> adjusted to 500
        # Tier 3 unconstrained EOQ = sqrt(2 * 1000 * 100 / 1.6) = 353.55 -> adjusted to 1000

        # Tier 1 Total Cost = (1000/316.23)*100 + (316.23/2)*2 + 10,000 = 632.46 + 10,000 = $10,632.46
        # Tier 2 Total Cost = (1000/500)*100 + (500/2)*1.8 + 9,000 = 200 + 450 + 9000 = $9,650.00
        # Tier 3 Total Cost = (1000/1000)*100 + (1000/2)*1.6 + 8,000 = 100 + 800 + 8000 = $8,900.00
        # Optimal is Tier 3 at Q=1000
        assert res.optimal_tier_index == 2
        assert res.optimal_quantity == 1000.0
        assert res.optimal_unit_price == 8.0
        assert pytest.approx(res.min_total_cost, abs=1e-2) == 8900.00
        assert pytest.approx(res.annual_savings, abs=1e-2) == (res.base_case_cost - 8900.00)
        assert res.verification.is_valid

    def test_quantity_discount_where_lower_tier_is_optimal(self):
        """Case where high holding costs make taking large discount suboptimal."""
        schedule = [
            {"min_qty": 1, "max_qty": 99, "unit_price": 100.0},
            {"min_qty": 100, "max_qty": None, "unit_price": 99.0},  # tiny discount, high holding
        ]
        res = calculate_quantity_discount(
            annual_demand=200,
            order_cost=10,
            discount_schedule=schedule,
            holding_cost=50.0,  # $50/unit fixed holding
        )
        # Tier 1 EOQ = sqrt(2 * 200 * 10 / 50) = sqrt(80) ~= 8.94 (feasible in 1-99)
        # Tier 1 TC = (200/8.94)*10 + (8.94/2)*50 + 20,000 = 447.21 + 20,000 = 20,447.21
        # Tier 2 Q=100 TC = (200/100)*10 + (100/2)*50 + 19,800 = 20 + 2500 + 19800 = 22,320
        # Optimal is Tier 1
        assert res.optimal_tier_index == 0
        assert pytest.approx(res.optimal_quantity, rel=1e-3) == 8.944

    def test_quantity_discount_boundary_conditions(self):
        """Test boundary conditions for feasible quantity determination across tiers:
        Case A: EOQ < min_qty -> clamped to min_qty
        Case B: min_qty <= EOQ <= max_qty -> feasible at EOQ
        Case C: EOQ > max_qty -> clamped to max_qty
        """
        # D=10,000, S=50, H=2.0 (constant fixed holding cost)
        # EOQ = sqrt(2 * 10,000 * 50 / 2.0) = sqrt(500,000) ~= 707.1068
        schedule = [
            # Tier 1: bracket [1, 500]. EOQ (707.1) > max_qty (500) -> feasible Q = 500
            {"min_qty": 1, "max_qty": 500, "unit_price": 10.0},
            # Tier 2: bracket [501, 1000]. EOQ (707.1) in [501, 1000] -> feasible Q = 707.1068
            {"min_qty": 501, "max_qty": 1000, "unit_price": 9.5},
            # Tier 3: bracket [1500, None]. EOQ (707.1) < min_qty (1500) -> feasible Q = 1500
            {"min_qty": 1500, "max_qty": None, "unit_price": 9.0},
        ]
        res = calculate_quantity_discount(
            annual_demand=10000,
            order_cost=50,
            discount_schedule=schedule,
            holding_cost=2.0,
        )
        assert len(res.tiers) == 3
        # Tier 1: clamped to max_qty (500)
        assert res.tiers[0].feasible_quantity == 500.0
        assert not res.tiers[0].is_feasible_at_eoq
        # Tier 2: feasible at unconstrained EOQ
        assert pytest.approx(res.tiers[1].feasible_quantity, rel=1e-4) == 707.1068
        assert res.tiers[1].is_feasible_at_eoq
        # Tier 3: clamped to min_qty (1500)
        assert res.tiers[2].feasible_quantity == 1500.0
        assert not res.tiers[2].is_feasible_at_eoq

        # Verify cost formulas
        # Tier 1 at Q=500: (10000/500)*50 + (500/2)*2 + 10000*10 = 1000 + 500 + 100000 = $101,500.00
        assert pytest.approx(res.tiers[0].total_annual_cost, abs=1e-2) == 101500.00
        # Tier 2 at Q=707.1068: (10000/707.1068)*50 + (707.1068/2)*2 + 10000*9.5 = 707.1068 + 707.1068 + 95000 = $96,414.21
        assert pytest.approx(res.tiers[1].total_annual_cost, abs=1e-2) == 96414.21
        # Tier 3 at Q=1500: (10000/1500)*50 + (1500/2)*2 + 10000*9 = 333.33 + 1500 + 90000 = $91,833.33
        assert pytest.approx(res.tiers[2].total_annual_cost, abs=1e-2) == 91833.33
        # Global minimum is Tier 3
        assert res.optimal_tier_index == 2
        assert res.optimal_quantity == 1500.0
        assert pytest.approx(res.min_total_cost, abs=1e-2) == 91833.33


# =====================================================================
# Model 3 Tests: Safety Stock & Reorder Point (ROP)
# =====================================================================

class TestSafetyStockROP:
    """Test suite for Safety Stock and Reorder Point calculations."""

    def test_variable_demand_constant_lead_time(self):
        """Case 1: d=50 units/day, sigma_d=8, L=5 days, sigma_L=0, CSL=95%."""
        res = calculate_safety_stock_rop(
            avg_demand=50,
            demand_std=8,
            lead_time=5,
            lead_time_std=0.0,
            service_level=0.95,
        )
        # z for 95% = 1.64485
        # Expected LTD = 50 * 5 = 250
        # sigma_LTD = 8 * sqrt(5) ~= 17.8885
        # SS = 1.64485 * 17.8885 ~= 29.424
        # ROP = 250 + 29.424 ~= 279.424
        assert pytest.approx(res.z_score, abs=1e-3) == 1.6449
        assert pytest.approx(res.expected_lead_time_demand, abs=1e-3) == 250.0
        assert pytest.approx(res.std_lead_time_demand, abs=1e-3) == 17.8885
        assert pytest.approx(res.safety_stock, abs=1e-2) == 29.42
        assert pytest.approx(res.reorder_point, abs=1e-2) == 279.42
        assert pytest.approx(res.stockout_probability, abs=1e-4) == 0.05
        assert res.verification.is_valid

    def test_both_demand_and_lead_time_variable(self):
        """Case 2: Both demand and lead time are stochastic:
        d=100, sigma_d=10, L=4, sigma_L=1, CSL=99%
        """
        res = calculate_safety_stock_rop(
            avg_demand=100,
            demand_std=10,
            lead_time=4,
            lead_time_std=1.0,
            service_level=0.99,
        )
        # Expected LTD = 100 * 4 = 400
        # Var(LTD) = 4 * 10^2 + 100^2 * 1^2 = 400 + 10,000 = 10,400
        # sigma_LTD = sqrt(10400) ~= 101.9804
        # z for 99% ~= 2.3263
        # SS = 2.3263 * 101.9804 ~= 237.24
        assert pytest.approx(res.std_lead_time_demand, abs=1e-3) == 101.9804
        assert pytest.approx(res.safety_stock, abs=1e-1) == 237.24
        assert pytest.approx(res.reorder_point, abs=1e-1) == 637.24
        assert res.verification.is_valid

    def test_safety_stock_rop_unit_normalization_days_and_weeks(self):
        """Test unit normalization when demand is in days and lead time is in weeks:
        Demand: 40 units/day, sigma_d = 8 units/day
        Lead time: 3 weeks, sigma_LT = 0.5 weeks
        Default workdays_per_week: 5.0
        Normalized lead time: 3 * 5 = 15 working days
        Normalized lead time std: 0.5 * 5 = 2.5 working days
        Expected LTD = 40 * 15 = 600 units
        Var(LTD) = 15 * 8^2 + 40^2 * 2.5^2 = 960 + 10,000 = 10,960
        sigma_LTD = sqrt(10,960) ~= 104.69
        CSL = 95% -> z ~= 1.6449
        Safety Stock = 1.64485 * 104.6899995 ~= 172.20
        ROP = 600 + 172.20 = 772.20
        """
        res = calculate_safety_stock_rop(
            avg_demand=40,
            demand_std=8,
            lead_time=3,
            lead_time_std=0.5,
            service_level=0.95,
            demand_time_unit="days",
            lead_time_unit="weeks",
            workdays_per_week=5.0,
        )
        assert res.raw_lead_time == 3.0
        assert res.raw_lead_time_std == 0.5
        assert res.lead_time == 15.0
        assert res.lead_time_std == 2.5
        assert res.unit_normalization_applied is True
        assert res.expected_lead_time_demand == 600.0
        assert pytest.approx(res.std_lead_time_demand, abs=0.01) == 104.69
        assert pytest.approx(res.z_score, abs=0.0001) == 1.6449
        assert pytest.approx(res.safety_stock, abs=0.01) == 172.20
        assert pytest.approx(res.reorder_point, abs=0.01) == 772.20
        assert res.verification.is_valid

    def test_service_level_boundary_validation(self):
        """Service level must be between 0 and 1 exclusive."""
        with pytest.raises(ValueError, match="Service level must be between 0 and 1 exclusive"):
            calculate_safety_stock_rop(avg_demand=10, demand_std=2, lead_time=5, service_level=1.0)
        with pytest.raises(ValueError, match="Service level must be between 0 and 1 exclusive"):
            calculate_safety_stock_rop(avg_demand=10, demand_std=2, lead_time=5, service_level=0.0)


# =====================================================================
# Model 4 Tests: Fixed-Period Review Model
# =====================================================================

class TestFixedPeriodReview:
    """Test suite for Fixed-Period Review (Order-Up-To Level)."""

    def test_fixed_period_review_order_up_to(self):
        """Periodic review problem:
        T=14 days, L=7 days -> Protection period = 21 days
        d=30 units/day, sigma_d=6, CSL=98% (z=2.0537)
        Current On-Hand=120, On-Order=50, Backorders=0
        """
        res = calculate_fixed_period_review(
            review_period=14,
            lead_time=7,
            avg_demand=30,
            demand_std=6,
            service_level=0.98,
            current_inventory=120,
            on_order=50,
            backorders=0,
        )
        assert res.protection_period == 21.0
        assert res.expected_demand_protection_period == 630.0  # 30 * 21
        # std = 6 * sqrt(21) ~= 27.4955
        assert pytest.approx(res.std_demand_protection_period, abs=1e-3) == 27.4955
        # SS = 2.0537 * 27.4955 ~= 56.469
        assert pytest.approx(res.safety_stock, abs=1e-2) == 56.47
        # M = 630 + 56.47 = 686.47
        assert pytest.approx(res.order_up_to_level, abs=1e-2) == 686.47
        # IP = 120 + 50 = 170
        assert res.inventory_position == 170.0
        # Q = 686.47 - 170 = 516.47
        assert pytest.approx(res.recommended_order_quantity, abs=1e-2) == 516.47
        assert res.verification.is_valid

    def test_fixed_period_inventory_exceeds_target_yields_zero_order(self):
        """If inventory position exceeds target M, order quantity is clamped to 0."""
        res = calculate_fixed_period_review(
            review_period=7,
            lead_time=3,
            avg_demand=10,
            demand_std=2,
            current_inventory=500,  # huge existing stock
        )
        assert res.recommended_order_quantity == 0.0
        assert any("already exceeds" in w for w in res.verification.warnings)

    def test_fixed_period_review_with_round_up_ceiling(self):
        """Verify round_up parameter applies integer ceiling to order quantity."""
        res_exact = calculate_fixed_period_review(
            review_period=7,
            lead_time=3,
            avg_demand=80,
            demand_std=12,
            service_level=0.95,
            current_inventory=600,
            on_order=150,
            backorders=40,
            round_up=False,
        )
        assert pytest.approx(res_exact.exact_order_quantity, abs=1e-2) == 152.42
        assert pytest.approx(res_exact.recommended_order_quantity, abs=1e-2) == 152.42

        res_ceiling = calculate_fixed_period_review(
            review_period=7,
            lead_time=3,
            avg_demand=80,
            demand_std=12,
            service_level=0.95,
            current_inventory=600,
            on_order=150,
            backorders=40,
            round_up=True,
        )
        assert pytest.approx(res_ceiling.exact_order_quantity, abs=1e-2) == 152.42
        assert res_ceiling.recommended_order_quantity == 153.0
        assert res_ceiling.final_answer["recommended_order_quantity"] == 153.0


# =====================================================================
# Model 5 Tests: Newsvendor Model
# =====================================================================

class TestNewsvendor:
    """Test suite for Newsvendor single-period stochastic model."""

    def test_classic_newsvendor_ski_jackets(self):
        """Classic ski jacket / seasonal fashion benchmark:
        Selling Price P=$250, Cost C=$100, Salvage S=$40.
        Demand Normal(mu=400, sigma=80).
        """
        res = calculate_newsvendor(
            selling_price=250.0,
            unit_cost=100.0,
            salvage_value=40.0,
            demand_mean=400.0,
            demand_std=80.0,
        )
        # Cu = 250 - 100 = 150
        # Co = 100 - 40 = 60
        # CR = 150 / (150 + 60) = 150 / 210 = 0.7142857
        assert res.underage_cost == 150.0
        assert res.overage_cost == 60.0
        assert pytest.approx(res.critical_fractile, abs=1e-4) == 0.7143

        # z* for 0.7142857 ~= 0.5659
        assert pytest.approx(res.z_score, abs=1e-3) == 0.5659

        # Optimal Q* = 400 + 0.5659 * 80 ~= 445.27
        assert pytest.approx(res.optimal_order_quantity, abs=1e-1) == 445.28

        # Expected sales, leftovers, and positive profit
        assert res.expected_sales < 400.0
        assert res.expected_leftover_units > 0.0
        assert res.expected_profit > 50000.0
        assert res.verification.is_valid

    def test_newsvendor_economic_boundaries(self):
        """Price <= Cost must raise ValueError."""
        with pytest.raises(ValueError, match="Selling price .* must be greater than unit cost"):
            calculate_newsvendor(selling_price=50, unit_cost=50, salvage_value=20, demand_mean=100, demand_std=15)

        with pytest.raises(ValueError, match="Unit cost .* cannot be less than salvage value"):
            calculate_newsvendor(selling_price=100, unit_cost=30, salvage_value=40, demand_mean=100, demand_std=15)


# =====================================================================
# End-to-End Agent Pipeline Tests
# =====================================================================

class TestInventoryDecisionAgent:
    """Test suite for the 6-stage intelligent decision agent."""

    def setup_method(self):
        self.agent = InventoryDecisionAgent(use_llm_if_available=False)

    def test_agent_eoq_query(self):
        query = "We sell 10,000 units per year. Each order costs $50, and our holding cost is $2 per unit per year. What is our optimal order quantity?"
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.EOQ
        assert resp.selected_tool_name == "calculate_eoq"
        assert resp.calculation_result is not None
        assert pytest.approx(resp.calculation_result.eoq, rel=1e-3) == 707.11
        assert "707" in resp.explanation
        assert resp.verification_result.is_valid
        assert len(resp.steps_log) == 6

    def test_agent_quantity_discount_query(self):
        query = (
            "Annual demand is 1,000 units, order cost is $100, holding cost rate is 20%. "
            "Supplier discount schedule: 1-499 units for $10, 500-999 units for $9, and 1000+ units for $8. "
            "What quantity should we order?"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.QUANTITY_DISCOUNT
        assert resp.selected_tool_name == "calculate_quantity_discount"
        assert resp.calculation_result.optimal_quantity == 1000.0
        assert resp.calculation_result.optimal_unit_price == 8.0
        assert "1,000" in resp.explanation or "1000" in resp.explanation
        assert resp.verification_result.is_valid

    def test_agent_safety_stock_rop_query(self):
        query = "Average daily demand is 50 units with standard deviation of 8. Replenishment lead time is 5 days. Target service level is 95%. Calculate safety stock and reorder point."
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.SAFETY_STOCK_ROP
        assert resp.selected_tool_name == "calculate_safety_stock_rop"
        assert pytest.approx(resp.calculation_result.reorder_point, abs=1e-1) == 279.4
        assert pytest.approx(resp.calculation_result.safety_stock, abs=1e-1) == 29.4
        assert "Reorder Point" in resp.explanation
        assert resp.verification_result.is_valid

    def test_agent_safety_stock_rop_unit_normalization_days_and_weeks(self):
        query = (
            "A product has average demand of 40 units per day with a standard deviation of 8 units per day. "
            "Average lead time is 3 weeks with a standard deviation of 0.5 weeks. "
            "Determine the safety stock and reorder point for a 95% service level."
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.SAFETY_STOCK_ROP
        assert resp.selected_tool_name == "calculate_safety_stock_rop"
        res = resp.calculation_result
        assert res.lead_time == 15.0
        assert res.lead_time_std == 2.5
        assert res.expected_lead_time_demand == 600.0
        assert pytest.approx(res.std_lead_time_demand, abs=0.01) == 104.69
        assert pytest.approx(res.z_score, abs=0.0001) == 1.6449
        assert pytest.approx(res.safety_stock, abs=0.01) == 172.20
        assert pytest.approx(res.reorder_point, abs=0.01) == 772.20
        assert res.unit_normalization_applied is True
        assert res.verification.is_valid
        assert "172.20" in resp.explanation
        assert "772.20" in resp.explanation
        assert "Unit Normalization Applied" in resp.explanation

    def test_agent_fixed_period_review_query(self):
        query = (
            "We operate a periodic review system with a review period of 14 days and lead time of 7 days. "
            "Daily demand averages 30 units with std of 6. Desired service level is 98%. "
            "Currently we have 120 units in stock and 50 units on order. How much should we order?"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.FIXED_PERIOD_REVIEW
        assert resp.selected_tool_name == "calculate_fixed_period_review"
        assert pytest.approx(resp.calculation_result.order_up_to_level, abs=1e-1) == 686.5
        assert pytest.approx(resp.calculation_result.recommended_order_quantity, abs=1e-1) == 516.5
        assert resp.verification_result.is_valid

    def test_agent_newsvendor_query(self):
        query = (
            "We sell ski jackets for $250 retail. We buy them from the manufacturer for $100. "
            "Any unsold jackets at end of season are salvaged to an outlet for $40. "
            "Demand is normally distributed with mean 400 and standard deviation 80. "
            "What is our optimal newsvendor order quantity?"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.NEWSVENDOR
        assert resp.selected_tool_name == "calculate_newsvendor"
        assert pytest.approx(resp.calculation_result.optimal_order_quantity, abs=1e-1) == 445.3
        assert "445" in resp.explanation
        assert resp.verification_result.is_valid

    def test_agent_missing_inputs_triggers_clarification(self):
        query = "Calculate our EOQ if annual demand is 10,000 units."
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.EOQ
        assert resp.needs_clarification is True
        assert "order_cost" in resp.missing_inputs[0] or "holding_cost" in resp.missing_inputs[1]
        assert "Clarification" in resp.steps_log[-1].step_name
        assert "Missing Parameters Required" in resp.explanation

    def test_agent_unrecognized_query_guidance(self):
        query = "What is the best way to design a warehouse layout?"
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.UNKNOWN
        assert resp.needs_clarification is True
        assert "5 core inventory optimization models" in resp.explanation


# =====================================================================
# Deterministic Tools Contract Tests (Explicit Functional Requirements)
# =====================================================================

class TestDeterministicToolsContract:
    """Rigorous contract tests verifying all 6 requirements:
    1. Validate inputs
    2. Return structured results
    3. Return intermediate calculations
    4. Return the final answer
    5. Identify assumptions
    6. Never invent missing numerical inputs
    """

    # --- Tool 1: calculate_eoq() ---
    def test_calculate_eoq_contract_and_structure(self):
        """Verify calculate_eoq contract, intermediate calculations, final answer, and assumptions."""
        res = calculate_eoq(annual_demand=12000, order_cost=60, holding_cost=2.5)

        # 1. Structured result
        assert isinstance(res, EOQResult)
        assert isinstance(res.to_dict(), dict)

        # 2. Final answer
        expected_eoq = math.sqrt(2 * 12000 * 60 / 2.5)  # sqrt(576,000) = 758.9466
        assert pytest.approx(res.final_answer, rel=1e-4) == expected_eoq
        assert res.final_answer == res.eoq

        # 3. Intermediate calculations returned
        assert "annual_ordering_cost" in res.intermediate_calculations
        assert "annual_holding_cost" in res.intermediate_calculations
        assert "total_annual_policy_cost" in res.intermediate_calculations
        assert "orders_per_year" in res.intermediate_calculations
        assert "cycle_time_days" in res.intermediate_calculations
        assert pytest.approx(res.intermediate_calculations["annual_ordering_cost"], rel=1e-3) == pytest.approx(res.intermediate_calculations["annual_holding_cost"], rel=1e-3)

        # 4. Explicit assumptions identified
        assert isinstance(res.assumptions, list)
        assert len(res.assumptions) >= 5
        assert any("constant" in a.lower() for a in res.assumptions)
        assert any("holding cost" in a.lower() for a in res.assumptions)

    def test_calculate_eoq_never_invents_missing_numerical_inputs(self):
        """Verify calculate_eoq never invents missing inputs."""
        with pytest.raises(ValueError, match="Missing required input 'annual_demand'"):
            calculate_eoq(annual_demand=None, order_cost=50, holding_cost=2)

        with pytest.raises(ValueError, match="Missing required input 'order_cost'"):
            calculate_eoq(annual_demand=1000, order_cost=None, holding_cost=2)

        with pytest.raises(ValueError, match="Must provide either a positive holding_cost"):
            calculate_eoq(annual_demand=1000, order_cost=50, holding_cost=None, holding_cost_rate=None)

    # --- Tool 2: calculate_quantity_discount() ---
    def test_calculate_quantity_discount_fixed_holding_cost(self):
        """Verify quantity discount with fixed holding cost across tiers."""
        schedule = [
            {"min_qty": 1, "max_qty": 499, "unit_price": 20.0},
            {"min_qty": 500, "max_qty": None, "unit_price": 18.0},
        ]
        res = calculate_quantity_discount(
            annual_demand=1000,
            order_cost=50,
            discount_schedule=schedule,
            holding_cost=5.0,  # Fixed holding cost
        )

        assert isinstance(res, QuantityDiscountResult)
        assert res.final_answer["optimal_quantity"] == 500.0
        assert res.final_answer["optimal_unit_price"] == 18.0
        assert "tier_evaluations" in res.intermediate_calculations
        assert res.intermediate_calculations["holding_cost_type"] == "fixed"
        assert len(res.assumptions) >= 4
        assert any("all-units" in a.lower() for a in res.assumptions)

    def test_calculate_quantity_discount_percentage_holding_cost(self):
        """Verify quantity discount with percentage holding cost (h * C_i)."""
        schedule = [
            {"min_qty": 1, "max_qty": 499, "unit_price": 10.0},
            {"min_qty": 500, "max_qty": 999, "unit_price": 9.0},
            {"min_qty": 1000, "max_qty": None, "unit_price": 8.0},
        ]
        res = calculate_quantity_discount(
            annual_demand=1000,
            order_cost=100,
            discount_schedule=schedule,
            holding_cost_rate=0.20,  # 20% holding cost rate
        )

        assert res.final_answer["optimal_quantity"] == 1000.0
        assert res.final_answer["optimal_unit_price"] == 8.0
        assert res.intermediate_calculations["holding_cost_type"] == "percentage"
        # Check tier holding costs: Tier 1: 0.2*10=2.0, Tier 2: 0.2*9=1.8, Tier 3: 0.2*8=1.6
        assert res.tiers[0].holding_cost == 2.0
        assert res.tiers[1].holding_cost == 1.8
        assert res.tiers[2].holding_cost == 1.6

    def test_calculate_quantity_discount_never_invents_missing_numerical_inputs(self):
        """Verify quantity discount rejects missing or invalid inputs."""
        schedule = [{"min_qty": 1, "unit_price": 10.0}]
        with pytest.raises(ValueError, match="Missing required input 'annual_demand'"):
            calculate_quantity_discount(annual_demand=None, order_cost=50, discount_schedule=schedule, holding_cost=2)

        with pytest.raises(ValueError, match="discount_schedule must be a non-empty list"):
            calculate_quantity_discount(annual_demand=1000, order_cost=50, discount_schedule=[], holding_cost=2)

        with pytest.raises(ValueError, match="Must provide either a fixed holding_cost"):
            calculate_quantity_discount(annual_demand=1000, order_cost=50, discount_schedule=schedule)

    # --- Tool 3: calculate_rop() supporting all 4 cases ---
    def test_calculate_rop_case1_constant_demand_constant_lead_time(self):
        """Case 1: Constant demand and constant lead time (sigma_d=0, sigma_L=0)."""
        res = calculate_rop(avg_demand=25.0, lead_time=6.0, demand_std=0.0, lead_time_std=0.0)

        assert isinstance(res, ROPResult)
        assert res.case == "constant_demand_constant_lead_time"
        assert res.final_answer["safety_stock"] == 0.0
        assert res.final_answer["reorder_point"] == 150.0  # 25 * 6
        assert res.safety_stock == 0.0
        assert res.reorder_point == 150.0
        assert res.std_lead_time_demand == 0.0
        assert len(res.assumptions) >= 5

    def test_calculate_rop_case2_variable_demand_constant_lead_time(self):
        """Case 2: Variable demand and constant lead time (sigma_d>0, sigma_L=0)."""
        res = calculate_rop(avg_demand=50.0, lead_time=9.0, demand_std=6.0, lead_time_std=0.0, service_level=0.95)

        assert res.case == "variable_demand_constant_lead_time"
        # sigma_LT = sigma_d * sqrt(L) = 6 * 3 = 18.0
        assert res.intermediate_calculations["sigma_LT"] == 18.0
        expected_z = 1.6448536
        expected_ss = expected_z * 18.0
        assert pytest.approx(res.safety_stock, abs=1e-2) == expected_ss
        assert pytest.approx(res.reorder_point, abs=1e-2) == (50 * 9 + expected_ss)

    def test_calculate_rop_case3_constant_demand_variable_lead_time(self):
        """Case 3: Constant demand and variable lead time (sigma_d=0, sigma_L>0)."""
        res = calculate_rop(avg_demand=30.0, lead_time=5.0, demand_std=0.0, lead_time_std=2.0, service_level=0.95)

        assert res.case == "constant_demand_variable_lead_time"
        # sigma_LT = d * sigma_L = 30 * 2 = 60.0
        assert res.intermediate_calculations["sigma_LT"] == 60.0
        expected_z = 1.6448536
        expected_ss = expected_z * 60.0
        assert pytest.approx(res.safety_stock, abs=1e-2) == expected_ss
        assert pytest.approx(res.reorder_point, abs=1e-2) == (30 * 5 + expected_ss)

    def test_calculate_rop_case4_variable_demand_variable_lead_time(self):
        """Case 4: Variable demand and variable lead time (sigma_d>0, sigma_L>0).
        Formula: sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        SS = z * sigma_LT
        ROP = d * L + SS
        """
        d = 40.0
        L = 4.0
        sigma_d = 5.0
        sigma_L = 1.5
        csl = 0.98

        res = calculate_rop(
            avg_demand=d,
            lead_time=L,
            demand_std=sigma_d,
            lead_time_std=sigma_L,
            service_level=csl,
        )

        assert res.case == "variable_demand_variable_lead_time"
        expected_var = L * (sigma_d ** 2) + (d ** 2) * (sigma_L ** 2)  # 4*25 + 1600*2.25 = 100 + 3600 = 3700
        expected_sigma_lt = math.sqrt(expected_var)  # sqrt(3700) ~= 60.8276

        assert pytest.approx(res.intermediate_calculations["lead_time_demand_variance"], abs=1e-3) == 3700.0
        assert pytest.approx(res.intermediate_calculations["sigma_LT"], abs=1e-3) == expected_sigma_lt

        # z for 0.98
        from statistics import NormalDist
        z = NormalDist().inv_cdf(0.98)
        expected_ss = z * expected_sigma_lt
        expected_rop = d * L + expected_ss

        assert pytest.approx(res.safety_stock, abs=1e-3) == expected_ss
        assert pytest.approx(res.reorder_point, abs=1e-3) == expected_rop
        assert res.final_answer["reorder_point"] == res.reorder_point
        assert res.final_answer["safety_stock"] == res.safety_stock

    def test_calculate_rop_never_invents_missing_numerical_inputs(self):
        """Verify calculate_rop rejects missing inputs."""
        with pytest.raises(ValueError, match="Missing required input 'avg_demand'"):
            calculate_rop(avg_demand=None, lead_time=5)

        with pytest.raises(ValueError, match="Missing required input 'lead_time'"):
            calculate_rop(avg_demand=10, lead_time=None)

        with pytest.raises(ValueError, match="Missing required input 'service_level'"):
            calculate_rop(avg_demand=10, lead_time=5, service_level=None)

    # --- Tool 4: calculate_periodic_review() ---
    def test_calculate_periodic_review_contract(self):
        """Verify protection period, target inventory, order quantity, intermediate calcs, and assumptions."""
        # Protection period = review period + lead time = 10 + 5 = 15
        # Target inventory = demand during protection period + safety stock
        # Order quantity = max(0, target inventory - inventory position)
        res = calculate_periodic_review(
            review_period=10,
            lead_time=5,
            avg_demand=20,
            demand_std=4,
            service_level=0.95,
            current_inventory=80,
            on_order=40,
            backorders=10,
        )

        assert isinstance(res, PeriodicReviewResult)
        # Protection period = 10 + 5 = 15
        assert res.protection_period == 15.0
        assert res.intermediate_calculations["protection_period"] == 15.0

        # Demand during protection period = 20 * 15 = 300
        assert res.intermediate_calculations["expected_demand_protection_period"] == 300.0

        # Std dev = 4 * sqrt(15) ~= 15.4919
        assert pytest.approx(res.std_demand_protection_period, abs=1e-3) == 4 * math.sqrt(15)

        # Safety stock = z * 15.4919 ~= 1.64485 * 15.4919 = 25.482
        assert pytest.approx(res.safety_stock, abs=1e-2) == 25.48

        # Target inventory = 300 + 25.48 = 325.48
        assert pytest.approx(res.target_inventory, abs=1e-2) == 325.48
        assert res.order_up_to_level == res.target_inventory

        # Inventory position = 80 + 40 - 10 = 110
        assert res.inventory_position == 110.0

        # Order quantity = 325.48 - 110 = 215.48
        assert pytest.approx(res.recommended_order_quantity, abs=1e-2) == 215.48
        assert res.final_answer["recommended_order_quantity"] == res.recommended_order_quantity
        assert res.final_answer["target_inventory"] == res.target_inventory

        # Assumptions
        assert len(res.assumptions) >= 5
        assert any("protection period" in a.lower() for a in res.assumptions)

    def test_calculate_periodic_review_clamped_order_quantity(self):
        """Verify order quantity is clamped to 0 if inventory position exceeds target."""
        res = calculate_periodic_review(
            review_period=7,
            lead_time=3,
            avg_demand=10,
            demand_std=2,
            service_level=0.95,
            current_inventory=500,  # Surplus stock
            on_order=0,
            backorders=0,
        )
        assert res.recommended_order_quantity == 0.0
        assert res.intermediate_calculations["unclamped_order_quantity"] < 0.0

    def test_calculate_periodic_review_never_invents_missing_numerical_inputs(self):
        """Verify periodic review strictly requires all inputs."""
        with pytest.raises(ValueError, match="Missing required input 'review_period'"):
            calculate_periodic_review(review_period=None, lead_time=5, avg_demand=20)

        with pytest.raises(ValueError, match="Missing required input 'lead_time'"):
            calculate_periodic_review(review_period=7, lead_time=None, avg_demand=20)

    # --- Tool 5: calculate_newsvendor() ---
    def test_calculate_newsvendor_contract(self):
        """Verify underage cost, overage cost, critical ratio, and recommended order quantity."""
        res = calculate_newsvendor(
            selling_price=120.0,
            unit_cost=50.0,
            salvage_value=20.0,
            demand_mean=300.0,
            demand_std=60.0,
            customer_penalty=10.0,
        )

        assert isinstance(res, NewsvendorResult)

        # Underage cost Cu = P - C + penalty = 120 - 50 + 10 = 80.0
        assert res.underage_cost == 80.0
        assert res.intermediate_calculations["underage_cost"] == 80.0

        # Overage cost Co = C - S = 50 - 20 = 30.0
        assert res.overage_cost == 30.0
        assert res.intermediate_calculations["overage_cost"] == 30.0

        # Critical ratio CR = Cu / (Cu + Co) = 80 / (80 + 30) = 80 / 110 ~= 0.72727
        assert pytest.approx(res.critical_ratio, abs=1e-4) == 80.0 / 110.0
        assert res.critical_fractile == res.critical_ratio

        # z-score and recommended order quantity
        from statistics import NormalDist
        expected_z = NormalDist().inv_cdf(80.0 / 110.0)
        expected_q = 300.0 + (expected_z * 60.0)

        assert pytest.approx(res.z_score, abs=1e-4) == expected_z
        assert pytest.approx(res.optimal_order_quantity, abs=1e-2) == expected_q
        assert res.final_answer == res.optimal_order_quantity

        # Intermediate expected financials
        assert "expected_sales" in res.intermediate_calculations
        assert "expected_leftover_units" in res.intermediate_calculations
        assert "expected_profit" in res.intermediate_calculations
        assert res.expected_profit > 0.0

        # Assumptions
        assert len(res.assumptions) >= 5
        assert any("single-period" in a.lower() for a in res.assumptions)
        assert any("critical ratio" in a.lower() for a in res.assumptions)

    def test_calculate_newsvendor_never_invents_missing_numerical_inputs(self):
        """Verify newsvendor rejects missing inputs or invalid economics."""
        with pytest.raises(ValueError, match="Missing required input 'selling_price'"):
            calculate_newsvendor(selling_price=None, unit_cost=50, salvage_value=20, demand_mean=300, demand_std=60)

        with pytest.raises(ValueError, match="Missing required input 'salvage_value'"):
            calculate_newsvendor(selling_price=100, unit_cost=50, salvage_value=None, demand_mean=300, demand_std=60)

        with pytest.raises(ValueError, match="Selling price .* must be greater than unit cost"):
            calculate_newsvendor(selling_price=40, unit_cost=50, salvage_value=20, demand_mean=300, demand_std=60)

        with pytest.raises(ValueError, match="Unit cost .* cannot be less than salvage value"):
            calculate_newsvendor(selling_price=100, unit_cost=30, salvage_value=40, demand_mean=300, demand_std=60)


# =====================================================================
# Enhanced Natural-Language Input Extraction and Disambiguation Tests
# =====================================================================

class TestNaturalLanguageInputExtraction:
    """Rigorous tests for input extraction, terminology recognition, and disambiguation."""

    def setup_method(self):
        self.agent = InventoryDecisionAgent()

    # --- EOQ: Ordering Cost Synonyms ---
    @pytest.mark.parametrize(
        "phrase",
        [
            "ordering cost of $50",
            "order cost of $50",
            "setup cost of $50",
            "setup expense of $50",
            "cost per order of $50",
            "cost of placing an order is $50",
            "ordering/setup cost of $50",
        ],
    )
    def test_eoq_ordering_cost_synonyms(self, phrase):
        """All 7 specified ordering cost phrases must be mapped to order_cost (S)."""
        query = f"Our annual demand is 10,000 units, holding cost is $2 per unit, and {phrase}. What is the EOQ?"
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.EOQ
        assert resp.needs_clarification is False
        assert resp.extracted_inputs["order_cost"] == 50.0
        assert resp.extracted_inputs["annual_demand"] == 10000.0
        assert resp.extracted_inputs["holding_cost"] == 2.0
        assert pytest.approx(resp.calculation_result.eoq, rel=1e-3) == 707.107

    # --- Safety Stock & ROP: Disambiguation ---
    def test_safety_stock_rop_disambiguation(self):
        """Strictly distinguish average demand, demand std, average lead time, lead-time std, service level."""
        query = (
            "We have an average demand of 40 units per day with a demand standard deviation of 8 units. "
            "Our average lead time is 6 days with a lead-time standard deviation of 1.5 days. "
            "Our target cycle service level is 98%."
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.SAFETY_STOCK_ROP
        assert resp.needs_clarification is False
        inputs = resp.extracted_inputs
        assert inputs["avg_demand"] == 40.0
        assert inputs["demand_std"] == 8.0
        assert inputs["lead_time"] == 6.0
        assert inputs["lead_time_std"] == 1.5
        assert inputs["service_level"] == 0.98

        res = resp.calculation_result
        assert res.case in ("variable_demand_variable_lead_time", "Case 4: Variable Demand, Variable Lead Time")
        expected_sigma_lt = math.sqrt(6.0 * (8.0**2) + (40.0**2) * (1.5**2))
        assert pytest.approx(res.std_lead_time_demand, abs=1e-2) == expected_sigma_lt
        assert pytest.approx(res.safety_stock, abs=1e-2) == res.z_score * expected_sigma_lt
        assert pytest.approx(res.reorder_point, abs=1e-2) == (40.0 * 6.0) + res.safety_stock

    # --- Quantity Discount: Disambiguation ---
    def test_quantity_discount_percentage_holding_cost(self):
        """Quantity discount with percentage holding cost and 3 discount tiers."""
        query = (
            "Annual demand is 5,000 units. Ordering cost is $40 per order. "
            "Percentage holding cost is 20%. "
            "Discount schedule: 1-499 units for $25, 500-999 units for $22, 1000+ units for $19. "
            "What quantity should we order?"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.QUANTITY_DISCOUNT
        assert resp.needs_clarification is False
        assert resp.extracted_inputs["annual_demand"] == 5000.0
        assert resp.extracted_inputs["order_cost"] == 40.0
        assert resp.extracted_inputs["holding_cost_rate"] == 0.20
        assert len(resp.extracted_inputs["discount_schedule"]) == 3
        assert resp.calculation_result.optimal_quantity >= 1000.0

    def test_quantity_discount_fixed_holding_cost_and_breaks(self):
        """Quantity discount with fixed holding cost and quantity breaks."""
        query = (
            "Annual demand is 2,400 units, setup expense is $50. Fixed holding cost is $3.50 per unit. "
            "Quantity break at 500 units gives unit price of $18, quantity break at 1000 units gives unit price of $16, "
            "and baseline unit purchase price is $20. What is the optimal order quantity?"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.QUANTITY_DISCOUNT
        assert resp.needs_clarification is False
        assert resp.extracted_inputs["annual_demand"] == 2400.0
        assert resp.extracted_inputs["order_cost"] == 50.0
        assert resp.extracted_inputs["holding_cost"] == 3.50
        assert len(resp.extracted_inputs["discount_schedule"]) == 3
        assert resp.calculation_result is not None

    def test_quantity_discount_three_tiers_extraction_and_global_optimization(self):
        """User benchmark inquiry with 3 tiers:
        - Annual demand: 10,000 units
        - Ordering cost: $50
        - Holding cost rate: 20%
        - Tiers: 1-49 units for $9.00, 50-99 units for $8.50, 100+ units for $8.00
        Verify extraction of all 3 tiers, evaluation of all 3 brackets, and selection of Tier 3.
        """
        query = (
            "Our annual demand is 10,000 units and ordering cost is $50. Our holding cost rate is 20%. "
            "The supplier offers quantity discounts: 1-49 units for $9.00, 50-99 units for $8.50, and 100+ units for $8.00. "
            "What quantity should we order?"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.QUANTITY_DISCOUNT
        assert resp.needs_clarification is False
        assert resp.extracted_inputs["annual_demand"] == 10000.0
        assert resp.extracted_inputs["order_cost"] == 50.0
        assert resp.extracted_inputs["holding_cost_rate"] == 0.20
        assert len(resp.extracted_inputs["discount_schedule"]) == 3

        res = resp.calculation_result
        assert res is not None
        assert len(res.tiers) == 3

        # Verify Tier 1 calculations (the single tier previously observed)
        t1 = res.tiers[0]
        assert t1.min_quantity == 1.0
        assert t1.max_quantity == 49.0
        assert t1.unit_price == 9.00
        assert pytest.approx(t1.unconstrained_eoq, abs=1e-1) == 745.4
        assert t1.feasible_quantity == 49.0
        assert pytest.approx(t1.annual_ordering_cost, abs=1e-2) == 10204.08
        assert pytest.approx(t1.annual_holding_cost, abs=1e-2) == 44.10
        assert pytest.approx(t1.annual_purchase_cost, abs=1e-2) == 90000.00
        assert pytest.approx(t1.total_annual_cost, abs=1e-2) == 100248.18

        # Verify Tier 2 calculations
        t2 = res.tiers[1]
        assert t2.min_quantity == 50.0
        assert t2.max_quantity == 99.0
        assert t2.unit_price == 8.50
        assert pytest.approx(t2.unconstrained_eoq, abs=1e-1) == 767.0
        assert t2.feasible_quantity == 99.0
        assert pytest.approx(t2.annual_ordering_cost, abs=1e-2) == 5050.51
        assert pytest.approx(t2.annual_holding_cost, abs=1e-2) == 84.15
        assert pytest.approx(t2.annual_purchase_cost, abs=1e-2) == 85000.00
        assert pytest.approx(t2.total_annual_cost, abs=1e-2) == 90134.66

        # Verify Tier 3 calculations (global optimum)
        t3 = res.tiers[2]
        assert t3.min_quantity == 100.0
        assert t3.max_quantity is None
        assert t3.unit_price == 8.00
        assert pytest.approx(t3.unconstrained_eoq, abs=1e-1) == 790.6
        assert pytest.approx(t3.feasible_quantity, abs=1e-1) == 790.6
        assert pytest.approx(t3.annual_ordering_cost, abs=1e-2) == 632.46
        assert pytest.approx(t3.annual_holding_cost, abs=1e-2) == 632.46
        assert pytest.approx(t3.annual_purchase_cost, abs=1e-2) == 80000.00
        assert pytest.approx(t3.total_annual_cost, abs=1e-2) == 81264.91

        # Global optimal decision
        assert res.optimal_tier_index == 2
        assert res.optimal_unit_price == 8.00
        assert pytest.approx(res.optimal_quantity, abs=1e-1) == 790.6
        assert pytest.approx(res.min_total_cost, abs=1e-2) == 81264.91
        assert pytest.approx(res.annual_savings, abs=1e-2) == 18983.27

        # Check explanation mentions all tiers and evaluated count
        assert "Evaluated 3 pricing brackets" in resp.explanation
        assert "Tier 1" in resp.explanation
        assert "Tier 2" in resp.explanation
        assert "Tier 3" in resp.explanation
        assert "$81,264.91" in resp.explanation

    def test_quantity_discount_phrasing_variations_extract_all_three_tiers(self):
        """Ensure various natural language phrasings of tiers are robustly extracted."""
        phrasings = [
            # Parenthesized prices
            "Tier 1: 1-49 units ($9.00), Tier 2: 50-99 units ($8.50), Tier 3: 100+ units ($8.00)",
            # "units or more" wording
            "1-49 units: $9.00, 50-99 units: $8.50, 100 units or more: $8.00",
            # "under 50" wording
            "under 50 units: $9.00, 50 to 99 units: $8.50, 100 or more units: $8.00",
            # Concise notation
            "1-49: $9.00, 50-99: $8.50, 100+: $8.00",
            # Unit suffixes and colons
            "1 - 49 units: $9.00 per unit, 50 - 99 units: $8.50 per unit, 100+ units: $8.00 per unit",
        ]
        for p in phrasings:
            query = f"Annual demand is 10,000 units, ordering cost is $50, holding cost rate is 20%. Quantity discount schedule: {p}."
            resp = self.agent.run(query)
            assert resp.identified_model == InventoryModelType.QUANTITY_DISCOUNT
            sched = resp.extracted_inputs.get("discount_schedule", [])
            assert len(sched) == 3, f"Failed on phrasing: {p}, got: {sched}"
            assert sched[0]["min_qty"] == 1.0 and sched[0]["max_qty"] == 49.0 and sched[0]["unit_price"] == 9.0
            assert sched[1]["min_qty"] == 50.0 and sched[1]["max_qty"] == 99.0 and sched[1]["unit_price"] == 8.5
            assert sched[2]["min_qty"] == 100.0 and sched[2]["max_qty"] is None and sched[2]["unit_price"] == 8.0
            assert len(resp.calculation_result.tiers) == 3

    def test_quantity_discount_user_regression_inequality_tiers(self):
        """User regression problem with inequality boundary tiers:
        - Annual demand: 5,000 units
        - Ordering cost: $60/order
        - Holding cost rate: 20%
        - Tier 1: Q < 500, price = $12.00/unit
        - Tier 2: 500 <= Q < 1,000, price = $11.50/unit
        - Tier 3: Q >= 1,000, price = $11.00/unit

        Required logic validation:
        1. Calculate EOQ for each tier using unit price and holding cost.
        2. Verify feasibility within exact tier quantity boundaries.
        3. If EOQ < tier minimum, evaluate at tier minimum quantity.
        4. If EOQ > tier maximum, evaluate at tier maximum quantity.
        5. Never apply a tier's price to an order quantity that does not qualify for that tier (e.g. $11.00 CANNOT apply to 522 units).
        6. Compute ordering, holding, and purchase costs using evaluated quantity.
        7. Global optimum is Tier 3 at Q = 1,000 units with Total Cost = $56,400.00.
        8. Actual tier boundaries match evaluated quantities.
        9. Comparison table includes candidate quantities and total costs.
        """
        query = (
            "Annual demand: 5,000 units\n"
            "Ordering cost: $60/order\n"
            "Holding cost rate: 20%\n"
            "Tier 1: Q < 500, price = $12.00/unit\n"
            "Tier 2: 500 <= Q < 1,000, price = $11.50/unit\n"
            "Tier 3: Q >= 1,000, price = $11.00/unit"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.QUANTITY_DISCOUNT
        assert resp.needs_clarification is False
        assert resp.extracted_inputs["annual_demand"] == 5000.0
        assert resp.extracted_inputs["order_cost"] == 60.0
        assert resp.extracted_inputs["holding_cost_rate"] == 0.20

        sched = resp.extracted_inputs.get("discount_schedule", [])
        assert len(sched) == 3
        assert sched[0]["min_qty"] == 1.0 and sched[0]["max_qty"] == 499.0 and sched[0]["unit_price"] == 12.00
        assert sched[1]["min_qty"] == 500.0 and sched[1]["max_qty"] == 999.0 and sched[1]["unit_price"] == 11.50
        assert sched[2]["min_qty"] == 1000.0 and sched[2]["max_qty"] is None and sched[2]["unit_price"] == 11.00

        res = resp.calculation_result
        assert res is not None
        assert len(res.tiers) == 3

        # Tier 1: Q < 500 -> evaluated at upper bound 499 units
        t1 = res.tiers[0]
        assert t1.min_quantity == 1.0
        assert t1.max_quantity == 499.0
        assert t1.unit_price == 12.00
        assert pytest.approx(t1.unconstrained_eoq, abs=1e-2) == 500.00
        assert t1.feasible_quantity == 499.0
        assert not t1.is_feasible_at_eoq
        assert pytest.approx(t1.annual_ordering_cost, abs=1e-2) == 601.20
        assert pytest.approx(t1.annual_holding_cost, abs=1e-2) == 598.80
        assert pytest.approx(t1.annual_purchase_cost, abs=1e-2) == 60000.00
        assert pytest.approx(t1.total_annual_cost, abs=1e-2) == 61200.00

        # Tier 2: 500 <= Q < 1,000 -> EOQ = 510.75 is feasible within [500, 999]
        t2 = res.tiers[1]
        assert t2.min_quantity == 500.0
        assert t2.max_quantity == 999.0
        assert t2.unit_price == 11.50
        assert pytest.approx(t2.unconstrained_eoq, abs=1e-2) == 510.75
        assert pytest.approx(t2.feasible_quantity, abs=1e-2) == 510.75
        assert t2.is_feasible_at_eoq
        assert pytest.approx(t2.annual_ordering_cost, abs=1e-2) == 587.37
        assert pytest.approx(t2.annual_holding_cost, abs=1e-2) == 587.37
        assert pytest.approx(t2.annual_purchase_cost, abs=1e-2) == 57500.00
        assert pytest.approx(t2.total_annual_cost, abs=1e-2) == 58674.73

        # Tier 3: Q >= 1,000 -> EOQ = 522.23 is BELOW 1,000; clamped to tier minimum 1,000 units
        t3 = res.tiers[2]
        assert t3.min_quantity == 1000.0
        assert t3.max_quantity is None
        assert t3.unit_price == 11.00
        assert pytest.approx(t3.unconstrained_eoq, abs=1e-2) == 522.23
        assert t3.feasible_quantity == 1000.0
        assert not t3.is_feasible_at_eoq
        assert pytest.approx(t3.annual_ordering_cost, abs=1e-2) == 300.00
        assert pytest.approx(t3.annual_holding_cost, abs=1e-2) == 1100.00
        assert pytest.approx(t3.annual_purchase_cost, abs=1e-2) == 55000.00
        assert pytest.approx(t3.total_annual_cost, abs=1e-2) == 56400.00

        # Global optimal decision
        assert res.optimal_tier_index == 2
        assert res.optimal_quantity == 1000.0
        assert res.optimal_unit_price == 11.00
        assert pytest.approx(res.annual_purchase_cost, abs=1e-2) == 55000.00
        assert pytest.approx(res.annual_ordering_cost, abs=1e-2) == 300.00
        assert pytest.approx(res.annual_holding_cost, abs=1e-2) == 1100.00
        assert pytest.approx(res.total_annual_policy_cost, abs=1e-2) == 1400.00
        assert pytest.approx(res.min_total_cost, abs=1e-2) == 56400.00
        assert pytest.approx(res.annual_savings, abs=1e-2) == 4800.00

        # Verification audit
        assert res.verification.is_valid
        assert len(res.verification.errors) == 0
        assert any("strictly qualifies for Tier 3" in c for c in res.verification.checks_passed)
        assert any("clamped to boundary quantity" in c for c in res.verification.checks_passed)

        # Explanation audit: must never recommend 522 units at $11.00
        assert "1,000 units" in resp.explanation
        assert "$55,000.00" in resp.explanation
        assert "$300.00" in resp.explanation
        assert "$1,100.00" in resp.explanation
        assert "$1,400.00" in resp.explanation
        assert "$56,400.00" in resp.explanation
        assert "$4,800.00" in resp.explanation
        assert "510.75" in resp.explanation
        assert "499" in resp.explanation
        assert "$61,200.00" in resp.explanation
        assert "$58,674.73" in resp.explanation
        assert "522" not in resp.explanation or "falls below minimum qualifying threshold" in resp.explanation

    def test_quantity_discount_direct_tool_regression(self):
        """Verify calculate_quantity_discount tool directly with the regression parameters."""
        schedule = [
            {"min_qty": 1.0, "max_qty": 499.0, "unit_price": 12.00},
            {"min_qty": 500.0, "max_qty": 999.0, "unit_price": 11.50},
            {"min_qty": 1000.0, "max_qty": None, "unit_price": 11.00},
        ]
        res = calculate_quantity_discount(
            annual_demand=5000,
            order_cost=60,
            holding_cost_rate=0.20,
            discount_schedule=schedule,
        )

        assert res.optimal_quantity == 1000.0
        assert res.optimal_unit_price == 11.00
        assert res.optimal_tier_index == 2
        assert pytest.approx(res.annual_purchase_cost, abs=1e-2) == 55000.00
        assert pytest.approx(res.annual_ordering_cost, abs=1e-2) == 300.00
        assert pytest.approx(res.annual_holding_cost, abs=1e-2) == 1100.00
        assert pytest.approx(res.total_annual_policy_cost, abs=1e-2) == 1400.00
        assert pytest.approx(res.min_total_cost, abs=1e-2) == 56400.00
        assert pytest.approx(res.annual_savings, abs=1e-2) == 4800.00
        assert res.tiers[0].feasible_quantity == 499.0
        assert pytest.approx(res.tiers[1].feasible_quantity, abs=1e-2) == 510.75
        assert res.tiers[2].feasible_quantity == 1000.0
        assert res.verification.is_valid

    def test_quantity_discount_inequality_phrasings_matrix(self):
        """Verify various phrasing variations of inequality brackets."""
        variations = [
            # Words with inequalities
            "Tier 1: Q < 500, price = $12.00; Tier 2: 500 <= Q < 1000, price = $11.50; Tier 3: Q >= 1000, price = $11.00",
            # Price first with inequalities
            "price is $12.00 for Q < 500, price = $11.50 for 500 <= Q < 1000, price = $11.00 for Q >= 1000",
            # Under and between
            "orders under 500 units: $12.00, orders between 500 and 999 units: $11.50, orders of 1000 or more units: $11.00",
            # Less than and at least
            "less than 500 units @ $12.00, 500 to 999 units @ $11.50, at least 1000 units @ $11.00",
        ]
        for v in variations:
            query = f"Annual demand is 5,000 units, order cost is $60, holding rate is 20%. Quantity discount schedule: {v}."
            resp = self.agent.run(query)
            assert resp.identified_model == InventoryModelType.QUANTITY_DISCOUNT, f"Failed model identification on: {v}"
            sched = resp.extracted_inputs.get("discount_schedule", [])
            assert len(sched) == 3, f"Failed tier extraction on: {v}, got: {sched}"
            assert resp.calculation_result.optimal_quantity == 1000.0
            assert resp.calculation_result.optimal_unit_price == 11.00
            assert pytest.approx(resp.calculation_result.min_total_cost, abs=1e-2) == 56400.00

    def test_quantity_discount_user_test_case_with_variable_labels_and_below_syntax(self):
        """Verify the exact user inquiry with (D), (S), below syntax, through syntax, and 'units or more'."""
        query = (
            "Annual Demand (D): 5,000 units\n"
            "Ordering Cost (S): $60 per order\n"
            "Annual Holding Rate: 20% of unit purchase price\n\n"
            "Supplier pricing:\n"
            "- Tier 1: Quantity below 500 units = $12.00/unit\n"
            "- Tier 2: Quantity from 500 through 999 units = $11.50/unit\n"
            "- Tier 3: Quantity of 1,000 units or more = $11.00/unit"
        )
        resp = self.agent.run(query)
        assert resp.identified_model == InventoryModelType.QUANTITY_DISCOUNT
        assert resp.needs_clarification is False
        assert len(resp.missing_inputs) == 0

        res = resp.calculation_result
        assert len(res.tiers) == 3

        # Tier 1: [1, 499] @ $12.00, EOQ=500 -> candidate Q=499, TC=$61,200.00
        t1 = res.tiers[0]
        assert t1.min_quantity == 1.0
        assert t1.max_quantity == 499.0
        assert t1.unit_price == 12.00
        assert pytest.approx(t1.holding_cost, rel=1e-4) == 2.40
        assert pytest.approx(t1.unconstrained_eoq, rel=1e-4) == 500.00
        assert t1.feasible_quantity == 499.0
        assert pytest.approx(t1.total_annual_cost, abs=0.01) == 61200.00

        # Tier 2: [500, 999] @ $11.50, EOQ~510.75 -> candidate Q=510.75, TC=$58,674.73
        t2 = res.tiers[1]
        assert t2.min_quantity == 500.0
        assert t2.max_quantity == 999.0
        assert t2.unit_price == 11.50
        assert pytest.approx(t2.holding_cost, rel=1e-4) == 2.30
        assert pytest.approx(t2.unconstrained_eoq, abs=0.01) == 510.75
        assert pytest.approx(t2.feasible_quantity, abs=0.01) == 510.75
        assert pytest.approx(t2.total_annual_cost, abs=0.01) == 58674.73

        # Tier 3: [1000, inf) @ $11.00, EOQ~522.23 -> candidate Q=1000, TC=$56,400.00
        t3 = res.tiers[2]
        assert t3.min_quantity == 1000.0
        assert t3.max_quantity is None
        assert t3.unit_price == 11.00
        assert pytest.approx(t3.holding_cost, rel=1e-4) == 2.20
        assert pytest.approx(t3.unconstrained_eoq, abs=0.01) == 522.23
        assert t3.feasible_quantity == 1000.0
        assert pytest.approx(t3.total_annual_cost, abs=0.01) == 56400.00

        # Global optimal breakdown
        assert res.optimal_quantity == 1000.0
        assert res.optimal_unit_price == 11.00
        assert res.annual_purchase_cost == 55000.00
        assert pytest.approx(res.annual_ordering_cost, abs=0.01) == 300.00
        assert pytest.approx(res.annual_holding_cost, abs=0.01) == 1100.00
        assert pytest.approx(res.total_annual_policy_cost, abs=0.01) == 1400.00
        assert pytest.approx(res.min_total_cost, abs=0.01) == 56400.00
        assert pytest.approx(res.annual_savings, abs=0.01) == 4800.00

        # Strict price qualification and boundary verification
        assert res.verification.is_valid

    # --- Fixed-Period Review: Disambiguation ---
    def test_fixed_period_direct_safety_stock_and_position(self):
        """Distinguish direct safety stock and direct inventory position; do not report missing."""
        query = (
            "We have a periodic review system with review period of 7 days and lead time of 3 days. "
            "Average daily demand is 25 units. Target safety stock is 50 units. "
            "Direct inventory position is 120 units. How many units should we order?"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.FIXED_PERIOD_REVIEW
        assert resp.needs_clarification is False
        assert len(resp.missing_inputs) == 0
        assert resp.extracted_inputs["review_period"] == 7.0
        assert resp.extracted_inputs["lead_time"] == 3.0
        assert resp.extracted_inputs["avg_demand"] == 25.0
        assert resp.extracted_inputs["safety_stock"] == 50.0
        assert resp.extracted_inputs["inventory_position"] == 120.0
        assert resp.calculation_result.order_up_to_level == 300.0
        assert resp.calculation_result.recommended_order_quantity == 180.0

    def test_fixed_period_components_disambiguation(self):
        """Distinguish on-hand inventory, on-order inventory, backorders, and protection interval."""
        query = (
            "We operate a fixed-period review model. Review period is 10 days, lead time is 4 days. "
            "Mean demand is 15 units/day with demand standard deviation of 4 units. Service level is 95%. "
            "On-hand inventory is 80 units, on-order inventory is 40 units, and customer backorders are 10 units. "
            "What is our order quantity?"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.FIXED_PERIOD_REVIEW
        assert resp.needs_clarification is False
        assert resp.extracted_inputs["current_inventory"] == 80.0
        assert resp.extracted_inputs["on_order"] == 40.0
        assert resp.extracted_inputs["backorders"] == 10.0
        assert resp.calculation_result.inventory_position == 110.0
        assert resp.calculation_result.protection_period == 14.0

    def test_fixed_period_review_original_user_inquiry(self):
        """Exact user regression test:
        'A warehouse reviews its inventory every 7 days. Average daily demand is 80 units, with a standard deviation of 12 units per day. The supplier has a lead time of 3 days. The company wants to maintain a 95% cycle service level. Currently, there are 600 units on hand, 150 units already on order, and 40 units of backorders. Assume daily demand is independent and normally distributed. Calculate the target inventory level, safety stock, and recommended order quantity using the fixed-period review model. Round the recommended order quantity up to the nearest whole unit.'
        """
        query = (
            "A warehouse reviews its inventory every 7 days. Average daily demand is 80 units, with a standard deviation of 12 units per day. "
            "The supplier has a lead time of 3 days. The company wants to maintain a 95% cycle service level. "
            "Currently, there are 600 units on hand, 150 units already on order, and 40 units of backorders. "
            "Assume daily demand is independent and normally distributed. Calculate the target inventory level, "
            "safety stock, and recommended order quantity using the fixed-period review model. "
            "Round the recommended order quantity up to the nearest whole unit."
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.FIXED_PERIOD_REVIEW
        assert resp.needs_clarification is False
        assert len(resp.missing_inputs) == 0

        # Parameter extraction assertions
        assert resp.extracted_inputs["review_period"] == 7.0
        assert resp.extracted_inputs["lead_time"] == 3.0
        assert resp.extracted_inputs["avg_demand"] == 80.0
        assert resp.extracted_inputs["demand_std"] == 12.0
        assert resp.extracted_inputs["service_level"] == 0.95
        assert resp.extracted_inputs["current_inventory"] == 600.0
        assert resp.extracted_inputs["on_order"] == 150.0
        assert resp.extracted_inputs["backorders"] == 40.0
        assert resp.extracted_inputs.get("round_up") is True

        # Intermediate & final calculations
        res = resp.calculation_result
        assert res.protection_period == 10.0
        assert res.expected_demand_protection_period == 800.0
        assert pytest.approx(res.safety_stock, abs=1e-2) == 62.42
        assert pytest.approx(res.target_inventory, abs=1e-2) == 862.42
        assert pytest.approx(res.order_up_to_level, abs=1e-2) == 862.42
        assert res.inventory_position == 710.0
        assert pytest.approx(res.exact_order_quantity, abs=1e-2) == 152.42
        assert res.recommended_order_quantity == 153.0

        # Explanation content
        assert "153 units" in resp.explanation
        assert "152.42" in resp.explanation
        assert "710" in resp.explanation
        assert "62.42" in resp.explanation
        assert "862" in resp.explanation

    def test_fixed_period_on_order_and_backorders_phrasing_variations(self):
        """Test diverse natural language variations for on-order and backorders."""
        phrasings = [
            ("150 units on order", "40 units of backorders"),
            ("150 units in transit", "40 units backordered"),
            ("150 units already on order", "40 units on backorder"),
            ("scheduled receipts of 150 units", "customer backorders of 40 units"),
            ("open orders of 150 units", "40 units in backlog"),
        ]
        for oo_text, bo_text in phrasings:
            query = (
                f"Review period is 7 days, lead time is 3 days, average demand is 80 units/day, "
                f"std dev is 12 units, 95% service level. There are 600 units in stock, {oo_text}, "
                f"and {bo_text}. Round the order quantity up."
            )
            resp = self.agent.run(query)
            assert resp.extracted_inputs["on_order"] == 150.0, f"Failed for on_order: '{oo_text}'"
            assert resp.extracted_inputs["backorders"] == 40.0, f"Failed for backorders: '{bo_text}'"
            assert resp.calculation_result.inventory_position == 710.0
            assert resp.calculation_result.recommended_order_quantity == 153.0

    # --- Newsvendor: Disambiguation ---
    def test_newsvendor_direct_underage_overage_costs(self):
        """Distinguish direct underage cost and overage cost; do not report P or C missing."""
        query = (
            "For our seasonal line, demand is normally distributed with mean demand of 500 units "
            "and demand standard deviation of 100 units. Underage cost is $30 and overage cost is $10. "
            "What is our recommended newsvendor order quantity?"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.NEWSVENDOR
        assert resp.needs_clarification is False
        assert len(resp.missing_inputs) == 0
        assert resp.extracted_inputs["underage_cost"] == 30.0
        assert resp.extracted_inputs["overage_cost"] == 10.0
        assert resp.extracted_inputs["demand_mean"] == 500.0
        assert resp.extracted_inputs["demand_std"] == 100.0
        assert pytest.approx(resp.calculation_result.critical_ratio, abs=1e-4) == 0.75
        assert resp.calculation_result.optimal_order_quantity > 500.0

    def test_newsvendor_selling_price_purchase_cost_salvage(self):
        """Distinguish selling price, purchase cost, salvage value, mean demand, demand std."""
        query = (
            "Selling price is $120 per unit. Unit purchase cost is $60. Salvage value is $20. "
            "Expected mean demand is 350 units with demand standard deviation of 50 units. "
            "Calculate the optimal newsvendor order."
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.NEWSVENDOR
        assert resp.needs_clarification is False
        assert resp.extracted_inputs["selling_price"] == 120.0
        assert resp.extracted_inputs["unit_cost"] == 60.0
        assert resp.extracted_inputs["salvage_value"] == 20.0
        assert resp.extracted_inputs["demand_mean"] == 350.0
        assert resp.extracted_inputs["demand_std"] == 50.0
        assert pytest.approx(resp.calculation_result.critical_ratio, abs=1e-4) == 0.60

    # --- Entire Question Examination & Negative Tests ---
    def test_examine_entire_question_never_declares_present_parameter_missing(self):
        """Ensure no parameter present anywhere in the text is reported as missing."""
        query = (
            "Please determine the order quantity. Our annual consumption is 12,000 units. "
            "The cost of placing an order is $75. Unit purchase cost is $50, and percentage holding cost is 15%."
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.EOQ
        assert resp.needs_clarification is False
        assert len(resp.missing_inputs) == 0
        assert resp.extracted_inputs["annual_demand"] == 12000.0
        assert resp.extracted_inputs["order_cost"] == 75.0
        assert resp.extracted_inputs["holding_cost_rate"] == 0.15
        assert resp.extracted_inputs["unit_cost"] == 50.0

    def test_genuinely_missing_parameter_triggers_clarification(self):
        """When a required parameter is genuinely omitted, the agent requests clarification."""
        query = "What is our EOQ if annual demand is 8,000 units and setup expense is $45?"
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.EOQ
        assert resp.needs_clarification is True
        assert any("holding_cost" in m for m in resp.missing_inputs)
        assert resp.clarification_message is not None
        assert "annual_demand" in resp.extracted_inputs
        assert "order_cost" in resp.extracted_inputs

    def test_agent_eoq_operating_days_cycle_time_regression(self):
        """Exact user regression test:
        - Annual demand: 12,000 units
        - Ordering cost: $75
        - Holding cost: $3 per unit annually
        - Operating days: 300 days per year
        Verify EOQ=774.60, order frequency=15.49 orders/yr, order cycle time=19.36 working days.
        """
        query = (
            "In my EOQ test, I specified:\n"
            "- Annual demand: 12,000 units\n"
            "- Ordering cost: $75\n"
            "- Holding cost: $3 per unit annually\n"
            "- Operating days: 300 days per year"
        )
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.EOQ
        assert resp.needs_clarification is False
        assert len(resp.missing_inputs) == 0

        # Verify extracted inputs
        assert resp.extracted_inputs["annual_demand"] == 12000.0
        assert resp.extracted_inputs["order_cost"] == 75.0
        assert resp.extracted_inputs["holding_cost"] == 3.0
        assert resp.extracted_inputs["workdays_per_year"] == 300

        # Verify calculation results
        assert resp.calculation_result is not None
        assert pytest.approx(resp.calculation_result.eoq, abs=1e-2) == 774.60
        assert pytest.approx(resp.calculation_result.orders_per_year, abs=1e-2) == 15.49
        assert resp.calculation_result.workdays_per_year == 300
        assert pytest.approx(resp.calculation_result.cycle_time_days, abs=1e-2) == 19.36

        # Verify explanation text displays exactly 19.36 working days and math formula
        assert "19.36 working days" in resp.explanation
        assert "300" in resp.explanation
        assert "15.49" in resp.explanation

    def test_agent_eoq_unspecified_operating_days_uses_documented_default(self):
        """When operating days are omitted, agent must apply and document default 250 working days."""
        query = "Annual demand is 12,000 units, ordering cost is $75, and holding cost is $3 per unit annually. What is the EOQ?"
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.EOQ
        assert resp.needs_clarification is False
        assert resp.calculation_result.workdays_per_year == 250
        assert pytest.approx(resp.calculation_result.cycle_time_days, abs=1e-2) == 16.14
        assert "16.14 working days" in resp.explanation
        assert "250 working days (standard documented default: 50 weeks × 5 days/week)" in resp.explanation

    def test_agent_eoq_monthly_demand_normalization_exact_user_case(self):
        """Exact user regression test inquiry:
        'Find EOQ. Demand is 1,000 units per month, ordering cost is $25 per order, and holding cost is $0.50 per unit per year.'
        Expected:
        - Original demand: 1,000 units/month
        - Annual demand: 12,000 units/year
        - EOQ: 1,095.45 units
        - Annual ordering cost: $273.86
        - Annual holding cost: $273.86
        - Total annual inventory policy cost: $547.72
        - Order frequency: 10.95 orders/year
        - Order cycle time: approximately 22.82 working days using 250 operating days/year.
        """
        query = "Find EOQ. Demand is 1,000 units per month, ordering cost is $25 per order, and holding cost is $0.50 per unit per year."
        resp = self.agent.run(query)

        assert resp.identified_model == InventoryModelType.EOQ
        assert resp.selected_tool_name == "calculate_eoq"
        assert resp.needs_clarification is False
        assert len(resp.missing_inputs) == 0

        res = resp.calculation_result
        assert res is not None
        assert res.raw_demand == 1000.0
        assert res.demand_period == "month"
        assert res.demand_unit == "units"
        assert res.annual_demand == 12000.0
        assert pytest.approx(res.eoq, abs=1e-2) == 1095.45
        assert pytest.approx(res.annual_ordering_cost, abs=1e-2) == 273.86
        assert pytest.approx(res.annual_holding_cost, abs=1e-2) == 273.86
        assert pytest.approx(res.total_annual_cost, abs=1e-2) == 547.72
        assert pytest.approx(res.orders_per_year, abs=1e-2) == 10.95
        assert pytest.approx(res.cycle_time_days, abs=1e-2) == 22.82
        assert res.workdays_per_year == 250
        assert res.demand_normalization_applied is True
        assert res.demand_normalization_factor == 12.0
        assert res.verification.is_valid

        # Verify explanation displays all expected metrics and formulas
        assert "1,095.45" in resp.explanation
        assert "273.86" in resp.explanation
        assert "547.72" in resp.explanation
        assert "10.95" in resp.explanation
        assert "22.82" in resp.explanation
        assert "Demand Period Normalization Applied" in resp.explanation
        assert "1,000.00 units per month" in resp.explanation
        assert "12,000.00 units per year" in resp.explanation

    def test_agent_eoq_other_demand_periods_weekly_quarterly_daily(self):
        """Test natural language queries for weekly, quarterly, and daily demand."""
        # Weekly demand
        q_wk = "Weekly demand is 250 units, ordering cost is $50, and holding cost is $2 per unit per year. Calculate EOQ."
        resp_wk = self.agent.run(q_wk)
        assert resp_wk.identified_model == InventoryModelType.EOQ
        assert resp_wk.calculation_result.raw_demand == 250.0
        assert resp_wk.calculation_result.demand_period == "week"
        assert resp_wk.calculation_result.annual_demand == 13000.0
        assert pytest.approx(resp_wk.calculation_result.eoq, abs=1e-2) == 806.23
        assert resp_wk.calculation_result.demand_normalization_applied is True

        # Quarterly demand
        q_qtr = "Quarterly demand is 3,000 units, ordering cost is $60, and holding cost is $2.50 per unit per year. Find EOQ."
        resp_qtr = self.agent.run(q_qtr)
        assert resp_qtr.identified_model == InventoryModelType.EOQ
        assert resp_qtr.calculation_result.raw_demand == 3000.0
        assert resp_qtr.calculation_result.demand_period == "quarter"
        assert resp_qtr.calculation_result.annual_demand == 12000.0
        assert pytest.approx(resp_qtr.calculation_result.eoq, abs=1e-2) == 758.95
        assert resp_qtr.calculation_result.demand_normalization_applied is True

        # Daily demand with custom operating days
        q_day = "Daily demand is 40 units per day, ordering cost is $75, holding cost is $3 per unit annually, with 300 operating days per year."
        resp_day = self.agent.run(q_day)
        assert resp_day.identified_model == InventoryModelType.EOQ
        assert resp_day.calculation_result.raw_demand == 40.0
        assert resp_day.calculation_result.demand_period == "day"
        assert resp_day.calculation_result.workdays_per_year == 300
        assert resp_day.calculation_result.annual_demand == 12000.0
        assert pytest.approx(resp_day.calculation_result.eoq, abs=1e-2) == 774.60
        assert pytest.approx(resp_day.calculation_result.cycle_time_days, abs=1e-2) == 19.36
        assert resp_day.calculation_result.demand_normalization_applied is True


# =====================================================================
# CSV Data Upload and Analysis Tests
# =====================================================================

class TestInventoryDataUploadAnalysis:
    """Test suite for CSV data upload processing, validation, and analytics."""

    def test_process_valid_inventory_dataframe(self):
        """Test calculation of Units, Unit Cost, Highest Units Product, Category grouping, and Inventory Value."""
        data = {
            "Product": ["Widget A", "Gadget B", "Tool C", "Widget D"],
            "Category": ["Electronics", "Hardware", "Hardware", "Electronics"],
            "Units": [150, 400, 250, 100],
            "Unit Cost": [20.0, 15.0, 30.0, 25.0],
        }
        df = pd.DataFrame(data)

        processed_df, results, missing = process_inventory_dataframe(df)

        assert missing is None
        assert processed_df is not None
        assert results is not None

        # 1. Total Units: 150 + 400 + 250 + 100 = 900
        assert results["total_units"] == 900.0

        # 2. Average Unit Cost: (20 + 15 + 30 + 25) / 4 = 22.5
        assert results["avg_unit_cost"] == 22.5

        # 3. Product with highest Units: Gadget B (400)
        assert results["highest_units_product"] == "Gadget B"
        assert results["highest_units_count"] == 400.0

        # 4. Group by Category and total Units:
        # Electronics: 150 + 100 = 250; Hardware: 400 + 250 = 650
        cat_df = results["category_units"]
        hw_units = cat_df.loc[cat_df["Category"] == "Hardware", "Total Units"].values[0]
        elec_units = cat_df.loc[cat_df["Category"] == "Electronics", "Total Units"].values[0]
        assert hw_units == 650.0
        assert elec_units == 250.0

        # 5. Inventory Value column: Units * Unit Cost
        expected_values = [150 * 20.0, 400 * 15.0, 250 * 30.0, 100 * 25.0]
        assert list(processed_df["Inventory Value"]) == expected_values

        # 6. Total Inventory Value across all products
        # 3000 + 6000 + 7500 + 2500 = 19000
        assert results["total_inventory_value"] == 19000.0

        # Dimensions
        assert results["num_rows"] == 4
        assert results["num_cols"] == 4
        assert "Product" in results["column_names"]

    def test_process_missing_required_columns(self):
        """Identify missing columns when required fields are absent."""
        data = {
            "Product": ["Widget A", "Widget B"],
            "Units": [100, 200],
            # Missing "Category" and "Unit Cost"
        }
        df = pd.DataFrame(data)

        processed_df, results, missing = process_inventory_dataframe(df)

        assert processed_df is None
        assert results is None
        assert missing is not None
        assert "Category" in missing
        assert "Unit Cost" in missing

    def test_process_numeric_coercion(self):
        """String formatted numbers should be coerced safely without crashing."""
        data = {
            "Product": ["Item 1", "Item 2"],
            "Category": ["Cat A", "Cat B"],
            "Units": ["50", "150"],
            "Unit Cost": ["10.50", "20.00"],
        }
        df = pd.DataFrame(data)

        processed_df, results, missing = process_inventory_dataframe(df)

        assert missing is None
        assert results["total_units"] == 200.0
        assert results["total_inventory_value"] == (50 * 10.50) + (150 * 20.00)


# =====================================================================
# Multicriteria Inventory Classification Tests (65% Sales / 35% POS Hits)
# =====================================================================

class TestMulticriteriaClassification:
    """Test suite for BAN517 Multicriteria Inventory Classification."""

    def test_multicriteria_classification_course_methodology_weights_and_scores(self):
        """Verify:
        - 65% Sales weight, 35% POS Hits weight
        - Pareto cutoffs: 60% (A), 80% (B), 90% (C), >90% (D)
        - Scores: A=4, B=3, C=2, D=1
        - Combined score = 0.65 * Sales_Score + 0.35 * POS_Score
        - Thresholds: Final A in [3.25, 4.00], Final B in [2.50, 3.25), Final C in [1.75, 2.50), Final D < 1.75
        """
        # Create a controlled 4-item dataset
        # Item 1: High Sales (A), High Hits (A) -> AA: score = 0.65(4) + 0.35(4) = 4.00 -> Final A
        # Item 2: Medium Sales (B), Medium Hits (B) -> BB: score = 0.65(3) + 0.35(3) = 3.00 -> Final B
        # Item 3: Low Sales (C), Low Hits (C) -> CC: score = 0.65(2) + 0.35(2) = 2.00 -> Final C
        # Item 4: Tail Sales (D), Tail Hits (D) -> DD: score = 0.65(1) + 0.35(1) = 1.00 -> Final D
        data = {
            "Product": ["Prod A", "Prod B", "Prod C", "Prod D"],
            "Sales": [6000.0, 2000.0, 1000.0, 1000.0],  # Total = 10,000; Cum: 60% (A), 80% (B), 90% (C), 100% (D)
            "POS Hits": [600.0, 200.0, 100.0, 100.0],  # Total = 1,000; Cum: 60% (A), 80% (B), 90% (C), 100% (D)
        }
        df = pd.DataFrame(data)

        res = calculate_multicriteria_classification(
            df,
            sales_col="Sales",
            pos_hits_col="POS Hits",
            sales_weight=0.65,
            pos_weight=0.35,
            pareto_cutoffs=(0.60, 0.80, 0.90),
            score_thresholds=(3.25, 2.50, 1.75),
        )

        assert isinstance(res, MulticriteriaClassificationResult)
        assert res.verification.is_valid is True

        c_df = res.classified_df
        assert list(c_df["Sales_Class"]) == ["A", "B", "C", "D"]
        assert list(c_df["Sales_Score"]) == [4, 3, 2, 1]
        assert list(c_df["POS_Class"]) == ["A", "B", "C", "D"]
        assert list(c_df["POS_Score"]) == [4, 3, 2, 1]

        expected_scores = [4.00, 3.00, 2.00, 1.00]
        for actual, exp in zip(c_df["Combined_Score"], expected_scores):
            assert pytest.approx(actual, abs=1e-3) == exp

        assert list(c_df["Final_Class"]) == ["A", "B", "C", "D"]

    def test_multicriteria_mixed_cross_classifications(self):
        """Verify cross-class combinations with 60/80/90 cutoffs and 3.25/2.50/1.75 thresholds.
        - AA: Sales A(4), Hits A(4) -> 4.00 -> Final A (>= 3.25)
        - BA: Sales B(3), Hits A(4) -> 3.35 -> Final A (>= 3.25)
        - BC: Sales B(3), Hits C(2) -> 2.65 -> Final B ([2.50, 3.25))
        - CB: Sales C(2), Hits B(3) -> 2.35 -> Final C ([1.75, 2.50))
        - CD: Sales D(1), Hits D(1) -> 1.00 -> Final D (< 1.75)
        """
        data = {
            "Product": ["Item AA", "Item BA", "Item BC", "Item CB", "Item CD"],
            "Sales":    [5000.0, 1000.0, 2000.0, 1000.0, 1000.0],
            "POS Hits": [ 300.0,  300.0,  100.0,  200.0,  100.0],
        }
        df = pd.DataFrame(data)

        res = calculate_multicriteria_classification(
            df,
            sales_col="Sales",
            pos_hits_col="POS Hits",
            sales_weight=0.65,
            pos_weight=0.35,
            pareto_cutoffs=(0.60, 0.80, 0.90),
            score_thresholds=(3.25, 2.50, 1.75),
        )

        c_df = res.classified_df.set_index("Product")
        # Item AA: 0.65(4) + 0.35(4) = 4.00 -> Final A
        assert c_df.loc["Item AA", "Sales_Class"] == "A"
        assert c_df.loc["Item AA", "POS_Class"] == "A"
        assert pytest.approx(c_df.loc["Item AA", "Combined_Score"], abs=1e-3) == 4.00
        assert c_df.loc["Item AA", "Final_Class"] == "A"

        # Item BA: Sales B(3), Hits A(4) -> 0.65(3) + 0.35(4) = 3.35 >= 3.25 -> Final A
        assert c_df.loc["Item BA", "Sales_Class"] == "B"
        assert c_df.loc["Item BA", "POS_Class"] == "A"
        assert pytest.approx(c_df.loc["Item BA", "Combined_Score"], abs=1e-3) == 3.35
        assert c_df.loc["Item BA", "Final_Class"] == "A"

        # Item BC: Sales B(3), Hits C(2) -> 0.65(3) + 0.35(2) = 2.65 -> Final B
        assert c_df.loc["Item BC", "Sales_Class"] == "B"
        assert c_df.loc["Item BC", "POS_Class"] == "C"
        assert pytest.approx(c_df.loc["Item BC", "Combined_Score"], abs=1e-3) == 2.65
        assert c_df.loc["Item BC", "Final_Class"] == "B"

        # Item CB: Sales C(2), Hits B(3) -> 0.65(2) + 0.35(3) = 2.35 -> Final C
        assert c_df.loc["Item CB", "Sales_Class"] == "C"
        assert c_df.loc["Item CB", "POS_Class"] == "B"
        assert pytest.approx(c_df.loc["Item CB", "Combined_Score"], abs=1e-3) == 2.35
        assert c_df.loc["Item CB", "Final_Class"] == "C"

        # Item CD: Sales D(1), Hits D(1) -> 0.65(1) + 0.35(1) = 1.00 -> Final D
        assert c_df.loc["Item CD", "Sales_Class"] == "D"
        assert c_df.loc["Item CD", "POS_Class"] == "D"
        assert pytest.approx(c_df.loc["Item CD", "Combined_Score"], abs=1e-3) == 1.00
        assert c_df.loc["Item CD", "Final_Class"] == "D"


    def test_multicriteria_summary_table_integrity(self):
        """Verify summary table accounts for 100% of items and 100% of sales."""
        data = {
            "Product": [f"P{i}" for i in range(10)],
            "Sales": [5000, 2500, 1000, 600, 400, 250, 150, 60, 30, 10],
            "POS Hits": [100, 80, 60, 40, 30, 20, 15, 10, 5, 2],
        }
        df = pd.DataFrame(data)

        res = calculate_multicriteria_classification(
            df,
            sales_col="Sales",
            pos_hits_col="POS Hits",
            pareto_cutoffs=(0.60, 0.80, 0.90),
            score_thresholds=(3.25, 2.50, 1.75),
        )

        summary_df = res.summary_df
        assert len(summary_df) == 4
        assert list(summary_df["Final_Class"]) == ["A", "B", "C", "D"]
        assert summary_df["Items_Count"].sum() == 10
        assert pytest.approx(summary_df["Pct_Items"].sum(), abs=1e-2) == 100.0
        assert pytest.approx(summary_df["Total_Sales"].sum(), abs=1e-2) == sum(data["Sales"])
        assert pytest.approx(summary_df["Pct_Sales"].sum(), abs=1e-2) == 100.0
        assert summary_df["Total_POS_Hits"].sum() == sum(data["POS Hits"])
        assert pytest.approx(summary_df["Pct_POS_Hits"].sum(), abs=1e-2) == 100.0

    def test_multicriteria_input_validation(self):
        """Verify parameter checks and error conditions."""
        df = pd.DataFrame({"Sales": [100], "POS Hits": [10]})

        # Invalid weights (sum != 1.0)
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            calculate_multicriteria_classification(df, sales_weight=0.50, pos_weight=0.30)

        # Negative sales value
        df_neg = pd.DataFrame({"Sales": [-10, 100], "POS Hits": [5, 10]})
        with pytest.raises(ValueError, match="contains negative values"):
            calculate_multicriteria_classification(df_neg)

        # Empty dataframe
        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            calculate_multicriteria_classification(pd.DataFrame())

    def test_multicriteria_dead_stock_assigned_to_d(self):
        """Zero sales or zero hits items must be classified as D."""
        data = {
            "Product": ["P1", "P2", "P3"],
            "Sales": [1000.0, 0.0, 500.0],
            "POS Hits": [50.0, 0.0, 25.0],
        }
        df = pd.DataFrame(data)
        res = calculate_multicriteria_classification(
            df,
            sales_col="Sales",
            pos_hits_col="POS Hits",
            pareto_cutoffs=(0.60, 0.80, 0.90),
            score_thresholds=(3.25, 2.50, 1.75),
        )
        c_df = res.classified_df.set_index("Product")
        assert c_df.loc["P2", "Sales_Class"] == "D"
        assert c_df.loc["P2", "POS_Class"] == "D"
        assert c_df.loc["P2", "Final_Class"] == "D"

    def test_process_inventory_dataframe_populates_multicriteria(self):
        """Ensure process_inventory_dataframe computes multicriteria results seamlessly."""
        data = {
            "Product": ["P1", "P2", "P3", "P4"],
            "Category": ["Office", "Office", "Electronics", "Electronics"],
            "Units": [200, 150, 80, 40],
            "Unit Cost": [50.0, 20.0, 100.0, 15.0],
        }
        df = pd.DataFrame(data)
        processed_df, results, missing = process_inventory_dataframe(df)

        assert missing is None
        assert "multicriteria" in results
        mc_res = results["multicriteria"]
        assert isinstance(mc_res, MulticriteriaClassificationResult)
        assert "Final_Class" in processed_df.columns
        assert "Combined_Score" in processed_df.columns

    def test_project_dataset_v2_if_available(self):
        """Verify real benchmark ProjectDataV2_UTF8.csv against exact BAN517 methodology:
        - Cutoffs: 60%, 80%, 90%
        - Thresholds: 3.25, 2.50, 1.75
        - Expected:
          Final A: 22 items (10.73%) | $54,788,937 (66.78% of sales)
          Final B: 18 items (8.78%)  | $10,450,897 (12.74% of sales)
          Final C: 23 items (11.22%) | $8,173,036  (9.96% of sales)
          Final D: 142 items (69.27%)| $8,633,280  (10.52% of sales)
          Total: 205 items           | $82,046,150 (100.0%)
        """
        import os
        proj_path = "/Users/jonahvoorheis/Downloads/ProjectDataV2_UTF8.csv"
        if not os.path.exists(proj_path):
            pytest.skip("ProjectDataV2_UTF8.csv not available in environment.")

        df = pd.read_csv(proj_path)
        processed_df, results, missing = process_inventory_dataframe(df)

        assert missing is None
        assert results["num_rows"] == 205
        mc = results["multicriteria"]
        assert mc is not None

        summary = mc.summary_df.set_index("Final_Class")

        # Class A: 22 items, $54,788,937.00 (66.78%)
        assert summary.loc["A", "Items_Count"] == 22
        assert pytest.approx(summary.loc["A", "Total_Sales"], abs=1e-1) == 54788937.0
        assert pytest.approx(summary.loc["A", "Pct_Sales"], abs=1e-1) == 66.78

        # Class B: 18 items, $10,450,897.00 (12.74%)
        assert summary.loc["B", "Items_Count"] == 18
        assert pytest.approx(summary.loc["B", "Total_Sales"], abs=1e-1) == 10450897.0
        assert pytest.approx(summary.loc["B", "Pct_Sales"], abs=1e-1) == 12.74

        # Class C: 23 items, $8,173,036.00 (9.96%)
        assert summary.loc["C", "Items_Count"] == 23
        assert pytest.approx(summary.loc["C", "Total_Sales"], abs=1e-1) == 8173036.0
        assert pytest.approx(summary.loc["C", "Pct_Sales"], abs=1e-1) == 9.96

        # Class D: 142 items, $8,633,280.00 (10.52%)
        assert summary.loc["D", "Items_Count"] == 142
        assert pytest.approx(summary.loc["D", "Total_Sales"], abs=1e-1) == 8633280.0
        assert pytest.approx(summary.loc["D", "Pct_Sales"], abs=1e-1) == 10.52

        # Conservation of all 205 items
        assert summary["Items_Count"].sum() == 205
        assert mc.verification.is_valid is True

    def test_process_inventory_dataframe_and_styler_formatting_project_data_v2(self):
        """Verify ProjectDataV2_UTF8.csv renders in pandas Styler without Unknown format code 'f'."""
        import os
        from app import build_column_format_dict, process_inventory_dataframe

        proj_path = "./ProjectDataV2_UTF8.csv"
        if not os.path.exists(proj_path):
            proj_path = "/Users/jonahvoorheis/Downloads/ProjectDataV2_UTF8.csv"
        if not os.path.exists(proj_path):
            pytest.skip("ProjectDataV2_UTF8.csv not available.")

        df = pd.read_csv(proj_path)
        processed_df, analysis, missing = process_inventory_dataframe(df)

        assert missing is None
        assert len(processed_df) == 205

        # Check numeric conversions
        assert pd.api.types.is_numeric_dtype(processed_df["Sales ($)"])
        assert pd.api.types.is_numeric_dtype(processed_df["Hits"])
        assert pd.api.types.is_numeric_dtype(processed_df["Units"])
        assert pd.api.types.is_numeric_dtype(processed_df["Unit Cost"])
        assert pd.api.types.is_numeric_dtype(processed_df["Inventory Value"])

        # Check text columns are strings
        assert not pd.api.types.is_numeric_dtype(processed_df["Item Name"])
        assert not pd.api.types.is_numeric_dtype(processed_df["UoM"])
        assert not pd.api.types.is_numeric_dtype(processed_df["Sales_Class"])
        assert not pd.api.types.is_numeric_dtype(processed_df["POS_Class"])
        assert not pd.api.types.is_numeric_dtype(processed_df["Final_Class"])

        # Construct available display columns as in Streamlit UI
        preferred_order = [
            "Product", "Category", "Sales ($)", "Sales_Class", "Sales_Score",
            "Hits", "POS_Class", "POS_Score", "Combined_Score", "Final_Class",
        ]
        available_cols = [c for c in preferred_order if c in processed_df.columns]
        for c in processed_df.columns:
            if c not in available_cols and not c.endswith("_CumPct"):
                available_cols.append(c)

        format_dict = build_column_format_dict(processed_df, available_cols)

        # Ensure text fields and class labels are strictly excluded from numeric formatters
        for text_col in ["Product", "Category", "Item Name", "UoM", "Sales_Class", "POS_Class", "Final_Class"]:
            if text_col in available_cols:
                assert text_col not in format_dict

        # Crucial test: Render to HTML via Pandas Styler
        # This was previously raising ValueError: Unknown format code 'f' for object of type 'str'
        styled = processed_df[available_cols].style.format(format_dict, na_rep="-")
        html_output = styled.to_html()
        assert len(html_output) > 10000
        assert "Unknown format code" not in html_output

        # Verify summary table formatting
        mc_result = analysis["multicriteria"]
        summary_display = mc_result.summary_df.copy()
        sum_fmt = build_column_format_dict(summary_display, list(summary_display.columns))
        assert "Final_Class" not in sum_fmt
        assert "managerial_guidance" not in sum_fmt
        sum_html = summary_display.style.format(sum_fmt, na_rep="-").to_html()
        assert len(sum_html) > 500

    def test_coerce_numeric_series_with_dirty_strings_and_missing_values(self):
        """Verify robust numeric coercion of dirty strings, currency symbols, and NaNs."""
        from inventory_tools import coerce_numeric_series

        dirty_series = pd.Series(["$1,200.50", "3,400", "€45.99", "£100.00", "N/A", "-", "None", "  50  ", None])
        coerced = coerce_numeric_series(dirty_series, column_name="TestCol", default=0.0)

        assert pd.api.types.is_numeric_dtype(coerced)
        assert pytest.approx(coerced.iloc[0]) == 1200.50
        assert pytest.approx(coerced.iloc[1]) == 3400.00
        assert pytest.approx(coerced.iloc[2]) == 45.99
        assert pytest.approx(coerced.iloc[3]) == 100.00
        assert coerced.iloc[4] == 0.0  # N/A -> default 0.0
        assert coerced.iloc[5] == 0.0  # - -> default 0.0
        assert coerced.iloc[6] == 0.0  # None string -> default 0.0
        assert pytest.approx(coerced.iloc[7]) == 50.00
        assert coerced.iloc[8] == 0.0  # None -> default 0.0

        # Entirely non-numeric string column must raise an informative ValueError identifying column
        invalid_text_series = pd.Series(["Warehouse A", "Warehouse B", "Warehouse C"])
        with pytest.raises(ValueError, match="Column 'Facility' expected numeric values"):
            coerce_numeric_series(invalid_text_series, column_name="Facility")

    def test_build_column_format_dict_safe_with_unconvertible_values(self):
        """Verify formatters safely fall back to str(x) without raising format errors."""
        from app import build_column_format_dict

        test_df = pd.DataFrame({
            "Product": ["Gadget A", "Gadget B"],
            "Sales": [15000.50, 25000.00],
            "Units": [100, 200],
            "Sales_Class": ["A", "B"],
            "Combined_Score": [3.65, 2.80],
        })

        fmt = build_column_format_dict(test_df, list(test_df.columns))
        assert "Product" not in fmt
        assert "Sales_Class" not in fmt
        assert "Sales" in fmt
        assert "Units" in fmt
        assert "Combined_Score" in fmt

        # Ensure currency and count formatting works
        assert fmt["Sales"](15000.50) == "$15,000.50"
        assert fmt["Units"](100) == "100"
        assert fmt["Combined_Score"](3.65) == "3.65"

        # Ensure that if unexpected string is passed to numeric formatter, it falls back without crashing
        assert fmt["Sales"]("Unexpected") == "Unexpected"
        assert fmt["Sales"](None) == "-"






