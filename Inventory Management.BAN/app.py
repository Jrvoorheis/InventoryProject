"""Supply Chain Analytics Inventory Decision Agent Application.

Provides a unified interface:
1. Interactive Command-Line Interface (CLI / REPL) with Rich formatting
2. Single-query execution mode (`--query "..."`)
3. Pre-configured benchmark demo runner (`--demo`)
4. Full interactive Streamlit Web Dashboard (`streamlit run app.py` or `python app.py --web`)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from inventory_agent import InventoryDecisionAgent, InventoryModelType
from inventory_tools import (
    MulticriteriaClassificationResult,
    calculate_multicriteria_classification,
    coerce_numeric_series,
)

# Try importing Rich for terminal styling
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


# =====================================================================
# Pre-Configured Benchmark Scenarios
# =====================================================================

DEMO_SCENARIOS = [
    {
        "model": "Economic Order Quantity (EOQ)",
        "question": (
            "We sell 10,000 units per year. Each order costs $50 to place, and our carrying cost "
            "is $2.00 per unit per year. Replenishment lead time is 10 days. What is our optimal "
            "order quantity, reorder point, and total annual policy cost?"
        ),
    },
    {
        "model": "Quantity Discount",
        "question": (
            "Our annual demand is 1,000 units and order setup cost is $100. Our holding cost rate is 20% "
            "of unit price. The supplier offers three price tiers: 1-499 units for $10, 500-999 units for $9, "
            "and 1000+ units for $8. Which tier minimizes our total annual inventory acquisition and holding cost?"
        ),
    },
    {
        "model": "Safety Stock & Reorder Point",
        "question": (
            "Average daily demand is 50 units with a standard deviation of 8 units. Supplier lead time is 5 days. "
            "We target a 95% cycle service level. What should our safety stock buffer and reorder point be?"
        ),
    },
    {
        "model": "Fixed-Period Review",
        "question": (
            "We operate a periodic review replenishment policy every 14 days with a lead time of 7 days. "
            "Average daily demand is 30 units with a standard deviation of 6 units. We target a 98% service level. "
            "Currently we have 120 units on hand and 50 units on order. How many units should we order today?"
        ),
    },
    {
        "model": "Newsvendor (Single-Period)",
        "question": (
            "We are ordering seasonal ski jackets that retail for $250. We purchase them from the vendor for $100. "
            "Any unsold jackets at the end of the season are liquidated to an outlet store for $40. "
            "Seasonal demand is normally distributed with a mean of 400 jackets and standard deviation of 80. "
            "How many jackets should we order to maximize expected profit?"
        ),
    },
]


# =====================================================================
# Terminal CLI Functions
# =====================================================================

def print_banner():
    """Print application banner and pipeline architecture."""
    architecture_text = (
        "[bold cyan]Customer Question[/bold cyan] ➔ "
        "[bold green]1. Identify Model[/bold green] ➔ "
        "[bold yellow]2. Extract Inputs[/bold yellow] ➔ "
        "[bold magenta]3. Select Python Tool[/bold magenta] ➔ "
        "[bold blue]4. Calculate[/bold blue] ➔ "
        "[bold red]5. Verify[/bold red] ➔ "
        "[bold white]6. Explain[/bold white]"
    )
    if HAS_RICH:
        console.print(Panel(
            f"[bold white]SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT[/bold white]\n"
            f"[dim]Autonomous Decision Optimization Engine for Inventory Operations[/dim]\n\n"
            f"{architecture_text}\n\n"
            f"[italic]Supported: EOQ | Quantity Discount | Safety Stock & ROP | Fixed-Period Review | Newsvendor[/italic]",
            title="[bold green]SCM Decision Agent v1.0[/bold green]",
            border_style="cyan"
        ))
    else:
        print("=" * 78)
        print(" SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT")
        print(" Architecture: Question -> Identify -> Extract -> Tool -> Calculate -> Verify -> Explain")
        print("=" * 78)


def display_agent_response(response):
    """Format and render agent pipeline steps and markdown explanation."""
    if HAS_RICH:
        # Display Step Audit Log Table
        step_table = Table(title="Agent Architectural Pipeline Execution Log", show_header=True, header_style="bold magenta")
        step_table.add_column("Stage", style="bold cyan", width=22)
        step_table.add_column("Status", width=12)
        step_table.add_column("Details & Output", style="dim")

        for s in response.steps_log:
            color = "green" if s.status == "SUCCESS" else ("yellow" if s.status == "WARNING" else "red")
            step_table.add_row(s.step_name, f"[{color}]{s.status}[/{color}]", s.summary)

        console.print(step_table)
        console.print("\n")

        # Display Main Explanation
        console.print(Panel(Markdown(response.explanation), title=f"[bold green]{response.identified_model.value}[/bold green]", border_style="green"))
    else:
        print("\n--- Pipeline Steps ---")
        for s in response.steps_log:
            print(f"[{s.status}] {s.step_name}: {s.summary}")
        print("\n" + "=" * 78)
        print(response.explanation)
        print("=" * 78 + "\n")


def run_interactive_cli(agent: InventoryDecisionAgent):
    """Run interactive REPL loop."""
    print_banner()
    if HAS_RICH:
        console.print("[dim]Type your natural language inventory question below. Type 'demo' to run benchmarks or 'quit' to exit.[/dim]\n")
    else:
        print("Type your question below. Type 'demo' to run benchmarks or 'quit' to exit.\n")

    while True:
        try:
            prompt = "\n[bold cyan]Supply Chain Question > [/bold cyan]" if HAS_RICH else "\nSupply Chain Question > "
            user_input = console.input(prompt) if HAS_RICH else input(prompt)
            query = user_input.strip()

            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                if HAS_RICH:
                    console.print("[bold yellow]Exiting Inventory Decision Agent. Goodbye![/bold yellow]")
                else:
                    print("Exiting. Goodbye!")
                break
            if query.lower() == "demo":
                run_demo_suite(agent)
                continue

            response = agent.run(query)
            display_agent_response(response)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


def run_demo_suite(agent: InventoryDecisionAgent):
    """Execute all benchmark supply chain scenarios."""
    print_banner()
    if HAS_RICH:
        console.print("[bold yellow]=== EXECUTING SUPPLY CHAIN BENCHMARK SUITE (5 MODELS) ===[/bold yellow]\n")
    else:
        print("=== EXECUTING SUPPLY CHAIN BENCHMARK SUITE (5 MODELS) ===\n")

    for idx, item in enumerate(DEMO_SCENARIOS, 1):
        if HAS_RICH:
            console.rule(f"[bold cyan]Case Study {idx}: {item['model']}[/bold cyan]")
            console.print(f"[bold]Question:[/bold] [italic]{item['question']}[/italic]\n")
        else:
            print(f"\n--- Case Study {idx}: {item['model']} ---")
            print(f"Question: {item['question']}\n")

        resp = agent.run(item["question"])
        display_agent_response(resp)


# =====================================================================
# Safe Styler Formatting Helper
# =====================================================================

def build_column_format_dict(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """Construct safe pandas Styler formatting dictionary.

    Guarantees:
    - Text columns (Product, Category, UoM, Item Name, Sales_Class, etc.) are never formatted with numeric formatters.
    - Decimal formatting like :.2f and currency formatting is only applied to verified numeric values.
    - Discrete scores (Sales_Score, POS_Score) and counts (Units, Hits) are formatted as whole numbers.
    - Formatters safely handle NumPy scalars (np.int64, np.float64) and fall back to str(val) on non-numeric types.
    """
    format_dict: Dict[str, Any] = {}
    known_text_cols = {
        "Product", "Category", "Item Name", "UoM", "Sales_Class", "POS_Class",
        "Final_Class", "Description", "SKU", "Class", "Type", "Family", "Department",
        "managerial_guidance", "Managerial_Guidance"
    }

    def _fmt_currency(x):
        if pd.isna(x):
            return "-"
        try:
            return f"${float(x):,.2f}"
        except (ValueError, TypeError):
            return str(x)

    def _fmt_integer(x):
        if pd.isna(x):
            return "-"
        try:
            return f"{float(x):,.0f}"
        except (ValueError, TypeError):
            return str(x)

    def _fmt_percent(x):
        if pd.isna(x):
            return "-"
        try:
            return f"{float(x):.2f}%"
        except (ValueError, TypeError):
            return str(x)

    def _fmt_decimal(x):
        if pd.isna(x):
            return "-"
        try:
            return f"{float(x):.2f}"
        except (ValueError, TypeError):
            return str(x)

    for col in columns:
        if col not in df.columns:
            continue

        # Strictly skip known text fields, class code columns, or non-numeric dtypes
        if col in known_text_cols or col.endswith("_Class") or not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Numeric column formatting
        if col == "Combined_Score":
            format_dict[col] = _fmt_decimal
        elif col in ["Sales_Score", "POS_Score"]:
            format_dict[col] = _fmt_integer
        elif col.endswith("_CumPct") or col.endswith("_Pct") or "Pct" in col or "Percent" in col:
            format_dict[col] = _fmt_percent
        elif col in ["Units", "Hits", "POS Hits", "Total Units", "Item Number", "Items_Count", "Orders"]:
            format_dict[col] = _fmt_integer
        elif any(term in col for term in ["Sales", "Cost", "Value", "Price", "Revenue", "Margin", "Total_Sales"]):
            format_dict[col] = _fmt_currency
        else:
            format_dict[col] = _fmt_decimal

    return format_dict


# =====================================================================
# Inventory Data CSV Analysis Function
# =====================================================================

def process_inventory_dataframe(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[List[str]]]:
    """Validate and analyze uploaded inventory CSV dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Uploaded pandas dataframe.

    Returns
    -------
    tuple
        (analyzed_df, analysis_results_dict, missing_columns_list)
    """
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return None, None, ["Empty or invalid CSV dataset uploaded."]

    required_columns = ["Product", "Category", "Units", "Unit Cost"]
    missing = [c for c in required_columns if c not in df.columns]

    # If missing standard columns, attempt intelligent column resolution
    # for schemas like (Item Number, Item Name, UoM, Sales ($), Hits)
    col_map = {}
    if missing:
        prod_candidates = ["Product", "Item Name", "Item", "Item Number", "Description", "SKU"]
        cat_candidates = ["Category", "UoM", "Department", "Family", "Class", "Type"]
        units_candidates = ["Units", "Hits", "POS Hits", "POS_Hits", "Quantity", "Qty", "Volume", "Demand"]
        cost_candidates = ["Unit Cost", "Cost", "Price", "Unit Price"]

        for col, cand in [
            ("Product", prod_candidates),
            ("Category", cat_candidates),
            ("Units", units_candidates),
            ("Unit Cost", cost_candidates),
        ]:
            for c in cand:
                if c in df.columns:
                    col_map[col] = c
                    break

        has_sales = any(s in df.columns for s in ["Sales", "Sales ($)", "Revenue", "Inventory Value"])
        if "Unit Cost" not in col_map and has_sales and ("Units" in col_map or "Hits" in df.columns):
            col_map["Unit Cost"] = "__derived__"

        missing = [c for c in required_columns if c not in col_map]
        if missing:
            return None, None, missing
    else:
        col_map = {c: c for c in required_columns}

    processed_df = df.copy()

    # Text fields: Ensure text fields are stringified and nulls handled cleanly
    if col_map.get("Product") != "Product":
        processed_df["Product"] = processed_df[col_map["Product"]].fillna("").astype(str)
    else:
        processed_df["Product"] = processed_df["Product"].fillna("").astype(str)

    if col_map.get("Category") != "Category":
        processed_df["Category"] = processed_df[col_map["Category"]].fillna("").astype(str)
    else:
        processed_df["Category"] = processed_df["Category"].fillna("").astype(str)

    if "Item Name" in processed_df.columns:
        processed_df["Item Name"] = processed_df["Item Name"].fillna("").astype(str)
    if "UoM" in processed_df.columns:
        processed_df["UoM"] = processed_df["UoM"].fillna("").astype(str)

    # Coerce Units to numeric
    if col_map.get("Units") != "Units":
        processed_df["Units"] = coerce_numeric_series(processed_df[col_map["Units"]], column_name="Units")
    else:
        processed_df["Units"] = coerce_numeric_series(processed_df["Units"], column_name="Units")

    # Correctly convert Sales ($) and Hits columns to numeric data types
    for col in ["Sales ($)", "Sales", "Total Sales", "Annual Sales", "Revenue", "Hits", "POS Hits", "POS_Hits", "Item Number"]:
        if col in processed_df.columns:
            processed_df[col] = coerce_numeric_series(processed_df[col], column_name=col)

    # Unit Cost derivation or coercion
    if col_map.get("Unit Cost") == "__derived__":
        sales_col = next(s for s in ["Sales ($)", "Sales", "Revenue", "Inventory Value"] if s in df.columns)
        processed_df["Unit Cost"] = (
            processed_df[sales_col] / processed_df["Units"].replace(0, 1)
        ).round(2)
    else:
        if col_map.get("Unit Cost") != "Unit Cost":
            processed_df["Unit Cost"] = coerce_numeric_series(processed_df[col_map["Unit Cost"]], column_name="Unit Cost")
        else:
            processed_df["Unit Cost"] = coerce_numeric_series(processed_df["Unit Cost"], column_name="Unit Cost")

    # 1. Total Units
    total_units = float(processed_df["Units"].sum())

    # 2. Average Unit Cost
    avg_unit_cost = float(processed_df["Unit Cost"].mean()) if len(processed_df) > 0 else 0.0

    # 3. Product with highest Units
    if len(processed_df) > 0:
        max_idx = processed_df["Units"].idxmax()
        highest_units_product = str(processed_df.loc[max_idx, "Product"])
        highest_units_count = float(processed_df.loc[max_idx, "Units"])
    else:
        highest_units_product = "N/A"
        highest_units_count = 0.0

    # 4. Group data by Category and calculate total Units
    category_units = processed_df.groupby("Category")["Units"].sum().reset_index()
    category_units.columns = ["Category", "Total Units"]
    category_units = category_units.sort_values(by="Total Units", ascending=False)

    # 5. Create new calculated column: Inventory Value = Units * Unit Cost
    sales_origin = next((s for s in ["Sales ($)", "Sales"] if s in processed_df.columns), None)
    if sales_origin and col_map.get("Unit Cost") == "__derived__":
        processed_df["Inventory Value"] = processed_df[sales_origin]
    else:
        processed_df["Inventory Value"] = processed_df["Units"] * processed_df["Unit Cost"]

    # 6. Calculate total Inventory Value across all products
    total_inventory_value = float(processed_df["Inventory Value"].sum())

    # 7. Execute Multicriteria Inventory Classification (65% Sales / 35% POS Hits)
    sales_col_for_mc = sales_origin if sales_origin else "Inventory Value"
    hits_col_for_mc = next((h for h in ["POS Hits", "POS_Hits", "Hits"] if h in processed_df.columns), "Units")

    mc_result = None
    mc_error = None
    try:
        mc_result = calculate_multicriteria_classification(
            processed_df,
            sales_col=sales_col_for_mc,
            pos_hits_col=hits_col_for_mc,
            product_col="Product",
            category_col="Category",
            sales_weight=0.65,
            pos_weight=0.35,
            pareto_cutoffs=(0.60, 0.80, 0.90),
            score_thresholds=(3.25, 2.50, 1.75),
        )
        for col in [
            "Sales_Pareto_CumPct",
            "Sales_Class",
            "Sales_Score",
            "POS_Pareto_CumPct",
            "POS_Class",
            "POS_Score",
            "Combined_Score",
            "Final_Class",
        ]:
            if col in mc_result.classified_df.columns:
                processed_df[col] = mc_result.classified_df[col]
    except Exception as err:
        mc_result = None
        mc_error = str(err)

    results = {
        "num_rows": int(len(df)),
        "num_cols": int(len(df.columns)),
        "column_names": list(df.columns),
        "total_units": total_units,
        "avg_unit_cost": avg_unit_cost,
        "highest_units_product": highest_units_product,
        "highest_units_count": highest_units_count,
        "category_units": category_units,
        "total_inventory_value": total_inventory_value,
        "multicriteria": mc_result,
        "multicriteria_error": mc_error,
        "sales_col_for_mc": sales_col_for_mc,
        "hits_col_for_mc": hits_col_for_mc,
    }

    return processed_df, results, None



# =====================================================================
# Streamlit Web Dashboard Mode
# =====================================================================

def run_streamlit_app():
    """Render full Streamlit interactive dashboard with decision agent and CSV data analysis."""
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📦 Supply Chain Analytics Inventory Decision Agent")
    st.markdown(
        """
        An autonomous decision support agent that accepts natural-language inventory questions,
        identifies the appropriate supply chain optimization model, executes pure Python calculations,
        verifies mathematical optimality conditions, and provides executive recommendations.
        """
    )

    # Pipeline Diagram
    st.info(
        "**Architecture**: 🗣️ Customer Question ➔ 🔍 **Identify Model** ➔ 📥 **Extract Inputs** "
        "➔ ⚙️ **Select Python Tool** ➔ 🧮 **Calculate** ➔ 🛡️ **Verify** ➔ 📊 **Explain**"
    )

    # Force reload of underlying agent and tools modules so hot-reloads always use latest code
    import importlib
    import inventory_agent
    import inventory_tools
    importlib.reload(inventory_tools)
    importlib.reload(inventory_agent)
    from inventory_agent import InventoryDecisionAgent, InventoryModelType

    # Always instantiate fresh agent
    agent = InventoryDecisionAgent()
    st.session_state.agent = agent

    # Sidebar: Case studies selector & configuration
    st.sidebar.header("🎯 Case Study Selector")
    st.sidebar.markdown("Choose a pre-configured problem or type your own custom question:")

    case_options = ["(Custom Question)"] + [f"{s['model']}: Case Study" for s in DEMO_SCENARIOS]
    selected_case = st.sidebar.selectbox("Load Example Scenario", case_options)

    default_question = ""
    if selected_case != "(Custom Question)":
        idx = case_options.index(selected_case) - 1
        default_question = DEMO_SCENARIOS[idx]["question"]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Supported Decision Models")
    st.sidebar.markdown(
        """
        1. **EOQ (Economic Order Quantity)**
        2. **Quantity Discount Optimization**
        3. **Safety Stock & Reorder Point (ROP)**
        4. **Fixed-Period Review (Order-Up-To)**
        5. **Newsvendor Single-Period Stochastic**
        """
    )

    # Sidebar: Data Upload Section (last section in the left sidebar)
    st.sidebar.markdown("---")
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    # Main Query Input
    query_text = st.text_area(
        "Enter your natural language inventory inquiry:",
        value=default_question,
        height=130,
        placeholder="e.g. We sell 10,000 units per year. Each order costs $50, and holding cost is $2 per unit per year. What is our optimal order quantity?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        run_btn = st.button("🚀 Analyze & Optimize", type="primary")

    if run_btn and query_text.strip():
        with st.spinner("Processing inquiry through 6-stage agent pipeline..."):
            agent = InventoryDecisionAgent()
            st.session_state.agent = agent
            resp = agent.run(query_text.strip())

        # Render Pipeline Execution Status
        with st.expander("🛠️ Agent Architecture Pipeline Execution Log", expanded=True):
            step_cols = st.columns(len(resp.steps_log))
            for i, step in enumerate(resp.steps_log):
                with step_cols[i]:
                    status_icon = "✅" if step.status == "SUCCESS" else ("⚠️" if step.status == "WARNING" else "❌")
                    st.metric(label=f"Step {i+1}", value=step.step_name, delta=f"{status_icon} {step.status}")
                    st.caption(step.summary)

        # Highlight Model Badge
        st.markdown(f"### 📋 Model: `{resp.identified_model.value}` (Confidence: `{resp.confidence:.0%}`)")

        # Render Explanation Markdown
        st.markdown(resp.explanation)

        # Download Report
        st.download_button(
            label="📥 Download Decision Report (.md)",
            data=resp.explanation,
            file_name=f"inventory_decision_report_{resp.identified_model.name.lower()}.md",
            mime="text/markdown",
        )

    # -----------------------------------------------------------------
    # Uploaded CSV Data Processing & Analysis Section
    # -----------------------------------------------------------------
    if uploaded_file is not None:
        st.markdown("---")
        st.header("📊 Inventory Data Upload & Analysis")

        try:
            raw_df = pd.read_csv(uploaded_file)
            processed_df, analysis, missing = process_inventory_dataframe(raw_df)

            if missing:
                st.error(
                    f"⚠️ **Validation Error: Missing Required Columns**\n\n"
                    f"The uploaded CSV file is missing the following required column(s): "
                    f"**{', '.join(missing)}**.\n\n"
                    f"Please ensure your CSV includes all four required columns: `Product`, `Category`, `Units`, `Unit Cost`."
                )
            else:
                # 1. Dataset Overview: rows, columns, column names
                st.subheader("📋 Dataset Overview")
                d_col1, d_col2, d_col3 = st.columns(3)
                d_col1.metric("Number of Rows", f"{analysis['num_rows']:,}")
                d_col2.metric("Number of Columns", f"{analysis['num_cols']}")
                d_col3.metric("Column Names", ", ".join(analysis["column_names"]))

                # 2. Key Results Summary:
                # - Total Units
                # - Average Unit Cost
                # - Product with Highest Units
                # - Total Inventory Value
                st.subheader("📈 Key Inventory Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Units", f"{analysis['total_units']:,.0f}")
                m2.metric("Average Unit Cost", f"${analysis['avg_unit_cost']:,.2f}")
                m3.metric("Product with Highest Units", f"{analysis['highest_units_product']}")
                m4.metric("Total Inventory Value", f"${analysis['total_inventory_value']:,.2f}")

                # 3. Total Units by Category
                st.subheader("🏷️ Total Units by Category")
                cat_col1, cat_col2 = st.columns([1, 1])
                with cat_col1:
                    cat_fmt = build_column_format_dict(analysis["category_units"], ["Total Units", "Units"])
                    st.dataframe(
                        analysis["category_units"].style.format(cat_fmt, na_rep="-"),
                        use_container_width=True,
                    )
                with cat_col2:
                    st.bar_chart(
                        data=analysis["category_units"].set_index("Category")["Total Units"],
                    )

                # 4. Uploaded DataFrame with calculated Inventory Value (formatted as currency)
                st.subheader("📦 Inventory Data with Calculated Inventory Value")
                st.caption("Calculated Column: Inventory Value = Units × Unit Cost")

                overview_cols = [c for c in ["Units", "Total Units", "Unit Cost", "Inventory Value", "Sales ($)", "Sales", "Hits", "POS Hits"] if c in processed_df.columns]
                overview_fmt = build_column_format_dict(processed_df, overview_cols)
                st.dataframe(
                    processed_df.style.format(overview_fmt, na_rep="-"),
                    use_container_width=True,
                )

                # -------------------------------------------------------------
                # 5. Multicriteria Inventory Classification (65% Sales / 35% POS Hits)
                # -------------------------------------------------------------
                mc_result = analysis.get("multicriteria")
                if mc_result is not None:
                    st.markdown("---")
                    st.header("🎯 Multicriteria Inventory Classification (65% Sales / 35% POS Hits)")
                    st.markdown(
                        """
                        **BAN517 Multicriteria Course Methodology**:
                        - **STEP 1: Separate Pareto Classifications**:
                          - Class A: Cumulative percentage from **0% through 60%** (cutoff 0.60)
                          - Class B: **Above 60% through 80%** (cutoff 0.80)
                          - Class C: **Above 80% through 90%** (cutoff 0.90)
                          - Class D: **Above 90% through 100%** (tail / inactive items)
                        - **STEP 2: Numerical Scoring**: **A = 4**, **B = 3**, **C = 2**, **D = 1**
                        - **STEP 3: Combined Weighted Score**: $\\text{Combined Weighted Score} = (0.65 \\times \\text{Sales Score}) + (0.35 \\times \\text{POS Hits Score})$
                        - **STEP 4: Final Inventory Classification**:
                          - **Final Class A**: $\\text{Score} \\ge 3.25 \\text{ and } \\le 4.00$ (Critical strategic priority; perpetual review; high service target)
                          - **Final Class B**: $\\text{Score} \\ge 2.50 \\text{ and } < 3.25$ (Moderate priority; periodic review; standard safety stock)
                          - **Final Class C**: $\\text{Score} \\ge 1.75 \\text{ and } < 2.50$ (Low priority; bulk/economic batches; automated replenishment)
                          - **Final Class D**: $\\text{Score} < 1.75$ (Tail / slow-moving; rationalization / obsolescence review)
                        """
                    )

                    fa = mc_result.final_answer

                    # 5a. Executive Metrics Summary by Final Classification
                    st.subheader("📌 Executive Classification Summary")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric(
                        "Class A (Strategic)",
                        f"{fa['class_a_items']:,} Items ({fa['summary_table'][0]['Pct_Items']:.1f}%)",
                        f"${fa['class_a_sales']:,.2f} ({fa['class_a_pct_sales']:.1f}% Sales)",
                    )
                    mc2.metric(
                        "Class B (Standard)",
                        f"{fa['class_b_items']:,} Items ({fa['summary_table'][1]['Pct_Items']:.1f}%)",
                        f"${fa['class_b_sales']:,.2f} ({fa['class_b_pct_sales']:.1f}% Sales)",
                    )
                    mc3.metric(
                        "Class C (Low Touch)",
                        f"{fa['class_c_items']:,} Items ({fa['summary_table'][2]['Pct_Items']:.1f}%)",
                        f"${fa['class_c_sales']:,.2f} ({fa['class_c_pct_sales']:.1f}% Sales)",
                    )
                    mc4.metric(
                        "Class D (Rationalize / Slow)",
                        f"{fa['class_d_items']:,} Items ({fa['summary_table'][3]['Pct_Items']:.1f}%)",
                        f"${fa['class_d_sales']:,.2f} ({fa['class_d_pct_sales']:.1f}% Sales)",
                    )

                    # 5b. Summary Table: Items Count and Total Sales for Each Final Classification
                    st.subheader("📋 Classification Breakdown & Strategic Policies")
                    st.caption("Summary of the number of items and total Sales for each final multicriteria classification:")
                    summary_display = mc_result.summary_df.copy()
                    sum_fmt = build_column_format_dict(summary_display, list(summary_display.columns))
                    st.dataframe(
                        summary_display.style.format(sum_fmt, na_rep="-"),
                        use_container_width=True,
                    )

                    # 5c. Comparative Visualizations
                    ch_col1, ch_col2 = st.columns(2)
                    with ch_col1:
                        st.subheader("💵 Total Sales ($) by Classification")
                        st.bar_chart(
                            data=summary_display.set_index("Final_Class")["Total_Sales"],
                        )
                    with ch_col2:
                        st.subheader("📦 Number of Items by Classification")
                        st.bar_chart(
                            data=summary_display.set_index("Final_Class")["Items_Count"],
                        )

                    # 5d. Interactive Filterable Classified Catalog
                    st.subheader("🔍 Detailed Item-by-Item Classification Results")
                    filter_col, info_col = st.columns([1, 2])
                    with filter_col:
                        selected_class_filter = st.selectbox(
                            "Filter by Final Classification:",
                            ["All", "Class A", "Class B", "Class C", "Class D"],
                        )

                    display_classified = processed_df.copy()
                    if selected_class_filter != "All":
                        chosen_tier = selected_class_filter.replace("Class ", "")
                        display_classified = display_classified[display_classified["Final_Class"] == chosen_tier]

                    preferred_order = [
                        "Product",
                        "Category",
                        analysis.get("sales_col_for_mc", "Sales"),
                        "Sales_Class",
                        "Sales_Score",
                        analysis.get("hits_col_for_mc", "POS Hits"),
                        "POS_Class",
                        "POS_Score",
                        "Combined_Score",
                        "Final_Class",
                    ]
                    available_cols = [c for c in preferred_order if c in display_classified.columns]
                    for c in display_classified.columns:
                        if c not in available_cols and not c.endswith("_CumPct"):
                            available_cols.append(c)

                    format_dict = build_column_format_dict(display_classified, available_cols)
                    st.dataframe(
                        display_classified[available_cols].style.format(format_dict, na_rep="-"),
                        use_container_width=True,
                    )

                    # 5e. Download Classified CSV
                    csv_bytes = processed_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Full Classified Inventory Dataset (.csv)",
                        data=csv_bytes,
                        file_name="multicriteria_inventory_classification.csv",
                        mime="text/csv",
                    )
                elif analysis.get("multicriteria_error"):
                    st.warning(f"⚠️ Multicriteria Inventory Classification could not be calculated: {analysis.get('multicriteria_error')}")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            err_type = type(e).__name__
            err_msg = str(e)
            st.error(f"❌ **Error processing uploaded CSV file ({err_type})**: {err_msg}")
            with st.expander("🔍 Diagnostic Error Details & Traceback", expanded=False):
                st.code(tb, language="python")


# =====================================================================
# Main Application Entry Point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
    parser.add_argument("--query", type=str, help="Run single natural language query and output result")
    parser.add_argument("--demo", action="store_true", help="Run benchmark case study suite across all 5 models")
    parser.add_argument("--web", action="store_true", help="Launch interactive Streamlit web application")

    args, unknown = parser.parse_known_args()

    # Detect if invoked through `streamlit run app.py`
    if any("streamlit" in arg for arg in sys.argv) or args.web:
        if args.web and not any("streamlit" in arg for arg in sys.argv):
            # Launch streamlit subprocess
            cmd = [sys.executable, "-m", "streamlit", "run", __file__] + unknown
            subprocess.run(cmd)
            return
        run_streamlit_app()
        return

    agent = InventoryDecisionAgent()

    if args.query:
        print_banner()
        resp = agent.run(args.query)
        display_agent_response(resp)
    elif args.demo:
        run_demo_suite(agent)
    else:
        run_interactive_cli(agent)


if __name__ == "__main__":
    main()

