# app.py

import streamlit as st
import pandas as pd
import re
from bs4 import BeautifulSoup

from backend import run_full_pipeline

st.set_page_config(page_title="[ai]udit – AI Tool for Tracking Blocks", layout="wide")

st.title("[ai]udit – AI Tool for Tracking Blocks")
st.write(
    "Paste your publisher app IDs in the sidebar, add optional exclusions, and your email + Claude API key. "
    "The tool will run the full analysis, optionally email the AI summary, and show detailed tables below."
)

# =====================================================================
# Helper: format spend / revenue columns as $ with no decimals
# =====================================================================
def format_money_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format any spend/revenue column into $#,### (no decimals).
    Now also includes *_rev_l7d columns used in the competitor matrix.
    Ensures the column stays as string so Streamlit doesn't override.
    Skips any column whose name contains 'rank'.
    """
    if df is None or df.empty:
        return df

    df_fmt = df.copy()
    money_cols = [
        c
        for c in df_fmt.columns
        if (
            any(key in c.lower() for key in ["spend", "revenue"])
            or c.lower().endswith("_rev_l7d")  # revenue columns in Table 4
        )
        and "rank" not in c.lower()  # avoid e.g. rank_missed_spend
    ]

    for col in money_cols:
        # Only numeric columns should be formatted
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].apply(
                lambda x: f"${int(round(x)):,}" if pd.notnull(x) else ""
            )
        # Force column to string to prevent Streamlit from auto-formatting
        df_fmt[col] = df_fmt[col].astype(str)

    return df_fmt


# =====================================================================
# Helper: specific formatting for Block Summary table (Table 3)
# =====================================================================
def format_block_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For block summary tables:
      - block_rate_diff_vs_peers: whole number (no decimals)
      - rank_* columns: integer (no $ formatting)
    Spend columns are still formatted via format_money_columns().
    """
    if df is None or df.empty:
        return df

    df_fmt = df.copy()

    # 1) block_rate_diff_vs_peers as whole number
    if "block_rate_diff_vs_peers" in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt["block_rate_diff_vs_peers"]):
            df_fmt["block_rate_diff_vs_peers"] = (
                df_fmt["block_rate_diff_vs_peers"].round(0).astype("Int64")
            )

    # 2) Any rank column as Int64
    for col in df_fmt.columns:
        if "rank" in col.lower():
            if pd.api.types.is_numeric_dtype(df_fmt[col]):
                df_fmt[col] = df_fmt[col].astype("Int64")

    return df_fmt


# =====================================================================
# Helper: format Summary Metrics table
# =====================================================================
def format_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    In summary_metrics:
      - Row 1, 2, 4 values are money -> $#,###
      - Row 3 is a count of blocks -> integer, no $
    Uses 'metric' column text to decide.
    """
    if df is None or df.empty:
        return df

    df_fmt = df.copy()

    money_metrics = {
        "Total L7D non-VX DSP spend across all blocks",
        "Total L30D global network advertiser spend across all blocks",
        "Total L7D competitor/similar apps revenue across all blocks",
    }

    if "metric" in df_fmt.columns and "value" in df_fmt.columns:
        def fmt_value(row):
            metric = row["metric"]
            val = row["value"]

            if pd.isna(val):
                return ""

            # Money rows: 1,2,4
            if metric in money_metrics:
                try:
                    return f"${int(round(float(val))):,}"
                except Exception:
                    return val

            # Count row (number of blocks)
            if "Number of blocks" in str(metric):
                try:
                    return int(round(float(val)))
                except Exception:
                    return val

            return val

        df_fmt["value"] = df_fmt.apply(fmt_value, axis=1).astype(str)

    return df_fmt


# =====================================================================
# Helper: render tables at narrower width
# =====================================================================
def render_narrow_table(df: pd.DataFrame, title: str = None, width_pct: int = 70):
    """
    Renders a table at a reduced width by wrapping in an HTML div.
    width_pct = 70 means 70% width relative to Streamlit container.
    """
    if df is None or df.empty:
        if title:
            st.markdown(f"**{title}**")
            st.caption("No data.")
        return

    if title:
        st.markdown(f"**{title}**")

    html = f"""
    <div style="width:{width_pct}%; margin-left:0;">
        {df.to_html(index=False)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =====================================================================
# Helper: parse AI summary HTML into native bullets + tables
# =====================================================================
def parse_ai_summary_html(html_str: str):
    """
    Parses the HTML summary from Claude into:
      - key_insights: list[str]
      - opportunities: list[str]
      - tables: list[(title, DataFrame)]
    """
    result = {"key_insights": [], "opportunities": [], "tables": []}
    if not html_str:
        return result

    soup = BeautifulSoup(html_str, "html.parser")

    # Key insights
    key_h3 = soup.find("h3", string=lambda s: s and "Key insights" in s)
    if key_h3:
        ul = key_h3.find_next("ul")
        if ul:
            result["key_insights"] = [
                li.get_text(strip=True) for li in ul.find_all("li")
            ]

    # Opportunities
    opp_h3 = soup.find("h3", string=lambda s: s and "Opportunities" in s)
    if opp_h3:
        ul = opp_h3.find_next("ul")
        if ul:
            result["opportunities"] = [
                li.get_text(strip=True) for li in ul.find_all("li")
            ]

    # Tables
    for table in soup.find_all("table"):
        # Try to get title from preceding <b> tag
        title_tag = table.find_previous("b")
        title = title_tag.get_text(strip=True) if title_tag else "Table"

        rows = []
        headers = []
        for i, tr in enumerate(table.find_all("tr")):
            cells = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
            if i == 0:
                headers = cells
            else:
                rows.append(cells)
        try:
            df = pd.DataFrame(rows, columns=headers if headers else None)
        except Exception:
            df = pd.DataFrame(rows)
        result["tables"].append((title, df))

    return result


# =====================================================================
# Helper: parse app IDs (Looker-ish UX)
# =====================================================================
def parse_app_ids(raw: str):
    """
    Accepts one-per-line or comma-/space-separated app IDs.
    Returns a list of cleaned IDs.
    """
    if not raw:
        return []
    tokens = re.split(r"[,\s]+", raw)
    return [t.strip() for t in tokens if t.strip()]


# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
with st.sidebar:
    st.header("Inputs")

    app_ids_input = st.text_area(
        "Publisher app IDs",
        placeholder="Paste IDs here:\n632cc7810ca02c6344d51822\n632cc70d35cc2d93ebf3b2d5\n...",
        height=140,
    )

    excluded_input = st.text_area(
        "Excluded block values (optional)",
        placeholder="dreamgames.com\n1234567890",
        height=100,
        help="Use to ignore specific advertiser domains or app market IDs in the block analysis.",
    )

    # Name kept as openai_api_key to match backend signature,
    # but label/help clearly indicate it's the Claude / Anthropic key.
    claude_api_key = st.text_input(
        "Claude API key",
        type="password",
        help=(
            "Paste your Claude (Anthropic) API key here (e.g. starts with `sk-ant-`). "
            "Leave blank to skip AI summary and only see tables."
        ),
    )

    recipient_email = st.text_input(
        "Recipient email (AI summary will be sent here)",
        value="",
    )

    gmail_app_password = st.text_input(
        "Gmail App Password",
        type="password",
        help=(
            "16-character Gmail App Password for sender `ssai@liftoff.io`. "
            "Leave blank to skip sending email and only show tables."
        ),
    )

    st.markdown("---")
    st.subheader("Scheduler (WIP)")

    scheduler_enabled = st.checkbox(
        "Enable schedule (WIP)",
        value=False,
        help="Visual only – does not actually schedule background runs yet.",
    )

    schedule_frequency = st.selectbox(
        "Frequency (WIP)",
        ["None", "Daily", "Weekly", "Monthly"],
        index=0,
        help="Future: how often the audit would run.",
    )

    schedule_time = st.time_input(
        "Preferred time (local, WIP)",
        help="Future: time of day for the scheduled audit.",
    )

    weekly_day = None
    if schedule_frequency == "Weekly":
        weekly_day = st.selectbox(
            "Day of week (WIP)",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=0,
            help="Future: weekly send-out day.",
        )

    run_button = st.button("Run audit & (optionally) send email", type="primary")

# Placeholder for results
results = None

# -------------------------------
# MAIN AREA – INTRO
# -------------------------------
st.markdown("### How to use this tool")
st.markdown(
    """
1. Paste one or more **publisher app IDs** in the sidebar (one per line or comma-separated).  
2. Optionally add **excluded domains / app market IDs** (e.g. house ads or test partners).  
3. Add your **Claude API key** if you want an AI-written email summary.  
4. Add **recipient email** and Gmail App Password if you want the summary emailed.  
5. Click **Run audit** to see combined and per-app tables.
"""
)

# -------------------------------
# RUN PIPELINE
# -------------------------------
if run_button:
    # Basic validation
    target_app_ids = parse_app_ids(app_ids_input)
    excluded_block_values = [
        x.strip() for x in re.split(r"[,\n]+", excluded_input) if x.strip()
    ]

    if not target_app_ids:
        st.error("Please provide at least one publisher app ID in the sidebar.")
    else:
        with st.spinner("Running full analysis..."):
            try:
                results = run_full_pipeline(
                    target_app_ids=target_app_ids,
                    excluded_block_values=excluded_block_values,
                    recipient_email=recipient_email,
                    # sender_email handled in backend (defaults to ssai@liftoff.io)
                    sender_email="",
                    gmail_app_password=gmail_app_password,
                    openai_api_key=claude_api_key,  # passes Claude key through
                )
            except Exception as e:
                st.error(f"Something went wrong while running the pipeline: {e}")
                results = None

# -------------------------------
# OUTPUTS
# -------------------------------
if results is not None:
    st.success("Analysis completed.")

    # Show a human-readable summary of the (fake) scheduler config
    if scheduler_enabled and schedule_frequency != "None":
        if schedule_frequency == "Weekly" and weekly_day:
            sched_text = (
                f"Scheduler (placeholder): would run **{schedule_frequency.lower()} on {weekly_day}** "
                f"around **{schedule_time.strftime('%H:%M')}**."
            )
        else:
            sched_text = (
                f"Scheduler (placeholder): would run **{schedule_frequency.lower()}** "
                f"around **{schedule_time.strftime('%H:%M')}**."
            )
        st.info(sched_text)
    else:
        st.caption(
            "Scheduler WIP: enable it in the sidebar and choose a frequency/time to see the planned run cadence."
        )

    # Email / AI summary status
    email_status = results.get("email_status")
    ai_error = results.get("ai_error")

    if email_status == "email_sent":
        st.info(f"Claude summary generated and email sent to **{recipient_email}**.")
    elif email_status == "summary_built":
        st.info("Claude summary generated (email not sent – missing Gmail app password or recipient).")
    elif email_status == "failed_ai_or_email":
        st.warning("Claude summary or email failed. Showing tables only.")
        if ai_error:
            with st.expander("Show AI/email error details"):
                st.code(ai_error, language="text")
    elif email_status == "ai_not_configured":
        st.info("Claude API key not provided – skipping AI summary/email and showing tables only.")
    else:
        st.info("AI summary/email not attempted (unknown status).")

    # ---------- TABS ----------
    tab_summary, tab_combined, tab_per_app, tab_metrics = st.tabs(
        ["AI Summary", "Combined Tables", "Per-App Tables", "Summary Metrics"]
    )

    # ---------- AI SUMMARY TAB ----------
    with tab_summary:
        html_summary = results.get("html_summary")

        if not html_summary:
            st.info("No AI summary available. Provide a valid Claude API key to generate one.")
        else:
            parsed = parse_ai_summary_html(html_summary)
            key_insights = parsed["key_insights"]
            opportunities = parsed["opportunities"]
            tables = parsed["tables"]

            st.subheader("Key insights")
            if key_insights:
                for bullet in key_insights:
                    st.markdown(f"- {bullet}")
            else:
                st.caption("No key insights found in summary.")

            st.subheader("Opportunities")
            if opportunities:
                for bullet in opportunities:
                    st.markdown(f"- {bullet}")
            else:
                st.caption("No opportunities found in summary.")

            if tables:
                st.markdown("#### Top tables from summary")
                for title, df in tables:
                    render_narrow_table(df, title=title, width_pct=70)

            with st.expander("View raw email HTML (debug)", expanded=False):
                st.code(html_summary, language="html")

    # ---------- COMBINED TABLES ----------
    with tab_combined:
        st.subheader("Combined Tables (All Selected Apps)")

        with st.expander("Legend (similar app mappings)", expanded=False):
            st.code(results["combined_legend"], language="text")

        render_narrow_table(
            format_money_columns(results["combined_blocks_with_spend"].head(50)),
            title="1. Blocks enriched with L7D DSP spend (app + domain)",
            width_pct=70,
        )

        render_narrow_table(
            format_money_columns(results["combined_blocks_with_global"].head(50)),
            title="2. Global advertiser network spend (L30D) for blocks",
            width_pct=70,
        )

        render_narrow_table(
            format_money_columns(
                format_block_summary_table(results["combined_summary_our"].head(50))
            ),
            title="3. Block summary per app (our apps only)",
            width_pct=70,
        )

        render_narrow_table(
            format_money_columns(results["combined_rev_matrix"].head(50)),
            title="4. Competitor revenue matrix (L7D per similar app)",
            width_pct=70,
        )

    # ---------- PER-APP TABLES ----------
    with tab_per_app:
        st.subheader("Per-App Detailed Tables")

        per_app_results = results.get("per_app_results", {})
        if not per_app_results:
            st.write("No per-app results available.")
        else:
            for app_id, tables in per_app_results.items():
                with st.expander(f"Publisher app: {app_id}", expanded=False):
                    legend_text = tables.get("legend_text", "")
                    if legend_text:
                        with st.expander("Similar app legend", expanded=False):
                            st.code(legend_text, language="text")

                    render_narrow_table(
                        format_money_columns(tables["blocks_with_spend"].head(20)),
                        title="Blocks enriched with L7D DSP spend (app + domain)",
                        width_pct=70,
                    )

                    render_narrow_table(
                        format_money_columns(tables["blocks_with_global"].head(20)),
                        title="Global advertiser network spend (L30D) for blocks",
                        width_pct=70,
                    )

                    render_narrow_table(
                        format_money_columns(
                            format_block_summary_table(tables["summary_per_app"].head(20))
                        ),
                        title="Block summary per app (advertiser L30D spend > 30,000)",
                        width_pct=70,
                    )

                    render_narrow_table(
                        format_money_columns(tables["competitor_rev_matrix"].head(20)),
                        title="Competitor revenue matrix (L7D per similar app)",
                        width_pct=70,
                    )

    # ---------- SUMMARY METRICS ----------
    with tab_metrics:
        st.subheader("High-level Summary Metrics")
        render_narrow_table(
            format_summary_metrics(results["summary_metrics"]),
            title="Summary Metrics",
            width_pct=70,
        )
        st.caption(
            "These aggregates are also used as input to the Claude summary (lost spend, competitor revenue, etc.)."
        )

else:
    st.info("Use the sidebar to configure inputs and click **Run audit & (optionally) send email** to start.")
