# app.py

import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

from backend import run_full_pipeline

st.set_page_config(page_title="[ai]udit – AI Tool for Tracking Blocks", layout="wide")

st.title("[ai]udit – AI Tool for Tracking Blocks")
st.write(
    "Enter your publisher app IDs on the left, optionally configure exclusions and email + Claude API key, "
    "and run the audit to see block, spend, and opportunity insights."
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
# Helper: interactive table with st-aggrid
# =====================================================================
def render_interactive_table(df: pd.DataFrame, height: int = 400, key: str = None):
    """
    Render an interactive, sortable, filterable table using st-aggrid.
    """
    if df is None or df.empty:
        st.write("No data to display.")
        return

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(
        sortable=True,
        filter=True,
        resizable=True,
    )
    gb.configure_grid_options(domLayout="normal")
    grid_options = gb.build()

    AgGrid(
        df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.NO_UPDATE,
        fit_columns_on_grid_load=True,
        height=height,
        key=key,
        enable_enterprise_modules=False,
        allow_unsafe_jscode=False,
    )


def render_centered_table(df: pd.DataFrame, title: str, key: str, height: int = 400):
    """
    Render a table centered on the page with ~70% width.
    """
    st.markdown(f"### {title}")
    left, mid, right = st.columns([0.15, 0.7, 0.15])
    with mid:
        render_interactive_table(df, height=height, key=key)


# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
with st.sidebar:
    st.header("Inputs")

    app_ids_input = st.text_area(
        "Publisher app IDs",
        placeholder="Paste one app ID per line\n632cc7810ca02c6344d51822\n632cc70d35cc2d93ebf3b2d5\n...",
        height=140,
        help="You can paste a column from Sheets/Looker. Newlines or commas are both accepted.",
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
        "Recipient email (summary will be sent here)",
        value="",
    )

    # Sender Gmail is fixed in code to ssai@liftoff.io
    st.text_input(
        "Sender Gmail (fixed)",
        value="ssai@liftoff.io",
        disabled=True,
        help="Sender is fixed for now; using ssai@liftoff.io with Gmail App Password below.",
    )

    gmail_app_password = st.text_input(
        "Gmail App Password",
        type="password",
        help="16-character Gmail App Password for ssai@liftoff.io. Leave blank to skip sending email and only show tables.",
    )

    st.markdown("### Scheduler (WIP)")

    st.caption(
        "Prototype controls for future automation. These settings are **not yet scheduling anything**, "
        "but can be used in demos to show how recurring audits might be configured."
    )

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
# RUN PIPELINE
# -------------------------------
if run_button:
    # Basic validation
    # Accept both newline and comma separated IDs (Looker-style paste)
    ids_normalized = app_ids_input.replace("\r", "\n").replace(",", "\n")
    target_app_ids = [x.strip() for x in ids_normalized.split("\n") if x.strip()]

    excluded_normalized = excluded_input.replace("\r", "\n").replace(",", "\n")
    excluded_block_values = [x.strip() for x in excluded_normalized.split("\n") if x.strip()]

    if not target_app_ids:
        st.error("Please provide at least one publisher app ID.")
    else:
        with st.spinner("Running full analysis..."):
            try:
                # Sender is fixed here
                sender_email = "ssai@liftoff.io"

                results = run_full_pipeline(
                    target_app_ids=target_app_ids,
                    excluded_block_values=excluded_block_values,
                    recipient_email=recipient_email,
                    sender_email=sender_email,
                    gmail_app_password=gmail_app_password,
                    openai_api_key=claude_api_key,  # passes Claude key through
                )
            except Exception as e:
                st.error(f"Something went wrong while running the pipeline: {e}")
                results = None

# -------------------------------
# MAIN OUTPUTS
# -------------------------------
st.markdown("### Enter your app IDs to track blocks")
st.caption(
    "Paste one or more app IDs in the sidebar, configure any exclusions and email, then click "
    "**Run audit & (optionally) send email**."
)

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
        st.info("Claude summary generated (email not sent – missing Gmail App Password or recipient).")
    elif email_status == "failed_ai_or_email":
        st.warning("Claude summary or email failed. Showing tables only.")
        if ai_error:
            with st.expander("Show AI/email error details"):
                st.code(ai_error, language="text")
    elif email_status == "ai_not_configured":
        st.info("Claude API key not provided – skipping AI summary/email and showing tables only.")
    else:
        st.info("AI summary/email not attempted (unknown status).")

    html_summary = results.get("html_summary")

    # ---------- TABS ----------
    tab_summary, tab_combined, tab_per_app, tab_metrics = st.tabs(
        ["AI Summary", "Combined Tables", "Per-App Tables", "Summary Metrics"]
    )

    # ----- AI Summary Tab -----
    with tab_summary:
        st.subheader("AI Summary (Claude)")
        if not html_summary:
            st.info("No AI summary available. Provide a valid Claude API key in the sidebar and rerun.")
        else:
            soup = BeautifulSoup(html_summary, "html.parser")

            # Render heading + bullet sections
            for h in soup.find_all("h3"):
                title = h.get_text(strip=True)
                st.markdown(f"#### {title}")
                ul = h.find_next_sibling("ul")
                if ul:
                    for li in ul.find_all("li"):
                        st.markdown(f"- {li.get_text(strip=True)}")

            # Render any tables from the HTML as native DataFrames with money formatting
            tables = soup.find_all("table")
            if tables:
                st.markdown("#### Tables from AI Summary")
                for i, table in enumerate(tables):
                    # Try to grab a title from the preceding <b> tag
                    title_tag = table.find_previous("b")
                    tbl_title = title_tag.get_text(strip=True) if title_tag else f"Table {i+1}"

                    rows = table.find_all("tr")
                    if not rows:
                        continue
                    header_cells = rows[0].find_all(["th", "td"])
                    headers = [hc.get_text(strip=True) for hc in header_cells]

                    data_rows = []
                    for tr in rows[1:]:
                        cells = tr.find_all(["td", "th"])
                        if cells:
                            data_rows.append([c.get_text(strip=True) for c in cells])

                    if not data_rows:
                        continue

                    df_tbl = pd.DataFrame(data_rows, columns=headers)
                    df_tbl = format_money_columns(df_tbl)

                    left, mid, right = st.columns([0.15, 0.7, 0.15])
                    with mid:
                        st.markdown(f"##### {tbl_title}")
                        render_interactive_table(df_tbl, height=260, key=f"ai_table_{i}")

            with st.expander("Show raw HTML email body"):
                st.code(html_summary, language="html")

    # ----- Combined Tables Tab -----
    with tab_combined:
        st.subheader("Combined Tables (All Selected Apps)")

        with st.expander("Legend (similar app mappings)", expanded=False):
            st.code(results["combined_legend"], language="text")

        # 1. Blocks with L7D DSP spend
        combined_blocks_with_spend = format_money_columns(
            results["combined_blocks_with_spend"].head(50)
        )
        render_centered_table(
            combined_blocks_with_spend,
            "1. Blocks enriched with L7D DSP spend (app + domain)",
            key="combined_blocks_with_spend",
        )

        # 2. Global advertiser network spend
        combined_blocks_with_global = format_money_columns(
            results["combined_blocks_with_global"].head(50)
        )
        render_centered_table(
            combined_blocks_with_global,
            "2. Global advertiser network spend (L30D) for blocks",
            key="combined_blocks_with_global",
        )

        # 3. Block summary per app
        combined_summary_our = format_money_columns(
            format_block_summary_table(
                results["combined_summary_our"].head(50)
            )
        )
        render_centered_table(
            combined_summary_our,
            "3. Block summary per app (our apps only)",
            key="combined_summary_our",
        )

        # 4. Competitor revenue matrix
        combined_rev_matrix = format_money_columns(
            results["combined_rev_matrix"].head(50)
        )
        render_centered_table(
            combined_rev_matrix,
            "4. Competitor revenue matrix (L7D per similar app)",
            key="combined_rev_matrix",
        )

    # ----- Per-App Tables Tab -----
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

                    # Blocks with L7D DSP spend
                    df_spend = format_money_columns(
                        tables["blocks_with_spend"].head(20)
                    )
                    render_centered_table(
                        df_spend,
                        "Blocks enriched with L7D DSP spend (app + domain)",
                        key=f"{app_id}_blocks_with_spend",
                        height=320,
                    )

                    # Global advertiser network spend
                    df_global = format_money_columns(
                        tables["blocks_with_global"].head(20)
                    )
                    render_centered_table(
                        df_global,
                        "Global advertiser network spend (L30D) for blocks",
                        key=f"{app_id}_blocks_with_global",
                        height=320,
                    )

                    # Block summary per app
                    df_summary = format_money_columns(
                        format_block_summary_table(
                            tables["summary_per_app"].head(20)
                        )
                    )
                    render_centered_table(
                        df_summary,
                        "Block summary per app (advertiser L30D spend > 30,000)",
                        key=f"{app_id}_summary_per_app",
                        height=320,
                    )

                    # Competitor revenue matrix
                    df_comp = format_money_columns(
                        tables["competitor_rev_matrix"].head(20)
                    )
                    render_centered_table(
                        df_comp,
                        "Competitor revenue matrix (L7D per similar app)",
                        key=f"{app_id}_competitor_rev_matrix",
                        height=320,
                    )

    # ----- Summary Metrics Tab -----
    with tab_metrics:
        st.subheader("High-level Summary Metrics")
        metrics_df = format_summary_metrics(results["summary_metrics"])

        left, mid, right = st.columns([0.15, 0.7, 0.15])
        with mid:
            render_interactive_table(metrics_df, height=220, key="summary_metrics")

        st.caption(
            "These aggregates are also used as input to the Claude summary (lost spend, competitor revenue, etc.)."
        )

else:
    st.info("Fill in the inputs in the left sidebar and click **Run audit & (optionally) send email** to start.")
