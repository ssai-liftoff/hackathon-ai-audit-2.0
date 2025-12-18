# app.py

import re
import streamlit as st
import pandas as pd

from backend import run_full_pipeline
from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    DataReturnMode,
)

st.set_page_config(page_title="[ai]udit – AI Tool for Tracking Blocks", layout="wide")

st.title("[ai]udit – AI Tool for Tracking Blocks")
st.write(
    "Enter publisher app IDs in the sidebar to analyse current blocks, missed spend, "
    "and competitor monetization. Optionally generate an AI summary email using Claude."
)

# Initialise session state for results so table interactions don't re-run the pipeline
if "results" not in st.session_state:
    st.session_state["results"] = None

# =====================================================================
# Helper: format spend / revenue columns as $ with no decimals
# =====================================================================
def format_money_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format any spend/revenue column into $#,### (no decimals).
    Includes *_rev_l7d columns used in the competitor matrix.
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
            or c.lower().endswith("_rev_l7d")
        )
        and "rank" not in c.lower()
    ]

    for col in money_cols:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].apply(
                lambda x: f"${int(round(float(x))):,}" if pd.notnull(x) else ""
            )
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

            if metric in money_metrics:
                try:
                    return f"${int(round(float(val))):,}"
                except Exception:
                    return val

            if "Number of blocks" in str(metric):
                try:
                    return int(round(float(val)))
                except Exception:
                    return val

            return val

        df_fmt["value"] = df_fmt.apply(fmt_value, axis=1).astype(str)

    return df_fmt


# =====================================================================
# Helper: AG Grid wrapper with selection + "Copy selected as CSV"
# =====================================================================
def aggrid_with_copy(df: pd.DataFrame, grid_key: str, title: str, height: int = 400):
    """
    Renders an AG Grid table with:
      - full-width columns
      - sorting/filtering
      - multi-row selection via checkboxes
      - 'Copy selected as CSV' button (shows CSV text to copy)
    """
    if df is None or df.empty:
        st.markdown(f"**{title}**")
        st.write("No data available.")
        return

    # Title + copy button placeholder on the same row
    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.markdown(f"**{title}**")
    copy_button_placeholder = header_col2.empty()

    df_display = df.copy()

    # AG Grid configuration
    gb = GridOptionsBuilder.from_dataframe(df_display)
    gb.configure_default_column(
        resizable=True,
        filter=True,
        sortable=True,
    )
    gb.configure_selection(
        selection_mode="multiple",
        use_checkbox=True,
    )
    gb.configure_grid_options(
        enableRangeSelection=True,
        enableRangeHandle=True,
        suppressCopyRowsToClipboard=False,
        copyHeadersToClipboard=True,
    )

    grid_options = gb.build()

    grid_response = AgGrid(
        df_display,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=height,
        theme="streamlit",
        key=grid_key,
    )

    selected_rows = grid_response.get("selected_rows", [])

    # If something is selected, show copy/export controls
    if selected_rows:
        selected_df = pd.DataFrame(selected_rows)
        with copy_button_placeholder:
            if st.button("Copy selected as CSV", key=f"{grid_key}_copy"):
                csv_text = selected_df.to_csv(index=False)
                st.success("Selected rows converted to CSV below – press Ctrl/Cmd + C to copy.")
                st.text_area(
                    "Copy from here:",
                    value=csv_text,
                    height=150,
                )
        st.caption(f"{len(selected_rows)} row(s) selected.")
    else:
        # Keep the header row visually aligned
        with copy_button_placeholder:
            st.button("Copy selected as CSV", key=f"{grid_key}_copy_disabled", disabled=True)


# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
sidebar = st.sidebar
sidebar.header("Inputs")

app_ids_input = sidebar.text_area(
    "Publisher app IDs",
    placeholder="Paste one App ID per line, or comma/space-separated.\n"
                "Example:\n632cc7810ca02c6344d51822\n632cc70d35cc2d93ebf3b2d5",
    height=140,
)

excluded_input = sidebar.text_area(
    "Excluded block values (optional)",
    placeholder="dreamgames.com\n1234567890",
    height=100,
    help="Use to ignore specific advertiser domains or app market IDs in the block analysis.",
)

# Name kept as openai_api_key to match backend signature,
# but label/help clearly indicate it's the Claude / Anthropic key.
openai_api_key = sidebar.text_input(
    "Claude API key",
    type="password",
    help=(
        "Paste your Claude (Anthropic) API key here (e.g. starts with `sk-ant-`). "
        "Leave blank to skip AI summary and only see tables."
    ),
)

recipient_email = sidebar.text_input(
    "Recipient email (summary will be sent here)",
    value="",
)

gmail_app_password = sidebar.text_input(
    "Gmail App Password",
    type="password",
    help="16-character Gmail App Password. Leave blank to skip sending email and only show tables.",
)

# -------------------------------
# Inline Scheduler (WIP – no real automation)
# -------------------------------
sidebar.markdown("### Scheduler (WIP)")

sidebar.caption(
    "Prototype controls for future automation. These settings are **not yet scheduling anything**, "
    "but can be used in demos to show how recurring audits might be configured."
)

sched_col1, sched_col2 = sidebar.columns(2)

with sched_col1:
    scheduler_enabled = sidebar.checkbox(
        "Enable",
        value=False,
        help="Visual only – does not actually schedule background runs yet.",
    )

with sched_col2:
    schedule_frequency = sidebar.selectbox(
        "Frequency",
        ["None", "Daily", "Weekly", "Monthly"],
        index=0,
        help="Future: how often the audit would run.",
    )

schedule_time = sidebar.time_input(
    "Preferred time (local)",
    help="Future: time of day for the scheduled audit.",
)

weekly_day = None
if schedule_frequency == "Weekly":
    weekly_day = sidebar.selectbox(
        "Day of week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        index=0,
        help="Future: weekly send-out day.",
    )

run_button = sidebar.button("Run audit & (optionally) send email", type="primary")

# -------------------------------
# RUN PIPELINE (only when button clicked)
# -------------------------------
if run_button:
    # Parse app IDs: allow commas, spaces, and newlines
    raw_ids = re.split(r"[,\s]+", app_ids_input.strip()) if app_ids_input.strip() else []
    target_app_ids = [x for x in raw_ids if x]

    excluded_raw = re.split(r"[,\s]+", excluded_input.strip()) if excluded_input.strip() else []
    excluded_block_values = [x for x in excluded_raw if x]

    if not target_app_ids:
        st.error("Please provide at least one publisher app ID.")
    else:
        with st.spinner("Running full analysis..."):
            try:
                # Sender email is fixed in backend to ssai@liftoff.io
                results = run_full_pipeline(
                    target_app_ids=target_app_ids,
                    excluded_block_values=excluded_block_values,
                    recipient_email=recipient_email,
                    sender_email=None,           # ignored by backend; kept for signature compatibility
                    gmail_app_password=gmail_app_password,
                    openai_api_key=openai_api_key,  # passes Claude key through
                )
                st.session_state["results"] = results
            except Exception as e:
                st.error(f"Something went wrong while running the pipeline: {e}")
                st.session_state["results"] = None

# -------------------------------
# OUTPUTS
# -------------------------------
results = st.session_state.get("results")

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
            "Scheduler WIP: enable it in the sidebar and choose a frequency/time "
            "to see the planned run cadence."
        )

    # Email / AI summary status
    email_status = results.get("email_status")
    ai_error = results.get("ai_error")

    if email_status == "email_sent":
        st.info(f"Claude summary generated and email sent to **{recipient_email}**.")
    elif email_status == "summary_built":
        st.info("Claude summary generated (email not sent – missing recipient or Gmail app password).")
    elif email_status == "failed_ai_or_email":
        st.warning("Claude summary or email failed. Showing tables only.")
        if ai_error:
            with st.expander("Show Claude / email error details"):
                st.code(ai_error, language="text")
    elif email_status == "ai_not_configured":
        st.info("Claude API key not provided – skipping AI summary/email and showing tables only.")
    else:
        st.info("AI summary/email not attempted (unknown status).")

    st.markdown("---")

    # ---------- TABS ----------
    tab_summary, tab_combined, tab_per_app, tab_metrics = st.tabs(
        ["AI Summary", "Combined Tables", "Per-App Tables", "Summary Metrics"]
    )

    # ----- AI Summary (no AG Grid here) -----
    with tab_summary:
        html_summary = results.get("html_summary")
        if html_summary:
            st.subheader("AI Summary (Claude)")
            st.caption("Rendered directly from the HTML email body.")
            # Wrap in a centered container with constrained width so tables don't stretch edge-to-edge
            wrapped_html = f"""
            <div style="display:flex; justify-content:flex-start;">
              <div style="max-width: 1000px; width: auto;">
                {html_summary}
              </div>
            </div>
            """
            st.components.v1.html(wrapped_html, height=600, scrolling=True)
        else:
            st.info("No AI summary available for this run.")

    # ----- Combined Tables -----
    with tab_combined:
        st.subheader("Combined Tables (All Selected Apps)")

        with st.expander("Legend (similar app mappings)", expanded=False):
            st.code(results["combined_legend"], language="text")

        # 1. Blocks with L7D DSP spend
        combined_blocks_with_spend = format_money_columns(
            results["combined_blocks_with_spend"].head(200)
        )
        aggrid_with_copy(
            combined_blocks_with_spend,
            grid_key="combined_blocks_with_spend",
            title="1. Blocks enriched with L7D DSP spend (app + domain)",
            height=400,
        )

        # 2. Global advertiser network spend (L30D)
        combined_blocks_with_global = format_money_columns(
            results["combined_blocks_with_global"].head(200)
        )
        aggrid_with_copy(
            combined_blocks_with_global,
            grid_key="combined_blocks_with_global",
            title="2. Global advertiser network spend (L30D) for blocks",
            height=400,
        )

        # 3. Block summary per app (our apps only)
        combined_summary_our = format_money_columns(
            format_block_summary_table(
                results["combined_summary_our"].head(200)
            )
        )
        aggrid_with_copy(
            combined_summary_our,
            grid_key="combined_summary_our",
            title="3. Block summary per app (our apps only)",
            height=400,
        )

        # 4. Competitor revenue matrix
        combined_rev_matrix = format_money_columns(
            results["combined_rev_matrix"].head(200)
        )
        aggrid_with_copy(
            combined_rev_matrix,
            grid_key="combined_rev_matrix",
            title="4. Competitor revenue matrix (L7D per similar app)",
            height=400,
        )

    # ----- Per-App Tables -----
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

                    # 1. Blocks with L7D DSP spend
                    per_blocks_with_spend = format_money_columns(
                        tables["blocks_with_spend"].head(200)
                    )
                    aggrid_with_copy(
                        per_blocks_with_spend,
                        grid_key=f"{app_id}_blocks_with_spend",
                        title="Blocks enriched with L7D DSP spend (app + domain)",
                        height=350,
                    )

                    # 2. Global advertiser network spend (L30D)
                    per_blocks_with_global = format_money_columns(
                        tables["blocks_with_global"].head(200)
                    )
                    aggrid_with_copy(
                        per_blocks_with_global,
                        grid_key=f"{app_id}_blocks_with_global",
                        title="Global advertiser network spend (L30D) for blocks",
                        height=350,
                    )

                    # 3. Block summary per app
                    per_summary = format_money_columns(
                        format_block_summary_table(
                            tables["summary_per_app"].head(200)
                        )
                    )
                    aggrid_with_copy(
                        per_summary,
                        grid_key=f"{app_id}_summary_per_app",
                        title="Block summary per app (advertiser L30D spend > 30,000)",
                        height=350,
                    )

                    # 4. Competitor revenue matrix
                    per_rev_matrix = format_money_columns(
                        tables["competitor_rev_matrix"].head(200)
                    )
                    aggrid_with_copy(
                        per_rev_matrix,
                        grid_key=f"{app_id}_competitor_rev_matrix",
                        title="Competitor revenue matrix (L7D per similar app)",
                        height=350,
                    )

    # ----- Summary Metrics -----
    with tab_metrics:
        st.subheader("High-level Summary Metrics")
        st.dataframe(
            format_summary_metrics(
                results["summary_metrics"]
            )
        )
        st.caption(
            "These aggregates are also used as input to the Claude summary "
            "(lost spend, competitor revenue, etc.)."
        )

else:
    st.info(
        "Use the controls in the left sidebar to enter app IDs and click "
        "**Run audit & (optionally) send email** to start."
    )
