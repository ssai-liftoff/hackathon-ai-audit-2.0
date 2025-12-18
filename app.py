# app.py

import re
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from backend import run_full_pipeline

# Fixed sender email for SMTP
DEFAULT_SENDER_EMAIL = "ssai@liftoff.io"

st.set_page_config(page_title="[ai]udit – AI Tool for Tracking Blocks", layout="wide")

st.title("[ai]udit – AI Tool for Tracking Blocks")

# Persistent, easy-to-read instructions (no mention of API key / Gmail password)
instructions_text = """
**How to use [ai]udit**

1. In the **sidebar**, paste one or more *publisher app IDs* (one per line or comma/space-separated).
2. (Optional) Add any **block values to exclude** (domains or app market IDs) to clean up the analysis.
3. (Optional) Enter a **recipient email** if you want the summary emailed after the run.
4. (Optional) Adjust the **Scheduler (WIP)** controls to show how recurring audits might work in the future.
5. Click **Run [ai]udit & (optionally) send email** in the sidebar.
6. Use the tabs at the top to explore:
   - **AI Summary** – narrative insights + supporting tables.
   - **Combined Tables** – interactive, sortable view across all selected apps.
   - **Per-App Tables** – drill-down per publisher app.
   - **Summary Metrics** – top-line aggregates (lost spend, competitor revenue, etc.).
"""

st.info(instructions_text)

# Session state for results so interactions (sorting/filtering) don't force re-run
if "results" not in st.session_state:
    st.session_state["results"] = None


# =====================================================================
# Helper: format Summary Metrics table (st.dataframe)
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
# Helper: specific formatting for Block Summary table (Table 3) – numeric only
# =====================================================================
def format_block_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For block summary tables (numeric-only version for AgGrid):
      - block_rate_diff_vs_peers: whole number (no decimals)
      - rank_* columns: integer
    Money columns stay numeric so AgGrid can sort correctly and format via JS.
    """
    if df is None or df.empty:
        return df

    df_fmt = df.copy()

    # 1) block_rate_diff_vs_peers as whole number
    if "block_rate_diff_vs_peers" in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt["block_rate_diff_vs_peers"]):
            df_fmt["block_rate_diff_vs_peers"] = df_fmt["block_rate_diff_vs_peers"].round(0)

    # 2) Any rank column as integer (keep numeric)
    for col in df_fmt.columns:
        if "rank" in col.lower():
            if pd.api.types.is_numeric_dtype(df_fmt[col]):
                df_fmt[col] = df_fmt[col].round(0)

    return df_fmt


# =====================================================================
# Helper: determine money columns for AgGrid
# =====================================================================
def get_money_columns(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    return [
        c
        for c in df.columns
        if (
            any(key in c.lower() for key in ["spend", "revenue"])
            or c.lower().endswith("_rev_l7d")
        )
        and "rank" not in c.lower()
    ]


# =====================================================================
# Helper: render interactive AgGrid for raw tables (numeric, sortable, copyable)
# =====================================================================
def render_aggrid(df: pd.DataFrame, height: int = 380):
    """
    Render a nice interactive AgGrid:
    - Sorting, filtering, resizing
    - Pagination
    - Range selection + clipboard (copyable)
    - Money columns formatted as $#,### client-side, but remain numeric for sorting
    """
    if df is None or df.empty:
        st.write("No data to display.")
        return

    df_ag = df.copy()

    gb = GridOptionsBuilder.from_dataframe(df_ag)
    gb.configure_default_column(
        resizable=True,
        filter=True,
        sortable=True,
    )
    gb.configure_pagination(enabled=True, paginationAutoPageSize=False, paginationPageSize=25)
    gb.configure_selection("multiple", use_checkbox=True)

    # Enable range selection + clipboard
    gb.configure_grid_options(
        enableRangeSelection=True,
        enableRangeHandle=True,
        suppressCopyRowsToClipboard=False,
        copyHeadersToClipboard=True,
    )

    # JS formatter for money columns (keeps underlying values numeric)
    money_cols = get_money_columns(df_ag)

    money_formatter = JsCode(
        """
        function(params) {
            if (params.value === null || params.value === undefined || isNaN(params.value)) {
                return '';
            }
            var v = Math.round(params.value);
            return '$' + v.toLocaleString();
        }
        """
    )

    for col in money_cols:
        gb.configure_column(
            col,
            type=["numericColumn"],
            valueFormatter=money_formatter,
        )

    gridOptions = gb.build()

    AgGrid(
        df_ag,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.NO_UPDATE,
        fit_columns_on_grid_load=True,
        use_container_width=True,
        enable_enterprise_modules=False,
        allow_unsafe_jscode=True,  # needed for JsCode formatter
        height=height,
    )


# =====================================================================
# Helper: parse & render AI summary HTML natively (NO AgGrid)
#   - Headings & bullets rendered as Markdown
#   - Tables rendered as raw HTML via st.markdown, so width is intrinsic
# =====================================================================
def render_ai_summary_native(html_summary: str):
    """
    Parse the Claude-generated HTML summary and render:
      - Headings & intro as Markdown
      - Bullet lists as Markdown bullets
      - Tables as raw HTML, so width fits the data (not forced full-screen)
    """
    if not html_summary:
        st.info("No AI summary is available.")
        return

    soup = BeautifulSoup(html_summary, "html.parser")

    # Top-level title (h2 in the HTML wrapper)
    h2 = soup.find("h2")
    if h2:
        st.subheader(h2.get_text(strip=True))

    # First paragraph (intro)
    first_p = soup.find("p")
    if first_p:
        st.write(first_p.get_text(" ", strip=True))

    # Key insights / Opportunities sections
    for h3 in soup.find_all("h3"):
        section_title = h3.get_text(strip=True)
        st.markdown(f"### {section_title}")

        ul = h3.find_next_sibling("ul")
        if ul:
            bullets = [li.get_text(" ", strip=True) for li in ul.find_all("li")]
            for bullet in bullets:
                st.markdown(f"- {bullet}")

    # Supporting tables: render the actual HTML tables so width fits the data
    tables = soup.find_all("table")
    if tables:
        st.markdown("### Supporting tables")

        for idx, table in enumerate(tables, start=1):
            # Try to get the bold title just before the table
            title_text = None
            prev = table.find_previous()
            while prev is not None:
                if prev.name == "b":
                    title_text = prev.get_text(strip=True)
                    break
                prev = prev.find_previous()

            # Build an HTML fragment: optional title + original table HTML
            if title_text:
                fragment = f"<b>{title_text}</b><br>{str(table)}"
            else:
                fragment = str(table)

            # Wrap in a div so table doesn't stretch full-width
            wrapped = f"""
            <div style="display:inline-block; margin-bottom: 16px;">
                {fragment}
            </div>
            """

            st.markdown(wrapped, unsafe_allow_html=True)


# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
with st.sidebar:
    st.subheader("Inputs")

    app_ids_input = st.text_area(
        "Publisher app IDs",
        placeholder="Paste one ID per line, or a column from Sheets.\n"
                    "Example:\n632cc7810ca02c6344d51822\n632cc70d35cc2d93ebf3b2d5",
        height=140,
    )

    excluded_input = st.text_area(
        "Excluded block values (optional)",
        placeholder="dreamgames.com\n1234567890",
        height=100,
        help="Paste domains or app market IDs to ignore in the block analysis (one per line or comma-separated).",
    )

    recipient_email = st.text_input(
        "Recipient email (summary will be sent here)",
        value="",
    )

    st.markdown("---")
    # -------------------------------
    # Inline Scheduler (WIP – no real automation)
    # -------------------------------
    st.subheader("Scheduler (WIP)")

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

    st.markdown("---")
    # -------------------------------
    # AI & Email (dev-only controls) – moved to bottom of sidebar
    # -------------------------------
    st.subheader("AI & Email (dev-only)")

    # Name kept as openai_api_key to match backend signature,
    # but label/help clearly indicate it's the Claude / Anthropic key.
    openai_api_key = st.text_input(
        "Claude API key",
        type="password",
        help=(
            "Paste your Claude (Anthropic) API key here (e.g. starts with `sk-ant-`). "
            "Leave blank to skip AI summary and only see tables."
        ),
    )

    # Sender email is fixed; show it as read-only info
    st.caption(f"Sender Gmail (fixed): **{DEFAULT_SENDER_EMAIL}**")

    gmail_app_password = st.text_input(
        "Gmail App Password",
        type="password",
        help="16-character Gmail App Password. Leave blank to skip sending email and only show tables.",
    )

    run_button = st.button("Run [ai]udit & (optionally) send email", type="primary")


# -------------------------------
# RUN PIPELINE (ONLY on button click)
# -------------------------------
if run_button:
    # Parse app IDs "Looker style": split on commas, newlines, and whitespace
    raw_ids = app_ids_input.strip()
    if raw_ids:
        target_app_ids = [x for x in re.split(r"[\s,]+", raw_ids) if x]
    else:
        target_app_ids = []

    # Exclusions can also be newline/comma separated
    raw_excl = excluded_input.strip()
    if raw_excl:
        excluded_block_values = [x for x in re.split(r"[\s,]+", raw_excl) if x]
    else:
        excluded_block_values = []

    if not target_app_ids:
        st.error("Please provide at least one publisher app ID.")
    else:
        with st.spinner("Running full analysis..."):
            try:
                results = run_full_pipeline(
                    target_app_ids=target_app_ids,
                    excluded_block_values=excluded_block_values,
                    recipient_email=recipient_email,
                    sender_email=DEFAULT_SENDER_EMAIL,  # fixed sender
                    gmail_app_password=gmail_app_password,
                    openai_api_key=openai_api_key,  # passes Claude key through
                )
                st.session_state["results"] = results
            except Exception as e:
                st.error(f"Something went wrong while running the pipeline: {e}")
                st.session_state["results"] = None

# Use stored results for display (so interactions don't clear them)
results = st.session_state["results"]

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
        st.info("Claude summary generated (email not sent – missing recipient or Gmail app password).")
    elif email_status == "failed_ai_or_email":
        st.warning("Claude summary or email failed. Showing tables only.")
        if ai_error:
            with st.expander("Show Claude error details"):
                st.code(ai_error, language="text")
    elif email_status == "ai_not_configured":
        st.info("Claude API key not provided – skipping AI summary/email and showing tables only.")
    else:
        st.info("AI summary/email not attempted (unknown status).")

    # ---------- TABS FOR SUMMARY + TABLES ----------
    tab_summary, tab_combined, tab_per_app, tab_metrics = st.tabs(
        ["AI Summary", "Combined Tables", "Per-App Tables", "Summary Metrics"]
    )

    # ----- AI Summary (native, NO AgGrid) -----
    with tab_summary:
        html_summary = results.get("html_summary")
        if html_summary:
            render_ai_summary_native(html_summary)
        else:
            st.info("No AI summary was generated. Check API key / error details above.")

    # ----- Combined Tables (AgGrid, full-width, copyable) -----
    with tab_combined:
        st.subheader("Combined Tables (All Selected Apps)")

        with st.expander("Legend (similar app mappings)", expanded=False):
            st.code(results["combined_legend"], language="text")

        st.markdown("### 1. Blocks enriched with L7D DSP spend (app + domain)")
        df1 = results["combined_blocks_with_spend"].head(50).copy()
        render_aggrid(df1)

        st.markdown("### 2. Global advertiser network spend (L30D) for blocks")
        df2 = results["combined_blocks_with_global"].head(50).copy()
        render_aggrid(df2)

        st.markdown("### 3. Block summary per app (our apps only)")
        df3 = results["combined_summary_our"].head(50)
        df3 = format_block_summary_table(df3)
        render_aggrid(df3)

        st.markdown("### 4. Competitor revenue matrix (L7D per similar app)")
        df4 = results["combined_rev_matrix"].head(50).copy()
        render_aggrid(df4)

    # ----- Per-App Tables (AgGrid, full-width, copyable) -----
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

                    st.markdown("**Blocks enriched with L7D DSP spend (app + domain)**")
                    df_pa1 = tables["blocks_with_spend"].head(20).copy()
                    render_aggrid(df_pa1)

                    st.markdown("**Global advertiser network spend (L30D) for blocks**")
                    df_pa2 = tables["blocks_with_global"].head(20).copy()
                    render_aggrid(df_pa2)

                    st.markdown("**Block summary per app (advertiser L30D spend > 30,000)**")
                    df_pa3 = tables["summary_per_app"].head(20)
                    df_pa3 = format_block_summary_table(df_pa3)
                    render_aggrid(df_pa3)

                    st.markdown("**Competitor revenue matrix (L7D per similar app)**")
                    df_pa4 = tables["competitor_rev_matrix"].head(20).copy()
                    render_aggrid(df_pa4)

    # ----- Summary Metrics -----
    with tab_metrics:
        st.subheader("High-level Summary Metrics")
        st.dataframe(
            format_summary_metrics(results["summary_metrics"])
        )
        st.caption(
            "These aggregates are also used as input to the Claude summary (lost spend, competitor revenue, etc.)."
        )

else:
    # No results yet – rely on the persistent instructions at the top
    st.info(
        "Use the sidebar to configure inputs, then click **Run [ai]udit & (optionally) send email** to start."
    )
