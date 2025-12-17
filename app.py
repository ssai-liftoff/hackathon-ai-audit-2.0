# app.py

import streamlit as st
import pandas as pd

from backend import run_full_pipeline

st.set_page_config(page_title="Liftoff DSP AI Audit", layout="wide")

st.title("Liftoff DSP – AI Block/Unblock Audit")
st.write(
    "Provide publisher app IDs, optional exclusions, and your email + API key. "
    "The app will run the full analysis, optionally email the AI summary, and show detailed tables below."
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


# -------------------------------
# INPUTS
# -------------------------------
st.subheader("Inputs")

col1, col2 = st.columns(2)

with col1:
    app_ids_input = st.text_area(
        "Publisher app IDs (comma-separated)",
        placeholder="632cc7810ca02c6344d51822, 632cc70d35cc2d93ebf3b2d5, ...",
        height=100,
    )

    excluded_input = st.text_area(
        "Excluded block values (optional, comma-separated)",
        placeholder="dreamgames.com, 1234567890",
        height=80,
        help="Use to ignore specific advertiser domains or app market IDs in the block analysis.",
    )

with col2:
    openai_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Paste your `sk-...` or `sk-proj-...` key here. Leave blank to skip AI summary and only see tables.",
    )

    recipient_email = st.text_input(
        "Recipient email (summary will be sent here)",
        value="",
    )
    sender_email = st.text_input(
        "Sender Gmail (SMTP account)",
        value="",
        help="Gmail address used to send the email. Must have an App Password configured."
    )
    gmail_app_password = st.text_input(
        "Gmail App Password",
        type="password",
        help="16-character Gmail App Password. Leave blank to skip sending email and only show tables.",
    )

# -------------------------------
# Inline Scheduler (WIP – no real automation)
# -------------------------------
st.markdown("### Scheduler (WIP)")

st.caption(
    "Prototype controls for future automation. These settings are **not yet scheduling anything**, "
    "but can be used in demos to show how recurring audits might be configured."
)

sched_col1, sched_col2, sched_col3 = st.columns([1, 1, 1.2])

with sched_col1:
    scheduler_enabled = st.checkbox(
        "Enable schedule (WIP)",
        value=False,
        help="Visual only – does not actually schedule background runs yet.",
    )

with sched_col2:
    schedule_frequency = st.selectbox(
        "Frequency (WIP)",
        ["None", "Daily", "Weekly", "Monthly"],
        index=0,
        help="Future: how often the audit would run.",
    )

with sched_col3:
    schedule_time = st.time_input(
        "Preferred time (local, WIP)",
        help="Future: time of day for the scheduled audit.",
    )

# Weekly-specific extra control (still WIP)
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

if run_button:
    # Basic validation
    target_app_ids = [x.strip() for x in app_ids_input.split(",") if x.strip()]
    excluded_block_values = [x.strip() for x in excluded_input.split(",") if x.strip()]

    if not target_app_ids:
        st.error("Please provide at least one publisher app ID.")
    else:
        with st.spinner("Running full analysis..."):
            try:
                results = run_full_pipeline(
                    target_app_ids=target_app_ids,
                    excluded_block_values=excluded_block_values,
                    recipient_email=recipient_email,
                    sender_email=sender_email,
                    gmail_app_password=gmail_app_password,
                    openai_api_key=openai_api_key,
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
            "Scheduler WIP: enable it above and choose a frequency/time to see the planned run cadence."
        )

    # Email / AI summary status
    email_status = results.get("email_status")
    ai_error = results.get("ai_error")

    if email_status == "email_sent":
        st.info(f"AI summary generated and email sent to **{recipient_email}**.")
    elif email_status == "summary_built":
        st.info("AI summary generated (email not sent – missing sender/recipient or Gmail app password).")
    elif email_status == "failed_ai_or_email":
        st.warning("AI summary or email failed. Showing tables only.")
        if ai_error:
            with st.expander("Show AI/email error details"):
                st.code(ai_error, language="text")
    elif email_status == "ai_not_configured":
        st.info("OpenAI API key not provided – skipping AI summary/email and showing tables only.")
    else:
        st.info("AI summary/email not attempted (unknown status).")

    # ---------- AI SUMMARY PREVIEW ----------
    html_summary = results.get("html_summary")
    if html_summary:
        st.subheader("AI Summary – Email Preview")
        st.caption("This is exactly what is sent in the email body.")
        st.components.v1.html(html_summary, height=500, scrolling=True)

    st.markdown("---")

    # ---------- TABS FOR TABLES ----------
    tab_combined, tab_per_app, tab_metrics = st.tabs(
        ["Combined Tables", "Per-App Tables", "Summary Metrics"]
    )

    # ----- Combined Tables -----
    with tab_combined:
        st.subheader("Combined Tables (All Selected Apps)")

        st.markdown("**Legend (similar app mappings)**")
        st.code(results["combined_legend"], language="text")

        st.markdown("### 1. Blocks enriched with L7D DSP spend (app + domain)")
        st.dataframe(
            format_money_columns(
                results["combined_blocks_with_spend"].head(50)
            )
        )

        st.markdown("### 2. Global advertiser network spend (L30D) for blocks")
        st.dataframe(
            format_money_columns(
                results["combined_blocks_with_global"].head(50)
            )
        )

        st.markdown("### 3. Block summary per app (our apps only)")
        st.dataframe(
            format_money_columns(
                format_block_summary_table(
                    results["combined_summary_our"].head(50)
                )
            )
        )

        st.markdown("### 4. Competitor revenue matrix (L7D per similar app)")
        st.dataframe(
            format_money_columns(
                results["combined_rev_matrix"].head(50)
            )
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
                        st.markdown("**Similar app legend**")
                        st.code(legend_text, language="text")

                    st.markdown("**Blocks enriched with L7D DSP spend (app + domain)**")
                    st.dataframe(
                        format_money_columns(
                            tables["blocks_with_spend"].head(20)
                        )
                    )

                    st.markdown("**Global advertiser network spend (L30D) for blocks**")
                    st.dataframe(
                        format_money_columns(
                            tables["blocks_with_global"].head(20)
                        )
                    )

                    st.markdown("**Block summary per app (advertiser L30D spend > 30,000)**")
                    st.dataframe(
                        format_money_columns(
                            format_block_summary_table(
                                tables["summary_per_app"].head(20)
                            )
                        )
                    )

                    st.markdown("**Competitor revenue matrix (L7D per similar app)**")
                    st.dataframe(
                        format_money_columns(
                            tables["competitor_rev_matrix"].head(20)
                        )
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
            "These aggregates are also used as input to the AI summary (lost spend, competitor revenue, etc.)."
        )

else:
    st.info("Fill in the inputs above and click **Run audit & (optionally) send email** to start.")
