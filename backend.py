import os
import json
import pandas as pd

from anthropic import Anthropic  # <-- Claude client
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# =====================================================================
# 0. CONFIG – DATA PATH
# =====================================================================

# In Streamlit / GitHub, your CSVs live in ./data
DATA_DIR = "data/"


# =====================================================================
# 1. LOAD & CLEAN DATA (RUN ON IMPORT)
# =====================================================================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    return df


t1 = pd.read_csv(DATA_DIR + "spot_non-vxspend_appsonly_7d.csv", skiprows=1, low_memory=False)
t2 = pd.read_csv(DATA_DIR + "spot_non-vxspend_domainsonly_7d.csv", skiprows=1, low_memory=False)
t3 = pd.read_csv(DATA_DIR + "VXoverview_topadvertisers_onlydomains_networklevel_7d.csv", low_memory=False)
t4 = pd.read_csv(DATA_DIR + "VXoverview_topadvertisers_onlyapps_networklevel_7d.csv", low_memory=False)
t5 = pd.read_csv(DATA_DIR + "VXoverview_topadvertisers_networklevel_7d.csv", low_memory=False)
t6 = pd.read_csv(DATA_DIR + "VXoverview_applevel_blockedappsdoaminsdata.csv", low_memory=False)
t7 = pd.read_csv(DATA_DIR + "MBR_appsbasedata_l30d.csv", low_memory=False)
t8 = pd.read_csv(DATA_DIR + "VXoverview_applevel_pubapps+advertiser_combospend_7d.csv", low_memory=False)
t9 = pd.read_csv(DATA_DIR + "VX_appsdata_forsimilarapps_7d.csv", low_memory=False)

t1 = clean_columns(t1)
t2 = clean_columns(t2)
t3 = clean_columns(t3)
t4 = clean_columns(t4)
t5 = clean_columns(t5)
t6 = clean_columns(t6)
t7 = clean_columns(t7)
t8 = clean_columns(t8)
t9 = clean_columns(t9)

# Rename last 2 columns of t1/t2 to standard names
t1_cols = list(t1.columns)
t1_cols[-2] = "non_vx_spend"
t1_cols[-1] = "vx_spend"
t1.columns = t1_cols

t2_cols = list(t2.columns)
t2_cols[-2] = "non_vx_spend"
t2_cols[-1] = "vx_spend"
t2.columns = t2_cols

# Normalise domain/app-id columns
for df in [t1, t2, t3, t4, t5, t6, t7, t8, t9]:
    for col in df.columns:
        if "domain" in col or "adomain" in col:
            df[col] = df[col].astype(str).str.strip().str.lower()
    for col in df.columns:
        if ("app" in col and ("id" in col or "name" in col)) or ("publisher_app" in col):
            df[col] = df[col].astype(str).str.strip()


# =====================================================================
# 2. SHARED HELPERS
# =====================================================================

def get_t6_row_for_app(app_id: str) -> pd.Series:
    row = t6[t6["publisher__app_id"] == app_id]
    if not row.empty:
        return row.iloc[0]

    t7_match = t7[t7["publisher_app_id"] == app_id]
    if not t7_match.empty:
        market_id = t7_match.iloc[0]["publisher_app_market_id"]
        row = t6[t6["publisher__app_market_id"] == str(market_id)]
        if not row.empty:
            return row.iloc[0]

    raise ValueError(f"Could not find app_id {app_id} in t6 or via t7.")


def can_find_in_t6(app_id: str) -> bool:
    try:
        _ = get_t6_row_for_app(app_id)
        return True
    except ValueError:
        return False


def parse_block_list(value):
    if pd.isna(value):
        return []
    items = str(value).split(",")
    items = [x.strip() for x in items if x and x.strip().lower() != "nan"]
    return items


def normalize_blocks_for_app(app_id: str, app_label: str = None) -> pd.DataFrame:
    row = get_t6_row_for_app(app_id)
    app_market_id = str(row.get("publisher__app_market_id", "")).strip()
    app_name = str(row.get("publisher__app_name", "")).strip()

    account_blocked_apps = parse_block_list(row.get("publisher_account_blocked_apps"))
    app_blocked_apps = parse_block_list(row.get("publisher__app_blocked_advertiser_market_ids"))
    account_blocked_domains = parse_block_list(row.get("publisher_account_blocked_adomains"))
    app_blocked_domains = parse_block_list(row.get("publisher__app_blocked_adomains"))

    records = []

    def add_record(v, block_type, block_level):
        rec = {
            "publisher_app_id": app_id,
            "publisher_app_market_id": app_market_id,
            "publisher_app_name": app_name,
            "block_type": block_type,
            "block_value": str(v).strip().lower() if block_type == "domain" else str(v).strip(),
            "block_level": block_level,
        }
        if app_label is not None:
            rec["app_label"] = app_label
        records.append(rec)

    for v in account_blocked_apps:
        add_record(v, "app", "account")
    for v in app_blocked_apps:
        add_record(v, "app", "app")
    for v in account_blocked_domains:
        add_record(v, "domain", "account")
    for v in app_blocked_domains:
        add_record(v, "domain", "app")

    df = pd.DataFrame(records)
    if df.empty:
        return df

    subset_cols = ["publisher_app_id", "block_type", "block_value", "block_level"]
    if app_label is not None:
        subset_cols.append("app_label")

    df = df.drop_duplicates(subset=subset_cols).reset_index(drop=True)
    return df


def find_similar_apps(
    target_app_id: str,
    n_similar: int = 5,
    min_spend: float = 0.0,
    df_apps: pd.DataFrame = None,
) -> pd.DataFrame:
    if df_apps is None:
        df_apps = t9

    df = df_apps.copy()
    spend_col = "transaction_event_ad_spend_(unified)"
    df[spend_col] = pd.to_numeric(df[spend_col], errors="coerce").fillna(0)

    target_row = df[df["publisher__app_id"] == target_app_id]
    if target_row.empty:
        raise ValueError(f"target_app_id {target_app_id} not found in t9")

    target_row = target_row.iloc[0]
    target_platform = target_row["publisher__app_platform"]
    target_category = target_row["publisher__app_metadata_liftoff_category"]
    target_account_id = target_row["publisher_account_id"]
    target_spend = float(target_row[spend_col])

    candidates = df[
        (df["publisher__app_platform"] == target_platform)
        & (df["publisher__app_metadata_liftoff_category"] == target_category)
        & (df["publisher_account_id"] != target_account_id)
        & (df["publisher__app_id"] != target_app_id)
    ].copy()

    candidates = candidates[candidates[spend_col] >= min_spend]

    if candidates.empty:
        return candidates

    candidates["spend_distance"] = (candidates[spend_col] - target_spend).abs()
    candidates = candidates.sort_values("spend_distance", ascending=True)

    top_similar = candidates.head(n_similar).copy()
    top_similar.insert(
        0,
        "similar_label",
        [f"Similar app {i+1}" for i in range(len(top_similar))]
    )

    cols = [
        "similar_label",
        "publisher__app_id",
        "publisher__app_name",
        "publisher_account_id",
        "publisher__app_market_id",
        "publisher__app_platform",
        "publisher__app_metadata_liftoff_category",
        spend_col,
        "spend_distance",
    ]
    top_similar = top_similar[cols]

    return top_similar


def apply_block_exclusions(df: pd.DataFrame, excluded_block_values) -> pd.DataFrame:
    """
    Filter out rows where block_value is in excluded_block_values.
    Comparison is case-insensitive on str(block_value).
    """
    if not excluded_block_values or df.empty:
        return df

    ex_set = {str(x).strip().lower() for x in excluded_block_values if pd.notna(x)}
    if not ex_set or "block_value" not in df.columns:
        return df

    tmp = df.copy()
    tmp["__bv_norm"] = tmp["block_value"].astype(str).str.strip().str.lower()
    tmp = tmp[~tmp["__bv_norm"].isin(ex_set)].drop(columns="__bv_norm")
    tmp = tmp.reset_index(drop=True)
    return tmp


# =====================================================================
# 3. PER-APP ANALYSIS
# =====================================================================

def run_full_analysis(
    target_app_id: str,
    n_similar: int = 5,
    min_spend: float = 0.0,
    excluded_block_values=None,
):
    # Normalise exclusions once
    if excluded_block_values is None:
        excluded_block_values = []
    ex_set = {str(x).strip().lower() for x in excluded_block_values if pd.notna(x)}

    # 3.1 Find similar apps from t9
    similar_apps_df_raw = find_similar_apps(target_app_id, n_similar=n_similar, min_spend=min_spend)

    # 3.2 Filter similar apps to those we can resolve in t6/t7
    if not similar_apps_df_raw.empty:
        similar_apps_df_raw["can_use"] = similar_apps_df_raw["publisher__app_id"].apply(can_find_in_t6)
        similar_apps_df = similar_apps_df_raw[similar_apps_df_raw["can_use"]].copy()
    else:
        similar_apps_df = similar_apps_df_raw.copy()

    # Build legend only for usable similar apps
    if similar_apps_df.empty:
        legend_text = f"Publisher app {target_app_id} similar apps: (none usable from t6/t7)\n"
    else:
        legend_lines = [
            f"{row['similar_label']} = {row['publisher__app_name']} ({row['publisher__app_id']})"
            for _, row in similar_apps_df.iterrows()
        ]
        legend_text = f"Publisher app {target_app_id} similar apps:\n" + "\n".join(legend_lines)

    # 3.3 Normalized blocks for *publisher* app, with exclusions
    df_blocks_normalized = normalize_blocks_for_app(target_app_id)
    df_blocks_normalized = apply_block_exclusions(df_blocks_normalized, ex_set)

    # ---------- Stage 3A: DSP L7D spend (t1/t2) ----------
    app_blocks = df_blocks_normalized[df_blocks_normalized["block_type"] == "app"].copy()
    domain_blocks = df_blocks_normalized[df_blocks_normalized["block_type"] == "domain"].copy()

    app_blocks_enriched = app_blocks.merge(
        t1,
        how="left",
        left_on=["publisher_app_market_id", "block_value"],
        right_on=[
            "analytics_data_source_app_app_store_id",
            "salesforce_info_app_play_store_id_(market_id)",
        ],
        suffixes=("", "_t1"),
    )
    app_blocks_enriched["advertiser_app_name"] = app_blocks_enriched["analytics_data_advertiser_app_name"]

    domain_blocks_enriched = domain_blocks.merge(
        t2,
        how="left",
        left_on=["publisher_app_market_id", "block_value"],
        right_on=[
            "analytics_data_source_app_app_store_id",
            "advertiser_(destination)_app_advertiser_domain",
        ],
        suffixes=("", "_t2"),
    )
    domain_blocks_enriched["advertiser_app_name"] = pd.NA

    app_view = app_blocks_enriched[
        [
            "publisher_app_id",
            "publisher_app_market_id",
            "publisher_app_name",
            "block_type",
            "block_value",
            "block_level",
            "advertiser_app_name",
            "non_vx_spend",
            "vx_spend",
        ]
    ].copy()

    domain_view = domain_blocks_enriched[
        [
            "publisher_app_id",
            "publisher_app_market_id",
            "publisher_app_name",
            "block_type",
            "block_value",
            "block_level",
            "advertiser_app_name",
            "non_vx_spend",
            "vx_spend",
        ]
    ].copy()

    df_blocks_with_spend = pd.concat([app_view, domain_view], ignore_index=True)

    df_blocks_with_spend["non_vx_spend"] = pd.to_numeric(
        df_blocks_with_spend["non_vx_spend"], errors="coerce"
    ).fillna(0)
    df_blocks_with_spend["vx_spend"] = pd.to_numeric(
        df_blocks_with_spend["vx_spend"], errors="coerce"
    ).fillna(0)

    df_blocks_with_spend = df_blocks_with_spend.rename(
        columns={"non_vx_spend": "non_vx_spend_l7d", "vx_spend": "vx_spend_l7d"}
    )

    df_blocks_with_spend = df_blocks_with_spend.sort_values(
        "non_vx_spend_l7d", ascending=False
    ).reset_index(drop=True)

    # ---------- Stage 3B: Global advertiser network spend L30D (t3/t4) ----------
    app_blocks_global = app_blocks.merge(
        t4,
        how="left",
        left_on="block_value",
        right_on="advertiser_app_market_id",
        suffixes=("", "_t4"),
    )
    domain_blocks_global = domain_blocks.merge(
        t3,
        how="left",
        left_on="block_value",
        right_on="vx_overview_adomain",
        suffixes=("", "_t3"),
    )

    app_view_global = app_blocks_global[
        [
            "publisher_app_id",
            "publisher_app_market_id",
            "publisher_app_name",
            "block_value",
            "block_type",
            "advertiser_app_title",
            "vx_overview_unified_ad_spend",
        ]
    ].copy()

    domain_view_global = domain_blocks_global[
        [
            "publisher_app_id",
            "publisher_app_market_id",
            "publisher_app_name",
            "block_value",
            "block_type",
            "vx_overview_unified_ad_spend",
        ]
    ].copy()
    domain_view_global["advertiser_app_title"] = pd.NA

    domain_view_global = domain_view_global[
        [
            "publisher_app_id",
            "publisher_app_market_id",
            "publisher_app_name",
            "block_value",
            "block_type",
            "advertiser_app_title",
            "vx_overview_unified_ad_spend",
        ]
    ]

    df_blocks_with_global = pd.concat(
        [app_view_global, domain_view_global], ignore_index=True
    )

    df_blocks_with_global["vx_overview_unified_ad_spend"] = pd.to_numeric(
        df_blocks_with_global["vx_overview_unified_ad_spend"], errors="coerce"
    ).fillna(0)

    df_blocks_with_global = df_blocks_with_global.rename(
        columns={"vx_overview_unified_ad_spend": "vx_overview_ad_spend_l30d"}
    )

    df_blocks_with_global = df_blocks_with_global.sort_values(
        "vx_overview_ad_spend_l30d", ascending=False
    ).reset_index(drop=True)

    # ---------- Stage 3C: Block summary per app (publisher + similar apps) ----------
    main_row_t6 = get_t6_row_for_app(target_app_id)
    main_app_label = f"{main_row_t6['publisher__app_name']} ({target_app_id})"

    apps_to_process = [(target_app_id, main_app_label)]
    for _, row in similar_apps_df.iterrows():
        apps_to_process.append((row["publisher__app_id"], row["similar_label"]))

    all_blocks_list = []
    for app_id, label in apps_to_process:
        try:
            df_app_blocks = normalize_blocks_for_app(app_id, app_label=label)
        except ValueError:
            continue
        if not df_app_blocks.empty:
            df_app_blocks = apply_block_exclusions(df_app_blocks, ex_set)
            all_blocks_list.append(df_app_blocks)

    if not all_blocks_list:
        raise ValueError(f"No blocks found for any app in the list for publisher {target_app_id}.")

    df_all_blocks = pd.concat(all_blocks_list, ignore_index=True)

    app_blocks_all = df_all_blocks[df_all_blocks["block_type"] == "app"].copy()
    domain_blocks_all = df_all_blocks[df_all_blocks["block_type"] == "domain"].copy()

    app_blocks_enriched_all = app_blocks_all.merge(
        t4[["advertiser_app_market_id", "vx_overview_unified_ad_spend"]],
        how="left",
        left_on="block_value",
        right_on="advertiser_app_market_id",
        suffixes=("", "_t4"),
    )
    domain_blocks_enriched_all = domain_blocks_all.merge(
        t3[["vx_overview_adomain", "vx_overview_unified_ad_spend"]],
        how="left",
        left_on="block_value",
        right_on="vx_overview_adomain",
        suffixes=("", "_t3"),
    )

    app_blocks_enriched_all["vx_overview_unified_ad_spend"] = pd.to_numeric(
        app_blocks_enriched_all["vx_overview_unified_ad_spend"], errors="coerce"
    )
    domain_blocks_enriched_all["vx_overview_unified_ad_spend"] = pd.to_numeric(
        domain_blocks_enriched_all["vx_overview_unified_ad_spend"], errors="coerce"
    )

    blocks_enriched_all = pd.concat(
        [app_blocks_enriched_all, domain_blocks_enriched_all], ignore_index=True
    )
    blocks_enriched_all["vx_overview_unified_ad_spend"] = blocks_enriched_all[
        "vx_overview_unified_ad_spend"
    ].fillna(0)

    blocks_enriched_all = blocks_enriched_all.rename(
        columns={"vx_overview_unified_ad_spend": "advertiser_spend_l30d"}
    )

    blocks_unique_all = blocks_enriched_all.drop_duplicates(
        subset=["publisher_app_id", "app_label", "block_type", "block_value"]
    ).reset_index(drop=True)

    spend_threshold = 30000
    blocks_filtered = blocks_unique_all[
        blocks_unique_all["advertiser_spend_l30d"] > spend_threshold
    ].copy()

    summary = (
        blocks_filtered.groupby(["publisher_app_id", "app_label"], as_index=False)
        .agg(
            total_high_value_blocks=("block_value", "nunique"),
            missed_spend_opportunity_l30d=("advertiser_spend_l30d", "sum"),
        )
    )

    for app_id, label in apps_to_process:
        if not ((summary["publisher_app_id"] == app_id) & (summary["app_label"] == label)).any():
            summary = pd.concat(
                [
                    summary,
                    pd.DataFrame(
                        [
                            {
                                "publisher_app_id": app_id,
                                "app_label": label,
                                "total_high_value_blocks": 0,
                                "missed_spend_opportunity_l30d": 0.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    competitor_mask = summary["app_label"].str.startswith("Similar app")
    competitor_blocks = summary[competitor_mask]["total_high_value_blocks"]

    if competitor_blocks.empty:
        competitor_mean = 0.0
        competitor_std = 0.0
    else:
        competitor_mean = competitor_blocks.mean()
        competitor_std = competitor_blocks.std(ddof=0)

    summary["block_rate_diff_vs_peers"] = summary["total_high_value_blocks"] - competitor_mean

    if competitor_std == 0:
        summary["block_rate_zscore"] = 0.0
    else:
        summary["block_rate_zscore"] = (
            (summary["total_high_value_blocks"] - competitor_mean) / competitor_std
        )

    summary["block_rate_zscore"] = summary["block_rate_zscore"].round(2)

    summary["rank_total_blocks"] = summary["total_high_value_blocks"].rank(
        method="min", ascending=False
    ).astype(int)
    summary["rank_missed_spend"] = summary["missed_spend_opportunity_l30d"].rank(
        method="min", ascending=False
    ).astype(int)

    summary_final = summary[
        [
            "publisher_app_id",
            "app_label",
            "total_high_value_blocks",
            "missed_spend_opportunity_l30d",
            "block_rate_diff_vs_peers",
            "block_rate_zscore",
            "rank_total_blocks",
            "rank_missed_spend",
        ]
    ].copy()

    # ---------- Stage 3D: Competitor revenue matrix (t8 + global spend) ----------
    t8_app = t8[~t8["advertiser_app_market_id"].isna()].copy()
    t8_app_agg = (
        t8_app
        .groupby(
            ["advertiser_app_market_id", "publisher__app_vungle_app_id"],
            as_index=False
        )["vx_overview_publisher_revenue"]
        .sum()
        .rename(columns={"vx_overview_publisher_revenue": "publisher_revenue_l7d"})
    )

    t8_domain = t8[~t8["vx_overview_adomain"].isna()].copy()
    t8_domain_agg = (
        t8_domain
        .groupby(
            ["vx_overview_adomain", "publisher__app_vungle_app_id"],
            as_index=False
        )["vx_overview_publisher_revenue"]
        .sum()
        .rename(columns={"vx_overview_publisher_revenue": "publisher_revenue_l7d"})
    )

    matrix = df_blocks_with_global[
        ["block_value", "block_type", "vx_overview_ad_spend_l30d"]
    ].drop_duplicates().copy()

    apps_for_revenue = [(target_app_id, "Our app")]
    for _, row in similar_apps_df.iterrows():
        apps_for_revenue.append((row["publisher__app_id"], row["similar_label"]))

    main_row_t6 = get_t6_row_for_app(target_app_id)
    publisher_app_name_main = main_row_t6["publisher__app_name"]

    rev_matrix = matrix.copy()
    rev_matrix.insert(0, "publisher_app_name", publisher_app_name_main)
    rev_matrix.insert(0, "publisher_app_id", target_app_id)

    for app_id, label in apps_for_revenue:
        if app_id == target_app_id:
            continue
        if not can_find_in_t6(app_id):
            continue

        col_name = f"{label}_rev_l7d"
        rev_matrix[col_name] = 0.0

        mask_app = rev_matrix["block_type"] == "app"
        if mask_app.any():
            app_side = rev_matrix.loc[mask_app, ["block_value"]].merge(
                t8_app_agg[t8_app_agg["publisher__app_vungle_app_id"] == app_id],
                left_on="block_value",
                right_on="advertiser_app_market_id",
                how="left",
            )
            rev_matrix.loc[mask_app, col_name] = (
                app_side["publisher_revenue_l7d"].fillna(0).values
            )

        mask_domain = rev_matrix["block_type"] == "domain"
        if mask_domain.any():
            domain_side = rev_matrix.loc[mask_domain, ["block_value"]].merge(
                t8_domain_agg[t8_domain_agg["publisher__app_vungle_app_id"] == app_id],
                left_on="block_value",
                right_on="vx_overview_adomain",
                how="left",
            )
            rev_matrix.loc[mask_domain, col_name] = (
                domain_side["publisher_revenue_l7d"].fillna(0).values
            )

    title_map = df_blocks_with_global[
        ["block_value", "block_type", "advertiser_app_title"]
    ].drop_duplicates()

    rev_matrix = rev_matrix.merge(
        title_map,
        how="left",
        on=["block_value", "block_type"]
    )

    rev_matrix["block_value_title"] = rev_matrix["advertiser_app_title"]
    rev_matrix.loc[
        rev_matrix["block_type"] == "domain", "block_value_title"
    ] = rev_matrix["block_value"]

    rev_matrix = rev_matrix.drop(columns=["advertiser_app_title"])

    rev_matrix["vx_overview_ad_spend_l30d"] = pd.to_numeric(
        rev_matrix["vx_overview_ad_spend_l30d"], errors="coerce"
    ).fillna(0)

    competitor_labels = [
        label for (app_id, label) in apps_for_revenue if app_id != target_app_id and can_find_in_t6(app_id)
    ]

    for label in competitor_labels:
        col_name = f"{label}_rev_l7d"
        rev_matrix[col_name] = pd.to_numeric(
            rev_matrix.get(col_name, 0), errors="coerce"
        ).fillna(0)

    if competitor_labels:
        rev_matrix["total_competitor_rev_l7d"] = rev_matrix[
            [f"{label}_rev_l7d" for label in competitor_labels]
        ].sum(axis=1)
    else:
        rev_matrix["total_competitor_rev_l7d"] = 0.0

    rev_matrix = rev_matrix.sort_values(
        "total_competitor_rev_l7d", ascending=False
    ).reset_index(drop=True)

    cols_order = (
        [
            "publisher_app_id",
            "publisher_app_name",
            "block_value",
            "block_type",
            "block_value_title",
        ]
        + [f"{label}_rev_l7d" for label in competitor_labels]
        + ["total_competitor_rev_l7d"]
    )
    rev_matrix = rev_matrix[cols_order]

    return (
        similar_apps_df,
        df_blocks_with_spend,
        df_blocks_with_global,
        summary_final,
        rev_matrix,
        legend_text,
    )


# =====================================================================
# 4. AI SUMMARY + EMAIL (CLAUDE)
# =====================================================================

def df_to_json_top(df, n=10):
    if df is None or df.empty:
        return "[]"
    return df.head(n).to_json(orient="records")


def build_ai_summary_html(
    target_app_ids,
    combined_legend,
    combined_blocks_with_spend,
    combined_blocks_with_global,
    combined_summary_our,
    combined_rev_matrix,
    summary_metrics,
    openai_api_key: str,  # reused name; this is actually the Claude key now
):
    """
    Build HTML AI summary using Claude (Anthropic).
    `openai_api_key` here is treated as the Anthropic API key for this Claude-only app.
    """
    claude_api_key = (openai_api_key or "").strip()
    if not claude_api_key:
        raise RuntimeError("Claude API key not provided.")

    client = Anthropic(api_key=claude_api_key)

    ai_payload = {
        "legend_text": combined_legend,
        "blocks_with_spend": df_to_json_top(combined_blocks_with_spend, 10),
        "blocks_with_global": df_to_json_top(combined_blocks_with_global, 10),
        "summary_per_app": df_to_json_top(combined_summary_our, 10),
        "competitor_rev_matrix": df_to_json_top(combined_rev_matrix, 10),
        "summary_metrics": summary_metrics.to_json(orient="records"),
        "target_app_ids": target_app_ids,
    }

    payload_json = json.dumps(ai_payload)

    system_prompt = """
You are a Strategy Analyst at Liftoff Mobile working on Supply side Vungle Exchange.
You analyse advertiser blocks, DSP spend, competitor monetization, and missed opportunities.

You will receive 5 JSON tables:
1. blocks_with_spend – L7D DSP spend on blocked advertisers.
2. blocks_with_global – L30D global network advertiser spend.
3. summary_per_app – block aggressiveness metrics vs similar apps.
4. competitor_rev_matrix – similar-app revenue on advertisers we block.
5. summary_metrics – numeric aggregates (lost spend, competitor revenue, etc.).

OUTPUT FORMAT (STRICT):

• Your entire output MUST be valid HTML.
• Use this exact structure:

  <h3>Key insights</h3>
  <ul>
    <li>...</li>
    <li>...</li>
  </ul>

  <h3>Opportunities</h3>
  <ul>
    <li>...</li>
    <li>...</li>
  </ul>

• Bullet point rules:
  - 3–5 bullets under Key insights.
  - 2–3 bullets under Opportunities.
  - Each bullet MUST be short, sharp, 1 sentence only.
  - No explanations longer than 15–20 words.
  - No compound sentences. No fluff. Just the insight.

• Table rules (EMAIL SAFE):
  - Include AT LEAST ONE and AT MOST THREE HTML tables.
  - Use only this exact structure:

    <b>Table title here</b><br>
    <table border="1" cellpadding="4" cellspacing="0"
           style="border-collapse: collapse; font-size: 13px; margin-top: 8px; margin-bottom: 16px;">
        <tr><th>Column1</th><th>Column2</th>...</tr>
        <tr><td>Value1</td><td>Value2</td>...</tr>
    </table>

  - Show only the 3–5 highest-impact rows.
  - Choose the columns that best illustrate the insight (e.g., advertiser + L30D spend).
  - For any spend or revenue columns, format values as US dollars:
      • Add a leading "$"
      • No decimals (round to nearest whole dollar)
      • You MAY use short notation for large numbers, e.g. $325K, $1.2M, $18M.
  - NEVER embed JSON inside the table.

CONTENT RULES:
• NEVER invent numbers.
• ONLY use values from the JSON payload.
• Refer to apps using their labels or app names exactly as provided.
• Tone: concise, data-driven, action-ready.
"""

    user_prompt = f"""
Here is the JSON data for the following publisher apps: {target_app_ids}

DATA:
{payload_json}

Summarise according to the system rules above.
"""

    # Claude messages.create call
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        temperature=0.25,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ],
    )

    # Extract text content
    ai_summary_text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            ai_summary_text_parts.append(block.text)
    ai_summary_text = "".join(ai_summary_text_parts).strip()

    html_wrapper = f"""
<html>
<body style="font-family: Arial, sans-serif; font-size: 14px; line-height: 1.4;">
<h2>Blocks Tracker - Summary</h2>

<p>Below is your automated analysis summary generated from the block/unblock audit tool.</p>

<hr>
{ai_summary_text}
<hr>

<p style="color: #888; font-size: 12px;">
This report was auto-generated by the Liftoff AI Audit System.
</p>

</body>
</html>
"""
    return html_wrapper


def send_email_summary(
    html_body: str,
    to_email: str,
    subject: str = "[ai]udit - Summary of current blocks",
    from_email: str = "your_gmail@gmail.com",        # Filled from Streamlit UI
    gmail_app_password: str = "YOUR_GMAIL_APP_PASSWORD_HERE",  # Filled from Streamlit UI
):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email

    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(from_email, gmail_app_password)
        server.send_message(msg)


# =====================================================================
# 5. PUBLIC ENTRYPOINT FOR STREAMLIT
# =====================================================================

def run_full_pipeline(
    target_app_ids,
    excluded_block_values,
    recipient_email,
    sender_email,
    gmail_app_password,
    openai_api_key,
):
    """
    Main function to be called by Streamlit.
    - Runs full multi-app analysis
    - Optionally builds AI HTML summary (if Claude key provided)
    - Optionally sends email (if summary built + SMTP info provided)
    - Always returns:
        * combined tables (all apps)
        * per-app tables (4 main tables per app)
        * html_summary (or None on failure / not configured)
        * email_status + ai_error (for UX/debugging)
    """

    all_legends = []
    all_blocks_with_spend = []
    all_blocks_with_global = []
    all_summary_raw = []
    all_rev_matrices = []

    # Per-app results for Streamlit
    per_app_results = {}

    # ---------- PER-APP LOOP ----------
    for app_id in target_app_ids:
        (
            similar_apps_df,
            df_blocks_with_spend,
            df_blocks_with_global,
            summary_raw,
            rev_matrix,
            legend_text,
        ) = run_full_analysis(
            app_id,
            n_similar=5,
            min_spend=0.0,
            excluded_block_values=excluded_block_values,
        )

        all_legends.append(legend_text)
        all_blocks_with_spend.append(df_blocks_with_spend)
        all_blocks_with_global.append(df_blocks_with_global)
        all_summary_raw.append(summary_raw)
        all_rev_matrices.append(rev_matrix)

        # 4 key tables per app
        per_app_results[app_id] = {
            "legend_text": legend_text,
            "blocks_with_spend": df_blocks_with_spend,
            "blocks_with_global": df_blocks_with_global,
            "summary_per_app": summary_raw,
            "competitor_rev_matrix": rev_matrix,
        }

    # ---------- COMBINED TABLES (ALL APPS) ----------
    combined_legend = "\n\n".join(all_legends)

    # Table 1: Blocks with L7D DSP spend
    combined_blocks_with_spend = pd.concat(
        all_blocks_with_spend, ignore_index=True, sort=False
    )
    combined_blocks_with_spend = combined_blocks_with_spend.sort_values(
        "non_vx_spend_l7d", ascending=False
    ).reset_index(drop=True)

    # Table 2: Global network spend L30D
    combined_blocks_with_global = pd.concat(
        all_blocks_with_global, ignore_index=True, sort=False
    )
    combined_blocks_with_global = combined_blocks_with_global.sort_values(
        "vx_overview_ad_spend_l30d", ascending=False
    ).reset_index(drop=True)

    # Table 3: Summary per app (our apps only, raw numbers for AI)
    combined_summary_raw = pd.concat(all_summary_raw, ignore_index=True, sort=False)
    mask_our_app = ~combined_summary_raw["app_label"].str.startswith("Similar app")
    combined_summary_our = combined_summary_raw[mask_our_app].copy()
    combined_summary_our = combined_summary_our.sort_values(
        "block_rate_zscore", ascending=False
    ).reset_index(drop=True)

    # Table 4: Competitor revenue matrix
    combined_rev_matrix = pd.concat(all_rev_matrices, ignore_index=True, sort=False)
    combined_rev_matrix = combined_rev_matrix.sort_values(
        "total_competitor_rev_l7d", ascending=False
    ).reset_index(drop=True)

    # Table 5: High-level summary metrics
    total_non_vx_l7d = combined_blocks_with_spend["non_vx_spend_l7d"].sum()
    total_network_spend_l30d = combined_blocks_with_global["vx_overview_ad_spend_l30d"].sum()

    if not combined_rev_matrix.empty:
        similar_rev_cols = [
            c
            for c in combined_rev_matrix.columns
            if c.endswith("_rev_l7d") and c.startswith("Similar app")
        ]

        if similar_rev_cols:
            n_similar = len(similar_rev_cols)
            similar_spend = combined_rev_matrix[similar_rev_cols] > 0
            blocked_by_count = similar_spend.sum(axis=1)

            less_than_half_mask = blocked_by_count < (n_similar / 2.0)
            num_blocks_less_than_half_similar = int(less_than_half_mask.sum())

            total_competitor_revenue = combined_rev_matrix["total_competitor_rev_l7d"].sum()
        else:
            n_similar = 0
            num_blocks_less_than_half_similar = 0
            total_competitor_revenue = 0.0
    else:
        n_similar = 0
        num_blocks_less_than_half_similar = 0
        total_competitor_revenue = 0.0

    summary_metrics = pd.DataFrame(
        [
            {
                "metric": "Total L7D non-VX DSP spend across all blocks",
                "value": total_non_vx_l7d,
            },
            {
                "metric": "Total L30D global network advertiser spend across all blocks",
                "value": total_network_spend_l30d,
            },
            {
                "metric": "Number of blocks where fewer than half of similar apps see revenue",
                "value": num_blocks_less_than_half_similar,
            },
            {
                "metric": "Total L7D competitor/similar apps revenue across all blocks",
                "value": total_competitor_revenue,
            },
        ]
    )

    # ---------- AI SUMMARY + EMAIL (SAFE WRAP) ----------
    html_summary = None
    ai_error = None

    claude_key_clean = (openai_api_key or "").strip()
    if not claude_key_clean:
        email_status = "ai_not_configured"
    else:
        try:
            html_summary = build_ai_summary_html(
                target_app_ids=target_app_ids,
                combined_legend=combined_legend,
                combined_blocks_with_spend=combined_blocks_with_spend,
                combined_blocks_with_global=combined_blocks_with_global,
                combined_summary_our=combined_summary_our,
                combined_rev_matrix=combined_rev_matrix,
                summary_metrics=summary_metrics,
                openai_api_key=claude_key_clean,
            )
            email_status = "summary_built"

            # Only try sending email if user actually provided creds
            if recipient_email and sender_email and gmail_app_password:
                send_email_summary(
                    html_body=html_summary,
                    to_email=recipient_email,
                    from_email=sender_email,
                    gmail_app_password=gmail_app_password,
                )
                email_status = "email_sent"

        except Exception as e:
            # We swallow AI/email errors here so tables still return
            ai_error = str(e)
            email_status = "failed_ai_or_email"

    # ---------- RETURN TO STREAMLIT ----------
    return {
        # AI summary / email info
        "html_summary": html_summary,          # may be None
        "email_status": email_status,          # 'email_sent', 'summary_built', 'failed_ai_or_email', 'ai_not_configured'
        "ai_error": ai_error,                  # error string or None

        # Combined outputs
        "combined_legend": combined_legend,
        "combined_blocks_with_spend": combined_blocks_with_spend,
        "combined_blocks_with_global": combined_blocks_with_global,
        "combined_summary_our": combined_summary_our,
        "combined_rev_matrix": combined_rev_matrix,
        "summary_metrics": summary_metrics,

        # Per-app tables (each app_id -> 4 tables)
        "per_app_results": per_app_results,
    }
