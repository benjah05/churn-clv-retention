""" local Streamlit dashboard: customer behavior, churn, customer value, model performance and retention """

import html
import io
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import (brier_score_loss, confusion_matrix, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)

st.set_page_config(page_title="Churn & Retention Dashboard", page_icon=":material/insights:",
                   layout="wide")

# constants
PRIORITY_ORDER = ["Retain now", "Monitor closely", "Low-cost action", "Maintain"]
SEGMENT_ORDER = ["High risk, high value", "Low risk, high value",
                 "High risk, low value", "Low risk, low value"]
LEVELS = ["low", "medium", "high"]

ACCENT = "#2f6fdf"
SECOND = "#e3a33b"
GREY = "#9aa5b1"
RISK_COLORS = alt.Scale(domain=LEVELS, range=["#5b9e6f", "#e3a33b", "#d1553f"])
RISK_HEX = dict(zip(LEVELS, ["#5b9e6f", "#e3a33b", "#d1553f"]))
PRIORITY_HEX = dict(zip(PRIORITY_ORDER, ["#d1553f", "#e3a33b", "#4c78a8", "#5b9e6f"]))

HORIZON_DAYS = 90             # the CLV model predicts revenue for the next 90 days, same as churn
DAYS_PER_MONTH = 365.25 / 12  # average month length
SCORING_DATE = "2 Jul 2026"

# figures from the notebook that are not in the exported files (update them if the notebook changes)
CLV_CUSTOMERS = 2053        # customers in the CLV dataset (CLV dataset cell)
RAW_AVERAGE_SCORE = 0.336   # average raw random forest score before calibration (calibration cell)
RAW_BRIER = 0.143           # Brier score before calibration (calibration cell)

FEATURE_LABELS = {
    "active_days_30d": "Active days, last 30 days",
    "revenue_30d": "Revenue, last 30 days",
    "recharge_count_30d": "Recharges, last 30 days",
    "active_day_share": "Share of days active",
    "avg_daily_revenue": "Average daily revenue",
    "revenue_change_30d": "Revenue change, last 30 days",
    "revenue_change_90d": "Revenue change, last 90 days",
    "recharge_count_change": "Change in recharges",
    "active_days_change": "Change in active days",
    "avg_recharge_amount": "Average recharge amount",
    "median_recharge_gap": "Typical days between recharges",
    "days_since_last_revenue": "Days since last revenue",
    "days_since_last_recharge": "Days since last recharge",
    "data_revenue_share": "Share of revenue from data",
    "voice_revenue_share": "Share of revenue from voice",
    "tenure_days": "Days since first seen",
    "revenue_61_180d": "Revenue 2-6 months ago",
    "revenue_change_vs_before_drop": "Revenue change vs 2-6 months ago",
    "revenue_trend": "Revenue trend, last 6 months",
}

MEASURE_LABELS = {
    "total_revenue": "Total revenue",
    "data_revenue": "Data revenue",
    "voice_revenue": "Voice revenue",
    "sms_revenue": "SMS revenue",
    "data_traffic": "Data traffic",
    "voice_traffic": "Voice traffic",
    "recharge_amount": "Recharge amount",
    "recharge_count": "Number of recharges",
    "active_customers": "Customers with revenue",
}

SUMMARY_LABELS = {
    "customers": "Customers",
    "share_of_customers": "Share of customers",
    "actual_churn_rate": "Actual churn rate",
    "avg_predicted_clv": "Avg predicted revenue (90 days)",
    "total_predicted_clv": "Total predicted revenue (90 days)",
    "revenue_at_risk": "Expected revenue at risk",
    "share_of_revenue_at_risk": "Share of revenue at risk",
}
SUMMARY_FORMATS = {
    "Customers": "{:,.0f}", "Share of customers": "{:.1%}", "Actual churn rate": "{:.1%}",
    "Avg predicted revenue (90 days)": "{:,.0f}", "Total predicted revenue (90 days)": "{:,.0f}",
    "Expected revenue at risk": "{:,.0f}", "Share of revenue at risk": "{:.1%}",
}

# customer-level files: not in the public repository, uploaded in the sidebar instead
CUSTOMER_FILES = {
    "retention_output.csv": "Churn probability, predicted revenue, risk, value and priority of every scored "
                            "customer. Needed by every page except Overview charts and Customer behavior.",
    "customer_activity.csv": "Monthly activity of each customer. Charts on Customer lookup, Customer value and "
                             "Campaign lists.",
    "customer_signals.csv": "Each customer's behavior at the scoring date. Tips on Customer lookup and the "
                            "behavior table on Campaign lists.",
    "churn_test.csv": "Churn model results on the test customers. Model performance page.",
    "clv_test.csv": "CLV model results on the test customers. Model performance page.",
}

# recommended action per priority: (color for st.markdown, action, reason)
ACTIONS = {
    "Retain now": ("red", "Contact within 7 days",
                   "High churn risk and valuable now or until recently. Contact them first, through the most "
                   "direct channel available, before they leave."),
    "Monitor closely": ("orange", "Send a targeted loyalty offer",
                        "Valuable customer showing early signs of churn. A personalized reward now can "
                        "prevent a loss. Review again next month."),
    "Low-cost action": ("blue", "Automated low-cost offer",
                        "At risk but with a lower value. Use cheap, automated contact such as an "
                        "SMS with a small recharge or bundle bonus."),
    "Maintain": ("green", "Business as usual",
                 "Low churn risk. Keep normal engagement; no special retention spend is needed."),
}

# styling
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .kpi {
        background: rgba(128, 128, 128, 0.07);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-left: 4px solid #2f6fdf;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .kpi-label {font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; opacity: 0.7;}
    .kpi-value {font-size: 1.6rem; font-weight: 700; line-height: 1.35;}
    .kpi-note {font-size: 0.8rem; opacity: 0.65; min-height: 1.15em;}
    .app-footer {
        position: fixed; right: 16px; bottom: 8px;
        font-size: 0.8rem; opacity: 0.6; z-index: 100;
    }
    </style>
    <div class="app-footer">MTN Rwanda</div>
    """,
    unsafe_allow_html=True,
)


# loading
@st.cache_data
def read_csv_cached(source, index_col=None):
    """ read a CSV from a file path or from uploaded bytes """
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    return pd.read_csv(source, index_col=index_col)


def load_optional(path, index_col=None):
    """ read a results file from the folder or from the uploads, or return None if it is not available """
    if Path(path).exists():
        return read_csv_cached(path, index_col)
    uploaded = st.session_state.get("uploads", {}).get(path)
    return read_csv_cached(uploaded, index_col) if uploaded is not None else None


def load_retention():
    """ read the retention results and restore the category orders, or None if not available """
    df = load_optional("retention_output.csv", index_col="customer_id")
    if df is None:
        return None
    df["retention_priority"] = pd.Categorical(df["retention_priority"], PRIORITY_ORDER, ordered=True)
    df["segment"] = pd.Categorical(df["segment"], SEGMENT_ORDER, ordered=True)
    for col in ["risk_category", "value_category", "predicted_value_category", "past_value_category"]:
        if col in df.columns:
            df[col] = pd.Categorical(df[col], LEVELS, ordered=True)
    return df


@st.cache_data
def load_json(path):
    """ read a JSON file, or return None if it hasn't been exported yet """
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


# small helper functions
def fmt_big(x):
    """ 46,662,210 -> 46.7M """
    if abs(x) >= 1e6:
        return f"{x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:,.1f}K"
    return f"{x:,.0f}"


def kpi(col, label, value, note="", accent=ACCENT):
    """ a dashboard card; all text is escaped because it is placed inside HTML """
    col.markdown(
        f'<div class="kpi" style="border-left-color:{accent}">'
        f'<div class="kpi-label">{html.escape(str(label))}</div>'
        f'<div class="kpi-value">{html.escape(str(value))}</div>'
        f'<div class="kpi-note">{html.escape(str(note))}</div></div>',
        unsafe_allow_html=True,
    )


def takeaway(text):
    """ one short insight under a chart: what to conclude from it """
    st.caption(f":material/lightbulb: {text}")


def card(title=None, caption=None):
    """ a bordered container for a chart or table, like a dashboard tile """
    box = st.container(border=True)
    if title:
        box.markdown(f"**{title}**")
    if caption:
        box.caption(caption)
    return box


def notes(title):
    """ explanation under a chart: a drop-down heading that can be clicked open """
    return st.expander(title)


def show(chart):
    """
    draw an Altair chart full width.
    labelLimit=0 stops long axis and legend labels being shortened with '...',
    and the top padding leaves room for numbers written above the tallest bar.
    """
    chart = chart.properties(padding={"top": 18, "right": 10, "bottom": 5, "left": 5})
    chart = (chart.configure_axis(labelLimit=0, titleLimit=0)
             .configure_legend(labelLimit=0, titleLimit=0))
    st.altair_chart(chart, width="stretch")


def for_chart(df):
    """ Altair works best with plain text, so turn category columns into strings """
    out = df.reset_index()
    for col in out.select_dtypes("category").columns:
        out[col] = out[col].astype(str)
    return out


def summarize(df, group_col):
    """ same idea as retention_summary in the notebook """
    s = df.groupby(group_col, observed=True).agg(
        customers=("predicted_clv", "size"),
        avg_predicted_clv=("predicted_clv", "mean"),
        total_predicted_clv=("predicted_clv", "sum"),
        revenue_at_risk=("revenue_at_risk", "sum"),
        actual_churn_rate=("churned", "mean"),
    )
    s["share_of_customers"] = s["customers"] / s["customers"].sum()
    s["share_of_revenue_at_risk"] = s["revenue_at_risk"] / s["revenue_at_risk"].sum()
    return s


def show_summary(s):
    """ summary table with readable column names and number formats """
    pretty = s[list(SUMMARY_LABELS)].rename(columns=SUMMARY_LABELS)
    pretty.index = pretty.index.astype(str)
    pretty.index.name = None
    st.dataframe(pretty.style.format(SUMMARY_FORMATS), width="stretch")


def share_comparison(df):
    """ clustered bars: share of customers vs share of revenue at risk per priority """
    s = summarize(df, "retention_priority")[["share_of_customers", "share_of_revenue_at_risk"]]
    long = for_chart(s).melt(id_vars="retention_priority", var_name="measure", value_name="share")
    long["measure"] = long["measure"].map({"share_of_customers": "Share of customers",
                                           "share_of_revenue_at_risk": "Share of revenue at risk"})
    bars = alt.Chart(long).mark_bar().encode(
        x=alt.X("retention_priority:N", sort=PRIORITY_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("measure:N"),
        y=alt.Y("share:Q", title=None, axis=alt.Axis(format="%")),
        color=alt.Color("measure:N", scale=alt.Scale(range=[ACCENT, SECOND]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["retention_priority", "measure", alt.Tooltip("share:Q", format=".1%")])
    labels = bars.mark_text(dy=-7, fontSize=11).encode(text=alt.Text("share:Q", format=".0%"))
    return (bars + labels).properties(height=300)


def rate_bars(df, x, order, x_title, color=None):
    """ bar chart of churn rates with the percentage written above each bar """
    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{x}:N", sort=order, title=x_title, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("churn_rate:Q", title="Actual churn rate", axis=alt.Axis(format="%")),
        color=color if color is not None else alt.value(ACCENT),
        tooltip=[x, alt.Tooltip("churn_rate:Q", format=".1%"), "customers"])
    labels = bars.mark_text(dy=-8).encode(text=alt.Text("churn_rate:Q", format=".1%"))
    return (bars + labels).properties(height=280)


def count_bars(df, label_col, value_col="customers", color=ACCENT, height=None, total=None):
    """ horizontal bars with the count (and share of total) written at the end of each bar """
    total = total or df[value_col].sum()
    df = df.assign(text=df[value_col].map(lambda v: f"{v:,.0f} ({v / total:.0%})"))
    base = alt.Chart(df).encode(
        y=alt.Y(f"{label_col}:N", sort=None, title=None),
        x=alt.X(f"{value_col}:Q", title="Customers", scale=alt.Scale(domainMax=df[value_col].max() * 1.3)),
        tooltip=[label_col, alt.Tooltip(f"{value_col}:Q", format=",.0f")])
    bars = base.mark_bar(color=color)
    # same color as the bars, so the numbers are readable in light and dark mode
    labels = base.mark_text(align="left", dx=4, color=color).encode(text="text:N")
    return (bars + labels).properties(height=height or 45 * len(df) + 30)


def priority_rule(risk, value):
    """ the prioritization rules from the notebook: risk first, then value """
    if risk == "low":
        return "Maintain"
    if value == "low":
        return "Low-cost action"
    return "Retain now" if risk == "high" else "Monitor closely"


def priority_grid(df):
    """ 3x3 grid of risk by value, each cell colored by its priority with the customer count """
    counts = pd.crosstab(df["risk_category"], df["value_category"])
    cells = pd.DataFrame([(r, v) for r in LEVELS for v in LEVELS], columns=["risk", "value"])
    cells["priority"] = [priority_rule(r, v) for r, v in zip(cells["risk"], cells["value"])]
    cells["customers"] = [int(counts.at[r, v]) if r in counts.index and v in counts.columns else 0
                          for r, v in zip(cells["risk"], cells["value"])]
    cells["count_label"] = cells["customers"].map("{:,} customers".format)
    base = alt.Chart(cells).encode(
        x=alt.X("value:N", sort=LEVELS, title="Value", axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("risk:N", sort=LEVELS[::-1], title="Risk"))
    rects = base.mark_rect(stroke="white", strokeWidth=3).encode(
        color=alt.Color("priority:N", legend=None,
                        scale=alt.Scale(domain=PRIORITY_ORDER, range=[PRIORITY_HEX[p] for p in PRIORITY_ORDER])),
        tooltip=["risk", "value", "priority", alt.Tooltip("customers:Q", format=",")])
    # dark text on the orange cells, white on the rest, so the text stays readable
    text_color = alt.condition(alt.datum.priority == "Monitor closely", alt.value("black"), alt.value("white"))
    names = base.mark_text(fontSize=14, fontWeight="bold", dy=-9).encode(text="priority:N", color=text_color)
    numbers = base.mark_text(fontSize=12, dy=10).encode(text="count_label:N", color=text_color)
    return (rects + names + numbers).properties(height=300)


def heatmap(df, x, y, x_order, y_order):
    """ grid of counts with the number written in each cell """
    base = alt.Chart(df).encode(x=alt.X(f"{x}:N", sort=x_order, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y(f"{y}:N", sort=y_order))
    cells = base.mark_rect().encode(
        color=alt.Color("customers:Q", scale=alt.Scale(scheme="blues"), legend=None))
    middle = float(df["customers"].max()) / 2
    labels = base.mark_text(fontSize=16).encode(
        text="customers:Q",
        color=alt.condition(alt.datum.customers > middle, alt.value("white"), alt.value("black")))
    return (cells + labels).properties(height=260)


def need_files(names):
    """ True when all files are available; otherwise list the missing ones and what they are for """
    missing = [n for n in names if load_optional(n) is None]
    if not missing:
        return True
    with card("Customer files needed",
              "This page uses private customer-level files that are not stored online. Upload them in the "
              "sidebar under Customer files (they are exported by the notebook)."):
        st.table(pd.DataFrame({"Used for": [CUSTOMER_FILES[n] for n in missing]},
                              index=pd.Index(missing, name="File to upload")))
    return False


def customer_upload_box():
    """
    sidebar upload for the customer-level files. Shown only when some are not in the folder
    (the deployed app); the files are kept in memory for this session only.
    """
    missing = [n for n in CUSTOMER_FILES if not Path(n).exists()]
    if not missing:
        return
    uploads = st.session_state.setdefault("uploads", {})
    with st.sidebar.expander("Customer files", icon=":material/upload_file:",
                             expanded=any(n not in uploads for n in missing)):
        st.caption("Upload these files exported by the notebook. They are private, used for this session "
                   "only and not stored online.")
        files = st.file_uploader("Customer files", type="csv", accept_multiple_files=True,
                                 key="customer_upload", label_visibility="collapsed")
        for f in files or []:
            if f.name in CUSTOMER_FILES:
                uploads[f.name] = f.getvalue()
            else:
                st.warning(f"{f.name} is not one of the files below. Keep the original file names.")
        for name in missing:
            mark = ":green[:material/check_circle:]" if name in uploads else ":red[:material/cancel:]"
            st.markdown(f"{mark} `{name}`")


def has_customers():
    """ show which file is missing, or a warning when the sidebar filters leave nobody """
    if data is None:
        need_files(["retention_output.csv"])
        return False
    if filtered.empty:
        st.warning("No customers match the sidebar filters.")
        return False
    return True


def recommend(row, signals):
    """ recommended action for one customer: priority first, then their own behavior """
    color, action, reason = ACTIONS[str(row["retention_priority"])]
    tips = []
    if signals is not None:
        days = signals["days_since_last_revenue"]
        if days >= 30:
            tips.append(f"No revenue for {days:.0f} days before the scoring date: lead with a small offer or a "
                        "win-back offer.")
        elif days >= 14:
            tips.append(f"Last paid {days:.0f} days before the scoring date: a free reminder is a good first step.")
        if signals["active_days_30d"] == 0:
            tips.append("No active days in the last 30 days before the scoring date.")
        previous = signals["revenue_60d"] - signals["revenue_30d"]
        if previous > 0:
            change = (signals["revenue_30d"] - previous) / previous
            if change <= -0.3:
                tips.append(f"Spending fell {abs(change):.0%} in the last 30 days compared with the 30 days before.")
            elif change >= 0.3:
                tips.append(f"Spending rose {change:.0%} in the last 30 days, a good moment to reinforce loyalty.")
        if signals["data_revenue_share"] >= 0.6:
            tips.append("Spends mostly on data, so a data bundle offer fits best.")
        elif signals["voice_revenue_share"] >= 0.6:
            tips.append("Spends mostly on voice, so a voice or minutes bundle fits best.")
    if "past_value_category" in row.index and "predicted_value_category" in row.index:
        before, predicted = row["past_value_category"], row["predicted_value_category"]
        if pd.notna(before) and pd.notna(predicted) and LEVELS.index(str(before)) > LEVELS.index(str(predicted)):
            tips.append(f"Was a {before} spender until about two months ago, but spending has dropped: "
                        "winning them back is worth more than the forecast suggests.")
    if str(row["retention_priority"]) == "Maintain" and str(row["value_category"]) == "high":
        tips.append("High-value, stable customer: a good candidate for loyalty rewards or upselling.")
    return color, action, reason, tips


# pages
def page_overview():
    st.title("Overview")
    st.caption(f"Whole dataset, Sep 2025 to Sep 2026. Scoring date: {SCORING_DATE}, the day customers are scored; "
               "the 90 days after it show who actually churned.")

    overview = load_json("eda_overview.json")
    facts = load_json("eda_facts.json")
    monthly = load_optional("eda_monthly.csv")
    revenue_mix = load_optional("eda_revenue_mix.csv")
    inactivity = load_optional("eda_inactivity.csv")
    recency = load_optional("eda_churn_by_recency.csv")
    churn_corr = load_optional("eda_churn_corr.csv")
    if any(f is None for f in [overview, monthly, revenue_mix, inactivity, recency, churn_corr]):
        st.info("Run the EDA export cell in the notebook to create the eda_ files.")
        return

    revenue_mix["share"] = revenue_mix["revenue"] / revenue_mix["revenue"].sum()
    revenue_mix["revenue_type"] = revenue_mix["revenue_type"].map(
        {"data_revenue": "Data", "voice_revenue": "Voice", "sms_revenue": "SMS"})
    share_of = dict(zip(revenue_mix["revenue_type"], revenue_mix["share"]))

    # KPI row
    k = st.columns(5)
    kpi(k[0], "Customers in dataset", f"{overview['customers']:,}",
        f"{pd.to_datetime(overview['start']):%b %Y} – {pd.to_datetime(overview['end']):%b %Y}, "
        f"{overview['days']} days")
    kpi(k[1], "Customers with revenue", f"{overview['paying_customers']:,}",
        f"{overview['paying_customers'] / overview['customers']:.0%} of all customers")
    kpi(k[2], "Total revenue", fmt_big(revenue_mix["revenue"].sum()),
        f"voice {share_of.get('Voice', 0):.0%} · data {share_of.get('Data', 0):.0%}")
    kpi(k[3], f"Active on {SCORING_DATE}", f"{overview['churn_customers']:,}",
        f"{overview['dormant_at_cutoff']:,} already dormant")
    kpi(k[4], "Churn rate", f"{overview['churn_rate']:.1%}", "no revenue in the next 90 days",
        accent=RISK_HEX["high"])

    # key insights
    came_back_90 = inactivity.loc[inactivity["days_inactive"] == 90, "came_back"].iloc[0]
    strongest = churn_corr.loc[churn_corr["correlation"].abs().idxmax()]
    strongest_name = FEATURE_LABELS.get(strongest["feature"], strongest["feature"])
    lines = [
        f"- **{overview['churn_rate']:.1%} of active customers churned** in the 90 days after the scoring date "
        f"({round(overview['churn_rate'] * overview['churn_customers']):,} of {overview['churn_customers']:,}).",
        f"- **Recency is the clearest warning sign:** customers who last paid {recency['group'].iloc[0]} days before "
        f"the scoring date churned at {recency['churn_rate'].iloc[0]:.0%}, against "
        f"{recency['churn_rate'].iloc[-1]:.0%} for {recency['group'].iloc[-1]} days.",
        f"- The feature most related to churn is **{strongest_name.lower()}** "
        f"(correlation {strongest['correlation']:.2f}).",
        f"- Revenue is split almost evenly between **voice ({share_of.get('Voice', 0):.0%})** and "
        f"**data ({share_of.get('Data', 0):.0%})**; SMS is negligible.",
        f"- Only **{came_back_90:.0%}** of inactive spells lasting 90 days or more ended with the customer "
        "paying again, which supports the 90-day churn definition.",
    ]
    if facts is not None:
        lines.append(
            f"- **{facts['data_without_data_revenue']:,} customers used data without any data revenue**, and "
            f"**{facts['gap_positive']:,}** recharged more than their recorded revenue. Part of the spending, possibly "
            "data bundles, seems not to be recorded as revenue (to be confirmed).")
    if data is not None:
        by_priority = summarize(data, "retention_priority")
        focus = by_priority.loc[["Retain now", "Monitor closely"]]
        lines.append(
            f"- **Retain now and Monitor closely** cover {focus['customers'].sum():,} customers "
            f"({focus['share_of_customers'].sum():.0%}) but {focus['share_of_revenue_at_risk'].sum():.0%} of the "
            "expected revenue at risk.")
    with card("Key insights"):
        st.markdown("\n".join(lines))

    # monthly performance + revenue mix
    left, right = st.columns([2, 1])
    with left, card("Monthly performance"):
        metric_labels = {"total_revenue": "Total revenue",
                         "paying_customers": "Customers with revenue",
                         "median_revenue_per_payer": "Median revenue per paying customer"}
        metric = st.radio("Show", list(metric_labels), format_func=metric_labels.get,
                          horizontal=True, label_visibility="collapsed")
        base = alt.Chart(monthly).encode(
            x=alt.X("month:N", sort=list(monthly["month"]), title=None,
                    axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
            y=alt.Y(f"{metric}:Q", title=metric_labels[metric]),
            tooltip=["month", alt.Tooltip(f"{metric}:Q", format=",.0f")])
        marks = base.mark_bar(color=ACCENT) if metric == "total_revenue" else base.mark_line(
            point=True, color=ACCENT)
        average = alt.Chart(monthly).mark_rule(strokeDash=[4, 4], color="gray").encode(
            y=f"mean({metric}):Q")
        show((marks + average).properties(height=300))
        best = monthly.loc[monthly[metric].idxmax()]
        worst = monthly.loc[monthly[metric].idxmin()]
        st.caption(f"Highest: {best['month']} ({best[metric]:,.0f}) · Lowest: {worst['month']} "
                   f"({worst[metric]:,.0f}) · dashed line = monthly average")
        first, last = monthly.iloc[0], monthly.iloc[-1]
        change = last[metric] / first[metric] - 1
        word = "higher" if change >= 0 else "lower"
        if metric == "total_revenue":
            takeaway(f"Revenue ended the year {abs(change):.0%} {word} than it started, with a one-off peak in "
                     f"{best['month']}.")
        elif metric == "paying_customers":
            takeaway(f"{abs(change):.0%} fewer customers pay than a year ago, yet revenue held up because each one "
                     "spends more." if change < 0 else f"{change:.0%} more customers pay than a year ago.")
        else:
            takeaway(f"A typical paying customer spends {abs(change):.0%} {'more' if change >= 0 else 'less'} than a "
                     "year ago.")

    with right, card("Revenue by type"):
        donut = alt.Chart(revenue_mix).mark_arc(innerRadius=70).encode(
            theta=alt.Theta("revenue:Q"),
            color=alt.Color("revenue_type:N", title=None, legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=["Voice", "Data", "SMS"], range=[ACCENT, SECOND, GREY])),
            tooltip=["revenue_type", alt.Tooltip("revenue:Q", format=",.0f"),
                     alt.Tooltip("share:Q", format=".1%")])
        show(donut.properties(height=330))

    # retention at a glance
    if data is None:
        need_files(["retention_output.csv"])
        return
    with card("Retention at a glance",
              f"All {len(data):,} customers active at the scoring date ({SCORING_DATE})."):
        show(share_comparison(data))
        ratio = by_priority["share_of_revenue_at_risk"] / by_priority["share_of_customers"]
        takeaway(f"{ratio.idxmax()} holds {ratio.max():.1f} times its share of customers in revenue at risk, so effort "
                 "there pays off most.")


def page_behavior():
    st.title("Customer behavior")
    st.caption("Exploratory analysis of all 3,500 customers, Sep 2025 to Sep 2026. Aggregated figures only.")

    overview = load_json("eda_overview.json")
    facts = load_json("eda_facts.json")
    daily = load_optional("eda_daily.csv")
    dist = load_optional("eda_distribution_summary.csv")
    hist = load_optional("eda_histograms.csv")
    corr = load_optional("eda_correlations.csv")
    active_share = load_optional("eda_active_share.csv")
    trends = load_optional("eda_trends.csv")
    inactivity = load_optional("eda_inactivity.csv")
    if any(f is None for f in [overview, facts, daily, dist, hist, corr, active_share, trends, inactivity]):
        st.info("Run both EDA export cells in the notebook (eda_overview.json, eda_facts.json, eda_daily.csv, "
                "eda_distribution_summary.csv, eda_histograms.csv, eda_correlations.csv, eda_active_share.csv).")
        return

    # customer base
    st.subheader("Customer base")
    never_paid = overview["customers"] - overview["paying_customers"]
    k = st.columns(4)
    kpi(k[0], "Customers", f"{overview['customers']:,}", f"{overview['days']} days of daily records")
    kpi(k[1], "Data completeness", f"{facts['rows'] / facts['expected_rows']:.1%}",
        f"{facts['expected_rows'] - facts['rows']:,} customer-days missing · {facts['missing_dates']} missing dates")
    kpi(k[2], "Full history", f"{facts['customers_full_history']:,}",
        f"customers with a row every day · shortest {facts['min_days_per_customer']} days")
    kpi(k[3], "Never generated revenue", f"{never_paid:,}",
        f"{never_paid / overview['customers']:.0%} of customers", accent=GREY)
    base = pd.DataFrame({
        "group": ["Active at the scoring date", "Paid before, then went quiet", "No revenue before the scoring date"],
        "customers": [overview["churn_customers"], facts["dormant_paid_before"], facts["dormant_never_paid"]],
    })
    with card("Where customers stood at the scoring date",
              f"Only customers with revenue in the 90 days before {SCORING_DATE} are scored for churn."):
        show(count_bars(base, "group"))
        takeaway(f"Only {overview['churn_customers'] / overview['customers']:.0%} of customers were still paying around "
                 "the scoring date, and the churn model covers only them.")

    # activity over time
    st.subheader("Activity over time across all customers")
    daily["date"] = pd.to_datetime(daily["date"])
    daily_cols = [c for c in ["total_revenue", "active_customers", "data_traffic", "voice_traffic",
                              "recharge_amount"] if c in daily.columns]
    with card("Daily totals", "Thin line = each day · thick line = 7-day average."):
        metric = st.radio("Daily metric", daily_cols, format_func=MEASURE_LABELS.get, horizontal=True,
                          label_visibility="collapsed")
        series = daily[["date", metric]].copy()
        series["average"] = series[metric].rolling(7, min_periods=1).mean()
        raw = alt.Chart(series).mark_line(color=GREY, strokeWidth=1, opacity=0.6).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b-%Y", labelAngle=-45)),
            y=alt.Y(f"{metric}:Q", title=MEASURE_LABELS[metric]),
            tooltip=[alt.Tooltip("date:T", format="%d %b %Y"), alt.Tooltip(f"{metric}:Q", format=",.0f")])
        smooth = alt.Chart(series).mark_line(color=ACCENT, strokeWidth=2.5).encode(
            x="date:T", y="average:Q")
        show((raw + smooth).properties(height=300))
        start_avg, end_avg = series[metric].head(30).mean(), series[metric].tail(30).mean()
        change = end_avg / start_avg - 1
        sentence = f"The last month was {abs(change):.0%} {'higher' if change >= 0 else 'lower'} than the first month."
        if metric in ("total_revenue", "active_customers") and {"total_revenue", "active_customers"} <= set(daily.columns):
            rev = daily["total_revenue"].tail(30).mean() / daily["total_revenue"].head(30).mean() - 1
            act = daily["active_customers"].tail(30).mean() / daily["active_customers"].head(30).mean() - 1
            sentence = (f"Comparing the last month with the first, daily revenue went {rev:+.0%} while paying "
                        f"customers went {act:+.0%}: fewer customers are each spending more.")
        gaps = int((series[metric] < 0.1 * series[metric].median()).sum())
        if gaps:
            sentence += (f" {gaps} day{'s' if gaps > 1 else ''} with almost nothing recorded "
                         f"{'are' if gaps > 1 else 'is'} probably missing data.")
        takeaway(sentence)

    # distributions
    st.subheader("Revenue and usage distributions")
    dist = dist.set_index("measure")
    pretty = pd.DataFrame({
        "Median": dist["median"], "Mean": dist["mean"], "Max": dist["max"],
        "Customers at zero": dist["zero_share_%"] / 100, "Skewness": dist["skewness"],
    })
    pretty.index = [MEASURE_LABELS.get(i, i) for i in pretty.index]
    left, right = st.columns([2, 3])
    with left, card("Per-customer totals over the whole period",
                    "Mean far above the median = a few heavy users pull the average up."):
        st.dataframe(pretty.style.format({"Median": "{:,.0f}", "Mean": "{:,.0f}", "Max": "{:,.0f}",
                                          "Customers at zero": "{:.0%}", "Skewness": "{:.1f}"}),
                     width="stretch")
    with right, card("Distribution per customer", "Customers above zero only · log scale: each step is ×10."):
        measures = list(dict.fromkeys(hist["measure"]))
        chosen = st.selectbox("Measure", measures, format_func=lambda m: MEASURE_LABELS.get(m, m),
                              label_visibility="collapsed")
        part = hist[hist["measure"] == chosen]
        show(alt.Chart(part).mark_bar(color=ACCENT).encode(
            x=alt.X("from:Q", scale=alt.Scale(type="log"), title=MEASURE_LABELS.get(chosen, chosen)),
            x2="to:Q",
            y=alt.Y("customers:Q", title="Customers"),
            tooltip=[alt.Tooltip("from:Q", format=",.0f"), alt.Tooltip("to:Q", format=",.0f"), "customers"],
        ).properties(height=260))

    # data usage without data revenue
    st.subheader("Data usage without data revenue")
    k = st.columns(4)
    kpi(k[0], "No data revenue", f"{facts['zero_data_revenue']:,}",
        f"{facts['zero_data_revenue'] / overview['customers']:.0%} of customers", accent=SECOND)
    kpi(k[1], "…but used data", f"{facts['data_without_data_revenue']:,}",
        "data traffic with no data revenue", accent=SECOND)
    kpi(k[2], "…and paid for voice", f"{facts['data_without_data_revenue_with_voice']:,}",
        "of those who used data")
    kpi(k[3], "…and paid nothing at all", f"{facts['data_without_any_revenue']:,}",
        "data traffic, zero revenue", accent=GREY)
    split = pd.DataFrame({
        "group": ["No data traffic either", "Used data, paid for voice", "Used data, no revenue at all"],
        "customers": [facts["zero_data_revenue_no_traffic"], facts["data_without_data_revenue_with_voice"],
                      facts["data_without_any_revenue"]],
    })
    with card(f"The {facts['zero_data_revenue']:,} customers with no data revenue"):
        show(count_bars(split, "group", color=SECOND))
    with notes("What it means"):
        st.markdown(
            "- These customers used data but no data revenue was recorded for them, and they recharged clearly "
            "more than their recorded revenue (see the spend gap below).\n"
            "- So, data revenue probably **understates** spending on data. To be confirmed with the data owners.")

    # recharge vs revenue
    st.subheader("Recharge vs revenue (spend gap)")
    k = st.columns(4)
    kpi(k[0], "Total recharged", fmt_big(facts["total_recharge"]), "all customers, whole period")
    kpi(k[1], "Total recorded revenue", fmt_big(facts["total_revenue"]),
        f"{facts['total_revenue'] / facts['total_recharge']:.0%} of the amount recharged")
    kpi(k[2], "Median spend gap", f"{facts['gap_median']:,.0f}", "recharge minus revenue, per customer")
    kpi(k[3], "Gap for data-without-revenue users", f"{facts['gap_median_bundle_users']:,.0f}",
        f"median · everyone else {facts['gap_median_others']:,.0f}", accent=SECOND)
    gap = pd.DataFrame({
        "group": ["Recharged more than revenue", "Recharge equals revenue", "Revenue above recharge"],
        "customers": [facts["gap_positive"], facts["gap_zero"], facts["gap_negative"]],
    })
    with card("Customers by spend gap"):
        show(count_bars(gap, "group"))
    with notes("What it means"):
        st.markdown(
            f"- **{facts['gap_zero_inactive']:,}** of the {facts['gap_zero']:,} 'equal' customers simply had "
            "no recharge and no revenue.\n"
            "- The gap is largest for customers who use data without data revenue.\n"
            "- Revenue above recharge suggests balance from other sources (for example transfers or credit), "
            "to be confirmed with the data owners.")

    # outliers and correlations
    st.subheader("Outliers and correlations")
    corr["measure_1"] = corr["measure_1"].map(lambda m: MEASURE_LABELS.get(m, m))
    corr["measure_2"] = corr["measure_2"].map(lambda m: MEASURE_LABELS.get(m, m))
    order = list(dict.fromkeys(corr["measure_1"]))
    left, right = st.columns([2, 3])
    with left, card("Outliers"):
        kpi(st, "Outlier customers", f"{facts['outlier_customers']:,}",
            "above the usual range in at least one measure (log scale, IQR rule)", accent=GREY)
        by_measure = {MEASURE_LABELS.get(m, m): n for m, n in facts["outliers_by_measure"].items() if n > 0}
        st.markdown("\n".join(f"- {m}: {n}" for m, n in by_measure.items()) or "- none")
        st.markdown("They were **kept**: they look like real heavy customers (for example, regular large "
                    "monthly recharges), not data errors.")
    with right, card("How the measures move together",
                     "Spearman correlation, customers with revenue. 1 = always rise together."):
        base_corr = alt.Chart(corr).encode(
            # labelOverlap=False: show every label instead of hiding every other one when they are close
            x=alt.X("measure_1:N", sort=order, title=None,
                    axis=alt.Axis(labelAngle=-40, labelOverlap=False, labelAlign="right")),
            y=alt.Y("measure_2:N", sort=order, title=None, axis=alt.Axis(labelOverlap=False)))
        cells = base_corr.mark_rect().encode(
            color=alt.Color("correlation:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
                            legend=None),
            tooltip=["measure_1", "measure_2", alt.Tooltip("correlation:Q", format=".2f")])
        text = base_corr.mark_text(fontSize=11).encode(
            text=alt.Text("correlation:Q", format=".2f"),
            color=alt.condition("abs(datum.correlation) > 0.6", alt.value("white"), alt.value("black")))
        show((cells + text).properties(height=380))
        pairs = corr[corr["measure_1"] < corr["measure_2"]]
        strongest, weakest = pairs.loc[pairs["correlation"].idxmax()], pairs.loc[pairs["correlation"].abs().idxmin()]
        takeaway(f"{strongest['measure_1']} and {strongest['measure_2'].lower()} rise and fall together, so they tell "
                 f"a similar story. {weakest['measure_1']} and {weakest['measure_2'].lower()} are almost unrelated.")

    # activity and inactivity
    st.subheader("Activity and inactivity")
    k = st.columns(4)
    kpi(k[0], "Typical inactive spell", f"{facts['streak_median_days']:.0f} days",
        f"median of {facts['streak_count']:,} spells without revenue")
    kpi(k[1], "Typical days between recharges", f"{facts['recharge_gap_median']:.0f} days",
        f"middle half: {facts['recharge_gap_q1']:.0f} to {facts['recharge_gap_q3']:.0f} days")
    kpi(k[2], "Went quiet, came back", f"{facts['dormant_paid_before_came_back']:.0%}",
        f"of {facts['dormant_paid_before']:,} who paid before the scoring date", accent=SECOND)
    kpi(k[3], "Never paid, came back", f"{facts['dormant_never_paid_came_back']:.0%}",
        f"of {facts['dormant_never_paid']:,} with no revenue before the scoring date", accent=GREY)
    left, right = st.columns(2)
    with left, card("Share of days with revenue", "How many customers were active on what share of days."):
        show(alt.Chart(active_share).mark_bar(color=ACCENT).encode(
            x=alt.X("active_share:N", sort=None, title="Share of days with revenue", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("customers:Q", title="Customers"),
            tooltip=["active_share", "customers"]).properties(height=280))
        lowest_bin = active_share.iloc[0]
        takeaway(f"{lowest_bin['customers'] / active_share['customers'].sum():.0%} of customers paid on fewer than 1 in "
                 "10 days, so a few quiet days alone are not a churn signal.")
    with right, card("Why churn means 90 days without revenue",
                     "Share of inactive spells that ended with the customer paying again."):
        line = alt.Chart(inactivity).mark_line(point=True, color=ACCENT).encode(
            x=alt.X("days_inactive:Q", title="Days without revenue"),
            y=alt.Y("came_back:Q", title="Came back later", axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=["days_inactive", alt.Tooltip("came_back:Q", format=".1%"), "streaks"])
        marker = alt.Chart(pd.DataFrame({"days_inactive": [90]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(x="days_inactive:Q")
        show((line + marker).properties(height=280))
        back_30 = inactivity.loc[inactivity["days_inactive"] == 30, "came_back"].iloc[0]
        back_90 = inactivity.loc[inactivity["days_inactive"] == 90, "came_back"].iloc[0]
        takeaway(f"After 90 days without paying only {back_90:.0%} come back, compared with {back_30:.0%} after 30 days. "
                 "That is why 90 days defines churn.")

    # trends over time
    st.subheader("Trends over time")
    trend_order = ["growing", "stable", "declining", "active mid-year only", "late joiner", "inactive all year"]
    trends = trends.set_index("trend").reindex([t for t in trend_order if t in set(trends["trend"])]).reset_index()
    with card("How each customer's spending changed",
                    "First three months compared with the last three months of the data."):
        show(count_bars(trends, "trend"))
    with notes("What it means"):
        counts = trends.set_index("trend")["customers"]
        growing, declining = counts.get("growing", 0), counts.get("declining", 0)
        lines = [f"- Growing ({growing:,}) and declining ({declining:,}) customers are almost balanced: "
                 f"{growing + declining:,} of {counts.sum():,} customers changed their spending by more than half."]
        monthly = load_optional("eda_monthly.csv")
        if monthly is not None:
            first, last = monthly.iloc[0], monthly.iloc[-1]
            lines.append(
                f"- Monthly revenue was {last['total_revenue'] / first['total_revenue'] - 1:.0%} higher at the end "
                f"than at the start, but paying customers fell "
                f"{1 - last['paying_customers'] / first['paying_customers']:.0%} while the median spend per "
                f"paying customer went from {first['median_revenue_per_payer']:,.0f} to "
                f"{last['median_revenue_per_payer']:,.0f}: revenue depends on fewer customers.")
        lines += ["- Customers who churned kept steady spending until about 60 days before the scoring date, then "
                  "cut it sharply (from the retention analysis).",
                  "- Recommendation: track paying customers and individual spending trends every month, not only "
                  "total revenue."]
        st.markdown("\n".join(lines))


def page_lookup():
    st.title("Customer lookup")
    st.caption(f"Pick customers to see their activity, churn probability, expected revenue and a recommended "
               f"action. Scores are as of the scoring date ({SCORING_DATE}).")
    if not need_files(["retention_output.csv"]):
        return

    activity = load_optional("customer_activity.csv")
    signals = load_optional("customer_signals.csv", index_col="customer_id")
    if activity is None or signals is None:
        st.info("For activity charts and personalized tips, upload customer_activity.csv and "
                "customer_signals.csv in the sidebar, or run the customer lookup export cell in the "
                "notebook (customer_activity.csv and customer_signals.csv).")

    all_ids = data.sort_values("priority_rank").index.tolist()
    st.session_state.setdefault("lookup_ids", [])

    def add_top():
        st.session_state["lookup_ids"] = all_ids[:5]

    def add_pasted():
        raw = st.session_state.get("pasted_ids", "")
        pasted = [p.strip() for p in raw.replace(",", " ").split() if p.strip()]
        valid = [p for p in pasted if p in data.index]
        st.session_state["lookup_ids"] = list(dict.fromkeys(st.session_state["lookup_ids"] + valid))
        st.session_state["paste_message"] = (f"Added {len(valid)} of {len(pasted)} IDs."
                                             + ("" if len(valid) == len(pasted)
                                                else " The others are not in the scored customers."))

    with card():
        pick, buttons = st.columns([4, 1], vertical_alignment="bottom")
        pick.multiselect(
            "Customers", all_ids, key="lookup_ids",
            format_func=lambda cid: f"{cid[:12]}…  ·  {data.at[cid, 'retention_priority']}",
            placeholder="Search by the first characters of a customer ID")
        buttons.button("Top 5 priority", on_click=add_top, width="stretch")
        with st.expander("Paste full customer IDs"):
            st.text_area("One ID per line, or separated by commas", key="pasted_ids")
            st.button("Add pasted IDs", on_click=add_pasted)
            if "paste_message" in st.session_state:
                st.caption(st.session_state["paste_message"])
        unit = st.radio("Expected revenue per", ["Day", "Month", "90 days"], index=1, horizontal=True)

    ids = st.session_state["lookup_ids"]
    if not ids:
        st.info("Pick customers above, or click **Top 5 priority**.")
        return

    factor = {"Day": 1 / HORIZON_DAYS, "Month": DAYS_PER_MONTH / HORIZON_DAYS, "90 days": 1}[unit]
    revenue_col = f"Expected revenue per {unit.lower()}" if unit != "90 days" else "Expected revenue, 90 days"
    selected = data.loc[ids]
    advice = {cid: recommend(selected.loc[cid],
                             signals.loc[cid] if signals is not None and cid in signals.index else None)
              for cid in ids}

    # comparison table
    table = pd.DataFrame({
        "Priority": selected["retention_priority"].astype(str),
        "Risk": selected["risk_category"].astype(str),
        "Churn probability": selected["churn_probability"],
        "Churn rate of similar customers": selected["band_churn_rate"],
        revenue_col: (selected["predicted_clv"] * factor).round(0),
        "Value": selected["value_category"].astype(str),
        "Revenue at risk (90 days)": selected["revenue_at_risk"].round(0),
        "Recommended action": [advice[cid][1] for cid in ids],
    }, index=selected.index)
    with card(f"{len(ids)} selected customer{'s' if len(ids) > 1 else ''}"):
        st.dataframe(table, width="stretch", column_config={
            "Churn probability": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent"),
            "Churn rate of similar customers": st.column_config.NumberColumn(format="percent"),
            revenue_col: st.column_config.NumberColumn(format="localized"),
            "Revenue at risk (90 days)": st.column_config.NumberColumn(format="localized"),
        })
        st.download_button("Download selection (CSV)", table.to_csv().encode("utf-8"),
                           file_name="customer_lookup.csv", mime="text/csv")

    # one detail card per customer
    max_cards = 10
    if len(ids) > max_cards:
        st.caption(f"Showing details for the first {max_cards} customers.")
    cutoff_month = "2026-07"
    for cid in ids[:max_cards]:
        row = selected.loc[cid]
        sig = signals.loc[cid] if signals is not None and cid in signals.index else None
        color, action, reason, tips = advice[cid]
        risk = str(row["risk_category"])
        priority = str(row["retention_priority"])

        if "past_value_category" in row.index:
            value_note = f"forecast {row['predicted_value_category']} · before the drop {row['past_value_category']}"
        else:
            value_note = f"past six months: {row['past_6m_revenue']:,.0f}"

        with st.container(border=True):
            st.markdown(f"**Customer** `{cid}`")
            k = st.columns(4)
            kpi(k[0], "Churn probability", f"{row['churn_probability']:.0%}",
                f"{risk} risk · {row['band_churn_rate']:.0%} of {risk}-risk customers churned",
                accent=RISK_HEX[risk])
            kpi(k[1], revenue_col, f"{row['predicted_clv'] * factor:,.0f}",
                f"{row['predicted_clv']:,.0f} over the next 90 days")
            kpi(k[2], "Value category", str(row["value_category"]).title(), value_note)
            kpi(k[3], "Retention priority", priority,
                f"rank {int(row['priority_rank']):,} of {len(data):,}", accent=PRIORITY_HEX[priority])

            left, right = st.columns([3, 2])
            with left:
                if activity is not None:
                    history = activity[activity["customer_id"] == cid].copy()
                    history["period"] = (history["month"] < cutoff_month).map(
                        {True: "Seen by the model", False: "After the scoring date"})
                    # "2025-09" -> "Sep-2025", same style as the Overview's monthly chart
                    history = history.sort_values("month")
                    history["month_label"] = pd.to_datetime(history["month"]).dt.strftime("%b-%Y")
                    st.markdown("**Monthly activity**")
                    show(alt.Chart(history).mark_bar().encode(
                        x=alt.X("month_label:N", sort=list(history["month_label"]), title=None,
                                axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                        y=alt.Y("revenue:Q", title="Monthly revenue", scale=alt.Scale(nice=True)),
                        color=alt.Color("period:N", title=None, legend=alt.Legend(orient="bottom"),
                                        scale=alt.Scale(domain=["Seen by the model", "After the scoring date"],
                                                        range=[ACCENT, "#b9c4d6"])),
                        tooltip=[alt.Tooltip("month_label:N", title="month"),
                                 alt.Tooltip("revenue:Q", format=",.0f"),
                                 "active_days", "recharges",
                                 alt.Tooltip("recharge_amount:Q", format=",.0f")],
                    ).properties(height=250))
                if sig is not None:
                    st.caption(
                        f"At the scoring date: last paid {sig['days_since_last_revenue']:.0f} days before · "
                        f"{sig['active_days_30d']:.0f} active days and {sig['recharge_count_30d']:.0f} recharges "
                        f"in the last 30 days · revenue last 30 days {sig['revenue_30d']:,.0f}")
            with right:
                st.markdown(f"#### :{color}[{action}]")
                st.write(reason)
                for tip in tips:
                    st.markdown(f"- {tip}")
                outcome = "churned" if row["churned"] == 1 else "stayed"
                st.caption(f"Historical check: this customer actually **{outcome}** after the scoring date "
                           "(known only because the data is historical).")


def page_churn():
    st.title("Churn")
    st.caption("What predicts churn, and the churn probability of each customer.")

    recency = load_optional("eda_churn_by_recency.csv")
    activity = load_optional("eda_churn_by_activity.csv")
    churn_corr = load_optional("eda_churn_corr.csv")
    medians = load_optional("eda_churn_medians.csv")

    if has_customers():
        churners = filtered["churned"].sum()
        in_high = filtered.loc[filtered["risk_category"] == "high", "churned"].sum()
        k = st.columns(4)
        kpi(k[0], "Customers (filtered)", f"{len(filtered):,}")
        kpi(k[1], "Actual churn rate", f"{filtered['churned'].mean():.1%}", f"{churners:,} churned",
            accent=RISK_HEX["high"])
        kpi(k[2], "High-risk customers", f"{(filtered['risk_category'] == 'high').sum():,}",
            f"{(filtered['risk_category'] == 'high').mean():.0%} of filtered customers", accent=RISK_HEX["high"])
        kpi(k[3], "Churners in the high-risk group", f"{in_high / churners:.0%}" if churners else "–",
            f"{in_high:,} of {churners:,} churners")

    st.subheader("What predicts churn")
    st.caption("Based on all 1,924 customers active at the scoring date. Not affected by the filters.")
    if any(f is None for f in [recency, activity, churn_corr, medians]):
        st.info("Run the EDA export cell in the notebook to create the eda_churn_ files.")
    else:
        left, right = st.columns(2)
        with left, card("Churn rate by days since last revenue"):
            show(rate_bars(recency, "group", list(recency["group"]), "Days since last revenue"))
            takeaway(f"Churn risk climbs the longer a customer goes without paying: {recency['churn_rate'].iloc[0]:.0%} "
                     f"for those who paid within the last week, {recency['churn_rate'].iloc[-1]:.0%} after two months.")
        with right, card("Churn rate by activity level", "Share of days active, four equal groups."):
            show(rate_bars(activity, "group", list(activity["group"]), "Activity level"))
            takeaway(f"Regular users rarely leave: {activity['churn_rate'].iloc[-1]:.0%} of the most active quarter "
                     f"churned, against {activity['churn_rate'].iloc[0]:.0%} of the least active.")

        churn_corr["feature"] = churn_corr["feature"].replace(FEATURE_LABELS)
        churn_corr["direction"] = churn_corr["correlation"].gt(0).map(
            {True: "Higher value, more churn", False: "Higher value, less churn"})
        left, right = st.columns([3, 2])
        with left, card("How each feature relates to churn", "Spearman correlation with churning."):
            show(alt.Chart(churn_corr).mark_bar().encode(
                x=alt.X("correlation:Q", title="Correlation with churn"),
                y=alt.Y("feature:N", sort="x", title=None),
                color=alt.Color("direction:N", title=None, legend=alt.Legend(orient="bottom"),
                                scale=alt.Scale(domain=["Higher value, more churn", "Higher value, less churn"],
                                                range=[RISK_HEX["high"], RISK_HEX["low"]])),
                tooltip=["feature", alt.Tooltip("correlation:Q", format=".3f")]).properties(height=420))
            top_pos = churn_corr.loc[churn_corr["correlation"].idxmax()]
            top_neg = churn_corr.loc[churn_corr["correlation"].idxmin()]
            takeaway(f"{top_pos['feature']} is the strongest warning sign; {top_neg['feature'].lower()} is the strongest "
                     "sign a customer will stay.")
        with right, card("Typical behavior before the scoring date", "Medians, stayed vs churned."):
            medians["feature"] = medians["feature"].replace(FEATURE_LABELS)
            medians = medians.set_index("feature")
            medians.columns = ["Stayed", "Churned"]
            medians.index.name = None
            st.dataframe(medians.style.format("{:,.2f}"), width="stretch", height=420)

    if data is None or filtered.empty:  # the message was already shown at the top of the page
        return
    st.subheader("Churn probabilities from the model")
    plot_df = for_chart(filtered)
    band = for_chart(filtered.groupby("risk_category", observed=True)
                     .agg(customers=("churned", "size"), churn_rate=("churned", "mean")))
    left, right = st.columns(2)
    with left, card("Distribution of churn probabilities",
                    "Risk: low under 10%, medium 10–30%, high 30% or more."):
        show(alt.Chart(plot_df).mark_bar().encode(
            x=alt.X("churn_probability:Q", bin=alt.Bin(maxbins=30), title="Churn probability (calibrated)",
                    axis=alt.Axis(format="%")),
            y=alt.Y("count():Q", title="Customers"),
            color=alt.Color("risk_category:N", scale=RISK_COLORS, title="Risk",
                            legend=alt.Legend(orient="bottom"))).properties(height=280))
    with right, card("Actual churn rate by risk category",
                     "Calibrated: each category's average probability is close to its actual churn rate."):
        show(rate_bars(band, "risk_category", LEVELS, "Risk category",
                       color=alt.Color("risk_category:N", scale=RISK_COLORS, legend=None)))
        rates = band.set_index("risk_category")["churn_rate"]
        if {"low", "high"} <= set(rates.index) and rates["low"] > 0:
            takeaway(f"High-risk customers really churned {rates['high'] / rates['low']:.0f} times as often as low-risk "
                     "ones, so the risk groups can be trusted for targeting.")


def page_value():
    st.title("Customer value")
    st.caption("Predicted revenue for the next 90 days from the CLV model, and value before the recent drop.")
    if not has_customers():
        return

    clv = filtered["predicted_clv"].sort_values(ascending=False)
    top_20 = clv.head(max(1, round(len(clv) * 0.2))).sum() / clv.sum()
    k = st.columns(4)
    kpi(k[0], "Total predicted revenue", fmt_big(clv.sum()), "next 90 days")
    kpi(k[1], "Average per customer", f"{clv.mean():,.0f}",
        f"≈ {clv.mean() / HORIZON_DAYS * DAYS_PER_MONTH:,.0f} per month")
    kpi(k[2], "Median per customer", f"{clv.median():,.0f}", "half of customers are below this")
    kpi(k[3], "Top 20% of customers", f"{top_20:.0%}", "of predicted revenue", accent=SECOND)

    # why value uses two measures: spending drops before churn
    activity = load_optional("customer_activity.csv")
    with card("Spending drops before customers churn",
                    "Average monthly revenue per customer in the six months before the scoring date. "
                    "All scored customers."):
        if activity is None:
            st.info("Upload customer_activity.csv in the sidebar to see this.")
        else:
            months = activity[(activity["month"] >= "2026-01") & (activity["month"] <= "2026-06")].copy()
            months["group"] = months["customer_id"].map(data["churned"]).map({1: "Churned", 0: "Stayed"})
            trend = months.dropna(subset=["group"]).groupby(["month", "group"], as_index=False)["revenue"].mean()
            trend["month_label"] = pd.to_datetime(trend["month"]).dt.strftime("%b-%Y")
            show(alt.Chart(trend).mark_line(point=True).encode(
                x=alt.X("month_label:N", sort=sorted(trend["month_label"].unique(),
                                                     key=lambda m: pd.to_datetime(m, format="%b-%Y")),
                        title=None, axis=alt.Axis(labelAngle=-45, labelOverlap=False),
                        scale=alt.Scale(padding=1.5)),
                y=alt.Y("revenue:Q", title="Average revenue per customer"),
                color=alt.Color("group:N", title=None, legend=alt.Legend(orient="bottom"),
                                scale=alt.Scale(domain=["Stayed", "Churned"],
                                                range=[RISK_HEX["low"], RISK_HEX["high"]])),
                tooltip=[alt.Tooltip("month_label:N", title="month"), "group",
                         alt.Tooltip("revenue:Q", format=",.0f")]).properties(height=300))
            churned = trend[trend["group"] == "Churned"].set_index("month")["revenue"]
            if len(churned) >= 6:
                drop = 1 - churned.iloc[-2:].mean() / churned.iloc[:4].mean()
                takeaway(f"Customers who churned spent {drop:.0%} less in May and June than in January to April, "
                         "while those who stayed kept spending.")
    with notes("How the value category handles it"):
        st.markdown(
            "- The 90-day forecast follows recent spending, so it is low for customers who have started to "
            "slow down, even if they were valuable until recently.\n"
            "- The value category therefore uses **two measures**: the forecast, and spending 61 to 180 days "
            "before the scoring date, which is before the drop.\n"
            "- A customer counts as valuable if **either** measure says so. Without this, most high-risk "
            "customers who used to spend a lot would be treated as low value.")

    columns = {"Customers": ("predicted_clv", "size"),
               "Avg predicted (90 days)": ("predicted_clv", "mean")}
    if "revenue_before_drop" in filtered.columns:
        columns["Avg revenue 61-180 days before"] = ("revenue_before_drop", "mean")
    columns["Avg past six months"] = ("past_6m_revenue", "mean")
    value_summary = filtered.groupby("value_category", observed=True).agg(**columns)
    value_summary.index = value_summary.index.astype(str)
    value_summary.index.name = None
    if "past_value_category" in filtered.columns:
        value_caption = ("Two measures, each split bottom 50% low, next 30% medium, top 20% high: predicted "
                         "revenue (next 90 days) and revenue 61–180 days before the scoring date, before the "
                         "drop that comes before churn. A customer's category is the higher of the two.")
    else:
        value_caption = "Bottom 50% low, next 30% medium, top 20% high, by predicted revenue."
    with card("Value categories", value_caption):
        st.dataframe(value_summary.style.format("{:,.0f}"), width="stretch")
        if "past_value_category" in filtered.columns:
            both = pd.crosstab(filtered["predicted_value_category"], filtered["past_value_category"])
            both.index = [f"forecast {i}" for i in both.index.astype(str)]
            both.columns = [f"before drop {c}" for c in both.columns.astype(str)]
            st.caption("How the two measures compare (rows: forecast, columns: before the drop).")
            st.dataframe(both, width="stretch")
            n_up = int((filtered["past_value_category"].cat.codes > filtered["predicted_value_category"].cat.codes).sum())
            takeaway(f"{n_up:,} customers rank higher by their earlier spending than by their forecast. They have cut back "
                     "recently, which is often a sign they are about to leave.")

    plot_df = for_chart(filtered)
    cap = plot_df["predicted_clv"].quantile(0.99)
    plot_df["predicted_clv_capped"] = plot_df["predicted_clv"].clip(upper=cap)
    plot_df["past_per_90_days"] = plot_df["past_6m_revenue"] / 2  # 180 days -> per 90 days
    left, right = st.columns(2)
    with left, card("Distribution of predicted 90-day revenue",
                    f"Values above {cap:,.0f} are grouped at the right."):
        show(alt.Chart(plot_df).mark_bar(color=ACCENT).encode(
            x=alt.X("predicted_clv_capped:Q", bin=alt.Bin(maxbins=40), title="Predicted revenue, next 90 days"),
            y=alt.Y("count():Q", title="Customers")).properties(height=320))

    positive = plot_df[(plot_df["past_per_90_days"] > 0) & (plot_df["predicted_clv"] > 0)]
    low = float(positive[["past_per_90_days", "predicted_clv"]].min().min())
    high = float(positive[["past_per_90_days", "predicted_clv"]].max().max())
    with right, card("Past vs predicted revenue, per 90 days",
                     "Past = average 90 days within the last six months. Below the dashed line = predicted "
                     "to spend less than before."):
        points = alt.Chart(positive).mark_circle(size=25, opacity=0.5).encode(
            x=alt.X("past_per_90_days:Q", scale=alt.Scale(type="log"), title="Past, average per 90 days"),
            y=alt.Y("predicted_clv:Q", scale=alt.Scale(type="log"), title="Predicted, next 90 days"),
            color=alt.Color("risk_category:N", scale=RISK_COLORS, title="Risk", legend=alt.Legend(orient="bottom")),
            tooltip=["risk_category", "value_category",
                     alt.Tooltip("past_per_90_days:Q", format=",.0f"),
                     alt.Tooltip("predicted_clv:Q", format=",.0f")])
        same = alt.Chart(pd.DataFrame({"past_per_90_days": [low, high], "predicted_clv": [low, high]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="past_per_90_days:Q", y="predicted_clv:Q")
        show((points + same).properties(height=320))
        below = positive["predicted_clv"] < positive["past_per_90_days"]
        high_risk = positive["risk_category"] == "high"
        if high_risk.any():
            takeaway(f"At-risk customers are expected to spend less than they used to: {below[high_risk].mean():.0%} of "
                     f"high-risk customers against {below[~high_risk].mean():.0%} of the others.")


def page_models():
    st.title("Model performance")
    st.caption("How the two models were built and how well they work on customers they never saw. "
               "Not affected by the filters.")

    if not need_files(["retention_output.csv", "churn_test.csv", "clv_test.csv"]):
        return
    churn_test = load_optional("churn_test.csv")
    clv_test = load_optional("clv_test.csv")
    importance = load_optional("feature_importance.csv")
    if importance is None:
        st.info("Run the export cell in the notebook to create feature_importance.csv.")
        return

    # how the models were built (dates and customer counts set in the notebook)
    st.subheader("How the models were built")
    churn_train = len(data) - len(churn_test)
    clv_train = CLV_CUSTOMERS - len(clv_test)
    setup = pd.DataFrame({
        "Churn model": [
            "Whether a customer has no revenue in the next 90 days",
            "Activity up to 2 Jul 2026, mostly the last 30 to 90 days",
            "3 Jul to 30 Sep 2026",
            "3 Jul to 30 Sep 2026",
            f"{len(data):,} with revenue in the 90 days before 2 Jul",
            f"{churn_train:,} to learn from, {len(churn_test):,} to test",
            "Probabilities calibrated to match real churn rates (below)",
        ],
        "CLV model": [
            "Revenue in the next 90 days",
            "The 180 days before 3 Apr 2026 (6 Oct 2025 to 3 Apr 2026)",
            "4 Apr to 2 Jul 2026 (earlier, so the forecast can be checked on a later period)",
            "3 Jul to 30 Sep 2026",
            f"{CLV_CUSTOMERS:,} with revenue in the 90 days before 3 Apr",
            f"{clv_train:,} to learn from, {len(clv_test):,} to test",
            "Forecasts made on 2 Jul compared with actual revenue up to 30 Sep",
        ],
    }, index=["Predicts", "Learns from", "Outcome it learns from", f"Used on {SCORING_DATE} to predict",
              "Customers", "Split", "Extra check"])
    with card("Time windows and customers"):
        st.table(setup)
        takeaway(f"Both models look 90 days ahead and are applied on {SCORING_DATE} to the {len(data):,} active "
                 "customers. Settings were tuned with 5-fold cross-validation, and every result below comes "
                 "from customers the models never saw while learning.")

    # churn model
    churn_models = {"Logistic regression": "logistic_regression", "Random forest": "random_forest"}
    y_true = churn_test["churned"]
    lr_auc = roc_auc_score(y_true, churn_test["logistic_regression"])
    rf_auc = roc_auc_score(y_true, churn_test["random_forest"])

    st.subheader("Churn model")
    k = st.columns(3)
    kpi(k[0], "ROC AUC, random forest", f"{rf_auc:.3f}", "1 = perfect, 0.5 = guessing")
    kpi(k[1], "ROC AUC, logistic regression", f"{lr_auc:.3f}", "simpler model, for comparison", accent=SECOND)
    kpi(k[2], "Test customers", f"{len(y_true):,}", f"{int(y_true.sum())} of them churned")

    with card():
        threshold = st.slider("Score threshold for predicting churn", 0.05, 0.95, 0.50, 0.05)
        rows, roc_parts = {}, []
        for label, col in churn_models.items():
            y_prob = churn_test[col]
            y_pred = (y_prob >= threshold).astype(int)
            auc = roc_auc_score(y_true, y_prob)
            rows[label] = {"ROC AUC": auc,
                           "Precision": precision_score(y_true, y_pred, zero_division=0),
                           "Recall": recall_score(y_true, y_pred, zero_division=0),
                           "Customers flagged": int(y_pred.sum())}
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_parts.append(pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr,
                                           "model": f"{label} (AUC {auc:.3f})"}))
        st.dataframe(pd.DataFrame(rows).T.style.format(
            {"ROC AUC": "{:.3f}", "Precision": "{:.1%}", "Recall": "{:.1%}", "Customers flagged": "{:,.0f}"}),
            width="stretch")
        st.caption("Precision: share of flagged customers who really churned. Recall: share of churners who were "
                   "flagged. Uses the raw model scores, before calibration.")

    roc_df = pd.concat(roc_parts)
    diagonal = pd.DataFrame({"false_positive_rate": [0, 1], "true_positive_rate": [0, 1]})
    left, right = st.columns(2)
    with left, card("ROC curves", "Dashed line = random guessing."):
        roc_chart = (
            alt.Chart(roc_df).mark_line().encode(
                x=alt.X("false_positive_rate:Q", title="Stayers wrongly flagged"),
                y=alt.Y("true_positive_rate:Q", title="Churners caught"),
                color=alt.Color("model:N", title=None, legend=alt.Legend(orient="bottom"),
                                scale=alt.Scale(range=[SECOND, ACCENT])))
            + alt.Chart(diagonal).mark_line(strokeDash=[4, 4], color="gray").encode(
                x="false_positive_rate:Q", y="true_positive_rate:Q"))
        show(roc_chart.properties(height=320))
        takeaway("Both models separate churners from stayers almost equally well, so the data matters more than the "
                 "choice of model. The random forest is used because it ranks customers slightly better.")
    with right, card(f"Confusion matrix at threshold {threshold:.2f}"):
        chosen = st.radio("Model", list(churn_models), index=1, horizontal=True, label_visibility="collapsed")
        y_pred = (churn_test[churn_models[chosen]] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cm = pd.DataFrame({"actual": ["stayed", "stayed", "churned", "churned"],
                           "predicted": ["stayed", "churned", "stayed", "churned"],
                           "customers": [tn, fp, fn, tp]})
        show(heatmap(cm, "predicted", "actual", ["stayed", "churned"], ["stayed", "churned"]))
        caught = tp / (tp + fn) if tp + fn else 0
        takeaway(f"At this threshold the model catches {caught:.0%} of churners but also flags {fp} customers who "
                 "stayed. A lower threshold catches more churners at the cost of more wasted contacts.")

    # calibration
    st.subheader("Calibration: turning scores into real probabilities")
    avg_prob, actual_rate = data["churn_probability"].mean(), data["churned"].mean()
    brier = brier_score_loss(data["churned"], data["churn_probability"])
    with card("Predicted vs actual churn by risk category", "All scored customers."):
        check = (data.groupby("risk_category", observed=True)
                 .agg(predicted=("churn_probability", "mean"), actual=("churned", "mean")))
        long = for_chart(check).melt(id_vars="risk_category", var_name="measure", value_name="rate")
        long["measure"] = long["measure"].map({"predicted": "Average predicted probability",
                                               "actual": "Actual churn rate"})
        bars = alt.Chart(long).mark_bar().encode(
            x=alt.X("risk_category:N", sort=LEVELS, title="Risk category", axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("measure:N"),
            y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%")),
            color=alt.Color("measure:N", scale=alt.Scale(range=[SECOND, ACCENT]),
                            legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["risk_category", "measure", alt.Tooltip("rate:Q", format=".1%")])
        labels = bars.mark_text(dy=-7, fontSize=11).encode(text=alt.Text("rate:Q", format=".0%"))
        show((bars + labels).properties(height=280))
        takeaway("Predicted and actual churn match in every risk category, so a probability can be read at face "
                 "value: 40% means about 4 in 10 such customers leave.")
    with notes("Why and how"):
        st.markdown(
            f"- Churners are only about 1 in 6 customers, so the random forest was trained to give them extra "
            f"weight. That made its raw scores too high: on average {RAW_AVERAGE_SCORE:.1%} against a real churn "
            f"rate of {actual_rate:.1%}.\n"
            "- A calibration step, fitted on customers the forest had not learned from, translates the scores into "
            "real probabilities. The order of customers stays the same, so the AUC does not change.\n"
            f"- **After calibration:** average probability {avg_prob:.1%} against {actual_rate:.1%} actual; "
            f"expected churners {data['churn_probability'].sum():,.0f} against {int(data['churned'].sum()):,} "
            f"actual. The probability error (Brier score) fell from {RAW_BRIER:.3f} to {brier:.3f}.")

    # CLV model
    st.subheader("CLV model")
    clv_rows = {}
    for model in ["baseline", "random_forest"]:
        pred = clv_test[model]
        clv_rows[model] = {
            "Average error (MAE)": mean_absolute_error(clv_test["actual"], pred),
            "Error with big misses weighted more (RMSE)": mean_squared_error(clv_test["actual"], pred) ** 0.5,
            "Share of differences explained (R²)": r2_score(clv_test["actual"], pred),
            "Ranking accuracy (Spearman)": clv_test["actual"].corr(pred, method="spearman"),
        }
    rf_m, base_m = clv_rows["random_forest"], clv_rows["baseline"]

    k = st.columns(3)
    kpi(k[0], "Average error per customer", f"{rf_m['Average error (MAE)']:,.0f}",
        f"random forest · baseline {base_m['Average error (MAE)']:,.0f}", accent=SECOND)
    kpi(k[1], "Ranking accuracy", f"{rf_m['Ranking accuracy (Spearman)']:.2f}",
        f"random forest · baseline {base_m['Ranking accuracy (Spearman)']:.2f} · 1 = perfect order", accent=SECOND)
    kpi(k[2], "Test customers", f"{len(clv_test):,}",
        f"typical 90-day revenue {clv_test['actual'].median():,.0f}")

    clv_metrics = pd.DataFrame(clv_rows).T
    clv_metrics.index = ["Baseline (last 90 days)", "Random forest"]
    top = float(max(clv_test["actual"].max(), clv_test["random_forest"].max()))
    left, right = st.columns([2, 3])
    with left, card("Test results"):
        st.dataframe(clv_metrics.style.format({
            "Average error (MAE)": "{:,.0f}", "Error with big misses weighted more (RMSE)": "{:,.0f}",
            "Share of differences explained (R²)": "{:.2f}", "Ranking accuracy (Spearman)": "{:.2f}"}),
            width="stretch")
        st.caption("Ranking accuracy shows how well a model puts customers in the right order, from biggest to "
                   "smallest spender. It matters most here, because value categories are based on that order, "
                   "not on exact amounts.")
        error_winner = "random forest" if rf_m["Average error (MAE)"] < base_m["Average error (MAE)"] else "baseline"
        rank_winner = ("random forest" if rf_m["Ranking accuracy (Spearman)"] > base_m["Ranking accuracy (Spearman)"]
                       else "baseline")
        takeaway(f"The {error_winner} is closer on amounts and the {rank_winner} is better at ordering customers. "
                 "Recent spending already explains most of future spending.")
    with right, card("Actual vs predicted revenue", "Dashed line = perfect prediction."):
        points = alt.Chart(clv_test).mark_circle(size=25, opacity=0.5, color=ACCENT).encode(
            x=alt.X("actual:Q", scale=alt.Scale(type="symlog"), title="Actual 90-day revenue"),
            y=alt.Y("random_forest:Q", scale=alt.Scale(type="symlog"), title="Predicted (random forest)"))
        perfect = alt.Chart(pd.DataFrame({"actual": [0, top], "random_forest": [0, top]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="actual:Q", y="random_forest:Q")
        show((points + perfect).properties(height=320))
        top_spenders = clv_test[clv_test["actual"] >= clv_test["actual"].quantile(0.95)]
        under = 1 - top_spenders["random_forest"].sum() / top_spenders["actual"].sum()
        takeaway(f"The model works well for typical customers but underestimates the biggest spenders: for the top 5%, "
                 f"predictions were {under:.0%} below what they actually spent. Treat high predictions as a minimum.")

    # feature importance
    st.subheader("Most important features")
    importance["feature"] = importance["feature"].replace(FEATURE_LABELS)
    left, right = st.columns(2)
    for col, model, title in [(left, "churn", "Churn model, top 10"), (right, "clv", "CLV model, top 10")]:
        top10 = importance[importance["model"] == model].nlargest(10, "importance")
        with col, card(title):
            show(alt.Chart(top10).mark_bar(color=ACCENT if model == "churn" else SECOND).encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title=None)).properties(height=300))
    st.caption("Importance shows how much each feature helped the model separate customers, "
               "not whether it raises or lowers the prediction.")

    # reasoning behind the model choice
    with notes("Why these models, and what would improve them"):
        st.markdown(
            f"- **A more complex model is unlikely to help much.** A simple model (logistic regression, "
            f"AUC {lr_auc:.3f}) and a flexible one (random forest, AUC {rf_auc:.3f}) score almost the same, so "
            "the limit is the information in the data, not the algorithm.\n"
            "- **The CLV model does not beat \"copy the last 90 days\" on amounts, only on ranking**, because "
            "most features describe recent "
            "revenue in different ways. Models mainly learn that recent spending continues.\n"
            "- **New information would help more:** features describing the spending drop before churn, and "
            "data such as subscription plan, customer satisfaction (CSAT), complaints and network quality, "
            "which can explain *why* customers leave.\n"
            "- **A longer history** would allow several training periods, more churn examples and testing "
            "on later periods. More complex models (for example gradient boosting) can be tested once that "
            "information is available.")


def page_priorities():
    st.title("Retention priorities")
    st.caption("Churn risk and customer value combined into a recommended action for each customer.")
    if not has_customers():
        return

    focus = filtered["retention_priority"].isin(["Retain now", "Monitor closely"])
    k = st.columns(4)
    kpi(k[0], "Customers (filtered)", f"{len(filtered):,}")
    kpi(k[1], "Expected revenue at risk", fmt_big(filtered["revenue_at_risk"].sum()), "next 90 days",
        accent=RISK_HEX["high"])
    kpi(k[2], "Retain now + Monitor closely", f"{focus.sum():,}",
        f"{focus.mean():.0%} of filtered customers", accent=PRIORITY_HEX["Retain now"])
    kpi(k[3], "Their share of revenue at risk",
        f"{filtered.loc[focus, 'revenue_at_risk'].sum() / filtered['revenue_at_risk'].sum():.0%}",
        "where effort pays off most", accent=PRIORITY_HEX["Monitor closely"])

    # the framework itself: how risk and value decide the priority, and why
    with card("How priorities are set",
                    "Each cell shows the priority for that mix of risk and value, and how many customers are in it."):
        show(priority_grid(filtered))
    high_risk = data[data["risk_category"] == "high"]
    high_value_n = int((high_risk["value_category"] == "high").sum())
    medium_value_n = int((high_risk["value_category"] == "medium").sum())
    with notes("Why it is built this way"):
        st.markdown(
            "- **Risk** (calibrated churn probability): below 10% low, 10–30% medium, above 30% high.\n"
            "- **Value**: bottom 50% low, next 30% medium, top 20% high, using the higher of the 90-day "
            "forecast and spending before the drop.\n"
            "- Risk says **who might leave**; value says **whether they are worth spending on** to keep.\n"
            "- Direct contact costs the most, so it goes only to high-risk customers who are worth keeping. "
            "At-risk customers with low value get cheap automated offers, and low-risk customers need no "
            "retention spend.\n"
            f"- **Why medium value counts too:** only {high_value_n:,} high-risk customers are high value. "
            "Medium value customers are still in the top half of spenders, so leaving them out ignores real "
            "revenue at risk; some of them only look medium because their spending is already falling; and a "
            "larger group can be split into contacted and control customers to measure whether the campaign "
            f"works. Adding them ({medium_value_n:,}) gives Retain now {high_value_n + medium_value_n:,} "
            "customers.\n"
            "- **Expected revenue at risk** = churn probability × predicted 90-day revenue. Within each "
            "priority, customers are ranked by it, so the list says who to contact first. It is an estimate "
            "for ranking, not an exact amount of money that will be lost.")

    with card("Summary by retention priority"):
        show_summary(summarize(filtered, "retention_priority"))

    left, right = st.columns([3, 2])
    with left, card("Share of customers vs share of revenue at risk",
                    "A group whose revenue share is taller than its customer share is worth more effort."):
        show(share_comparison(filtered))
        shares = summarize(filtered, "retention_priority")
        ratio = shares["share_of_revenue_at_risk"] / shares["share_of_customers"]
        takeaway(f"{ratio.idxmax()} has {shares.loc[ratio.idxmax(), 'share_of_customers']:.0%} of customers but "
                 f"{shares.loc[ratio.idxmax(), 'share_of_revenue_at_risk']:.0%} of the revenue at risk, so effort there "
                 f"pays off most. {ratio.idxmin()} has {shares.loc[ratio.idxmin(), 'share_of_customers']:.0%} of customers "
                 f"but only {shares.loc[ratio.idxmin(), 'share_of_revenue_at_risk']:.0%} of it.")
    with right, card("Customers by risk and value", "Rows: risk category. Columns: value category."):
        risk_value = (pd.crosstab(filtered["risk_category"], filtered["value_category"])
                      .stack().rename("customers").reset_index())
        risk_value.columns = ["risk", "value", "customers"]
        risk_value[["risk", "value"]] = risk_value[["risk", "value"]].astype(str)
        show(heatmap(risk_value, "value", "risk", LEVELS, LEVELS[::-1]))
        high_row = risk_value[risk_value["risk"] == "high"]
        if not high_row.empty:
            valuable = high_row.loc[high_row["value"] != "low", "customers"].sum()
            takeaway(f"Only {valuable:,} of {high_row['customers'].sum():,} high-risk customers are medium or high value: "
                     "risk and value rarely come together, so Retain now stays a focused group.")

    with card("Summary by segment"):
        show_summary(summarize(filtered, "segment"))

    with notes("Recommended actions"):
        for priority in PRIORITY_ORDER:
            color, action, reason = ACTIONS[priority]
            st.markdown(f"**:{color}[{priority}]: {action}.** {reason}")
        st.caption("See the Recommendations page for business-wide recommendations.")


def page_recommendations():
    st.title("Recommendations")
    st.caption("What the business can do with these results. Based on all scored customers, not the filters.")
    if not need_files(["retention_output.csv"]):
        return

    facts = load_json("eda_facts.json")
    recency = load_optional("eda_churn_by_recency.csv")
    by_priority = summarize(data, "retention_priority")

    def count(priority):
        return int(by_priority.loc[priority, "customers"]) if priority in by_priority.index else 0

    def risk_share(priority):
        return by_priority.loc[priority, "share_of_revenue_at_risk"] if priority in by_priority.index else 0

    recs = [
        ("Contact the Retain now group within 7 days",
         f"{count('Retain now'):,} customers at high risk who are valuable now or were until recently. "
         f"They hold {risk_share('Retain now'):.0%} of the expected revenue at risk.",
         "Contact them through the most direct channel available, starting with the highest revenue at risk "
         "(download the list on the Campaign lists page)."),
        ("Respond to silence step by step, not only after 90 days",
         (f"Churn rises from {recency['churn_rate'].iloc[0]:.0%} for customers who paid in the last week to "
          f"{recency['churn_rate'].iloc[1]:.0%} after 8-14 days, {recency['churn_rate'].iloc[2]:.0%} after "
          f"15-30 days and {recency['churn_rate'].iloc[3]:.0%} after 31-60 days, but most customers with short "
          "gaps come back on their own.") if recency is not None else
         "Churn rises the longer customers go without paying, but most short gaps end on their own.",
         "Send a free reminder after about two weeks without revenue, a small offer after 30 days and a win-back "
         "offer after 60 days. Contact valuable customers earlier and more directly. Where possible, trigger on silence "
         "that is unusually long for that customer."),
        ("Watch for falling spending",
         "Customers who churned kept steady spending until about 60 days before the scoring date, then cut it "
         "sharply. A falling spend is an early warning while the customer can still be reached.",
         "Refresh the churn scores every month and flag customers whose spending fell by 30% or more."),
        ("Spend little on low-value risk",
         f"{count('Low-cost action'):,} customers are at risk but with a low value.",
         "Use cheap, automated contact only (SMS with a small recharge or bundle bonus)."),
        ("Keep loyal customers loyal",
         f"{count('Maintain'):,} customers have a low churn risk.",
         "No retention spend; offer upgrades or loyalty rewards to high-value customers in this group."),
        ("Match the offer to how customers spend",
         (f"Revenue is split between voice and data, and {facts['data_without_data_revenue']:,} customers used "
          "data without any data revenue, possibly through bundles." if facts is not None else
          "Revenue is split between voice and data."),
         "Offer data bundles to data-heavy customers and minutes to voice-heavy customers (see Customer lookup)."),
        ("Win back customers who went quiet",
         (f"{facts['dormant_paid_before_came_back']:.0%} of the {facts['dormant_paid_before']:,} customers who "
          f"paid before and then went quiet came back on their own, against "
          f"{facts['dormant_never_paid_came_back']:.0%} of those who never paid." if facts is not None else
          "Customers who paid before are more likely to return than those who never paid."),
         "Target win-back campaigns at former payers rather than customers who never paid."),
        ("Measure what retention actually achieves",
         "Scores show who is likely to leave, not whether an offer changes that.",
         "Keep a small random control group in each campaign and compare churn between contacted and "
         "not contacted customers."),
        ("Track churn, CLV and CSAT together",
         "Churn rate and CLV show who leaves and what they are worth, but not why. Customer satisfaction "
         "(CSAT) is the third key retention KPI: dissatisfied customers often leave before their spending "
         "shows it. This dataset has no satisfaction data.",
         "Collect a short CSAT survey after customer care contacts and campaigns, report it next to churn "
         "and CLV, and link it to customer IDs so it can be used as a churn predictor."),
        ("Improve the data",
         (f"{facts['gap_positive']:,} customers recharged more than their recorded revenue, so recorded "
          "revenue may not show all spending." if facts is not None else
          "Recorded revenue does not include all bundle spending."),
         "Link recharges to the bundles they buy, and add each customer's subscription plan, CSAT and "
         "complaints, network quality data and a longer history."),
    ]

    for start in range(0, len(recs), 2):
        cols = st.columns(2)
        for number, col, (title, why, action) in zip(range(start + 1, start + 3), cols, recs[start:start + 2]):
            with col, st.container(border=True):
                st.markdown(f"**{number}. {title}**")
                st.markdown(f"*Why:* {why}")
                st.markdown(f"*Action:* {action}")


def page_groups():
    st.title("Campaign lists")
    st.caption("Pick a group to see how it behaves compared with all scored customers, then download its list. "
               f"Behavior is measured at the scoring date ({SCORING_DATE}).")
    if not need_files(["retention_output.csv"]):
        return

    group_types = {"Retention priority": "retention_priority", "Segment (risk and value)": "segment",
                   "Risk category": "risk_category", "Value category": "value_category"}
    pick_type, pick_group = st.columns([1, 2])
    group_col = group_types[pick_type.selectbox("Group by", list(group_types))]
    counts = data[group_col].value_counts(sort=False)
    counts = counts[counts > 0]
    group = pick_group.selectbox(
        "Group", list(counts.index),
        format_func=lambda g: f"{str(g).title() if group_col.endswith('category') else g}  ({counts[g]:,} customers)")
    members = data[data[group_col] == group]
    signals = load_optional("customer_signals.csv", index_col="customer_id")
    activity = load_optional("customer_activity.csv")

    # headline numbers: this group against all customers
    k = st.columns(4)
    kpi(k[0], "Customers", f"{len(members):,}", f"{len(members) / len(data):.0%} of scored customers")
    kpi(k[1], "Actual churn rate", f"{members['churned'].mean():.0%}",
        f"all customers {data['churned'].mean():.0%}", accent=RISK_HEX["high"])
    month = members["predicted_clv"].mean() / HORIZON_DAYS * DAYS_PER_MONTH
    kpi(k[2], "Expected revenue per month", f"{month:,.0f}",
        f"per customer · all customers {data['predicted_clv'].mean() / HORIZON_DAYS * DAYS_PER_MONTH:,.0f}")
    kpi(k[3], "Expected revenue at risk", fmt_big(members["revenue_at_risk"].sum()),
        f"{members['revenue_at_risk'].sum() / data['revenue_at_risk'].sum():.0%} of the total · next 90 days",
        accent=SECOND)

    left, right = st.columns([2, 3])
    with left, card("Typical behavior before the scoring date", "Medians unless stated."):
        if signals is None:
            st.info("Upload customer_signals.csv in the sidebar to see this.")
        else:
            def describe(ids):
                s = signals.reindex(ids)
                previous = s["revenue_60d"] - s["revenue_30d"]
                fell = (previous > 0) & (s["revenue_30d"] <= 0.7 * previous)
                return {
                    "Days since last revenue": f"{s['days_since_last_revenue'].median():,.0f}",
                    "Active days, last 30 days": f"{s['active_days_30d'].median():,.0f}",
                    "Recharges, last 30 days": f"{s['recharge_count_30d'].median():,.0f}",
                    "Revenue, last 30 days": f"{s['revenue_30d'].median():,.0f}",
                    "Spending fell 30% or more (share)": f"{fell.mean():.0%}",
                    "Data share of revenue (average)": f"{s['data_revenue_share'].mean():.0%}",
                }
            st.table(pd.DataFrame({"This group": describe(members.index), "All customers": describe(data.index)}))
            group_days = signals["days_since_last_revenue"].reindex(members.index).median()
            all_days = signals["days_since_last_revenue"].reindex(data.index).median()
            takeaway(f"This group churned at {members['churned'].mean():.0%} against {data['churned'].mean():.0%} "
                     f"overall, and its customers last paid a median of {group_days:,.0f} days before the scoring "
                     f"date against {all_days:,.0f} for everyone.")
    with right, card("Average monthly revenue per customer",
                     "This group against all scored customers. The dashed line is the scoring month."):
        if activity is None:
            st.info("Upload customer_activity.csv in the sidebar to see this.")
        else:
            def monthly_average(ids, label):
                rows = activity[activity["customer_id"].isin(ids)]
                out = (rows.groupby("month")["revenue"].sum() / len(ids)).rename("revenue").reset_index()
                return out.assign(who=label)
            trend = pd.concat([monthly_average(members.index, "This group"),
                               monthly_average(data.index, "All customers")])
            trend["month_label"] = pd.to_datetime(trend["month"]).dt.strftime("%b-%Y")
            order = list(trend.sort_values("month")["month_label"].unique())
            lines = alt.Chart(trend).mark_line(point=True).encode(
                x=alt.X("month_label:N", sort=order, title=None, axis=alt.Axis(labelAngle=-45, labelOverlap=False)),
                y=alt.Y("revenue:Q", title="Average revenue per customer"),
                color=alt.Color("who:N", title=None, legend=alt.Legend(orient="bottom"),
                                scale=alt.Scale(domain=["This group", "All customers"], range=[ACCENT, GREY])),
                tooltip=[alt.Tooltip("month_label:N", title="month"), "who",
                         alt.Tooltip("revenue:Q", format=",.0f")])
            scoring = alt.Chart(pd.DataFrame({"month_label": ["Jul-2026"]})).mark_rule(
                strokeDash=[4, 4], color="gray").encode(x=alt.X("month_label:N", sort=order))
            show((lines + scoring).properties(height=300))

    if group_col == "retention_priority":
        color, action, reason = ACTIONS[str(group)]
        with card("Recommended action"):
            st.markdown(f"**:{color}[{group}]: {action}.** {reason}")

    # the list itself is only downloaded, not shown
    ranked = members.sort_values("priority_rank")
    export = pd.DataFrame({
        "rank": ranked["priority_rank"],
        "retention_priority": ranked["retention_priority"].astype(str),
        "segment": ranked["segment"].astype(str),
        "risk_category": ranked["risk_category"].astype(str),
        "value_category": ranked["value_category"].astype(str),
        "churn_probability": ranked["churn_probability"].round(3),
        "predicted_revenue_90_days": ranked["predicted_clv"].round(0),
        "expected_revenue_at_risk": ranked["revenue_at_risk"].round(0),
    }, index=ranked.index)
    if signals is not None:
        export = export.join(signals[["days_since_last_revenue", "active_days_30d", "revenue_30d"]])
    with card("Download"):
        st.markdown(f"**{len(export):,} customers**, ranked by priority and then by expected revenue at risk.")
        name = str(group).lower().replace(",", "").replace(" ", "_")
        st.download_button("Download this group (CSV)", export.to_csv().encode("utf-8"),
                           file_name=f"customers_{name}.csv", mime="text/csv", icon=":material/download:")


# navigation, customer files and filters
pages = [
    st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(page_behavior, title="Customer behavior", icon=":material/query_stats:"),
    st.Page(page_lookup, title="Customer lookup", icon=":material/person_search:"),
    st.Page(page_churn, title="Churn", icon=":material/trending_down:"),
    st.Page(page_value, title="Customer value", icon=":material/payments:"),
    st.Page(page_models, title="Model performance", icon=":material/monitoring:"),
    st.Page(page_priorities, title="Retention priorities", icon=":material/flag:"),
    st.Page(page_groups, title="Campaign lists", icon=":material/groups:"),
    st.Page(page_recommendations, title="Recommendations", icon=":material/lightbulb:"),
]
current = st.navigation(pages)
customer_upload_box()
data = load_retention()

FILTERED_PAGES = {"Churn", "Customer value", "Retention priorities"}
filtered = data
if data is not None and current.title in FILTERED_PAGES:
    st.sidebar.header("Filters")
    priorities = st.sidebar.multiselect("Retention priority", PRIORITY_ORDER, default=PRIORITY_ORDER,
                                        key="filter_priority")
    risks = st.sidebar.multiselect("Risk category", LEVELS, default=LEVELS, key="filter_risk")
    values = st.sidebar.multiselect("Value category", LEVELS, default=LEVELS, key="filter_value")
    filtered = data[data["retention_priority"].isin(priorities)
                    & data["risk_category"].isin(risks)
                    & data["value_category"].isin(values)]
    st.sidebar.caption(f"{len(filtered):,} of {len(data):,} customers selected")

current.run()
