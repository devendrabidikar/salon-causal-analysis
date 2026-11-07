from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_data import build_processed_tables


RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATE_MIN = pd.Timestamp("2025-11-01")
DATE_MAX = pd.Timestamp("2026-10-31")
EXPERIMENT_START = pd.Timestamp("2026-09-01")
EXPERIMENT_END = pd.Timestamp("2026-10-31")
POLICY_START = pd.Timestamp("2026-07-01")
PRIMARY_DID_POST_END = pd.Timestamp("2026-08-31")

CSV_COLUMNS = {
    "appointments.csv": [
        "appointment_id", "client_id", "appointment_date", "outlet_id",
        "stylist_id", "service_id", "appointment_status",
        "appointment_revenue", "discount_amount", "service_duration_minutes",
        "booking_channel", "cancellation_reason",
    ],
    "clients.csv": [
        "client_id", "signup_date", "retention_segment",
        "preferred_outlet_id", "simulated_active_until",
    ],
    "outlets.csv": ["outlet_id", "outlet_name", "region", "price_index"],
    "stylists.csv": [
        "stylist_id", "outlet_id", "stylist_level", "simulated_start_date",
    ],
    "services.csv": [
        "service_id", "service_name", "base_price", "base_duration_minutes",
    ],
    "products.csv": [
        "product_id", "product_name", "product_category", "unit_price",
    ],
    "product_sales.csv": [
        "sale_id", "appointment_id", "client_id", "outlet_id", "product_id",
        "sale_date", "quantity", "unit_price", "discount_amount",
        "product_revenue",
    ],
    "experiment_assignments.csv": [
        "client_id", "assignment_group", "assigned_at",
        "experiment_start_date", "experiment_end_date",
    ],
    "did_outlet_policy.csv": [
        "outlet_id", "region", "policy_group", "is_treated",
        "policy_start_date", "policy_name",
    ],
    "data_dictionary.csv": [
        "file_name", "column_name", "description", "synthetic_notes",
    ],
}

DATE_COLUMNS = {
    "appointments.csv": ["appointment_date"],
    "clients.csv": ["signup_date", "simulated_active_until"],
    "stylists.csv": ["simulated_start_date"],
    "product_sales.csv": ["sale_date"],
}
NUMERIC_COLUMNS = {
    "appointments.csv": [
        "appointment_revenue", "discount_amount", "service_duration_minutes",
    ],
    "outlets.csv": ["price_index"],
    "services.csv": ["base_price", "base_duration_minutes"],
    "products.csv": ["unit_price"],
    "product_sales.csv": [
        "quantity", "unit_price", "discount_amount", "product_revenue",
    ],
    "did_outlet_policy.csv": ["is_treated"],
}


def _file_signature() -> tuple[tuple[str, int, int], ...]:
    """Include file metadata in the cache key so changed CSVs reload."""
    missing = [name for name in CSV_COLUMNS if not (RAW_DATA_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required synthetic CSV file(s) are missing from "
            f"{RAW_DATA_DIR}: {', '.join(missing)}"
        )
    return tuple(
        (name, (RAW_DATA_DIR / name).stat().st_size,
         (RAW_DATA_DIR / name).stat().st_mtime_ns)
        for name in CSV_COLUMNS
    )


@st.cache_data(show_spinner="Loading synthetic salon data...")
def _load_csv_data(
    signature: tuple[tuple[str, int, int], ...],
) -> dict[str, pd.DataFrame]:
    del signature
    tables: dict[str, pd.DataFrame] = {}
    for filename, expected_columns in CSV_COLUMNS.items():
        path = RAW_DATA_DIR / filename
        frame = pd.read_csv(path, low_memory=False)
        missing_columns = [c for c in expected_columns if c not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"{filename} is missing expected column(s): "
                f"{', '.join(missing_columns)}"
            )
        for column in DATE_COLUMNS.get(filename, []):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        for column in NUMERIC_COLUMNS.get(filename, []):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        tables[filename.removesuffix(".csv")] = frame
    tables.update(build_processed_tables(tables))
    return tables


def load_data() -> dict[str, pd.DataFrame]:
    try:
        return _load_csv_data(_file_signature())
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as exc:
        st.error(f"Could not load the dashboard data: {exc}")
        st.info(
            "Keep the original synthetic CSV files in "
            "`data/raw/`. This dashboard does not generate or modify them."
        )
        st.stop()


def money(value: float | int) -> str:
    if pd.isna(value):
        return "—"
    amount = float(value)
    sign = "-" if amount < 0 else ""
    whole, decimals = f"{abs(amount):.2f}".split(".")
    if len(whole) > 3:
        last_three = whole[-3:]
        prefix = whole[:-3]
        pairs = []
        while prefix:
            pairs.append(prefix[-2:])
            prefix = prefix[:-2]
        whole = f"{','.join(reversed(pairs))},{last_three}"
    return f"{sign}₹{whole}.{decimals}"


def as_date(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def filter_dates(
    appointments: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    return appointments.loc[
        appointments["appointment_date"].between(start, end, inclusive="both")
    ].copy()


def add_page_title(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def appointment_status_chart(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No appointments fall within the selected date range.")
        return
    status_counts = (
        frame["appointment_status"].fillna("Unknown")
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Appointments")
    )
    fig = px.bar(
        status_counts,
        x="Status",
        y="Appointments",
        color="Status",
        text="Appointments",
        title="Appointment status",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Rows")
    st.plotly_chart(fig, width="stretch")


def build_client_outcomes(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return the shared, zero-filled client-level analysis dataset."""
    cohort = tables["client_experiment_outcomes"].copy()
    cohort["experiment_revenue"] = cohort["experiment_service_revenue"]
    cohort["baseline_revenue"] = cohort["pre_experiment_service_revenue"]
    cohort["baseline_appointment_count"] = cohort[
        "pre_experiment_appointment_count"
    ]
    cohort["signup_age_days"] = cohort["signup_age_days_at_assignment"]
    return cohort


def welch_comparison(
    treatment: pd.Series | np.ndarray,
    control: pd.Series | np.ndarray,
) -> dict[str, float]:
    treatment_values = np.asarray(treatment, dtype=float)
    control_values = np.asarray(control, dtype=float)
    n_t, n_c = len(treatment_values), len(control_values)
    mean_t, mean_c = treatment_values.mean(), control_values.mean()
    var_t, var_c = treatment_values.var(ddof=1), control_values.var(ddof=1)
    standard_error = np.sqrt(var_t / n_t + var_c / n_c)
    if standard_error == 0:
        return {
            "treatment_mean": mean_t, "control_mean": mean_c,
            "effect": mean_t - mean_c, "ci_low": mean_t - mean_c,
            "ci_high": mean_t - mean_c, "p_value": 1.0,
            "n_treatment": n_t, "n_control": n_c,
        }
    df_num = (var_t / n_t + var_c / n_c) ** 2
    df_den = ((var_t / n_t) ** 2 / (n_t - 1)) + (
        (var_c / n_c) ** 2 / (n_c - 1)
    )
    degrees_freedom = df_num / df_den
    effect = mean_t - mean_c
    critical = stats.t.ppf(0.975, degrees_freedom)
    test = stats.ttest_ind(treatment_values, control_values, equal_var=False)
    return {
        "treatment_mean": mean_t,
        "control_mean": mean_c,
        "effect": effect,
        "ci_low": effect - critical * standard_error,
        "ci_high": effect + critical * standard_error,
        "p_value": float(test.pvalue),
        "n_treatment": n_t,
        "n_control": n_c,
    }


def build_outlet_month_panel(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    panel = tables["outlet_month_panel"].copy()
    panel["post"] = panel["post_policy"]
    panel["treated_post"] = panel["is_treated"] * panel["post"]
    return panel


def fit_did(panel: pd.DataFrame, outcome: str) -> Any:
    primary = panel.loc[
        (panel["month"] <= PRIMARY_DID_POST_END)
        & (panel["month"] <= DATE_MAX)
    ].copy()
    primary["period"] = np.where(
        primary["month"] < POLICY_START, "Pre", "Post"
    )
    primary["treated_post"] = (
        primary["is_treated"] * primary["period"].eq("Post").astype(int)
    )
    return smf.ols(
        f"{outcome} ~ treated_post + C(outlet_id) + C(month)",
        data=primary,
    ).fit(
        cov_type="cluster",
        cov_kwds={"groups": primary["region"], "use_correction": True},
        use_t=True,
    )


def render_overview(
    tables: dict[str, pd.DataFrame],
    period_appointments: pd.DataFrame,
) -> None:
    add_page_title(
        "Salon performance overview",
        "Summary for the appointment dates selected in the sidebar.",
    )
    clients, outlets, stylists = (
        tables["clients"], tables["outlets"], tables["stylists"]
    )
    kpi_left, kpi_right = st.columns(2, gap="large")
    kpi_left.metric("Appointments", f"{len(period_appointments):,}")
    kpi_right.metric("Clients in roster", f"{clients['client_id'].nunique():,}")
    kpi_left, kpi_right = st.columns(2, gap="large")
    kpi_left.metric("Outlets", f"{outlets['outlet_id'].nunique():,}")
    kpi_right.metric("Stylists", f"{stylists['stylist_id'].nunique():,}")
    kpi_left, kpi_right = st.columns(2, gap="large")
    kpi_left.metric(
        "Service revenue",
        money(period_appointments["appointment_revenue"].sum()),
    )
    left, right = st.columns([1.25, 1])
    with left:
        appointment_status_chart(period_appointments)
    with right:
        st.subheader("Data snapshot")
        st.write(
            f"Appointment window: **{period_appointments['appointment_date'].min().date() if not period_appointments.empty else 'No rows'}** "
            f"to **{period_appointments['appointment_date'].max().date() if not period_appointments.empty else 'No rows'}**"
        )
        st.write(f"Randomized clients: **{tables['experiment_assignments']['client_id'].nunique():,}**")
        st.write(f"Product sales revenue (all dates): **{money(tables['product_sales']['product_revenue'].sum())}**")
        st.caption(
            "Client and entity roster totals describe the full synthetic dimensions; "
            "the appointment and service-revenue metrics use the selected date range."
        )


def render_appointments(frame: pd.DataFrame) -> None:
    add_page_title(
        "Appointment analytics",
        "Daily and monthly counts include walk-ins; rates use all appointments in the selected date range.",
    )
    if frame.empty:
        st.info("No appointments fall within the selected date range.")
        return
    daily = frame.groupby(frame["appointment_date"].dt.floor("D")).size().rename("Appointments")
    monthly = (
        frame.groupby(frame["appointment_date"].dt.to_period("M").dt.to_timestamp())
        .size().rename("Appointments")
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.line(
                daily.rename_axis("Date").reset_index(),
                x="Date", y="Appointments", title="Daily appointment volume",
                markers=True,
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.bar(
                monthly.rename_axis("Month").reset_index(),
                x="Month", y="Appointments", title="Monthly appointment volume",
            ),
            width="stretch",
        )
    left, right = st.columns(2)
    with left:
        appointment_status_chart(frame)
    with right:
        cancellations = frame.loc[
            frame["appointment_status"].eq("Cancelled"), "cancellation_reason"
        ].dropna()
        if cancellations.empty:
            st.info("No cancellation reasons are recorded for the selected period.")
        else:
            reasons = cancellations.value_counts().rename_axis("Reason").reset_index(
                name="Appointments"
            )
            st.plotly_chart(
                px.bar(
                    reasons, x="Appointments", y="Reason", orientation="h",
                    title="Cancellation reasons",
                ),
                width="stretch",
            )
    statuses = frame["appointment_status"].value_counts()
    total = max(len(frame), 1)
    no_show_rate = statuses.get("No-show", 0) / total
    cancel_rate = statuses.get("Cancelled", 0) / total
    m1, m2 = st.columns(2)
    m1.metric("No-show rate", f"{no_show_rate:.1%}")
    m2.metric("Cancellation rate", f"{cancel_rate:.1%}")
    st.caption("Rates use all appointment rows in the selected date range as the denominator.")


def render_revenue(
    tables: dict[str, pd.DataFrame],
    frame: pd.DataFrame,
) -> None:
    add_page_title(
        "Revenue analytics",
        "Service revenue and product sales are shown separately.",
    )
    if frame.empty:
        st.info("No appointments fall within the selected date range.")
        return
    monthly = (
        frame.groupby(frame["appointment_date"].dt.to_period("M").dt.to_timestamp())
        ["appointment_revenue"].sum().rename_axis("Month").reset_index()
    )
    st.plotly_chart(
        px.line(
            monthly, x="Month", y="appointment_revenue", markers=True,
            title="Monthly net service revenue",
            labels={"appointment_revenue": "Service revenue (₹)"},
        ),
        width="stretch",
    )
    outlets = frame.groupby("outlet_id", as_index=False)["appointment_revenue"].sum()
    outlets = outlets.merge(
        tables["outlets"][["outlet_id", "outlet_name", "region"]],
        on="outlet_id", how="left", validate="many_to_one",
    ).sort_values("appointment_revenue", ascending=True)
    services = frame.groupby("service_id", as_index=False)["appointment_revenue"].sum()
    services = services.merge(
        tables["services"][["service_id", "service_name"]],
        on="service_id", how="left", validate="many_to_one",
    ).sort_values("appointment_revenue", ascending=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                outlets, x="appointment_revenue", y="outlet_name", color="region",
                orientation="h", title="Service revenue by outlet",
                labels={"appointment_revenue": "Service revenue (₹)", "outlet_name": "Outlet"},
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            px.bar(
                services, x="appointment_revenue", y="service_name",
                orientation="h", title="Service revenue by service",
                labels={"appointment_revenue": "Service revenue (₹)", "service_name": "Service"},
            ),
            width="stretch",
        )
    product_sales = tables["product_sales"]
    product_dates = product_sales.loc[
        product_sales["sale_date"].between(
            frame["appointment_date"].min(), frame["appointment_date"].max(),
            inclusive="both",
        )
    ]
    st.metric("Product sales revenue in selected date window", money(product_dates["product_revenue"].sum()))
    st.caption("Product sales are filtered by sale date and remain distinct from service revenue.")


def render_customers(
    tables: dict[str, pd.DataFrame],
    frame: pd.DataFrame,
) -> None:
    add_page_title(
        "Customer analytics",
        "Client summaries use identified clients. Walk-ins have no client ID, and retention segments are simulated labels.",
    )
    identified_completed = frame.loc[
        frame["client_id"].notna()
        & frame["appointment_status"].eq("Completed")
    ]
    visits = identified_completed.groupby("client_id")["appointment_id"].count()
    active_clients = int(visits.index.nunique())
    repeat_clients = int(visits.ge(2).sum())
    repeat_share = repeat_clients / active_clients if active_clients else 0.0
    m1, m2, m3 = st.columns(3)
    m1.metric("Clients with a completed visit", f"{active_clients:,}")
    m2.metric("Repeat clients (2+ completed visits)", f"{repeat_clients:,}")
    m3.metric("Repeat share among active clients", f"{repeat_share:.1%}")

    segment_counts = (
        tables["clients"]["retention_segment"].fillna("Unknown")
        .value_counts().rename_axis("Simulated segment").reset_index(name="Clients")
    )
    st.plotly_chart(
        px.bar(
            segment_counts, x="Simulated segment", y="Clients", color="Simulated segment",
            title="Client roster by simulated retention segment",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        ).update_layout(showlegend=False),
        width="stretch",
    )
    per_client = identified_completed.groupby("client_id", as_index=False).agg(
        completed_visits=("appointment_id", "count"),
        service_revenue=("appointment_revenue", "sum"),
    )
    per_client = per_client.merge(
        tables["clients"][["client_id", "retention_segment"]],
        on="client_id", how="left", validate="one_to_one",
    )
    if not per_client.empty:
        segment_summary = per_client.groupby("retention_segment", as_index=False).agg(
            active_clients=("client_id", "nunique"),
            average_completed_visits=("completed_visits", "mean"),
            service_revenue=("service_revenue", "sum"),
        )
        st.dataframe(
            segment_summary.assign(
                service_revenue=segment_summary["service_revenue"].map(money),
                average_completed_visits=segment_summary[
                    "average_completed_visits"
                ].round(2),
            ),
            width="stretch",
            hide_index=True,
        )
    st.caption(
        "Repeat share is computed within the selected date range. It is descriptive, "
        "not a causal retention estimate; walk-in sales are not attributed to clients."
    )


def render_ab(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    add_page_title(
        "A/B testing",
        "Client-level intention-to-treat comparison for September 1–October 31, 2026. Each assigned client is counted once.",
    )
    cohort = build_client_outcomes(tables)
    counts = cohort["assignment_group"].value_counts()
    if set(counts.index) != {"Treatment", "Control"}:
        st.error("Experiment assignments must contain Treatment and Control groups.")
        return cohort
    treatment = cohort.loc[
        cohort["assignment_group"].eq("Treatment"), "experiment_revenue"
    ]
    control = cohort.loc[
        cohort["assignment_group"].eq("Control"), "experiment_revenue"
    ]
    result = welch_comparison(treatment, control)
    lift = result["effect"] / result["control_mean"] if result["control_mean"] else np.nan
    zero_count = int(cohort["experiment_revenue"].eq(0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Treatment / Control clients", f"{len(treatment):,} / {len(control):,}")
    c2.metric("Treatment mean revenue / client", money(result["treatment_mean"]))
    c3.metric("Control mean revenue / client", money(result["control_mean"]))
    st.subheader("Client-level ITT estimate")
    st.write(
        f"Treatment minus Control: **{money(result['effect'])} per randomized client** "
        f"(95% Welch CI {money(result['ci_low'])} to {money(result['ci_high'])}; "
        f"p = {result['p_value']:.4g})."
    )
    st.write(f"Relative lift versus Control mean: **{lift:.2%}**" if pd.notna(lift) else "Relative lift is undefined because the Control mean is zero.")
    st.caption(
        f"All {len(cohort):,} assigned clients are included, including {zero_count:,} "
        "with zero experiment-period service revenue. Repeated appointment rows are "
        "aggregated to one outcome per randomized client; walk-ins without a client_id "
        "are not part of the randomized cohort."
    )
    chart_data = cohort[["assignment_group", "experiment_revenue"]].rename(
        columns={"assignment_group": "Group", "experiment_revenue": "Revenue per client"}
    )
    st.plotly_chart(
        px.box(
            chart_data, x="Group", y="Revenue per client", color="Group",
            points=False, title="Experiment-period service revenue per randomized client",
            color_discrete_map={"Control": "#4575b4", "Treatment": "#d73027"},
        ),
        width="stretch",
    )
    return cohort


def render_did(tables: dict[str, pd.DataFrame]) -> None:
    add_page_title(
        "Difference-in-Differences",
        "Outlet-month service revenue. July–August 2026 is the primary post-policy period; September–October overlaps the A/B experiment.",
    )
    panel = build_outlet_month_panel(tables)
    panel["period"] = np.where(
        panel["month"] < POLICY_START,
        "Pre (Nov 2025–Jun 2026)",
        np.where(
            panel["month"] <= PRIMARY_DID_POST_END,
            "Primary post (Jul–Aug 2026)",
            "A/B-overlap sensitivity (Sep–Oct 2026)",
        ),
    )
    summary = panel.loc[panel["month"] <= PRIMARY_DID_POST_END].groupby(
        ["is_treated", "period"], as_index=False
    ).agg(
        mean_service_revenue=("service_revenue", "mean"),
        mean_appointments=("appointment_count", "mean"),
        outlet_months=("outlet_id", "size"),
    )
    summary["Group"] = summary["is_treated"].map({1: "Treated", 0: "Control"})
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            px.bar(
                summary, x="period", y="mean_service_revenue", color="Group",
                barmode="group", title="Mean service revenue per outlet-month",
                labels={"mean_service_revenue": "Revenue (₹)", "period": ""},
            ),
            width="stretch",
        )
    with right:
        trends = panel.groupby(["month", "is_treated"], as_index=False)[
            "service_revenue"
        ].mean()
        trends["Group"] = trends["is_treated"].map({1: "Treated", 0: "Control"})
        fig = px.line(
            trends, x="month", y="service_revenue", color="Group", markers=True,
            title="Monthly mean service revenue per outlet",
            labels={"service_revenue": "Revenue (₹)", "month": "Month"},
        )
        fig.add_vline(x=POLICY_START, line_dash="dash", line_color="black")
        st.plotly_chart(fig, width="stretch")

    try:
        model = fit_did(panel, "service_revenue")
        effect = float(model.params["treated_post"])
        low, high = map(float, model.conf_int().loc["treated_post"])
        p_value = float(model.pvalues["treated_post"])
        st.subheader("Primary fixed-effects DiD estimate")
        st.write(
            f"Estimated treated × post contrast: **{money(effect)} per outlet-month** "
            f"(region-clustered 95% CI {money(low)} to {money(high)}; "
            f"reported model p = {p_value:.4g})."
        )
    except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
        st.warning(f"The DiD model could not be estimated for this data: {exc}")

    st.warning(
        "Inference limitation: policy assignment is at the region level, with only "
        "four synthetic regions (North and South treated; East and West control). "
        "Cluster-robust standard errors, p-values, and confidence intervals with four "
        "clusters are unreliable and must not be presented as strong causal evidence. "
        "Pre-trend diagnostics cannot prove parallel trends."
    )
    st.caption(
        "September–October 2026 is displayed as an overlap sensitivity period because "
        "the client-level A/B experiment also runs then. All outcomes are synthetic."
    )
    st.dataframe(
        summary[["Group", "period", "mean_service_revenue", "mean_appointments", "outlet_months"]]
        .rename(columns={
            "mean_service_revenue": "Mean revenue / outlet-month (₹)",
            "mean_appointments": "Mean appointments / outlet-month",
            "outlet_months": "Outlet-months",
        })
        .round(2),
        width="stretch",
        hide_index=True,
    )


def render_cuped_regression(
    tables: dict[str, pd.DataFrame],
    cohort: pd.DataFrame | None,
) -> None:
    add_page_title(
        "CUPED & regression adjustment",
        "Client-level estimates adjusted with information recorded before the experiment.",
    )
    if cohort is None:
        cohort, _ = build_client_outcomes(tables)
    treatment_mask = cohort["assignment_group"].eq("Treatment")
    control_mask = cohort["assignment_group"].eq("Control")
    y = cohort["experiment_revenue"].astype(float)
    x = cohort["baseline_revenue"].astype(float)
    variance_x = x.var(ddof=1)
    theta = float(np.cov(y, x, ddof=1)[0, 1] / variance_x) if variance_x > 0 else 0.0
    adjusted_y = y - theta * (x - x.mean())
    cuped = welch_comparison(adjusted_y[treatment_mask], adjusted_y[control_mask])
    raw = welch_comparison(y[treatment_mask], y[control_mask])
    variance_reduction = (
        100 * (1 - adjusted_y.var(ddof=1) / y.var(ddof=1))
        if y.var(ddof=1) > 0 else 0.0
    )

    regression_data = cohort.copy()
    regression_data["preferred_region"] = regression_data[
        "preferred_region"
    ].fillna("Unknown")
    regression_data["retention_segment"] = regression_data[
        "retention_segment"
    ].fillna("Unknown")
    formula = (
        "experiment_revenue ~ treatment + baseline_revenue + "
        "baseline_appointment_count + signup_age_days + "
        "C(retention_segment) + C(preferred_region)"
    )
    regression = smf.ols(formula, data=regression_data).fit(cov_type="HC3")
    reg_effect = float(regression.params["treatment"])
    reg_low, reg_high = map(float, regression.conf_int().loc["treatment"])
    reg_p = float(regression.pvalues["treatment"])

    st.caption(
        "Covariates are computed strictly before 2026-09-01. The primary outcome is "
        "service revenue per assigned client, with zero-revenue clients retained."
    )
    metrics = [
        ("Unadjusted ITT", raw["effect"], raw["ci_low"], raw["ci_high"], raw["p_value"]),
        ("CUPED", cuped["effect"], cuped["ci_low"], cuped["ci_high"], cuped["p_value"]),
        ("OLS with HC3", reg_effect, reg_low, reg_high, reg_p),
    ]
    comparison = pd.DataFrame(
        metrics,
        columns=["Method", "Effect (₹/client)", "CI low", "CI high", "p-value"],
    )
    left, right = st.columns([1, 1.2])
    with left:
        st.metric("CUPED θ", f"{theta:.5f}")
        st.metric("CUPED outcome variance reduction", f"{variance_reduction:.2f}%")
        st.metric("HC3 regression treatment coefficient", money(reg_effect))
    with right:
        fig = go.Figure()
        for _, row in comparison.iterrows():
            fig.add_trace(go.Scatter(
                x=[row["Effect (₹/client)"]],
                y=[row["Method"]],
                mode="markers",
                marker={"size": 11},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [row["CI high"] - row["Effect (₹/client)"]],
                    "arrayminus": [row["Effect (₹/client)"] - row["CI low"]],
                },
                name=row["Method"],
            ))
        fig.update_layout(
            title="Effect estimates with 95% confidence intervals",
            xaxis_title="Treatment effect (₹ per randomized client)",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")
    st.dataframe(comparison.round(4), width="stretch", hide_index=True)
    st.info(
        "These are recalculated estimates from the local synthetic CSVs, not imported "
        "or hard-coded findings. CUPED uses the pre-period service revenue covariate; "
        "the regression uses pre-period revenue, appointment count, signup age, "
        "simulated retention segment, and preferred region. Adjustment does not turn "
        "the synthetic result into real-world evidence."
    )


def render_decision(cohort: pd.DataFrame | None) -> None:
    add_page_title(
        "Final decision",
        "The documented recommendation and the limits of what this simulation can show.",
    )
    st.error("Documented recommendation: NO-GO for real-world rollout.")
    st.write(
        "The existing decision memo advises against operational launch or customer-impact "
        "decisions based on this project. The dataset and injected effects are fictional; "
        "any real decision requires authorized real data and a prospectively specified evaluation."
    )
    st.markdown("**Key limitations**")
    st.markdown(
        "- A/B estimates are synthetic client-level ITT calculations, not real business impact.\n"
        "- DiD policy assignment has only four regions (two treated and two control), "
        "so cluster-robust inference is unreliable; parallel trends are not established.\n"
        "- The A/B experiment overlaps the policy post-period in September–October 2026.\n"
        "- The generator deliberately injects artificial 6% A/B and 4% policy effects."
    )
    st.caption("Source: reports/One_Page_Decision_Memo.docx and the executed analysis notebooks.")
    if cohort is not None:
        result = welch_comparison(
            cohort.loc[cohort["assignment_group"].eq("Treatment"), "experiment_revenue"],
            cohort.loc[cohort["assignment_group"].eq("Control"), "experiment_revenue"],
        )
        st.metric("Current-data A/B ITT estimate", money(result["effect"]))
        st.caption(
            "A descriptive dashboard statistic only; it is not a rollout threshold "
            "or a real-world recommendation."
        )


def main() -> None:
    st.set_page_config(
        page_title="Salon Experiment & Causal Analysis",
        page_icon="✂️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container {padding: 1.5rem 2rem 2rem;}
        [data-testid="stMetric"] {
            background: #14213d;
            border: 1px solid #334b70;
            padding: 1.1rem 1.25rem;
            border-radius: 0.75rem;
            min-height: 6.4rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.16);
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {
            color: #dbeafe !important;
            font-weight: 600;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div {
            color: #ffffff !important;
            font-size: clamp(1.25rem, 2vw, 2rem);
            line-height: 1.2;
            white-space: normal !important;
            overflow: visible !important;
            overflow-wrap: anywhere;
            text-overflow: clip !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("Salon Analytics")
    st.sidebar.caption("Synthetic internship project")
    st.sidebar.info(
        "All records and effects are fictional and simulated. "
        "No real customer or salon data is represented."
    )
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Appointment Analytics",
            "Revenue Analytics",
            "Customer Analytics",
            "A/B Testing",
            "Difference-in-Differences",
            "CUPED & Regression",
            "Final Decision",
        ],
    )
    tables = load_data()
    appointments = tables["appointments"]
    observed_min = appointments["appointment_date"].min()
    observed_max = appointments["appointment_date"].max()
    default_min = max(observed_min, DATE_MIN) if pd.notna(observed_min) else DATE_MIN
    default_max = min(observed_max, DATE_MAX) if pd.notna(observed_max) else DATE_MAX
    date_range = st.sidebar.date_input(
        "Appointment date range",
        value=(default_min.date(), default_max.date()),
        min_value=DATE_MIN.date(),
        max_value=DATE_MAX.date(),
        help="Applies to descriptive overview, appointment, revenue, and customer pages. Experiment and DiD use their specified analysis windows.",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = map(as_date, date_range)
    else:
        start = end = as_date(date_range)
    if start > end:
        st.sidebar.error("Start date must be on or before end date.")
        st.stop()
    selected = filter_dates(appointments, start, end)

    if page == "Overview":
        render_overview(tables, selected)
    elif page == "Appointment Analytics":
        render_appointments(selected)
    elif page == "Revenue Analytics":
        render_revenue(tables, selected)
    elif page == "Customer Analytics":
        render_customers(tables, selected)
    elif page == "A/B Testing":
        render_ab(tables)
    elif page == "Difference-in-Differences":
        render_did(tables)
    elif page == "CUPED & Regression":
        cohort = build_client_outcomes(tables)
        render_cuped_regression(tables, cohort)
    elif page == "Final Decision":
        cohort = build_client_outcomes(tables)
        render_decision(cohort)


if __name__ == "__main__":
    main()
