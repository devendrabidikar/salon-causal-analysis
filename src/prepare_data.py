"""Build reusable client-level and outlet-month analysis datasets.

The raw synthetic CSVs are read only. Generated outputs are deterministic:
identical reruns are no-ops, while partial or differing existing outputs are
refused rather than overwritten.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

HISTORY_START = pd.Timestamp("2025-11-01")
HISTORY_END = pd.Timestamp("2026-10-31")
ASSIGNMENT_DATE = pd.Timestamp("2026-08-15")
EXPERIMENT_START = pd.Timestamp("2026-09-01")
EXPERIMENT_END = pd.Timestamp("2026-10-31")
POLICY_START = pd.Timestamp("2026-07-01")
PRIMARY_DID_POST_END = pd.Timestamp("2026-08-31")
PLANNING_BASELINE_START = pd.Timestamp("2026-05-01")
PLANNING_BASELINE_END = pd.Timestamp("2026-06-30")

CLIENT_OUTPUT = "client_experiment_outcomes.csv"
OUTLET_MONTH_OUTPUT = "outlet_month_panel.csv"
OUTPUT_FILES = (CLIENT_OUTPUT, OUTLET_MONTH_OUTPUT)

CLIENT_COLUMNS = (
    "client_id",
    "assignment_group",
    "treatment",
    "assigned_at",
    "experiment_start_date",
    "experiment_end_date",
    "signup_date",
    "signup_age_days_at_assignment",
    "retention_segment",
    "preferred_outlet_id",
    "preferred_region",
    "simulated_active_until",
    "pre_experiment_appointment_count",
    "pre_experiment_completed_count",
    "pre_experiment_service_revenue",
    "planning_baseline_appointment_count",
    "planning_baseline_completed_count",
    "planning_baseline_service_revenue",
    "planning_baseline_product_revenue",
    "experiment_appointment_count",
    "experiment_completed_count",
    "experiment_cancelled_count",
    "experiment_no_show_count",
    "experiment_service_revenue",
    "experiment_product_revenue",
    "completed_share_per_client",
    "cancelled_share_per_client",
    "no_show_share_per_client",
    "any_completed",
    "any_cancelled",
    "any_no_show",
    "repeat_completed",
)

OUTLET_MONTH_COLUMNS = (
    "outlet_id",
    "outlet_name",
    "region",
    "policy_group",
    "is_treated",
    "policy_start_date",
    "month",
    "month_label",
    "post_primary",
    "post_sensitivity",
    "post_policy",
    "service_revenue",
    "discount_amount",
    "appointment_count",
    "completed_count",
    "cancelled_count",
    "no_show_count",
    "product_revenue",
)

REQUIRED_COLUMNS: Mapping[str, set[str]] = {
    "appointments": {
        "appointment_id", "client_id", "appointment_date", "outlet_id",
        "appointment_status", "appointment_revenue", "discount_amount",
    },
    "clients": {
        "client_id", "signup_date", "retention_segment",
        "preferred_outlet_id", "simulated_active_until",
    },
    "outlets": {"outlet_id", "outlet_name", "region"},
    "experiment_assignments": {
        "client_id", "assignment_group", "assigned_at",
        "experiment_start_date", "experiment_end_date",
    },
    "did_outlet_policy": {
        "outlet_id", "region", "policy_group", "is_treated",
        "policy_start_date",
    },
    "product_sales": {
        "sale_id", "appointment_id", "client_id", "outlet_id", "product_id",
        "sale_date", "quantity", "unit_price", "discount_amount",
        "product_revenue",
    },
}


def _require_columns(tables: Mapping[str, pd.DataFrame]) -> None:
    for table_name, required in REQUIRED_COLUMNS.items():
        if table_name not in tables:
            raise ValueError(f"Missing required input table: {table_name}")
        missing = sorted(required - set(tables[table_name].columns))
        if missing:
            raise ValueError(
                f"{table_name} is missing required columns: {', '.join(missing)}"
            )


def _prepare_source_frames(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    _require_columns(tables)
    frames = {name: tables[name].copy() for name in REQUIRED_COLUMNS}
    for table_name, column in (
        ("appointments", "appointment_date"),
        ("clients", "signup_date"),
        ("clients", "simulated_active_until"),
        ("experiment_assignments", "assigned_at"),
        ("experiment_assignments", "experiment_start_date"),
        ("experiment_assignments", "experiment_end_date"),
        ("did_outlet_policy", "policy_start_date"),
        ("product_sales", "sale_date"),
    ):
        frames[table_name][column] = pd.to_datetime(
            frames[table_name][column], errors="coerce"
        )
    for table_name, columns in (
        ("appointments", ["appointment_revenue", "discount_amount"]),
        ("product_sales", ["product_revenue"]),
        ("did_outlet_policy", ["is_treated"]),
    ):
        for column in columns:
            frames[table_name][column] = pd.to_numeric(
                frames[table_name][column], errors="coerce"
            )
    return frames


def _aggregate_appointments(
    frame: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    return frame.groupby("client_id", as_index=False).agg(
        **{
            f"{prefix}_appointment_count": ("appointment_id", "size"),
            f"{prefix}_completed_count": (
                "appointment_status",
                lambda values: values.eq("Completed").sum(),
            ),
            **(
                {
                    f"{prefix}_cancelled_count": (
                        "appointment_status",
                        lambda values: values.eq("Cancelled").sum(),
                    ),
                    f"{prefix}_no_show_count": (
                        "appointment_status",
                        lambda values: values.eq("No-show").sum(),
                    ),
                }
                if prefix == "experiment"
                else {}
            ),
            f"{prefix}_service_revenue": ("appointment_revenue", "sum"),
        }
    )


def _validate_sources(frames: Mapping[str, pd.DataFrame]) -> None:
    appointments = frames["appointments"]
    clients = frames["clients"]
    outlets = frames["outlets"]
    assignments = frames["experiment_assignments"]
    policy = frames["did_outlet_policy"]
    sales = frames["product_sales"]

    if clients["client_id"].isna().any() or clients["client_id"].duplicated().any():
        raise ValueError("clients.client_id must be non-null and unique")
    if assignments["client_id"].isna().any() or assignments["client_id"].duplicated().any():
        raise ValueError(
            "experiment_assignments.client_id must be non-null and unique"
        )
    if set(assignments["client_id"]) != set(clients["client_id"]):
        raise ValueError("Assignments must cover exactly the client dimension")
    if set(assignments["assignment_group"].dropna()) != {"Control", "Treatment"}:
        raise ValueError("Assignment groups must be Control and Treatment")
    if assignments["assignment_group"].isna().any():
        raise ValueError("Assignment groups cannot be null")
    group_counts = assignments["assignment_group"].value_counts()
    if group_counts.get("Control", 0) != group_counts.get("Treatment", 0):
        raise ValueError("Client experiment assignments must be balanced 1:1")
    if (
        assignments["assigned_at"].isna().any()
        or assignments["assigned_at"].ne(ASSIGNMENT_DATE).any()
        or assignments["experiment_start_date"].isna().any()
        or assignments["experiment_start_date"].ne(EXPERIMENT_START).any()
        or assignments["experiment_end_date"].isna().any()
        or assignments["experiment_end_date"].ne(EXPERIMENT_END).any()
    ):
        raise ValueError("Experiment assignment metadata dates are invalid")
    if clients["signup_date"].isna().any() or clients["signup_date"].gt(ASSIGNMENT_DATE).any():
        raise ValueError("Client signups must be valid and on/before assignment")
    if clients["simulated_active_until"].isna().any():
        raise ValueError("Client active-until dates must be valid")
    if outlets["outlet_id"].isna().any() or outlets["outlet_id"].duplicated().any():
        raise ValueError("outlets.outlet_id must be non-null and unique")
    if policy["outlet_id"].isna().any() or policy["outlet_id"].duplicated().any():
        raise ValueError("Policy outlet IDs must be non-null and unique")
    if set(policy["outlet_id"]) != set(outlets["outlet_id"]):
        raise ValueError("Policy rows must cover exactly the outlet dimension")
    if policy["region"].isna().any() or policy["policy_start_date"].isna().any():
        raise ValueError("Policy region and start date cannot be null")
    if policy["policy_start_date"].ne(POLICY_START).any():
        raise ValueError("Policy start date must be 2026-07-01")
    if not policy["is_treated"].isin([0, 1]).all():
        raise ValueError("Policy is_treated values must be 0 or 1")
    if not policy["region"].isin(["North", "South", "East", "West"]).all():
        raise ValueError("Policy regions must be North, South, East, or West")
    expected_groups = np.where(policy["is_treated"].eq(1), "Treated", "Control")
    if not policy["policy_group"].eq(expected_groups).all():
        raise ValueError("Policy group and is_treated values disagree")
    expected_treatment = policy["region"].isin(["North", "South"]).astype("int8")
    if not policy["is_treated"].astype("int8").eq(expected_treatment).all():
        raise ValueError("Policy regions do not match the documented treated regions")
    outlet_regions = outlets.set_index("outlet_id")["region"]
    policy_regions = policy.set_index("outlet_id")["region"]
    if not outlet_regions.sort_index().equals(policy_regions.sort_index()):
        raise ValueError("Policy regions do not match outlet regions")

    if appointments["appointment_id"].isna().any() or appointments["appointment_id"].duplicated().any():
        raise ValueError("Appointment IDs must be non-null and unique")
    dates = appointments["appointment_date"]
    if dates.isna().any() or not dates.between(HISTORY_START, HISTORY_END).all():
        raise ValueError("Appointment dates are invalid or outside the history window")
    if not appointments["appointment_status"].isin(
        ["Completed", "Cancelled", "No-show"]
    ).all():
        raise ValueError("Appointment status contains unexpected values")
    if appointments["appointment_revenue"].isna().any() or appointments[
        "appointment_revenue"
    ].lt(0).any():
        raise ValueError("Appointment service revenue must be non-null and non-negative")
    if appointments["discount_amount"].isna().any() or appointments[
        "discount_amount"
    ].lt(0).any():
        raise ValueError("Appointment discounts must be non-null and non-negative")
    non_completed = appointments["appointment_status"].ne("Completed")
    if (
        appointments.loc[non_completed, "appointment_revenue"].ne(0).any()
        or appointments.loc[non_completed, "discount_amount"].ne(0).any()
    ):
        raise ValueError("Non-completed appointments must have zero revenue and discount")
    if not appointments["client_id"].dropna().isin(clients["client_id"]).all():
        raise ValueError("Appointment rows contain unknown client IDs")
    if not appointments["outlet_id"].isin(outlets["outlet_id"]).all():
        raise ValueError("Appointment rows contain unknown outlet IDs")
    if not sales["client_id"].dropna().isin(clients["client_id"]).all():
        raise ValueError("Product sales contain unknown non-null client IDs")
    if not sales["outlet_id"].isin(outlets["outlet_id"]).all():
        raise ValueError("Product sales contain unknown outlet IDs")
    if (
        sales["sale_date"].isna().any()
        or not sales["sale_date"].between(HISTORY_START, HISTORY_END).all()
        or sales["product_revenue"].isna().any()
    ):
        raise ValueError("Product-sale dates and revenue must be valid")
    if (
        sales["product_revenue"].lt(0).any()
        or sales["quantity"].isna().any()
        or pd.to_numeric(sales["quantity"], errors="coerce").le(0).any()
        or sales["unit_price"].isna().any()
        or pd.to_numeric(sales["unit_price"], errors="coerce").le(0).any()
        or sales["discount_amount"].isna().any()
        or pd.to_numeric(sales["discount_amount"], errors="coerce").lt(0).any()
    ):
        raise ValueError("Product-sale quantities, prices, discounts, or revenue are invalid")
    expected_revenue = (
        pd.to_numeric(sales["quantity"], errors="coerce")
        * pd.to_numeric(sales["unit_price"], errors="coerce")
        - pd.to_numeric(sales["discount_amount"], errors="coerce")
    )
    if (expected_revenue - sales["product_revenue"]).abs().gt(0.01).any():
        raise ValueError("Product-sale revenue does not reconcile within 0.01")
    if sales["sale_id"].isna().any() or sales["sale_id"].duplicated().any():
        raise ValueError("Product-sale IDs must be non-null and unique")
    for column in ("sale_id", "appointment_id", "outlet_id", "product_id"):
        if sales[column].isna().any() or sales[column].astype("string").str.strip().eq("").any():
            raise ValueError(f"Product-sale {column} values must be non-empty")
    appointment_lookup = appointments.set_index("appointment_id")
    if not sales["appointment_id"].isin(appointment_lookup.index).all():
        raise ValueError("Product sales contain unknown appointment IDs")
    sale_appointments = appointment_lookup.loc[sales["appointment_id"]]
    if (
        not sale_appointments["appointment_status"].eq("Completed").all()
        or not sale_appointments["outlet_id"].reset_index(drop=True).equals(
            sales["outlet_id"].reset_index(drop=True)
        )
        or not pd.to_datetime(
            sale_appointments["appointment_date"].reset_index(drop=True)
        ).equals(sales["sale_date"].reset_index(drop=True))
    ):
        raise ValueError("Product sales do not match their completed appointments")
    expected_clients = sale_appointments["client_id"].fillna("").astype(str).reset_index(drop=True)
    observed_clients = sales["client_id"].fillna("").astype(str).reset_index(drop=True)
    if not expected_clients.equals(observed_clients):
        raise ValueError("Product-sale client IDs do not match linked appointments")


def _client_outcomes(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    appointments = frames["appointments"]
    clients = frames["clients"]
    assignments = frames["experiment_assignments"]
    sales = frames["product_sales"]
    outlets = frames["outlets"]

    linked = appointments.loc[appointments["client_id"].notna()].copy()
    pre_experiment = linked.loc[
        linked["appointment_date"] < EXPERIMENT_START
    ]
    planning_baseline = linked.loc[
        linked["appointment_date"].between(
            PLANNING_BASELINE_START, PLANNING_BASELINE_END, inclusive="both"
        )
    ]
    experiment = linked.loc[
        linked["appointment_date"].between(
            EXPERIMENT_START, EXPERIMENT_END, inclusive="both"
        )
    ]

    client = assignments.merge(
        clients,
        on="client_id",
        how="left",
        validate="one_to_one",
    )
    region_by_outlet = outlets.set_index("outlet_id")["region"]
    client["preferred_region"] = client["preferred_outlet_id"].map(region_by_outlet)
    if client["preferred_region"].isna().any():
        raise ValueError("Client preferred outlet references an unknown outlet")
    client["treatment"] = client["assignment_group"].eq("Treatment").astype(int)
    client["signup_age_days_at_assignment"] = (
        ASSIGNMENT_DATE - client["signup_date"]
    ).dt.days.astype(int)

    aggregations = [
        (
            pre_experiment,
            "pre_experiment",
            (
                "pre_experiment_appointment_count",
                "pre_experiment_completed_count",
                "pre_experiment_service_revenue",
            ),
        ),
        (
            planning_baseline,
            "planning_baseline",
            (
                "planning_baseline_appointment_count",
                "planning_baseline_completed_count",
                "planning_baseline_service_revenue",
            ),
        ),
        (
            experiment,
            "experiment",
            (
                "experiment_appointment_count",
                "experiment_completed_count",
                "experiment_cancelled_count",
                "experiment_no_show_count",
                "experiment_service_revenue",
            ),
        ),
    ]
    for source, prefix, _ in aggregations:
        metrics = _aggregate_appointments(source, prefix)
        client = client.merge(
            metrics, on="client_id", how="left", validate="one_to_one"
        )

    client_sales = sales.loc[sales["client_id"].notna()].copy()
    planning_sales = client_sales.loc[
        client_sales["sale_date"].between(
            PLANNING_BASELINE_START, PLANNING_BASELINE_END, inclusive="both"
        )
    ].groupby("client_id")["product_revenue"].sum()
    experiment_sales = client_sales.loc[
        client_sales["sale_date"].between(
            EXPERIMENT_START, EXPERIMENT_END, inclusive="both"
        )
    ].groupby("client_id")["product_revenue"].sum()
    client["planning_baseline_product_revenue"] = (
        client["client_id"].map(planning_sales).fillna(0.0)
    )
    client["experiment_product_revenue"] = (
        client["client_id"].map(experiment_sales).fillna(0.0)
    )

    count_columns = [
        "pre_experiment_appointment_count",
        "pre_experiment_completed_count",
        "planning_baseline_appointment_count",
        "planning_baseline_completed_count",
        "experiment_appointment_count",
        "experiment_completed_count",
        "experiment_cancelled_count",
        "experiment_no_show_count",
    ]
    revenue_columns = [
        "pre_experiment_service_revenue",
        "planning_baseline_service_revenue",
        "planning_baseline_product_revenue",
        "experiment_service_revenue",
        "experiment_product_revenue",
    ]
    client[count_columns] = client[count_columns].fillna(0).astype("int64")
    client[revenue_columns] = client[revenue_columns].fillna(0.0).astype("float64")

    denominator = client["experiment_appointment_count"].to_numpy()
    client["completed_share_per_client"] = np.divide(
        client["experiment_completed_count"],
        denominator,
        out=np.zeros(len(client), dtype="float64"),
        where=denominator > 0,
    )
    client["cancelled_share_per_client"] = np.divide(
        client["experiment_cancelled_count"],
        denominator,
        out=np.zeros(len(client), dtype="float64"),
        where=denominator > 0,
    )
    client["no_show_share_per_client"] = np.divide(
        client["experiment_no_show_count"],
        denominator,
        out=np.zeros(len(client), dtype="float64"),
        where=denominator > 0,
    )
    client["any_completed"] = client["experiment_completed_count"].gt(0).astype("int8")
    client["any_cancelled"] = client["experiment_cancelled_count"].gt(0).astype("int8")
    client["any_no_show"] = client["experiment_no_show_count"].gt(0).astype("int8")
    client["repeat_completed"] = client["experiment_completed_count"].ge(2).astype("int8")

    result = client.loc[:, CLIENT_COLUMNS].sort_values(
        "client_id", kind="stable"
    ).reset_index(drop=True)
    if len(result) != len(assignments) or result["client_id"].duplicated().any():
        raise ValueError("Client output must contain exactly one row per assignment")
    if result[list(CLIENT_COLUMNS)].isna().any().any():
        raise ValueError("Client output contains missing values after zero-fill")
    return result


def _outlet_month_panel(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    appointments = frames["appointments"].copy()
    sales = frames["product_sales"].copy()
    outlets = frames["outlets"]
    policy = frames["did_outlet_policy"]

    appointments["month"] = (
        appointments["appointment_date"].dt.to_period("M").dt.to_timestamp()
    )
    observed = appointments.groupby(
        ["outlet_id", "month"], as_index=False, sort=True
    ).agg(
        service_revenue=("appointment_revenue", "sum"),
        discount_amount=("discount_amount", "sum"),
        appointment_count=("appointment_id", "size"),
        completed_count=(
            "appointment_status",
            lambda values: values.eq("Completed").sum(),
        ),
        cancelled_count=(
            "appointment_status",
            lambda values: values.eq("Cancelled").sum(),
        ),
        no_show_count=(
            "appointment_status",
            lambda values: values.eq("No-show").sum(),
        ),
    )
    sales["month"] = sales["sale_date"].dt.to_period("M").dt.to_timestamp()
    product = sales.groupby(
        ["outlet_id", "month"], as_index=False, sort=True
    ).agg(product_revenue=("product_revenue", "sum"))

    months = pd.date_range(HISTORY_START, HISTORY_END, freq="MS")
    grid = pd.MultiIndex.from_product(
        [outlets["outlet_id"].sort_values().tolist(), months],
        names=["outlet_id", "month"],
    ).to_frame(index=False)
    panel = grid.merge(
        observed, on=["outlet_id", "month"], how="left", validate="one_to_one"
    ).merge(
        product, on=["outlet_id", "month"], how="left", validate="one_to_one"
    )
    value_columns = [
        "service_revenue", "discount_amount", "appointment_count",
        "completed_count", "cancelled_count", "no_show_count", "product_revenue",
    ]
    panel[value_columns] = panel[value_columns].fillna(0)
    panel = panel.merge(
        outlets[["outlet_id", "outlet_name", "region"]],
        on="outlet_id",
        how="left",
        validate="many_to_one",
    ).merge(
        policy[[
            "outlet_id", "policy_group", "is_treated", "policy_start_date"
        ]],
        on="outlet_id",
        how="left",
        validate="many_to_one",
    )
    panel["month_label"] = panel["month"].dt.strftime("%Y-%m")
    panel["post_primary"] = panel["month"].between(
        POLICY_START, PRIMARY_DID_POST_END, inclusive="both"
    ).astype("int8")
    panel["post_sensitivity"] = panel["month"].between(
        POLICY_START, EXPERIMENT_END.replace(day=1) + pd.offsets.MonthEnd(0),
        inclusive="both",
    ).astype("int8")
    panel["post_policy"] = panel["month"].ge(POLICY_START).astype("int8")
    panel["is_treated"] = panel["is_treated"].astype("int8")
    panel["policy_start_date"] = panel["policy_start_date"].dt.strftime("%Y-%m-%d")
    result = panel.loc[:, OUTLET_MONTH_COLUMNS].sort_values(
        ["outlet_id", "month"], kind="stable"
    ).reset_index(drop=True)
    expected_rows = outlets["outlet_id"].nunique() * len(months)
    if len(result) != expected_rows:
        raise ValueError(f"Outlet-month output has {len(result)} rows; expected {expected_rows}")
    if result.duplicated(["outlet_id", "month"]).any():
        raise ValueError("Outlet-month output contains duplicate outlet/month keys")
    if result[list(OUTLET_MONTH_COLUMNS)].isna().any().any():
        raise ValueError("Outlet-month output contains missing values")
    return result


def build_processed_tables(
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build analysis-ready tables from already loaded raw tables."""
    frames = _prepare_source_frames(tables)
    _validate_sources(frames)
    return {
        CLIENT_OUTPUT.removesuffix(".csv"): _client_outcomes(frames),
        OUTLET_MONTH_OUTPUT.removesuffix(".csv"): _outlet_month_panel(frames),
    }


def load_raw_tables(raw_dir: Path = RAW_DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read only the raw tables required by this preprocessing pipeline."""
    filenames = {
        "appointments": "appointments.csv",
        "clients": "clients.csv",
        "outlets": "outlets.csv",
        "experiment_assignments": "experiment_assignments.csv",
        "did_outlet_policy": "did_outlet_policy.csv",
        "product_sales": "product_sales.csv",
    }
    missing = [filename for filename in filenames.values() if not (raw_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required raw CSV file(s) in {raw_dir}: {', '.join(missing)}"
        )
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in filenames.items():
        tables[key] = pd.read_csv(raw_dir / filename, low_memory=False)
    return tables


def _write_processed_tables(
    processed: Mapping[str, pd.DataFrame],
    processed_dir: Path,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    stage_parent = processed_dir
    existing = [filename for filename in OUTPUT_FILES if (processed_dir / filename).exists()]
    if existing:
        if len(existing) != len(OUTPUT_FILES):
            raise FileExistsError(
                "Refusing to overwrite partial processed outputs: "
                + ", ".join(existing)
            )
        with tempfile.TemporaryDirectory(
            prefix=".prepare_data_check_", dir=stage_parent
        ) as temp_name:
            stage = Path(temp_name)
            for filename in OUTPUT_FILES:
                processed[filename.removesuffix(".csv")].to_csv(
                    stage / filename, index=False, lineterminator="\n"
                )
                if _sha256(stage / filename) != _sha256(processed_dir / filename):
                    raise FileExistsError(
                        f"Refusing to overwrite differing processed output: {filename}"
                    )
        print("Processed outputs already match current raw data; no files changed.")
        return

    committed: list[Path] = []
    with tempfile.TemporaryDirectory(
        prefix=".prepare_data_stage_", dir=stage_parent
    ) as temp_name:
        stage = Path(temp_name)
        for filename in OUTPUT_FILES:
            processed[filename.removesuffix(".csv")].to_csv(
                stage / filename, index=False, lineterminator="\n"
            )
        try:
            for filename in OUTPUT_FILES:
                target = processed_dir / filename
                if target.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite processed output: {target}"
                    )
                (stage / filename).rename(target)
                committed.append(target)
        except Exception:
            for target in reversed(committed):
                if target.exists():
                    target.unlink()
            raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def prepare_data(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, pd.DataFrame]:
    """Build and safely persist reusable analysis datasets."""
    tables = load_raw_tables(raw_dir)
    processed = build_processed_tables(tables)
    _write_processed_tables(processed, processed_dir)
    for name, frame in processed.items():
        print(f"{name}.csv: {len(frame):,} rows, {len(frame.columns)} columns")
    return processed


if __name__ == "__main__":
    prepare_data()
