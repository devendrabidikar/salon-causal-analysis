from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 42
APPOINTMENT_COUNT = 300_000
CLIENT_COUNT = 24_000
CHUNK_SIZE = 25_000
HISTORY_START = pd.Timestamp("2025-11-01")
HISTORY_END = pd.Timestamp("2026-10-31")
EXPERIMENT_START = pd.Timestamp("2026-09-01")
EXPERIMENT_END = pd.Timestamp("2026-10-31")
ASSIGNMENT_DATE = pd.Timestamp("2026-08-15")
POLICY_START = pd.Timestamp("2026-07-01")
PRE_HISTORY_START = pd.Timestamp("2024-11-01")
PRODUCT_REVENUE_ROUNDING_TOLERANCE = 0.01

OUTPUT_FILES = (
    "appointments.csv",
    "clients.csv",
    "outlets.csv",
    "stylists.csv",
    "services.csv",
    "products.csv",
    "product_sales.csv",
    "experiment_assignments.csv",
    "did_outlet_policy.csv",
    "data_dictionary.csv",
)

APPOINTMENT_COLUMNS = (
    "appointment_id",
    "client_id",
    "appointment_date",
    "outlet_id",
    "stylist_id",
    "service_id",
    "appointment_status",
    "appointment_revenue",
    "discount_amount",
    "service_duration_minutes",
    "booking_channel",
    "cancellation_reason",
)
PRODUCT_SALE_COLUMNS = (
    "sale_id",
    "appointment_id",
    "client_id",
    "outlet_id",
    "product_id",
    "sale_date",
    "quantity",
    "unit_price",
    "discount_amount",
    "product_revenue",
)
EXPECTED_COLUMNS = {
    "appointments.csv": APPOINTMENT_COLUMNS,
    "clients.csv": (
        "client_id",
        "signup_date",
        "retention_segment",
        "preferred_outlet_id",
        "simulated_active_until",
    ),
    "outlets.csv": ("outlet_id", "outlet_name", "region", "price_index"),
    "stylists.csv": (
        "stylist_id",
        "outlet_id",
        "stylist_level",
        "simulated_start_date",
    ),
    "services.csv": (
        "service_id",
        "service_name",
        "base_price",
        "base_duration_minutes",
    ),
    "products.csv": ("product_id", "product_name", "product_category", "unit_price"),
    "product_sales.csv": PRODUCT_SALE_COLUMNS,
    "experiment_assignments.csv": (
        "client_id",
        "assignment_group",
        "assigned_at",
        "experiment_start_date",
        "experiment_end_date",
    ),
    "did_outlet_policy.csv": (
        "outlet_id",
        "region",
        "policy_group",
        "is_treated",
        "policy_start_date",
        "policy_name",
    ),
    "data_dictionary.csv": (
        "file_name",
        "column_name",
        "description",
        "synthetic_notes",
    ),
}


def _validate_schema(stage_dir: Path) -> None:
    missing_files = [
        file_name
        for file_name in OUTPUT_FILES
        if not (stage_dir / file_name).is_file()
    ]
    missing_columns: dict[str, list[str]] = {}
    unexpected_columns: dict[str, list[str]] = {}
    column_order_mismatches: dict[str, dict[str, list[str]]] = {}
    for file_name, expected_columns in EXPECTED_COLUMNS.items():
        file_path = stage_dir / file_name
        if not file_path.is_file():
            continue
        actual_columns = list(pd.read_csv(file_path, nrows=0).columns)
        actual_column_set = set(actual_columns)
        expected_column_set = set(expected_columns)
        missing = sorted(expected_column_set - actual_column_set)
        unexpected = sorted(actual_column_set - expected_column_set)
        if missing:
            missing_columns[file_name] = missing
        if unexpected:
            unexpected_columns[file_name] = unexpected
        if actual_columns != list(expected_columns):
            column_order_mismatches[file_name] = {
                "expected": list(expected_columns),
                "actual": actual_columns,
            }

    print("\nCSV SCHEMA VALIDATION")
    print(f"Missing files: {missing_files or 'none'}")
    print(f"Missing columns: {missing_columns or 'none'}")
    print(f"Unexpected columns: {unexpected_columns or 'none'}")
    print(f"Column order mismatches: {column_order_mismatches or 'none'}")
    if (
        missing_files
        or missing_columns
        or unexpected_columns
        or column_order_mismatches
    ):
        raise RuntimeError(
            "CSV schema validation failed; row-level validation was not run."
        )


def _make_dimensions(
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outlet_ids = np.array([f"OUT{i:02d}" for i in range(1, 23)])
    regions = np.array(["North", "South", "East", "West"])
    region_labels = np.repeat(regions, [6, 5, 5, 6])
    rng.shuffle(region_labels)
    treated_regions = {"North", "South"}

    outlets = pd.DataFrame(
        {
            "outlet_id": outlet_ids,
            "outlet_name": [f"Fictional Salon {i:02d}" for i in range(1, 23)],
            "region": region_labels,
            "price_index": np.round(rng.uniform(0.92, 1.10, size=len(outlet_ids)), 3),
        }
    )

    stylist_rows: list[dict[str, Any]] = []
    stylist_number = 1
    for outlet_id in outlet_ids:
        for _ in range(8):
            stylist_rows.append(
                {
                    "stylist_id": f"STY{stylist_number:04d}",
                    "outlet_id": outlet_id,
                    "stylist_level": rng.choice(
                        ["Junior", "Experienced", "Senior"],
                        p=[0.25, 0.55, 0.20],
                    ),
                    "simulated_start_date": (
                        HISTORY_START
                        - pd.Timedelta(days=int(rng.integers(30, 1_500)))
                    ).strftime("%Y-%m-%d"),
                }
            )
            stylist_number += 1
    stylists = pd.DataFrame(stylist_rows)

    services = pd.DataFrame(
        {
            "service_id": [f"SRV{i:02d}" for i in range(1, 11)],
            "service_name": [
                "Classic haircut",
                "Styling",
                "Root color",
                "Full color",
                "Highlights",
                "Manicure",
                "Pedicure",
                "Facial",
                "Blowout",
                "Consultation",
            ],
            "base_price": [42.0, 35.0, 78.0, 118.0, 145.0, 32.0, 48.0, 72.0, 38.0, 15.0],
            "base_duration_minutes": [45, 40, 100, 150, 180, 40, 60, 60, 45, 20],
        }
    )

    products = pd.DataFrame(
        {
            "product_id": [f"PRD{i:02d}" for i in range(1, 16)],
            "product_name": [
                "Fictional shampoo",
                "Fictional conditioner",
                "Repair mask",
                "Styling cream",
                "Heat protectant",
                "Finishing spray",
                "Color-care shampoo",
                "Hair oil",
                "Curl lotion",
                "Scalp serum",
                "Hand cream",
                "Cuticle oil",
                "Face cleanser",
                "Moisturizer",
                "Sunscreen",
            ],
            "product_category": [
                "Haircare", "Haircare", "Haircare", "Styling", "Styling",
                "Styling", "Haircare", "Haircare", "Styling", "Haircare",
                "Nailcare", "Nailcare", "Skincare", "Skincare", "Skincare",
            ],
            "unit_price": [
                18.0, 19.0, 26.0, 22.0, 21.0, 17.0, 20.0, 25.0, 23.0,
                29.0, 12.0, 14.0, 20.0, 27.0, 24.0,
            ],
        }
    )

    policy = outlets[["outlet_id", "region"]].copy()
    policy["policy_group"] = np.where(
        policy["region"].isin(treated_regions), "Treated", "Control"
    )
    policy["is_treated"] = policy["policy_group"].eq("Treated").astype("int8")
    policy["policy_start_date"] = POLICY_START.strftime("%Y-%m-%d")
    policy["policy_name"] = "Fictional service-pricing policy"

    return outlets, stylists, services, products, policy


def _make_clients(
    rng: np.random.Generator,
    outlet_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    client_ids = np.array([f"CL{i:06d}" for i in range(1, CLIENT_COUNT + 1)])
    pre_period_signup = np.zeros(CLIENT_COUNT, dtype=bool)
    pre_period_signup[
        rng.choice(CLIENT_COUNT, size=int(CLIENT_COUNT * 0.70), replace=False)
    ] = True

    history_start_ordinal = HISTORY_START.to_datetime64().astype("datetime64[D]").astype(int)
    history_end_ordinal = HISTORY_END.to_datetime64().astype("datetime64[D]").astype(int)
    assignment_ordinal = ASSIGNMENT_DATE.to_datetime64().astype("datetime64[D]").astype(int)
    pre_period_start_ordinal = (
        pd.Timestamp("2024-11-01").to_datetime64().astype("datetime64[D]").astype(int)
    )
    signup_ordinals = np.empty(CLIENT_COUNT, dtype=np.int64)
    signup_ordinals[pre_period_signup] = rng.integers(
        pre_period_start_ordinal,
        history_start_ordinal,
        size=int(pre_period_signup.sum()),
    )
    signup_ordinals[~pre_period_signup] = rng.integers(
        history_start_ordinal,
        assignment_ordinal + 1,
        size=int((~pre_period_signup).sum()),
    )

    segment_names = np.array(["Loyal", "Regular", "Occasional", "At-risk"])
    segment_codes = rng.choice(
        len(segment_names), size=CLIENT_COUNT, p=[0.20, 0.40, 0.25, 0.15]
    )
    retention_segments = segment_names[segment_codes]
    active_months = np.array([12, 10, 7, 4], dtype=np.int64)[segment_codes]
    active_end_ordinals = np.minimum(
        signup_ordinals + active_months * 30, history_end_ordinal
    )
    active_start_ordinals = np.maximum(signup_ordinals, history_start_ordinal)
    available = active_end_ordinals >= active_start_ordinals

    frequency_weights = np.array([2.4, 1.3, 0.7, 0.45])[segment_codes]
    frequency_weights = frequency_weights * available
    client_probabilities = frequency_weights / frequency_weights.sum()
    preferred_outlet_ids = rng.choice(outlet_ids, size=CLIENT_COUNT)

    clients = pd.DataFrame(
        {
            "client_id": client_ids,
            "signup_date": pd.to_datetime(signup_ordinals, unit="D").strftime("%Y-%m-%d"),
            "retention_segment": retention_segments,
            "preferred_outlet_id": preferred_outlet_ids,
            "simulated_active_until": pd.to_datetime(active_end_ordinals, unit="D").strftime(
                "%Y-%m-%d"
            ),
        }
    )
    return clients, client_probabilities, signup_ordinals, active_end_ordinals


def _make_experiment_assignments(
    rng: np.random.Generator,
    clients: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    assignment_by_client = np.zeros(CLIENT_COUNT, dtype=bool)
    randomized_order = rng.permutation(CLIENT_COUNT)
    assignment_by_client[randomized_order[: CLIENT_COUNT // 2]] = True
    assignments = pd.DataFrame(
        {
            "client_id": clients["client_id"].to_numpy()[randomized_order],
            "assignment_group": np.where(
                assignment_by_client[randomized_order], "Treatment", "Control"
            ),
            "assigned_at": ASSIGNMENT_DATE.strftime("%Y-%m-%d"),
            "experiment_start_date": EXPERIMENT_START.strftime("%Y-%m-%d"),
            "experiment_end_date": EXPERIMENT_END.strftime("%Y-%m-%d"),
        }
    )
    return assignments, assignment_by_client


def _make_data_dictionary() -> pd.DataFrame:
    definitions: dict[str, dict[str, str]] = {
        "appointments.csv": {
            "appointment_id": "Unique synthetic appointment identifier.",
            "client_id": "Synthetic client ID; blank for walk-ins.",
            "appointment_date": "Appointment date within the historical period.",
            "outlet_id": "Fictional salon outlet where the appointment occurred.",
            "stylist_id": "Synthetic stylist assigned to the same outlet.",
            "service_id": "Service delivered or booked.",
            "appointment_status": "Completed, Cancelled, or No-show.",
            "appointment_revenue": "Net service revenue after discount; zero if not completed.",
            "discount_amount": "Synthetic discount applied to the service; zero if not completed.",
            "service_duration_minutes": "Simulated service duration in minutes.",
            "booking_channel": "Synthetic booking channel, including Walk-in.",
            "cancellation_reason": "Synthetic reason for cancelled/no-show appointments; otherwise blank.",
        },
        "clients.csv": {
            "client_id": "Unique synthetic client identifier; not linked to a real person.",
            "signup_date": "Synthetic client signup date.",
            "retention_segment": "Simulated engagement segment used to shape visit frequency.",
            "preferred_outlet_id": "Synthetic preferred outlet.",
            "simulated_active_until": "Simulated end of the client's active period, capped at the dataset end.",
        },
        "outlets.csv": {
            "outlet_id": "Unique fictional outlet identifier.",
            "outlet_name": "Fictional outlet name.",
            "region": "Synthetic region used for the policy comparison.",
            "price_index": "Synthetic outlet-specific service price multiplier.",
        },
        "stylists.csv": {
            "stylist_id": "Unique synthetic stylist identifier.",
            "outlet_id": "Fictional outlet employing the stylist.",
            "stylist_level": "Synthetic experience level.",
            "simulated_start_date": "Synthetic stylist start date.",
        },
        "services.csv": {
            "service_id": "Unique synthetic service identifier.",
            "service_name": "Fictional service label.",
            "base_price": "Synthetic baseline service price before adjustments.",
            "base_duration_minutes": "Synthetic baseline service duration.",
        },
        "products.csv": {
            "product_id": "Unique synthetic product identifier.",
            "product_name": "Fictional product label.",
            "product_category": "Synthetic product category.",
            "unit_price": "Synthetic product unit price before discount.",
        },
        "product_sales.csv": {
            "sale_id": "Unique synthetic product-sale line identifier.",
            "appointment_id": "Completed appointment associated with the sale.",
            "client_id": "Synthetic client ID; blank for walk-in appointments.",
            "outlet_id": "Outlet where the sale occurred.",
            "product_id": "Product sold.",
            "sale_date": "Sale date, matching the linked appointment date.",
            "quantity": "Synthetic number of units sold.",
            "unit_price": "Synthetic product unit price.",
            "discount_amount": "Synthetic product discount amount.",
            "product_revenue": "Net synthetic product revenue.",
        },
        "experiment_assignments.csv": {
            "client_id": "Synthetic client assigned exactly once.",
            "assignment_group": "Balanced client-level randomized Control/Treatment assignment.",
            "assigned_at": "Synthetic assignment timestamp represented as a date.",
            "experiment_start_date": "Start of the artificial experiment effect period.",
            "experiment_end_date": "End of the artificial experiment effect period.",
        },
        "did_outlet_policy.csv": {
            "outlet_id": "Fictional outlet included in the policy comparison.",
            "region": "Synthetic treated or control region.",
            "policy_group": "Treated for North/South regions; Control for East/West.",
            "is_treated": "1 for policy-treated outlets and 0 for control outlets.",
            "policy_start_date": "Common synthetic policy start date.",
            "policy_name": "Fictional policy label.",
        },
        "data_dictionary.csv": {
            "file_name": "Name of the CSV dataset described.",
            "column_name": "Column name in the described dataset.",
            "description": "Meaning of the field.",
            "synthetic_notes": "Synthetic-data context for the described dataset.",
        },
    }

    notes = {
        "appointments.csv": (
            "All records are simulated. Revenue effects are artificial injected multipliers."
        ),
        "clients.csv": "Synthetic identifiers and attributes only; no real customer data.",
        "outlets.csv": "All outlet names, regions, and price indices are fictional.",
        "stylists.csv": "Synthetic stylist identifiers and attributes only.",
        "services.csv": "Fictional services and simulated prices/durations.",
        "products.csv": "Fictional products and simulated prices.",
        "product_sales.csv": "Simulated sales linked to completed synthetic appointments.",
        "experiment_assignments.csv": (
            "Artificial 1:1 randomized assignment; treatment effect is injected for testing."
        ),
        "did_outlet_policy.csv": (
            "Artificial treated/control policy assignment; post-policy effect is injected."
        ),
        "data_dictionary.csv": "Synthetic field definitions for the generated CSV files.",
    }
    rows = [
        {
            "file_name": file_name,
            "column_name": column_name,
            "description": description,
            "synthetic_notes": notes[file_name],
        }
        for file_name, fields in definitions.items()
        for column_name, description in fields.items()
    ]
    return pd.DataFrame(rows)


def _write_static_tables(
    stage_dir: Path,
    tables: dict[str, pd.DataFrame],
) -> None:
    for file_name, frame in tables.items():
        frame.to_csv(stage_dir / file_name, index=False)


def _validate_generated_data(
    stage_dir: Path,
    clients: pd.DataFrame,
    outlets: pd.DataFrame,
    stylists: pd.DataFrame,
    services: pd.DataFrame,
    products: pd.DataFrame,
) -> None:
    _validate_schema(stage_dir)

    checks: list[tuple[str, int, str]] = []

    def add_check(name: str, invalid_records: int, detail: str) -> None:
        checks.append((name, int(invalid_records), detail))

    null_counts: dict[str, dict[str, int]] = {}
    allowed_nulls = {
        "appointments.csv": {"client_id", "cancellation_reason"},
        "product_sales.csv": {"client_id"},
    }
    required_id_columns = {
        "appointments.csv": (
            "appointment_id",
            "outlet_id",
            "stylist_id",
            "service_id",
        ),
        "clients.csv": ("client_id", "preferred_outlet_id"),
        "outlets.csv": ("outlet_id",),
        "stylists.csv": ("stylist_id", "outlet_id"),
        "services.csv": ("service_id",),
        "products.csv": ("product_id",),
        "product_sales.csv": (
            "sale_id",
            "appointment_id",
            "outlet_id",
            "product_id",
        ),
        "experiment_assignments.csv": ("client_id",),
        "did_outlet_policy.csv": ("outlet_id",),
    }
    client_ids = set(clients["client_id"])
    outlet_ids = set(outlets["outlet_id"])
    stylist_outlets = dict(zip(stylists["stylist_id"], stylists["outlet_id"]))
    service_ids = set(services["service_id"])
    product_ids = set(products["product_id"])
    seen_appointment_ids: set[str] = set()
    appointment_lookup: dict[str, tuple[str, str, str, str]] = {}
    appointment_rows = 0
    duplicate_appointment_ids = 0
    appointment_min: pd.Timestamp | None = None
    appointment_max: pd.Timestamp | None = None
    appointment_invalid_counts = {
        "date": 0,
        "client_fk": 0,
        "outlet_fk": 0,
        "stylist_fk_or_outlet": 0,
        "service_fk": 0,
        "status": 0,
        "walk_in_identity": 0,
        "duration": 0,
        "revenue_or_discount": 0,
        "cancellation_reason": 0,
    }

    def normalize_nullable(value: Any) -> str:
        return "" if pd.isna(value) else str(value)

    def empty_id_mask(values: pd.Series) -> pd.Series:
        return values.isna() | values.astype("string").str.strip().eq("").fillna(True)

    empty_id_counts = {file_name: 0 for file_name in required_id_columns}
    client_dimension_invalid = (
        abs(len(clients) - CLIENT_COUNT)
        + int(clients["client_id"].duplicated(keep=False).sum())
    )
    add_check(
        "Client dimension size and uniqueness",
        client_dimension_invalid,
        f"{len(clients):,} rows; {len(client_ids):,} unique clients; expected {CLIENT_COUNT:,}",
    )
    add_check(
        "Stylist outlet references",
        int((~stylists["outlet_id"].isin(outlet_ids)).sum()),
        "stylist rows referencing unknown outlets",
    )

    for file_name in OUTPUT_FILES:
        counts: dict[str, int] = {}
        file_path = stage_dir / file_name
        for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
            for column, count in chunk.isna().sum().items():
                counts[column] = counts.get(column, 0) + int(count)
            for column in required_id_columns.get(file_name, ()):
                empty_id_counts[file_name] += int(empty_id_mask(chunk[column]).sum())
            if file_name != "appointments.csv":
                continue

            appointment_rows += len(chunk)
            ids = chunk["appointment_id"].fillna("").astype(str)
            for appointment_id in ids:
                if appointment_id in seen_appointment_ids:
                    duplicate_appointment_ids += 1
                seen_appointment_ids.add(appointment_id)

            dates = pd.to_datetime(chunk["appointment_date"], errors="coerce")
            valid_dates = dates.dropna()
            if not valid_dates.empty:
                chunk_min, chunk_max = valid_dates.min(), valid_dates.max()
                appointment_min = chunk_min if appointment_min is None else min(appointment_min, chunk_min)
                appointment_max = chunk_max if appointment_max is None else max(appointment_max, chunk_max)
            appointment_invalid_counts["date"] += int(
                (dates.isna() | dates.lt(HISTORY_START) | dates.gt(HISTORY_END)).sum()
            )
            client_missing = chunk["client_id"].isna()
            appointment_invalid_counts["client_fk"] += int(
                (~client_missing & ~chunk["client_id"].isin(client_ids)).sum()
            )
            appointment_invalid_counts["outlet_fk"] += int(
                (~chunk["outlet_id"].isin(outlet_ids)).sum()
            )
            is_walk_in = chunk["booking_channel"].eq("Walk-in")
            stylist_outlet_for_rows = chunk["stylist_id"].map(stylist_outlets)
            appointment_invalid_counts["stylist_fk_or_outlet"] += int(
                (
                    stylist_outlet_for_rows.isna()
                    | stylist_outlet_for_rows.ne(chunk["outlet_id"])
                ).sum()
            )
            appointment_invalid_counts["service_fk"] += int(
                (~chunk["service_id"].isin(service_ids)).sum()
            )
            status = chunk["appointment_status"]
            valid_status = status.isin(["Completed", "Cancelled", "No-show"])
            appointment_invalid_counts["status"] += int((~valid_status).sum())
            appointment_invalid_counts["walk_in_identity"] += int(
                (is_walk_in.ne(client_missing)).sum()
            )
            appointment_invalid_counts["client_fk"] += int(
                (client_missing & ~is_walk_in).sum()
            )
            duration = pd.to_numeric(
                chunk["service_duration_minutes"], errors="coerce"
            )
            appointment_invalid_counts["duration"] += int(
                (duration.isna() | duration.le(0)).sum()
            )
            revenue = pd.to_numeric(chunk["appointment_revenue"], errors="coerce")
            discount = pd.to_numeric(chunk["discount_amount"], errors="coerce")
            completed = status.eq("Completed")
            appointment_invalid_counts["revenue_or_discount"] += int(
                (
                    revenue.isna()
                    | discount.isna()
                    | revenue.lt(0)
                    | discount.lt(0)
                    | (~completed & (revenue.ne(0) | discount.ne(0)))
                ).sum()
            )
            cancellation_missing = chunk["cancellation_reason"].isna()
            appointment_invalid_counts["cancellation_reason"] += int(
                (completed & ~cancellation_missing).sum()
                + (~completed & cancellation_missing).sum()
            )

            normalized_dates = dates.dt.strftime("%Y-%m-%d").fillna("")
            for appointment_id, row_status, client_id, outlet_id, date in zip(
                ids,
                status,
                chunk["client_id"],
                chunk["outlet_id"],
                normalized_dates,
            ):
                if appointment_id not in appointment_lookup:
                    appointment_lookup[appointment_id] = (
                        str(row_status),
                        normalize_nullable(client_id),
                        normalize_nullable(outlet_id),
                        str(date),
                    )

        null_counts[file_name] = counts

    for file_name, invalid_ids in empty_id_counts.items():
        add_check(
            f"Non-empty IDs in {file_name}",
            invalid_ids,
            f"{invalid_ids:,} required ID cells are null or blank",
        )

    add_check(
        "Appointment row count",
        abs(appointment_rows - APPOINTMENT_COUNT),
        f"found {appointment_rows:,}; expected {APPOINTMENT_COUNT:,}",
    )
    add_check(
        "Appointment ID uniqueness",
        duplicate_appointment_ids,
        f"{duplicate_appointment_ids:,} duplicate appointment rows",
    )
    appointment_check_details = {
        "Appointment date range": "invalid or out-of-period appointment dates",
        "Appointment client references": "missing non-walk-in or unknown client IDs",
        "Appointment outlet references": "unknown outlet IDs",
        "Stylist references and outlet ownership": "unknown stylists or outlet mismatches",
        "Appointment service references": "unknown service IDs",
        "Appointment statuses": "values outside Completed, Cancelled, No-show",
        "Walk-in/client consistency": "Walk-in channel and blank client ID do not agree",
        "Appointment duration": "missing or non-positive service durations",
        "Appointment revenue and discounts": (
            "missing/negative values or nonzero amounts for non-completed visits"
        ),
        "Cancellation reasons": (
            "completed appointments with a reason or non-completed appointments without one"
        ),
    }
    for key, name in zip(appointment_invalid_counts, appointment_check_details):
        add_check(
            name,
            appointment_invalid_counts[key],
            appointment_check_details[name],
        )

    assignments = pd.read_csv(stage_dir / "experiment_assignments.csv")
    assignment_duplicates = int(assignments["client_id"].duplicated(keep=False).sum())
    assigned_client_ids = set(assignments["client_id"].dropna())
    missing_clients = client_ids - assigned_client_ids
    unknown_clients = assigned_client_ids - client_ids
    assignment_invalid = (
        assignment_duplicates
        + len(missing_clients)
        + len(unknown_clients)
        + int(assignments["client_id"].isna().sum())
    )
    add_check(
        "Experiment client assignment coverage",
        assignment_invalid,
        (
            f"{len(assignments):,} rows; {len(assigned_client_ids):,} unique known clients; "
            f"{len(missing_clients):,} missing, {len(unknown_clients):,} unknown, "
            f"{assignment_duplicates:,} duplicate rows"
        ),
    )
    group_counts = assignments["assignment_group"].value_counts(dropna=False)
    control_count = int(group_counts.get("Control", 0))
    treatment_count = int(group_counts.get("Treatment", 0))
    invalid_groups = int((~assignments["assignment_group"].isin(["Control", "Treatment"])).sum())
    add_check(
        "Experiment assignment groups",
        invalid_groups,
        "assignment groups must be Control or Treatment",
    )
    group_imbalance = (
        abs(control_count - CLIENT_COUNT // 2)
        + abs(treatment_count - CLIENT_COUNT // 2)
    )
    add_check(
        "Experiment 1:1 balance",
        group_imbalance,
        (
            f"Control={control_count:,}, Treatment={treatment_count:,}; "
            "expected 12,000 in each group"
        ),
    )

    assignment_dates = pd.to_datetime(assignments["assigned_at"], errors="coerce")
    experiment_start_dates = pd.to_datetime(
        assignments["experiment_start_date"], errors="coerce"
    )
    experiment_end_dates = pd.to_datetime(
        assignments["experiment_end_date"], errors="coerce"
    )
    experiment_metadata_invalid = int(
        (assignment_dates.ne(ASSIGNMENT_DATE)).sum()
        + (experiment_start_dates.ne(EXPERIMENT_START)).sum()
        + (experiment_end_dates.ne(EXPERIMENT_END)).sum()
    )
    add_check(
        "Experiment assignment metadata dates",
        experiment_metadata_invalid,
        (
            "assigned_at must be 2026-08-15, experiment_start_date "
            "2026-09-01, and experiment_end_date 2026-10-31"
        ),
    )
    signup_dates_by_client = clients.drop_duplicates("client_id").set_index("client_id")[
        "signup_date"
    ]
    assigned_signup_dates = pd.to_datetime(
        assignments["client_id"].map(signup_dates_by_client), errors="coerce"
    )
    assignment_eligibility_invalid = int(
        (
            assigned_signup_dates.isna()
            | assignment_dates.isna()
            | assigned_signup_dates.gt(assignment_dates)
            | assignment_dates.ne(ASSIGNMENT_DATE)
            | assignment_dates.ge(EXPERIMENT_START)
        ).sum()
    )
    client_signup_dates = pd.to_datetime(clients["signup_date"], errors="coerce")
    is_prehistory_signup = client_signup_dates.between(
        PRE_HISTORY_START, HISTORY_START - pd.Timedelta(days=1)
    )
    is_historical_signup = client_signup_dates.between(HISTORY_START, ASSIGNMENT_DATE)
    prehistory_signup_count = int(is_prehistory_signup.sum())
    historical_signup_count = int(is_historical_signup.sum())
    expected_pre_history = int(CLIENT_COUNT * 0.70)
    expected_historical = CLIENT_COUNT - expected_pre_history
    signup_proportion_invalid = (
        abs(prehistory_signup_count - expected_pre_history)
        + abs(historical_signup_count - expected_historical)
        + int((client_signup_dates.isna() | ~(is_prehistory_signup | is_historical_signup)).sum())
    )
    add_check(
        "Experiment assignment timing and signup eligibility",
        assignment_eligibility_invalid,
        (
            f"{assignment_eligibility_invalid:,} ineligible or mistimed records; "
            f"assignment date is {ASSIGNMENT_DATE.date()}"
        ),
    )
    add_check(
        "Client signup-period proportions",
        signup_proportion_invalid,
        (
            f"pre-history={prehistory_signup_count:,} (expected {expected_pre_history:,}); "
            f"historical={historical_signup_count:,} (expected {expected_historical:,}); "
            "pre-history dates must be 2024-11-01 through 2025-10-31 and historical "
            "dates 2025-11-01 through 2026-08-15"
        ),
    )

    policy = pd.read_csv(stage_dir / "did_outlet_policy.csv")
    expected_policy_by_region = {
        "North": ("Treated", 1),
        "South": ("Treated", 1),
        "East": ("Control", 0),
        "West": ("Control", 0),
    }
    expected_outlet_regions = dict(zip(outlets["outlet_id"], outlets["region"]))
    policy_region_mismatch = 0
    for outlet_id, region, group, is_treated in zip(
        policy["outlet_id"],
        policy["region"],
        policy["policy_group"],
        policy["is_treated"],
    ):
        expected = expected_policy_by_region.get(region)
        if (
            expected is None
            or expected_outlet_regions.get(outlet_id) != region
            or (group, is_treated) != expected
        ):
            policy_region_mismatch += 1
    outlet_duplicate_count = int(outlets["outlet_id"].duplicated(keep=False).sum())
    add_check(
        "Outlet dimension size and uniqueness",
        abs(len(outlets) - 22) + outlet_duplicate_count,
        f"{len(outlets):,} rows; {outlets['outlet_id'].nunique():,} unique outlets; expected 22",
    )
    for column, dimension_name, frame in (
        ("stylist_id", "Stylist", stylists),
        ("service_id", "Service", services),
        ("product_id", "Product", products),
    ):
        duplicate_count = int(frame[column].duplicated(keep=False).sum())
        add_check(
            f"{dimension_name} ID uniqueness",
            duplicate_count,
            f"{duplicate_count:,} rows have a duplicated {column}",
        )
    preferred_outlet_invalid = int(
        (~clients["preferred_outlet_id"].isin(outlet_ids)).sum()
    )
    add_check(
        "Client preferred outlet references",
        preferred_outlet_invalid,
        f"{preferred_outlet_invalid:,} client rows reference an unknown outlet",
    )
    add_check(
        "Stylist dimension row count",
        abs(len(stylists) - 176),
        f"{len(stylists):,} rows; expected 176",
    )
    add_check(
        "Service dimension row count",
        abs(len(services) - 10),
        f"{len(services):,} rows; expected 10",
    )
    add_check(
        "Product dimension row count",
        abs(len(products) - 15),
        f"{len(products):,} rows; expected 15",
    )
    valid_retention_segments = {"Loyal", "Regular", "Occasional", "At-risk"}
    add_check(
        "Client retention segments",
        int((~clients["retention_segment"].isin(valid_retention_segments)).sum()),
        "retention segment must be Loyal, Regular, Occasional, or At-risk",
    )
    service_prices = pd.to_numeric(services["base_price"], errors="coerce")
    add_check(
        "Positive service base prices",
        int((service_prices.isna() | service_prices.le(0)).sum()),
        "service base prices must be present and greater than zero",
    )
    product_prices = pd.to_numeric(products["unit_price"], errors="coerce")
    add_check(
        "Positive product prices",
        int((product_prices.isna() | product_prices.le(0)).sum()),
        "product unit prices must be present and greater than zero",
    )
    outlet_price_indices = pd.to_numeric(outlets["price_index"], errors="coerce")
    add_check(
        "Positive outlet price indices",
        int((outlet_price_indices.isna() | outlet_price_indices.le(0)).sum()),
        "outlet price indices must be present and greater than zero",
    )
    valid_stylist_levels = {"Junior", "Experienced", "Senior"}
    add_check(
        "Stylist levels",
        int((~stylists["stylist_level"].isin(valid_stylist_levels)).sum()),
        "stylist level must be Junior, Experienced, or Senior",
    )
    service_durations = pd.to_numeric(
        services["base_duration_minutes"], errors="coerce"
    )
    add_check(
        "Positive service base durations",
        int((service_durations.isna() | service_durations.le(0)).sum()),
        "service base durations must be present and greater than zero",
    )
    policy_duplicates = int(policy["outlet_id"].duplicated(keep=False).sum())
    policy_coverage_invalid = (
        abs(len(policy) - 22)
        + policy_duplicates
        + len(outlet_ids - set(policy["outlet_id"].dropna()))
        + len(set(policy["outlet_id"].dropna()) - outlet_ids)
    )
    add_check(
        "Policy outlet coverage and uniqueness",
        policy_coverage_invalid,
        f"{len(policy):,} rows; {policy['outlet_id'].nunique():,} unique policy outlets",
    )
    add_check(
        "Policy record count",
        abs(len(policy) - 22),
        f"{len(policy):,} records; expected 22",
    )
    policy_groups = policy["policy_group"].value_counts(dropna=False)
    treated_count = int(policy_groups.get("Treated", 0))
    control_count_policy = int(policy_groups.get("Control", 0))
    invalid_policy_groups = int(
        (~policy["policy_group"].isin(["Treated", "Control"])).sum()
    )
    policy_balance_invalid = (
        abs(treated_count - 11)
        + abs(control_count_policy - 11)
        + invalid_policy_groups
    )
    add_check(
        "Policy treated/control balance",
        policy_balance_invalid,
        f"Treated={treated_count:,}, Control={control_count_policy:,}; expected 11 each",
    )
    add_check(
        "Policy region assignment",
        policy_region_mismatch,
        "North/South must be Treated and East/West must be Control",
    )
    policy_treated_values = pd.to_numeric(policy["is_treated"], errors="coerce")
    invalid_treated_values = int(
        (~policy_treated_values.isin([0, 1])).sum()
    )
    add_check(
        "Policy is_treated values",
        invalid_treated_values,
        "is_treated must be 0 or 1",
    )
    expected_treated_values = policy["policy_group"].map(
        {"Treated": 1, "Control": 0}
    )
    policy_group_indicator_mismatch = int(
        (
            expected_treated_values.isna()
            | policy_treated_values.ne(expected_treated_values)
        ).sum()
    )
    add_check(
        "Policy group/is_treated agreement",
        policy_group_indicator_mismatch,
        "Treated must map to 1 and Control must map to 0",
    )
    policy_date_invalid = int(
        policy["policy_start_date"].ne(POLICY_START.strftime("%Y-%m-%d")).sum()
    )
    add_check(
        "Policy start date",
        policy_date_invalid,
        f"expected {POLICY_START.date()}",
    )

    product_sales_rows = 0
    seen_sale_ids: set[str] = set()
    duplicate_sale_ids = 0
    sales_invalid_counts = {
        "appointment_link": 0,
        "client_match": 0,
        "client_fk": 0,
        "outlet_match": 0,
        "date_match": 0,
        "product_fk": 0,
        "quantity": 0,
        "unit_price": 0,
        "revenue_or_discount": 0,
        "revenue_reconciliation": 0,
    }
    for chunk in pd.read_csv(stage_dir / "product_sales.csv", chunksize=CHUNK_SIZE):
        product_sales_rows += len(chunk)
        sale_ids = chunk["sale_id"].fillna("").astype(str)
        for sale_id in sale_ids:
            if sale_id in seen_sale_ids:
                duplicate_sale_ids += 1
            seen_sale_ids.add(sale_id)
        linked_appointments = [
            appointment_lookup.get(str(appointment_id))
            for appointment_id in chunk["appointment_id"].fillna("")
        ]
        sales_invalid_counts["appointment_link"] += sum(
            linked is None or linked[0] != "Completed"
            for linked in linked_appointments
        )
        sales_invalid_counts["client_match"] += sum(
            linked is not None
            and normalize_nullable(client_id) != linked[1]
            for linked, client_id in zip(linked_appointments, chunk["client_id"])
        )
        nonmissing_sale_clients = chunk["client_id"].dropna()
        sales_invalid_counts["client_fk"] += int(
            (~nonmissing_sale_clients.isin(client_ids)).sum()
        )
        sales_invalid_counts["outlet_match"] += sum(
            linked is not None and normalize_nullable(outlet_id) != linked[2]
            for linked, outlet_id in zip(linked_appointments, chunk["outlet_id"])
        )
        sales_invalid_counts["date_match"] += sum(
            linked is not None and normalize_nullable(sale_date) != linked[3]
            for linked, sale_date in zip(linked_appointments, chunk["sale_date"])
        )
        sales_invalid_counts["product_fk"] += int(
            (~chunk["product_id"].isin(product_ids)).sum()
        )
        sale_quantity = pd.to_numeric(chunk["quantity"], errors="coerce")
        sales_invalid_counts["quantity"] += int(
            (sale_quantity.isna() | sale_quantity.le(0)).sum()
        )
        sale_unit_price = pd.to_numeric(chunk["unit_price"], errors="coerce")
        sales_invalid_counts["unit_price"] += int(
            (sale_unit_price.isna() | sale_unit_price.le(0)).sum()
        )
        sale_revenue = pd.to_numeric(chunk["product_revenue"], errors="coerce")
        sale_discount = pd.to_numeric(chunk["discount_amount"], errors="coerce")
        sales_invalid_counts["revenue_or_discount"] += int(
            (
                sale_revenue.isna()
                | sale_discount.isna()
                | sale_revenue.lt(0)
                | sale_discount.lt(0)
            ).sum()
        )
        expected_sale_revenue = sale_quantity * sale_unit_price - sale_discount
        sales_invalid_counts["revenue_reconciliation"] += int(
            (
                sale_revenue.isna()
                | sale_discount.isna()
                | sale_quantity.isna()
                | sale_unit_price.isna()
                | (
                    (sale_revenue - expected_sale_revenue).abs()
                    > PRODUCT_REVENUE_ROUNDING_TOLERANCE
                )
            ).sum()
        )

    add_check(
        "Product sale ID uniqueness",
        duplicate_sale_ids,
        f"{duplicate_sale_ids:,} duplicate sale rows",
    )
    product_sale_check_details = {
        "Product sale appointment link": "appointment missing or not Completed",
        "Product sale client match": "sale client ID differs from linked appointment",
        "Product sale client references": "unknown non-null client IDs",
        "Product sale outlet match": "sale outlet differs from linked appointment",
        "Product sale date match": "sale date differs from linked appointment date",
        "Product sale product reference": "unknown product IDs",
        "Product sale quantity": "missing or non-positive quantities",
        "Product sale unit price": "missing or non-positive unit prices",
        "Product sale revenue and discounts": "missing or negative revenue/discount values",
        "Product sale revenue reconciliation": (
            "revenue differs from quantity x unit price - discount by more than "
            f"{PRODUCT_REVENUE_ROUNDING_TOLERANCE:.2f}"
        ),
    }
    for key, name in zip(sales_invalid_counts, product_sale_check_details):
        add_check(
            name,
            sales_invalid_counts[key],
            product_sale_check_details[name],
        )

    unexpected_missing_values = 0
    for file_name, counts in null_counts.items():
        for column, count in counts.items():
            if count and column not in allowed_nulls.get(file_name, set()):
                unexpected_missing_values += count
    add_check(
        "Unexpected missing values",
        unexpected_missing_values,
        "missing cells in columns that do not permit null values",
    )

    print("\nSYNTHETIC DATA VALIDATION REPORT")
    print(
        "Observed appointment date range: "
        f"{appointment_min.date() if appointment_min is not None else 'none'} to "
        f"{appointment_max.date() if appointment_max is not None else 'none'}"
    )
    print(f"Product sales rows checked: {product_sales_rows:,}")
    print("\nValidation categories:")
    failed_checks = 0
    for name, invalid_records, detail in checks:
        result = "PASS" if invalid_records == 0 else "FAIL"
        failed_checks += invalid_records > 0
        print(f"[{result}] {name}: {invalid_records:,} invalid record(s) - {detail}")
    print("\nMissing values by file:")
    for file_name, counts in null_counts.items():
        nonzero = {column: count for column, count in counts.items() if count}
        print(f"  {file_name}: {nonzero or 'none'}")
    if failed_checks:
        print("Overall validation: FAILED")
        failed_categories = [
            f"{name}: {invalid_records} invalid record(s)"
            for name, invalid_records, _ in checks
            if invalid_records
        ]
        raise RuntimeError(
            "Dataset validation failed:\n- " + "\n- ".join(failed_categories)
        )
    print("Overall validation: PASSED")


def _write_appointments_and_sales(
    stage_dir: Path,
    rng: np.random.Generator,
    clients: pd.DataFrame,
    client_probabilities: np.ndarray,
    signup_ordinals: np.ndarray,
    active_end_ordinals: np.ndarray,
    client_is_treatment: np.ndarray,
    outlets: pd.DataFrame,
    stylists: pd.DataFrame,
    services: pd.DataFrame,
    products: pd.DataFrame,
    policy: pd.DataFrame,
) -> None:
    history_start_ordinal = HISTORY_START.to_datetime64().astype("datetime64[D]").astype(int)
    dates = pd.date_range(HISTORY_START, HISTORY_END, freq="D")
    daily_weights = np.ones(len(dates), dtype=float)
    daily_weights += (dates.dayofweek >= 5) * 0.30
    daily_weights += (dates.month == 12) * 0.12
    daily_weights += (dates.month == 8) * 0.06
    daily_cumulative = np.concatenate(([0.0], np.cumsum(daily_weights)))

    client_start_ordinals = np.maximum(signup_ordinals, history_start_ordinal)
    client_end_ordinals = np.minimum(
        active_end_ordinals,
        HISTORY_END.to_datetime64().astype("datetime64[D]").astype(int),
    )
    outlet_ids = outlets["outlet_id"].to_numpy()
    outlet_indexes = {outlet_id: index for index, outlet_id in enumerate(outlet_ids)}
    outlet_price_indexes = outlets["price_index"].to_numpy(dtype=float)
    preferred_outlet_indexes = np.array(
        [outlet_indexes[outlet_id] for outlet_id in clients["preferred_outlet_id"]]
    )
    treated_outlet_ids = set(policy.loc[policy["is_treated"].eq(1), "outlet_id"])
    outlet_is_treated = np.array([outlet_id in treated_outlet_ids for outlet_id in outlet_ids])

    staff_by_outlet = {
        outlet_id: group.reset_index(drop=True)
        for outlet_id, group in stylists.groupby("outlet_id", sort=False)
    }
    stylist_ids_by_outlet = {
        outlet_id: group["stylist_id"].to_numpy() for outlet_id, group in staff_by_outlet.items()
    }
    stylist_multiplier_by_id = dict(
        zip(
            stylists["stylist_id"],
            stylists["stylist_level"].map(
                {"Junior": 0.94, "Experienced": 1.0, "Senior": 1.07}
            ),
        )
    )
    service_ids = services["service_id"].to_numpy()
    service_prices = services["base_price"].to_numpy(dtype=float)
    service_durations = services["base_duration_minutes"].to_numpy(dtype=int)
    product_ids = products["product_id"].to_numpy()
    product_prices = products["unit_price"].to_numpy(dtype=float)
    sale_probabilities = np.array([1.0] * len(product_ids), dtype=float)
    sale_probabilities /= sale_probabilities.sum()

    appointment_path = stage_dir / "appointments.csv"
    sales_path = stage_dir / "product_sales.csv"
    sales_written = False
    sales_count = 0
    experiment_start_ordinal = EXPERIMENT_START.to_datetime64().astype("datetime64[D]").astype(int)
    experiment_end_ordinal = EXPERIMENT_END.to_datetime64().astype("datetime64[D]").astype(int)
    policy_start_ordinal = POLICY_START.to_datetime64().astype("datetime64[D]").astype(int)

    for batch_start in range(0, APPOINTMENT_COUNT, CHUNK_SIZE):
        batch_end = min(batch_start + CHUNK_SIZE, APPOINTMENT_COUNT)
        row_count = batch_end - batch_start
        client_indexes = rng.choice(
            CLIENT_COUNT, size=row_count, replace=True, p=client_probabilities
        )
        walk_in = rng.random(row_count) < 0.08

        client_ids = clients["client_id"].to_numpy()[client_indexes].astype(object)
        client_ids[walk_in] = None

        start_indexes = (client_start_ordinals[client_indexes] - history_start_ordinal).clip(
            0, len(dates) - 1
        )
        end_indexes = (client_end_ordinals[client_indexes] - history_start_ordinal).clip(
            0, len(dates) - 1
        )
        client_targets = daily_cumulative[start_indexes] + rng.random(row_count) * (
            daily_cumulative[end_indexes + 1] - daily_cumulative[start_indexes]
        )
        date_indexes = np.searchsorted(daily_cumulative, client_targets, side="right") - 1
        walk_in_targets = rng.random(row_count) * daily_cumulative[-1]
        walk_in_date_indexes = (
            np.searchsorted(daily_cumulative, walk_in_targets, side="right") - 1
        )
        date_indexes[walk_in] = walk_in_date_indexes[walk_in]
        appointment_ordinals = history_start_ordinal + date_indexes
        appointment_dates = HISTORY_START + pd.to_timedelta(date_indexes, unit="D")

        outlet_indexes_for_rows = rng.integers(0, len(outlet_ids), size=row_count)
        use_preferred = (~walk_in) & (rng.random(row_count) < 0.78)
        outlet_indexes_for_rows[use_preferred] = preferred_outlet_indexes[
            client_indexes[use_preferred]
        ]
        appointment_outlets = outlet_ids[outlet_indexes_for_rows]

        stylist_ids = np.empty(row_count, dtype=object)
        for outlet_index, outlet_id in enumerate(outlet_ids):
            mask = outlet_indexes_for_rows == outlet_index
            if mask.any():
                stylist_ids[mask] = rng.choice(
                    stylist_ids_by_outlet[outlet_id], size=int(mask.sum())
                )
        stylist_multipliers = pd.Series(stylist_ids).map(stylist_multiplier_by_id).to_numpy(
            dtype=float
        )

        service_indexes = rng.choice(
            len(service_ids),
            size=row_count,
            p=np.array([0.20, 0.12, 0.12, 0.10, 0.08, 0.10, 0.07, 0.08, 0.08, 0.05]),
        )
        appointment_services = service_ids[service_indexes]
        base_prices = service_prices[service_indexes]

        completion_draw = rng.random(row_count)
        completed = completion_draw < np.where(walk_in, 0.96, 0.89)
        cancelled = (~completed) & (
            completion_draw < np.where(walk_in, 0.985, 0.96)
        )
        statuses = np.where(completed, "Completed", np.where(cancelled, "Cancelled", "No-show"))

        treatment = np.zeros(row_count, dtype=bool)
        identified_clients = ~walk_in
        treatment[identified_clients] = client_is_treatment[
            client_indexes[identified_clients]
        ]
        experiment_window = (
            (appointment_ordinals >= experiment_start_ordinal)
            & (appointment_ordinals <= experiment_end_ordinal)
        )
        treatment_multiplier = np.where(
            treatment & experiment_window, 1.06, 1.0
        )
        policy_multiplier = np.where(
            outlet_is_treated[outlet_indexes_for_rows]
            & (appointment_ordinals >= policy_start_ordinal),
            1.04,
            1.0,
        )
        gross_amount = (
            base_prices
            * outlet_price_indexes[outlet_indexes_for_rows]
            * stylist_multipliers
            * rng.normal(1.0, 0.04, size=row_count).clip(0.80, 1.20)
            * treatment_multiplier
            * policy_multiplier
        )
        discount_rates = rng.choice(
            [0.0, 0.05, 0.10, 0.15], size=row_count, p=[0.70, 0.16, 0.10, 0.04]
        )
        discount_amounts = np.where(completed, gross_amount * discount_rates, 0.0)
        revenues = np.where(completed, gross_amount - discount_amounts, 0.0)
        duration_noise = rng.normal(1.0, 0.08, size=row_count).clip(0.75, 1.25)
        durations = np.maximum(
            10, np.rint(service_durations[service_indexes] * duration_noise)
        ).astype(int)

        channels = rng.choice(
            ["Website", "Mobile app", "Phone", "In person"],
            size=row_count,
            p=[0.28, 0.40, 0.22, 0.10],
        ).astype(object)
        channels[walk_in] = "Walk-in"
        reason_values = np.array(
            ["Schedule conflict", "Illness", "Travel", "No response", "Other"]
        )
        reasons = np.full(row_count, None, dtype=object)
        not_completed = ~completed
        reasons[not_completed] = rng.choice(
            reason_values, size=int(not_completed.sum())
        )

        appointment_ids = [
            f"APT{appointment_number:06d}"
            for appointment_number in range(batch_start + 1, batch_end + 1)
        ]
        appointments = pd.DataFrame(
            {
                "appointment_id": appointment_ids,
                "client_id": client_ids,
                "appointment_date": appointment_dates.strftime("%Y-%m-%d"),
                "outlet_id": appointment_outlets,
                "stylist_id": stylist_ids,
                "service_id": appointment_services,
                "appointment_status": statuses,
                "appointment_revenue": np.round(revenues, 2),
                "discount_amount": np.round(discount_amounts, 2),
                "service_duration_minutes": durations,
                "booking_channel": channels,
                "cancellation_reason": reasons,
            },
            columns=APPOINTMENT_COLUMNS,
        )
        appointments.to_csv(
            appointment_path,
            mode="w" if batch_start == 0 else "a",
            header=batch_start == 0,
            index=False,
        )

        sale_mask = completed & (rng.random(row_count) < 0.30)
        sale_indexes = np.flatnonzero(sale_mask)
        sale_count = len(sale_indexes)
        sale_products = rng.choice(
            len(product_ids), size=sale_count, p=sale_probabilities
        )
        quantities = rng.choice([1, 2], size=sale_count, p=[0.88, 0.12])
        sale_unit_prices = product_prices[sale_products]
        sale_discount_rates = rng.choice(
            [0.0, 0.05, 0.10], size=sale_count, p=[0.82, 0.13, 0.05]
        )
        sale_discounts = sale_unit_prices * quantities * sale_discount_rates
        sales = pd.DataFrame(
            {
                "sale_id": [
                    f"SALE{sale_number:07d}"
                    for sale_number in range(sales_count + 1, sales_count + sale_count + 1)
                ],
                "appointment_id": np.asarray(appointment_ids, dtype=object)[sale_indexes],
                "client_id": client_ids[sale_indexes],
                "outlet_id": appointment_outlets[sale_indexes],
                "product_id": product_ids[sale_products],
                "sale_date": appointment_dates[sale_indexes].strftime("%Y-%m-%d"),
                "quantity": quantities,
                "unit_price": sale_unit_prices,
                "discount_amount": np.round(sale_discounts, 2),
                "product_revenue": np.round(
                    sale_unit_prices * quantities - sale_discounts, 2
                ),
            },
            columns=PRODUCT_SALE_COLUMNS,
        )
        sales.to_csv(
            sales_path,
            mode="w" if not sales_written else "a",
            header=not sales_written,
            index=False,
        )
        sales_written = True
        sales_count += sale_count


def _commit_staged_files(stage_dir: Path, output_dir: Path) -> None:
    moved: list[Path] = []
    try:
        for file_name in OUTPUT_FILES:
            target = output_dir / file_name
            if target.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing output: {target}"
                )
            staged = stage_dir / file_name
            staged.rename(target)
            moved.append(target)
    except Exception:
        for target in reversed(moved):
            if target.exists():
                target.rename(stage_dir / target.name)
        raise


def generate_salon_data(output_dir: Path | None = None, seed: int = SEED) -> None:
    """Build, validate, and write all synthetic CSVs without overwriting files."""
    project_root = Path(__file__).resolve().parents[1]
    destination = output_dir or project_root / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)

    existing_files = [name for name in OUTPUT_FILES if (destination / name).exists()]
    if existing_files:
        raise FileExistsError(
            "Refusing to overwrite existing dataset files: "
            + ", ".join(str(destination / name) for name in existing_files)
        )

    rng = np.random.default_rng(seed)
    outlets, stylists, services, products, policy = _make_dimensions(rng)
    clients, client_probabilities, signup_ordinals, active_end_ordinals = _make_clients(
        rng, outlets["outlet_id"].to_numpy()
    )
    assignments, client_is_treatment = _make_experiment_assignments(rng, clients)
    tables = {
        "clients.csv": clients,
        "outlets.csv": outlets,
        "stylists.csv": stylists,
        "services.csv": services,
        "products.csv": products,
        "experiment_assignments.csv": assignments,
        "did_outlet_policy.csv": policy,
        "data_dictionary.csv": _make_data_dictionary(),
    }

    with tempfile.TemporaryDirectory(
        prefix=".salon_data_build_", dir=destination
    ) as temporary_directory:
        stage_dir = Path(temporary_directory)
        _write_static_tables(stage_dir, tables)
        _write_appointments_and_sales(
            stage_dir,
            rng,
            clients,
            client_probabilities,
            signup_ordinals,
            active_end_ordinals,
            client_is_treatment,
            outlets,
            stylists,
            services,
            products,
            policy,
        )
        _validate_generated_data(
            stage_dir, clients, outlets, stylists, services, products
        )
        _commit_staged_files(stage_dir, destination)

    print(f"\nSynthetic CSV files created in: {destination}")
    print("No real customer or business data was used.")


if __name__ == "__main__":
    generate_salon_data()
