from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.prepare_data import (
    CLIENT_COLUMNS,
    CLIENT_OUTPUT,
    OUTLET_MONTH_COLUMNS,
    OUTLET_MONTH_OUTPUT,
    build_processed_tables,
    prepare_data,
)


@pytest.fixture
def source_tables():
    outlets = pd.DataFrame(
        [
            {"outlet_id": "O1", "outlet_name": "North outlet", "region": "North"},
            {"outlet_id": "O2", "outlet_name": "South outlet", "region": "South"},
            {"outlet_id": "O3", "outlet_name": "East outlet", "region": "East"},
            {"outlet_id": "O4", "outlet_name": "West outlet", "region": "West"},
        ]
    )
    clients = pd.DataFrame(
        [
            {
                "client_id": f"C{i}",
                "signup_date": "2025-01-01",
                "retention_segment": "Regular",
                "preferred_outlet_id": f"O{i}",
                "simulated_active_until": "2026-10-31",
            }
            for i in range(1, 5)
        ]
    )
    assignments = pd.DataFrame(
        [
            {
                "client_id": f"C{i}",
                "assignment_group": group,
                "assigned_at": "2026-08-15",
                "experiment_start_date": "2026-09-01",
                "experiment_end_date": "2026-10-31",
            }
            for i, group in enumerate(
                ["Treatment", "Control", "Treatment", "Control"], start=1
            )
        ]
    )
    policy = pd.DataFrame(
        [
            {
                "outlet_id": f"O{i}",
                "region": region,
                "policy_group": "Treated" if treated else "Control",
                "is_treated": int(treated),
                "policy_start_date": "2026-07-01",
            }
            for i, (region, treated) in enumerate(
                [("North", True), ("South", True), ("East", False), ("West", False)],
                start=1,
            )
        ]
    )
    appointments = pd.DataFrame(
        [
            {
                "appointment_id": "A1", "client_id": "C1",
                "appointment_date": "2026-05-10", "outlet_id": "O1",
                "appointment_status": "Completed", "appointment_revenue": 100.0,
                "discount_amount": 10.0,
            },
            {
                "appointment_id": "A2", "client_id": "C1",
                "appointment_date": "2026-09-10", "outlet_id": "O1",
                "appointment_status": "Completed", "appointment_revenue": 120.0,
                "discount_amount": 5.0,
            },
            {
                "appointment_id": "A3", "client_id": "C2",
                "appointment_date": "2026-09-11", "outlet_id": "O2",
                "appointment_status": "Cancelled", "appointment_revenue": 0.0,
                "discount_amount": 0.0,
            },
            {
                "appointment_id": "A4", "client_id": "C3",
                "appointment_date": "2026-09-12", "outlet_id": "O3",
                "appointment_status": "No-show", "appointment_revenue": 0.0,
                "discount_amount": 0.0,
            },
            {
                "appointment_id": "A5", "client_id": None,
                "appointment_date": "2025-11-03", "outlet_id": "O4",
                "appointment_status": "Completed", "appointment_revenue": 40.0,
                "discount_amount": 0.0,
            },
        ]
    )
    sales = pd.DataFrame(
        [
            {
                "sale_id": "S1", "appointment_id": "A1", "client_id": "C1",
                "outlet_id": "O1", "product_id": "P1", "sale_date": "2026-05-10",
                "quantity": 2, "unit_price": 20.0, "discount_amount": 5.0,
                "product_revenue": 35.0,
            },
            {
                "sale_id": "S2", "appointment_id": "A2", "client_id": "C1",
                "outlet_id": "O1", "product_id": "P1", "sale_date": "2026-09-10",
                "quantity": 1, "unit_price": 20.0, "discount_amount": 0.0,
                "product_revenue": 20.0,
            },
            {
                "sale_id": "S3", "appointment_id": "A5", "client_id": None,
                "outlet_id": "O4", "product_id": "P1", "sale_date": "2025-11-03",
                "quantity": 1, "unit_price": 10.0, "discount_amount": 0.0,
                "product_revenue": 10.0,
            },
        ]
    )
    return {
        "appointments": appointments,
        "clients": clients,
        "outlets": outlets,
        "experiment_assignments": assignments,
        "did_outlet_policy": policy,
        "product_sales": sales,
    }


def test_processed_tables_have_expected_schema_row_counts_and_zero_fill(source_tables):
    outputs = build_processed_tables(source_tables)
    clients = outputs[CLIENT_OUTPUT.removesuffix(".csv")]
    panel = outputs[OUTLET_MONTH_OUTPUT.removesuffix(".csv")]

    assert list(clients.columns) == list(CLIENT_COLUMNS)
    assert len(clients) == 4
    assert clients["client_id"].is_unique
    assert clients["assignment_group"].value_counts().to_dict() == {
        "Treatment": 2,
        "Control": 2,
    }
    assert clients.loc[clients["client_id"].eq("C1"), "experiment_service_revenue"].item() == 120
    assert clients.loc[clients["client_id"].eq("C1"), "planning_baseline_service_revenue"].item() == 100
    assert clients.loc[clients["client_id"].eq("C1"), "planning_baseline_product_revenue"].item() == 35
    assert clients.loc[clients["client_id"].eq("C1"), "experiment_product_revenue"].item() == 20
    zero_client = clients.loc[clients["client_id"].eq("C4")].iloc[0]
    assert zero_client["experiment_service_revenue"] == 0
    assert zero_client["experiment_appointment_count"] == 0
    assert zero_client["no_show_share_per_client"] == 0

    assert list(panel.columns) == list(OUTLET_MONTH_COLUMNS)
    assert len(panel) == 4 * 12
    assert not panel.duplicated(["outlet_id", "month"]).any()
    assert panel.groupby("outlet_id")["month"].nunique().eq(12).all()
    assert panel["appointment_count"].sum() == len(source_tables["appointments"])
    assert panel["service_revenue"].sum() == pytest.approx(
        source_tables["appointments"]["appointment_revenue"].sum()
    )
    assert panel["product_revenue"].sum() == pytest.approx(
        source_tables["product_sales"]["product_revenue"].sum()
    )
    walk_in_month = panel.loc[
        panel["outlet_id"].eq("O4") & panel["month_label"].eq("2025-11")
    ].iloc[0]
    assert walk_in_month["appointment_count"] == 1
    assert walk_in_month["product_revenue"] == 10


def _write_raw_fixture(raw_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    raw_dir.mkdir(parents=True)
    filenames = {
        "appointments": "appointments.csv",
        "clients": "clients.csv",
        "outlets": "outlets.csv",
        "experiment_assignments": "experiment_assignments.csv",
        "did_outlet_policy": "did_outlet_policy.csv",
        "product_sales": "product_sales.csv",
    }
    for key, filename in filenames.items():
        tables[key].to_csv(raw_dir / filename, index=False)


def test_prepare_data_is_idempotent_and_refuses_changed_output_overwrite(
    tmp_path, source_tables
):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_raw_fixture(raw_dir, source_tables)

    first = prepare_data(raw_dir, processed_dir)
    first_bytes = {
        name: (processed_dir / name).read_bytes()
        for name in (CLIENT_OUTPUT, OUTLET_MONTH_OUTPUT)
    }
    second = prepare_data(raw_dir, processed_dir)

    assert (processed_dir / CLIENT_OUTPUT).is_file()
    assert (processed_dir / OUTLET_MONTH_OUTPUT).is_file()
    assert first[CLIENT_OUTPUT.removesuffix(".csv")].equals(
        second[CLIENT_OUTPUT.removesuffix(".csv")]
    )
    assert first[OUTLET_MONTH_OUTPUT.removesuffix(".csv")].equals(
        second[OUTLET_MONTH_OUTPUT.removesuffix(".csv")]
    )
    assert first_bytes == {
        name: (processed_dir / name).read_bytes()
        for name in (CLIENT_OUTPUT, OUTLET_MONTH_OUTPUT)
    }

    appointments_path = raw_dir / "appointments.csv"
    changed = pd.read_csv(appointments_path)
    changed.loc[changed["appointment_id"].eq("A2"), "appointment_revenue"] = 121.0
    changed.to_csv(appointments_path, index=False)
    client_output_before = (processed_dir / CLIENT_OUTPUT).read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite differing"):
        prepare_data(raw_dir, processed_dir)

    assert (processed_dir / CLIENT_OUTPUT).read_bytes() == client_output_before


def test_processed_schema_constants_are_unique():
    assert len(CLIENT_COLUMNS) == len(set(CLIENT_COLUMNS))
    assert len(OUTLET_MONTH_COLUMNS) == len(set(OUTLET_MONTH_COLUMNS))


def test_multiple_product_sale_rows_can_link_to_one_completed_appointment(source_tables):
    extra_sale = source_tables["product_sales"].loc[
        source_tables["product_sales"]["sale_id"].eq("S2")
    ].copy()
    extra_sale.loc[:, "sale_id"] = "S4"
    extra_sale.loc[:, "product_id"] = "P2"
    extra_sale.loc[:, "quantity"] = 1
    extra_sale.loc[:, "unit_price"] = 5.0
    extra_sale.loc[:, "product_revenue"] = 5.0
    source_tables["product_sales"] = pd.concat(
        [source_tables["product_sales"], extra_sale], ignore_index=True
    )

    outputs = build_processed_tables(source_tables)
    client = outputs[CLIENT_OUTPUT.removesuffix(".csv")].set_index("client_id")
    panel = outputs[OUTLET_MONTH_OUTPUT.removesuffix(".csv")]

    assert client.loc["C1", "experiment_product_revenue"] == 25.0
    assert panel["product_revenue"].sum() == 70.0
