from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import generate_salon_data as generator


def _write_schema_headers(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, columns in generator.EXPECTED_COLUMNS.items():
        pd.DataFrame(columns=columns).to_csv(directory / filename, index=False)


def test_schema_validation_accepts_exact_expected_columns(tmp_path):
    _write_schema_headers(tmp_path)

    generator._validate_schema(tmp_path)


def test_schema_validation_reports_missing_files(tmp_path, capsys):
    _write_schema_headers(tmp_path)
    (tmp_path / "products.csv").unlink()

    with pytest.raises(RuntimeError, match="CSV schema validation failed"):
        generator._validate_schema(tmp_path)

    output = capsys.readouterr().out
    assert "Missing files:" in output
    assert "products.csv" in output


def test_schema_validation_reports_column_differences_and_order(tmp_path, capsys):
    _write_schema_headers(tmp_path)
    expected = generator.EXPECTED_COLUMNS["clients.csv"]
    actual = [column for column in expected if column != "signup_date"]
    actual.insert(0, "unexpected_column")
    pd.DataFrame(columns=actual).to_csv(tmp_path / "clients.csv", index=False)

    with pytest.raises(RuntimeError, match="CSV schema validation failed"):
        generator._validate_schema(tmp_path)

    output = capsys.readouterr().out
    assert "Missing columns:" in output
    assert "signup_date" in output
    assert "Unexpected columns:" in output
    assert "unexpected_column" in output
    assert "Column order mismatches:" in output


def test_dimension_and_client_assignment_generators_preserve_design():
    rng = np.random.default_rng(generator.SEED)
    outlets, stylists, services, products, policy = generator._make_dimensions(rng)

    assert len(outlets) == 22
    assert len(stylists) == 176
    assert len(services) == 10
    assert len(products) == 15
    assert stylists["stylist_id"].is_unique
    assert services["service_id"].is_unique
    assert products["product_id"].is_unique
    assert stylists["outlet_id"].isin(outlets["outlet_id"]).all()
    assert policy["outlet_id"].is_unique

    clients, _, _, _ = generator._make_clients(
        rng, outlets["outlet_id"].to_numpy()
    )
    assignments, _ = generator._make_experiment_assignments(rng, clients)

    assert len(clients) == generator.CLIENT_COUNT
    assert clients["client_id"].is_unique
    assert clients["signup_date"].le(
        generator.ASSIGNMENT_DATE.strftime("%Y-%m-%d")
    ).all()
    signup_dates = pd.to_datetime(clients["signup_date"])
    pre_history = signup_dates < generator.HISTORY_START
    assert int(pre_history.sum()) == 16_800
    assert int((~pre_history).sum()) == 7_200
    assert len(assignments) == generator.CLIENT_COUNT
    assert assignments["client_id"].is_unique
    assert assignments["assignment_group"].value_counts().to_dict() == {
        "Treatment": 12_000,
        "Control": 12_000,
    }


def test_generation_refuses_existing_outputs_without_overwriting(tmp_path):
    existing = tmp_path / "appointments.csv"
    existing.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generator.generate_salon_data(output_dir=tmp_path)

    assert existing.read_text(encoding="utf-8") == "sentinel\n"


def test_staged_commit_rolls_back_after_overwrite_conflict(tmp_path):
    stage_dir = tmp_path / "stage"
    output_dir = tmp_path / "output"
    stage_dir.mkdir()
    output_dir.mkdir()
    for filename in generator.OUTPUT_FILES:
        (stage_dir / filename).write_text(
            f"staged {filename}", encoding="utf-8"
        )
    existing = output_dir / "clients.csv"
    existing.write_text("existing client data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generator._commit_staged_files(stage_dir, output_dir)

    assert not (output_dir / "appointments.csv").exists()
    assert (stage_dir / "appointments.csv").read_text(encoding="utf-8") == (
        "staged appointments.csv"
    )
    assert existing.read_text(encoding="utf-8") == "existing client data"
