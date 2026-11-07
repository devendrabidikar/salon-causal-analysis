from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_overview_metrics_and_all_navigation_pages():
    app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    assert not app.exception
    metrics = {metric.label: metric.value for metric in app.metric}
    assert set(metrics) == {
        "Appointments",
        "Clients in roster",
        "Outlets",
        "Stylists",
        "Service revenue",
    }
    assert metrics["Appointments"] == "300,000"
    assert metrics["Clients in roster"] == "24,000"
    assert metrics["Outlets"] == "22"
    assert metrics["Stylists"] == "176"
    assert metrics["Service revenue"]

    for page in [
        "Appointment Analytics",
        "Revenue Analytics",
        "Customer Analytics",
        "A/B Testing",
        "Difference-in-Differences",
        "CUPED & Regression",
        "Final Decision",
        "Overview",
    ]:
        app.radio[0].set_value(page).run(timeout=120)
        assert not app.exception, f"{page}: {[error.message for error in app.exception]}"
