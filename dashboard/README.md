# Salon Experiment & Causal Analysis Dashboard

The dashboard reads the existing synthetic CSVs in `data/raw/`; it does not
generate or change them. All records and injected effects are fictional, not
evidence about real salons or customers.

## Requirements

Use the project virtual environment and install the root requirements:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The requirements include Streamlit and Plotly. The dashboard needs all ten
source CSVs in `data/raw/` and reports missing or malformed tables in the app.
It builds the client-level and outlet-month tables in memory. To save these
derived tables for notebooks or other analysis, run this from the project root:

```powershell
.\.venv\Scripts\python.exe .\src\prepare_data.py
```

See `docs/data_processing.md` for their schemas.

## Run

From the project root:

```powershell
streamlit run .\dashboard\app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Pages

- **Overview:** date-filtered appointment and service-revenue totals, entity counts,
  and status distribution.
- **Appointment Analytics:** daily/monthly appointment volume, status, no-show,
  cancellation rates, and cancellation reasons.
- **Revenue Analytics:** monthly, outlet, and service-level service revenue;
  product sales revenue is shown separately.
- **Customer Analytics:** completed-visit repeat summaries and simulated client
  retention segments. Walk-ins without a client ID are excluded from client-level
  calculations.
- **A/B Testing:** client-level ITT service revenue for September–October 2026.
  Every assigned client is retained, including zero-revenue clients. Appointment
  rows are not treated as independent randomized observations.
- **Difference-in-Differences:** outlet-month treated/control trends and a
  two-way fixed-effects estimate for the primary July–August post-policy window.
  There are only four policy-assignment regions; conventional cluster-robust
  inference is unreliable and is not strong causal evidence.
- **CUPED & Regression:** client-level estimates and intervals recalculated from
  the CSVs with pre-experiment covariates.
- **Final Decision:** the existing documented NO-GO for real-world rollout and
  its synthetic-data limitations.

Revenue is displayed in Indian Rupees (₹), the currency convention used in this
simulation. Dashboard metrics describe the generated records; they are not
empirical findings or deployment recommendations.
