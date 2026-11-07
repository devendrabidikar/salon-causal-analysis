# Salon Experiment & Causal Analysis

This internship project uses a fictional salon setting to practice data
validation, exploratory analysis, A/B testing, Difference-in-Differences (DiD),
and regression adjustment. **All records are synthetic and all intervention
effects are deliberately simulated.** The project contains no real customer,
employee, salon, or business data, and its findings are not evidence about real
business outcomes.

## Project structure

```text
salon-causal-analysis/
├── data/
│   ├── raw/             # Ten synthetic source tables
│   └── processed/       # Client-level and outlet-month analysis tables
├── docs/                # Simulation assumptions and data documentation
├── notebooks/           # Four analysis notebooks
├── reports/             # Final report, design document, memo, and audits
├── dashboard/           # Streamlit dashboard
├── src/                 # Data generator and preprocessing pipeline
├── tests/               # Generator, pipeline, and dashboard tests
├── requirements.txt
└── README.md
```

## Dataset and design

The generator defines 300,000 appointments dated from November 1, 2025, through
October 31, 2026. The synthetic tables include 24,000 clients, 22 outlets in
four regions, stylists, services, products, sales, experiment assignments,
and policy assignments.

Clients are assigned 1:1 to the A/B groups (12,000 per group) for September and
October 2026. The simulated regional policy starts July 1, 2026; North and
South are treated, while East and West are controls. The generator injects a
6% service-revenue multiplier for the A/B treatment and a 4% multiplier for
treated outlets after the policy date. These are artificial settings for
testing analysis methods, not measured effects or business recommendations.
The A/B and policy periods overlap in September–October.

The generator uses seed 42 by default and creates appointments in 25,000-row
batches. It validates staged output and refuses to overwrite existing CSVs.
No generation manifest is available for the current files, so their exact
historical seed and source revision are not verified. See
[docs/simulation_assumptions.md](docs/simulation_assumptions.md) before
interpreting the results.

## Processed analysis tables

Two derived tables consolidate repeated analysis preparation without copying
the raw appointment rows:

- `client_experiment_outcomes.csv` keeps every randomized client, including
  clients with zero experiment-period revenue, and includes pre-period
  covariates.
- `outlet_month_panel.csv` contains the complete 22-outlet by 12-month panel
  used for DiD.

Run or verify the pipeline from the project root:

```powershell
.\.venv\Scripts\python.exe .\src\prepare_data.py
```

Matching outputs are left unchanged on reruns. Partial or different existing
outputs are not overwritten. See
[docs/data_processing.md](docs/data_processing.md) for definitions and schemas.

## Analysis notebooks

1. [01_data_quality_and_eda.ipynb](notebooks/01_data_quality_and_eda.ipynb) —
   checks the source tables and summarizes their distributions and trends.
2. [02_ab_experiment_design_and_analysis.ipynb](notebooks/02_ab_experiment_design_and_analysis.ipynb) —
   checks assignment and baseline balance, plans power, and estimates
   client-level intention-to-treat (ITT) effects.
3. [03_difference_in_differences_analysis.ipynb](notebooks/03_difference_in_differences_analysis.ipynb) —
   examines pre-policy trends and estimates fixed-effects DiD models.
4. [04_cuped_regression_robustness.ipynb](notebooks/04_cuped_regression_robustness.ipynb) —
   compares unadjusted, CUPED, and HC3 regression estimates.

The A/B notebooks use one row per randomized client; repeat appointment rows
are not treated as independent observations. DiD treatment is assigned across
only four regions (two treated and two control), which makes conventional
cluster-robust inference unreliable. September–October also overlaps the A/B
experiment. These limitations are discussed in the notebooks and reports.

## Reports

- [Final analysis report](reports/Salon_Experiment_Causal_Analysis_Final_Report.docx)
- [Experiment design document](reports/Experiment_Design_Document.docx)
- [One-page decision memo](reports/One_Page_Decision_Memo.docx)
- [Project description](docs/PROJECT_DESCRIPTION.md)

The documented recommendation is **NO-GO for real-world rollout**. It applies
to using this simulation as decision evidence; it does not claim that a real
salon intervention would or would not work.

## Fresh-clone setup (Windows PowerShell)

Clone the repository, open PowerShell in the `salon-causal-analysis` project
folder, and run these commands from the project root. Python 3.10 or newer is
required.

Create and activate a virtual environment, then install the project
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

The raw dataset is excluded from Git and is generated locally by the project.
Read [docs/simulation_assumptions.md](docs/simulation_assumptions.md), then run
the generator once on a fresh clone:

```powershell
python .\src\generate_salon_data.py
```

It creates the ten synthetic CSV files in `data/raw/` using seed 42 by default.
The generator refuses to overwrite existing outputs; skip this step if the raw
CSV files already exist. No external data or personal information is
downloaded.

Build the two reusable analysis tables, then run the test suite:

```powershell
python .\src\prepare_data.py
python -m pytest -q
```

To open and execute the analyses, start JupyterLab from the project root:

```powershell
jupyter lab
```

In the JupyterLab file browser, open each notebook in `notebooks/` and select
**Run → Run All Cells**:

1. `01_data_quality_and_eda.ipynb`
2. `02_ab_experiment_design_and_analysis.ipynb`
3. `03_difference_in_differences_analysis.ipynb`
4. `04_cuped_regression_robustness.ipynb`

Run them in order. The notebooks read the raw CSV files and do not modify them.
They locate the project data when JupyterLab is started from the project root.

## Dashboard

The Streamlit app reads the synthetic source CSVs and provides overview,
appointment, revenue, customer, A/B, DiD, adjustment, and decision pages. Its
metrics are calculated from the CSVs. Start the dashboard from the project root after generating the raw data:

```powershell
streamlit run .\dashboard\app.py
```

See [dashboard/README.md](dashboard/README.md) for dashboard-specific
instructions.

## Interpretation limits

The data patterns and intervention effects are chosen by construction. The
pre-trend checks cannot establish the DiD parallel-trends assumption, and four
region clusters are too few for reliable conventional cluster-robust inference.
The A/B and policy windows overlap in September–October 2026. These results are
for practicing analysis methods, not for estimating real salon performance or
making operational decisions.
