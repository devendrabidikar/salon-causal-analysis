# Salon Experiment & Causal Analysis

## Project overview

This project uses a fictional salon scenario to practice the main steps of a
small data science analysis: checking source data, exploring patterns, planning
an experiment, estimating effects, and explaining what the evidence can and
cannot support. **The dataset is entirely synthetic.** The records describe no
real clients, staff, outlets, or business, and the project does not use external
APIs or downloaded personal data.

The generator defines 300,000 appointments dated from November 1, 2025, through
October 31, 2026. Related tables describe 24,000 fictional clients, 22 outlets
across four regions, stylists, services, products, product sales, experiment
assignments, and policy assignments. Appointments are generated in batches to
limit memory use. The generator defaults to NumPy seed 42, validates its
staged files, and refuses to overwrite existing CSVs. The current files do not
have a generation manifest, so their exact historical seed and source revision
are not independently verified.

## Simulated interventions

Clients are assigned once, at random, to the A/B groups: 12,000 Control and
12,000 Treatment. The experiment window is September–October 2026. The
generator applies an artificial 6% multiplier to gross service revenue for
Treatment-assigned clients during that period, before discounts. It does not
directly change appointment volume, retention, cancellation or no-show status,
or product revenue. This multiplier is included to exercise analysis code; it
is not an observed business result.

A separate simulated policy starts on July 1, 2026. North and South are the
treated regions; East and West are controls. The generator applies an
artificial 4% gross service-revenue multiplier to completed appointments at
treated outlets after the policy date. September and October fall in both the
policy period and the A/B experiment. The DiD analysis therefore uses
July–August as its primary post-policy period and treats September–October as
sensitivity analysis. Policy assignment covers four regions only, which is
too few for reliable conventional cluster-robust inference.

## Analysis workflow

The four notebooks move from data checks and descriptive summaries to
client-level A/B analysis, outlet-month DiD, and CUPED/regression adjustment.
The experiment analysis retains all assigned clients, including clients with
zero experiment-period service revenue. Appointment rows are aggregated to
clients before treatment-group comparisons; they are not treated as
independent randomized units. The A/B and DiD analyses also discuss baseline
balance, uncertainty, pre-trend diagnostics, overlap, and sensitivity checks.
The Streamlit dashboard presents the same synthetic source data and calculated
summaries. The final report, experiment design document, and decision memo
summarize the notebook results.

## How to interpret the results

Numerical results describe this generated dataset only. The injected effects
are simulation settings, not empirical estimates, forecasts, or evidence of
customer behavior. DiD relies on a parallel-trends assumption that cannot be
established by these plots and short pre-period diagnostics. Four policy
regions make conventional cluster-based intervals and p-values unreliable.
Because the A/B and policy periods overlap, the project keeps July–August as
the primary DiD post-period and labels September–October as a sensitivity
period.

The project is suitable for practicing data validation and statistical
workflows. It does not support a real-world rollout decision. Such a decision
would require authorized real data, a pre-specified evaluation, and a policy
design with enough independent regions for credible inference.
