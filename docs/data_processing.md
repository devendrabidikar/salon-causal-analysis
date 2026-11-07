# Reusable processed analysis tables

The CSVs in `data/raw/` remain the source of truth. The pipeline builds two
tables used by several analyses, so notebooks and the dashboard do not need to
repeat the same client- and outlet-level aggregations:

1. `data/processed/client_experiment_outcomes.csv` — one record for each
   randomized client, including clients with no appointments or no revenue in
   the experiment. It consolidates the ITT outcome/guardrails used in A/B
   analysis and the pre-treatment covariates used by CUPED and regression.
2. `data/processed/outlet_month_panel.csv` — a complete outlet-by-month panel
   from November 2025 through October 2026 for DiD and trend analysis. The
   current design produces 22 × 12 = 264 records. Service revenue and product
   revenue remain separate.

These tables are derived from the existing CSVs; they do not add simulated
observations or effects. They use the project date windows. Walk-ins remain in
outlet-month summaries, but appointments without a client ID cannot be assigned
to a randomized client.

## Generate or verify

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe .\src\prepare_data.py
```

The pipeline reads six required raw files:
`appointments.csv`, `clients.csv`, `outlets.csv`,
`experiment_assignments.csv`, `did_outlet_policy.csv`, and
`product_sales.csv`. It validates schemas, dates, assignment metadata and
balance, foreign keys, policy consistency, non-negative amounts, and product
sales links/arithmetic before creating outputs.

The pipeline is deterministic and safe to rerun:

- The first successful run stages and validates both files before committing
  them.
- If both existing processed CSVs are byte-identical to the current
  transformations, a rerun is a no-op.
- If only one output exists, or either output differs from the current raw
  data, the pipeline refuses to overwrite anything. Back up and deliberately
  remove both generated processed outputs before regenerating after changing
  raw inputs or transformation logic.
- Raw files are read-only and are never regenerated, replaced, or removed.

## Output schemas and transformations

### `client_experiment_outcomes.csv`

| Columns | Definition |
|---|---|
| `client_id`, `assignment_group`, `treatment` | Unique randomized roster key, Control/Treatment label, and 0/1 indicator. |
| `assigned_at`, `experiment_start_date`, `experiment_end_date` | Assignment and experiment metadata copied from the assignment table. |
| `signup_date`, `signup_age_days_at_assignment`, `retention_segment`, `preferred_outlet_id`, `preferred_region`, `simulated_active_until` | Client dimension attributes; signup age is measured at assignment. |
| `pre_experiment_appointment_count`, `pre_experiment_completed_count`, `pre_experiment_service_revenue` | Identified-client appointments before 2026-09-01, aggregated per client. Revenue is the sum of the source appointment revenue. |
| `planning_baseline_appointment_count`, `planning_baseline_completed_count`, `planning_baseline_service_revenue`, `planning_baseline_product_revenue` | Per-client May–June 2026 baseline used by the A/B notebook's balance and power-planning steps. |
| `experiment_appointment_count`, `experiment_completed_count`, `experiment_cancelled_count`, `experiment_no_show_count`, `experiment_service_revenue`, `experiment_product_revenue` | Per-client outcomes from 2026-09-01 through 2026-10-31; only identified clients are attributed. |
| `completed_share_per_client`, `cancelled_share_per_client`, `no_show_share_per_client` | Status counts divided by that client's experiment appointment count; set to zero when the denominator is zero. |
| `any_completed`, `any_cancelled`, `any_no_show`, `repeat_completed` | Binary indicators for at least one such event, or at least two completed experiment appointments. |

Every assignment is retained. Missing client-level aggregate values are
zero-filled only after joining the full randomized roster. This implements the
cohort shape; statistical inference remains the responsibility of the analysis
notebooks.

### `outlet_month_panel.csv`

| Columns | Definition |
|---|---|
| `outlet_id`, `outlet_name`, `region`, `policy_group`, `is_treated`, `policy_start_date` | Outlet and policy dimensions. |
| `month`, `month_label` | Calendar-month start timestamp and `YYYY-MM` display label. |
| `post_primary` | 1 for July–August 2026; 0 otherwise. |
| `post_sensitivity` | 1 for July–October 2026; 0 otherwise. |
| `post_policy` | 1 for July 2026 onward; 0 before policy start. |
| `service_revenue`, `discount_amount` | Monthly sums from appointments; service revenue uses the source net-revenue field. |
| `appointment_count`, `completed_count`, `cancelled_count`, `no_show_count` | Monthly row/status counts. Walk-ins are included. |
| `product_revenue` | Product-sale revenue aggregated separately by outlet and sale month. |

The complete 22-outlet × 12-month grid is retained, with zero metrics for
outlet-months without activity. Region is the policy assignment level; the four
region clusters still make conventional cluster-robust DiD inference
unreliable. These processed outcomes do not remove that design limitation.
