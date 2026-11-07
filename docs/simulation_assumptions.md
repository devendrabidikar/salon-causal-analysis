
All identifiers, outlets, services, stylists, client attributes, appointments,
and transactions in this project are fictional and generated in code. The data
does not describe real salons or customers and does not use external APIs or
downloaded personal information.

The generator is for internship practice and analysis testing. Its patterns
and intervention effects are chosen by construction, not measured from salon
operations.

## Reproducibility and execution

The generator uses NumPy's `default_rng` with seed `42` by default and pandas
for tabular output. Appointments are generated in batches of 25,000 to limit
peak memory use on a typical Windows laptop.
The generation function accepts a seed override. Because no generation
manifest was supplied for the current CSV files, their exact historical seed
and source revision are not independently verified.

From the project root in PowerShell, after reviewing the generator and
installing the project requirements, run:

```powershell
.\.venv\Scripts\Activate.ps1
python .\src\generate_salon_data.py
```

The script writes CSVs into `data\raw\`, validates them, and only then moves the
staged files into place. It refuses to run if any requested output CSV already
exists, rather than overwriting it. Move or rename existing files yourself
before rerunning if you intend to replace them.

## Coverage and entities

- The appointment table contains exactly 300,000 rows with dates within the
  inclusive range 2025-11-01 to 2026-10-31. The generator validates that dates
  fall within this range; it does not require either endpoint to be observed.
- There are 22 fictional outlets across four synthetic regions and 24,000
  synthetic client records.
- Each client is assigned once to a client-level experiment group. Assignment
  is randomized and exactly balanced: 12,000 Control and 12,000 Treatment.
- Exactly 70% of clients (16,800) have pre-history signup dates from
  2024-11-01 through 2025-10-31. The remaining 30% (7,200) sign up during the
  historical period from 2025-11-01 through 2026-08-15.
- Every client signs up on or before the fixed assignment date, 2026-08-15,
  which precedes the experiment start date of 2026-09-01. Therefore all
  24,000 clients are eligible before the experiment begins.
- Client signup dates, engagement segments, simulated active windows, and
  outlet preferences shape repeat-visit patterns. Appointment counts are
  sampled according to segment-level visit propensities. Some appointment rows
  are walk-ins and therefore have a blank `client_id`.
- Simulated active windows approximate months using 30-day periods:
  `active_months * 30`, capped at the end of the historical period. These are
  approximate durations, not calendar-month calculations.
- Stylists belong to one outlet; appointments link to valid outlets, stylists,
  and services. Cancelled and No-show appointments have zero service revenue.
- Product sales are sampled from completed appointments and link to a valid
  appointment, outlet, product, and (where applicable) client.
- Nullable values are intentional: walk-ins have no client ID, and completed
  appointments have no cancellation reason.

## Artificial experiment effect

The experiment period is 2026-09-01 through 2026-10-31. For appointments in
that period belonging to Treatment-assigned clients, the generator multiplies
the simulated gross service amount by **1.06** before applying any discount.
The same multiplier does not apply to Control clients, appointments outside
the experiment period, or walk-ins.

This is an artificial 6% uplift injected solely to test A/B analysis code. It
is not a measured effect, a real-world estimate, or a claim about customer
behavior. It changes service revenue only. It does not directly change
retention, appointment volume, no-show rate, cancellation rate, or product
revenue. Client-level assignment applies to the full 24,000-client table even
when a client has no appointment during the experiment window.

## Artificial Difference-in-Differences policy effect

North and South are the two treated synthetic regions; East and West are
control regions. The outlet allocation is balanced at 11 treated and 11
control outlets. The policy start date is 2026-07-01.

For completed appointments at treated outlets on or after the policy start
date, the generator multiplies the simulated gross service amount by **1.04**
before applying discounts. Control outlets and treated-outlet appointments
before the start date do not receive this multiplier.

This is an artificial 4% post-policy effect injected to exercise
Difference-in-Differences workflows, not a real policy impact or empirical
finding. Where the experiment and policy windows overlap, their multipliers
are applied multiplicatively (1.06 × 1.04) before discounts.

The policy post-period overlaps the A/B experiment in September and October
2026. For a primary DiD analysis that avoids this overlap, use July–August
2026 as the post-policy period. Treat September–October as a sensitivity
analysis and account for client-level A/B assignment in those months.

## Revenue and appointment behavior

- Service prices, outlet price indices, stylist levels, discounts, duration,
  booking channels, completion status, and product purchases are randomly
  simulated using fixed parameter distributions in the generator.
- Completed appointment revenue is non-negative net service revenue after
  discounts. Cancelled and No-show appointments have zero service revenue.
- Walk-ins are more likely to be completed and always use the `Walk-in` booking
  channel.
- Product revenue is recorded separately from appointment service revenue.
- Product-sale revenue is calculated as `quantity × unit_price − discount`.
  Revenue and discount are stored to two decimal places; validation allows an
  absolute reconciliation difference of at most 0.01 currency units because
  those fields are rounded separately.

## Statistical limitations

- Experiment assignment is randomized at the client level. A client may have
  multiple appointment rows, so repeated appointments from the same client are
  not independent randomized observations. For an intention-to-treat A/B
  analysis, use service revenue per randomized client as the primary outcome:
  construct one row per randomized client and sum that client's service
  revenue during 2026-09-01 through 2026-10-31. Include every assigned client,
  including clients with no experiment-period appointments, with zero revenue.
  Compare Treatment and Control at the client level and report the effect size
  with a 95% confidence interval. Do not treat repeated appointment rows as
  independent randomized observations.
- Difference-in-Differences policy assignment is at the region level: North
  and South are treated, while East and West are controls. There are only four
  synthetic regions. Inference with only four region clusters is limited and
  must not be presented as strong real-world causal evidence. Use July–August
  2026 as the primary clean post-policy period; use September–October as a
  sensitivity analysis that accounts for A/B assignment.
- The 6% experiment uplift and 4% post-policy uplift are deliberately injected
  simulation effects. They are for testing analysis code only, not real-world
  effects, estimates, or findings.

## Validation performed by the generator

Before final CSVs are committed, the script prints a validation report and
prints an individual PASS/FAIL result and invalid-record count for each
category. It first checks all ten CSV files for missing files, missing
columns, and unexpected columns; row-level validation starts only if all
schemas match. It then checks:

- Exactly 300,000 appointment rows, unique appointment IDs, and dates within
  the specified historical period.
- Valid appointment client, outlet, stylist, and service references, including
  stylist/outlet ownership; valid appointment statuses; consistent Walk-in
  channel and missing client IDs; and positive service duration.
- Non-negative appointment revenue and discounts; zero revenue and discount
  for cancelled/no-show appointments; and cancellation reasons present only
  for non-completed appointments.
- Exactly 24,000 unique client assignments, with each client assigned once,
  valid assignment groups and dates, all signups on or before 2026-08-15, and
  exactly 12,000 Control and 12,000 Treatment.
- Exactly 22 unique outlets and matching policy rows; North/South are Treated,
  East/West are Control, with 11 outlets in each group and the 2026-07-01
  policy start date. Policy group labels and `is_treated` values must agree.
- Non-null, non-empty required IDs; unique client, outlet, stylist, service,
  product, appointment, sale, assignment, and policy identifiers as
  applicable; and valid foreign-key relationships.
- Product-sale IDs are unique. Each sale links to a completed appointment and
  matches that appointment's client ID (including blank walk-in IDs), outlet,
  and sale date. Product references are valid, quantities and unit prices are
  positive, discounts and revenue are non-negative, and revenue reconciles to
  quantity times unit price less discount within 0.01 currency units.
- Missing values, allowing only the nullable fields documented above.

If validation fails, the script reports the issue and does not move the staged
CSV files into the output directory.
