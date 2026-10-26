# Disclaimer: 

Educational portfolio project. It runs on synthetic data and does not provide financial advice. Merchant and bank names in the simulated data are used only for realism; this project is not affiliated with any of them.

# Intelligent Personal Finance Advisor

An AI + analytics personal-finance assistant on a **synthetic Indian transaction dataset** (realistic merchants, UPI/card descriptions, festivals,
EMIs, injected ground-truth events), so every model is evaluated against known truth.

> **Login:** the app and API require a login. Set `JWT_SECRET` in `.env` before starting (generate one with
> `python -c "import secrets; print(secrets.token_urlsafe(48))"`). Set `DEMO_CUSTOMER_ID=C0001` for a one-click demo account, and use
> `MAIL_BACKEND=console` in development so verification and reset links are printed in the terminal. An existing database needs
> `sql/migrations/002_auth_extras.sql`. Details are in *Phase 9* at the end of this file.

## Quickstart (Ubuntu + Docker + VS Code)

```bash
cd finance-advisor
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                                           # then set JWT_SECRET in .env (see Phase 9)

docker compose up -d db                                        # PostgreSQL 16 with the schema
python -m data_gen.generate --customers 300 --months 36 --seed 42    # ~80 s, writes data/raw/*.csv
python -m data_gen.load_to_postgres --reset                    # loads CSVs, creates the analytics views
python -m ml.anomaly.run_detector --preset balanced            # overspending alerts  -> alerts table
python -m core.health_score                                    # health scores        -> health_scores table
python -m ml.forecasting.run_forecast --source db              # next-month forecasts -> forecasts table
uvicorn api.main:app --reload                                  # REST API + docs at http://localhost:8000/docs
streamlit run app/streamlit_app.py                             # dashboard + chat at http://localhost:8501 (log in, or use the demo account)
python -m pytest                                               # full test-suite (several minutes: builds simulated datasets)
```

On **Windows (PowerShell)** create the environment with `python -m venv .venv; .\.venv\Scripts\Activate.ps1` and copy the env file with
`Copy-Item .env.example .env`; the remaining commands are unchanged. Docker Desktop must be running.

Optional extras
```bash
python -m ml.categorizer.train --data-dir data/raw --out models/categorizer   # ~10 min; then:
python -m ml.categorizer.predict --dir models/categorizer                      # fills predicted_category for blank-label rows
#   (re-run the detector and forecast jobs afterwards so they use the improved categories)
pip install statsmodels prophet torch                          # enables ARIMA, Prophet and LSTM in the forecast comparison
python -m ml.forecasting.compare --data-dir data/raw --max-series 30 --origins 6
docker compose --profile api up -d --build                     # run the API in a container instead of uvicorn
docker compose --profile app up -d --build                     # run the Streamlit app in a container (after loading the data)
```
In VS Code: *Terminal > Run Task...* has every step above (task **"Setup database end to end (1-6)"** runs the first six), and *Run and Debug* has
launch configs for the API, the data loader, the forecast comparison and the current test file. Copy `.env.example` to `.env` first.
Chat works without any key (offline demo mode); set `ANTHROPIC_API_KEY` in `.env` to use a real model.

## Status: what is verified and what is not

**Run and tested** on Windows with Docker Desktop (Python 3.14, PostgreSQL 16, Streamlit 1.58, FastAPI 0.159, Plotly 6.7, pandas 3.0, SQLAlchemy 2.0, PyJWT 2.13):

| Area | Status |
|---|---|
| Synthetic data generation, loading into PostgreSQL 16, analytics views | Run |
| Streamlit dashboard (overview, spending, health & savings, alerts & insights, chat, statement import, model lab) against PostgreSQL | Run in a browser |
| FastAPI app under uvicorn, plus HTTP-level tests with `TestClient` | Run (15 HTTP tests pass) |
| Sign-up, email verification (console mail backend), login, refresh, password reset, lockout, account deletion, statement import and duplicate check | Click-tested in a real browser |
| Auth, flows, statement import, HTTP API and isolation tests | 152 tests pass |
| Data generator, SQL view logic (SQLite plus a runner that returns PostgreSQL-style types), overspend detector, categoriser, health score, savings planner, insights, chat service / tools / agent loop, forecasting baselines and gradient boosting | Automated tests (`python -m pytest`) |

**Not yet verified**

| Area | Status |
|---|---|
| Row-level security (`sql/rls.sql`) | Written; proven only when `tests/test_rls_postgres.py` is run against a real PostgreSQL with the role applied (instructions in the file) |
| Production deployment (`docker-compose.prod.yml`, Caddy HTTPS, SMTP email, backups, nightly jobs) | Written, not yet run end to end; email has only been tried with the console backend |
| Real LLM chat (`AnthropicClient`) | Not run. Chat uses the offline rule-based client unless `ANTHROPIC_API_KEY` is set |
| XGBoost stage 2; ARIMA, Prophet and LSTM forecasters | Implemented, not run |
| Power BI dashboard | Not built |

Everything here is simulated data with injected patterns; none of the accuracy figures describe real-world performance.

---

# Intelligent Personal Finance Advisor — Phase 1: Synthetic Data Layer

A simulated customer world (300 customers x 36 months, ~740K transactions) with realistic Indian
banking behaviour and **injected ground-truth events**, so every later model can be evaluated honestly.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m data_gen.generate --customers 300 --months 36 --seed 42   # writes data/raw/*.csv (~80s); --no-ambiguity for easy descriptions

docker compose up -d db                                             # PostgreSQL 16 + schema
python -m data_gen.load_to_postgres --reset                         # COPY CSVs into Postgres
psql postgresql://finance:finance@localhost:5432/finance_advisor -f sql/validation_queries.sql
```

## What is simulated
| Layer | Detail |
|---|---|
| Personas | 8 archetypes (young techie, single-earner family, frugal saver, student, freelancer, high-earner spender, govt employee, retiree) with their own income range, category mix, savings rate, EMI/SIP habits |
| Geography | 3 city tiers; rent scales by tier |
| Income | Salary (1st / month-end), April hikes, March bonus, lumpy freelance invoices, pension, student allowance |
| Spending | Poisson counts x lognormal tickets per category, monthly noise per customer, 6% annual inflation |
| Seasonality | Diwali / Holi / Eid / Christmas windows, summer travel & electricity, monsoon health, wedding season |
| Weekday effects | Dining, delivery, entertainment and shopping skew to Fri-Sun; transport to weekdays |
| Descriptions | UPI / POS / ECOM / ACH / NEFT / ATM strings with truncation, typos, aggregator prefixes (RAZORPAY*, PTM*) |
| Label noise | 8% of debits unlabelled, 3% mislabelled with realistic confusions (Food Delivery <-> Dining Out) |
| Ledger | Running balance, month-start sweeps to FD/RD, top-ups so balance never goes negative |
| SMS | Bank-specific alert formats (HDFC, SBI, ICICI, Axis, Kotak) + OTP / promo noise, linked to `true_txn_id` |

## Ground truth (for evaluation only — never use as model features)
* `transactions.true_category` — real category behind each label
* `injected_events` — `category_creep`, `category_spike`, `job_loss`, `income_raise`, `medical_emergency`,
  always starting after 6 clean months of history
* `sms_raw.true_txn_id`, `is_transaction`
* `customers.persona`

## Things to remember downstream
* The categoriser must only use `raw_description` (+ amount/channel/time) — not `merchant`.
* Exclude `Self Transfer` (sweeps/top-ups), `Investments` and income categories when computing "spending".
  `dim_category.is_expense` encodes this.
* Split by **time** (e.g. train 2023-24, test 2025), never randomly.
* Results are on simulated data with injected patterns; do not claim real-world accuracy.


---

# Phase 2: SQL Analytics Layer

`sql/views.sql` (created automatically by the loader) is the single source of truth every later module reads from.

| View | Purpose |
|---|---|
| `v_expense_txn` | Clean spending ledger: debits only, excludes investments / self-transfers, final category = `COALESCE(category, predicted_category, 'Uncategorized')` |
| `v_monthly_category_spend` | Customer x month x category totals |
| `v_category_month_grid` | Same, zero-filled so rolling statistics are not biased by missing months |
| `v_monthly_summary` | Income, expenses, EMI, discretionary spend, month-end balance, savings reserve, savings rate |
| `v_category_baseline` | Trailing 6-month mean/std per category (excludes the current month), % vs baseline, z-score |
| `v_overspend_alerts` | Naive rule-based alerts (>=25%, z>=2, >=Rs500): the **benchmark** to beat |
| `v_health_features` | Trailing-6-month inputs for the health score: savings rate, debt ratio, emergency-fund months, expense/income CV |

`sql/analytics_queries.sql` holds 15 named, parametrised queries (top categories, month comparison, weekday pattern,
seasonal index, top "unnecessary" merchants, spend drivers for "why did I spend more?", savings candidates, detector
evaluation ...). `core/queries.py` loads them by name; FastAPI endpoints and LLM tools will call these.

## Verifying
```bash
python -m tests.run_sql_checks
```
Runs the same SQL under SQLite (a smoke-test harness, not a replacement for Postgres) and asserts that expense totals,
6-month rolling mean/std, monthly income and reserves match independent pandas calculations.

## First finding: the naive detector is recall-rich and precision-poor
On an 80-customer sample, scored against `injected_events`:

| Event type | Detected |
|---|---|
| category_creep | ~91% |
| medical_emergency | ~90% |
| category_spike | ~70% |

...but only **~2.7% of alerts** fall inside a real event window (~1.5 alerts per customer-month).
Two things explain this, and both are worth saying in an interview:
1. **Base rate.** Events cover ~1.2% of customer-category-months, so even a decent detector has low precision.
2. **Small-sample z-scores.** A z-score from 6 monthly points has heavy tails, so normal months cross z>=2 about 10% of the time,
   even for stable categories like Groceries. Gating on category volatility did *not* fix it (tested).

Planned fixes for `ml/anomaly/`: persistence rules (e.g. 2 of last 3 months elevated), robust statistics (median/MAD),
seasonality-aware baselines (same month last year / festival calendar), an alert budget (top-k per customer-month ranked by
rupee impact), and reporting precision@k and time-to-detection instead of raw precision.
Caveat: "false" alerts are not necessarily wrong (a Diwali shopping spike is a real spike), only unlabeled.

## If you already created the Docker volume in Phase 1
`dim_category` gained an `is_primary_income` column. Recreate the DB: `docker compose down -v && docker compose up -d db`.


---

# Phase 3: Improved Overspend Detector (`ml/anomaly/`)

| File | Purpose |
|---|---|
| `overspend.py` | Vectorised detector (numpy/pandas): robust reference, peer adjustment, persistence and level-shift rules, spike rule, frequency check, alert budget, templated messages. Presets: `naive`, `balanced`, `low_noise` |
| `evaluate.py` | Scores any detector against `injected_events` (alert precision, event recall by type, time-to-detection, threshold-free average precision) |
| `compare.py` | Dev -> test comparison; writes `docs/overspend_results.md` |
| `run_detector.py` | Reads `v_category_month_grid`, writes the `alerts` table (`--latest` for production-style monthly runs) |

```bash
python -m ml.anomaly.compare --customers 150     # ~2 minutes, deterministic
python -m ml.anomaly.run_detector --preset balanced
```

## Method
1. **Robust reference**: median / trimmed mean of the previous 6 months, so one old spike does not poison the baseline.
2. **Peer adjustment** (`low_noise`): expected spend is scaled by how similar customers' spend in that category moved *that same month*, absorbing festivals and inflation without leaking future data.
3. **Level-shift test** (`balanced`): is the average of the last 4 months abnormal? The reference skips those months so a creeping habit is not absorbed into its own baseline, and averaging cuts noise by 2x.
4. **Spike rule**: a single month alerts alone only at z >= 7.
5. **Frequency check**: purchase count must also rise (a single big ticket is a "large purchase", not a habit).
6. **Alert budget**: top-k alerts per customer-month, ranked by rupee impact.

Example output (from the planted-creep unit-test data): *"Food Delivery spending is 117% higher than your 6-month average (Rs 13,435 vs Rs 6,196); the last 4 months average 123% above normal, so this looks like a sustained shift."*

## Results (150 customers per world)
DEV = the world the operating points were chosen on (seed 11). TEST = an unseen world (seed 22). **Quote the TEST rows.**

| World | Detector | Alerts / cust-month | Alert precision | Event recall | Creep recall | Spike recall | Months to detect creep (median) | Avg. precision (ranking) |
|---|---|---|---|---|---|---|---|---|
| DEV | naive (SQL rule) | 1.53 | 2.5% | 83.3% | 91.4% | 65.0% | 2 | 0.029 |
| DEV | balanced | 0.27 | 7.8% | 48.2% | 55.2% | 27.5% | 3 | 0.029 |
| DEV | low_noise | 0.08 | 10.3% | 23.7% | 32.8% | 12.5% | 3 | 0.039 |
| **TEST** | naive (SQL rule) | 1.51 | 1.8% | 71.1% | 87.8% | 48.8% | 3 | 0.024 |
| **TEST** | **balanced** | **0.27** | **5.8%** | **39.2%** | **53.7%** | 14.0% | 4 | 0.026 |
| **TEST** | low_noise | 0.09 | 7.1% | 20.6% | 34.1% | 9.3% | 4 | 0.037 |

Base rate of event cells: ~0.8% (a random scorer gets ~0.008 average precision).

## Honest reading
* **It is a trade, not a free win.** On unseen data `balanced` sends **5.6x fewer alerts** at **~3x the precision**, but recall falls
  from 71% to 39%. Most of the loss is one-off spikes (49% -> 14%) and creep is caught later (median 4 vs 3 months). Creep, the
  "overspending pattern" the product is about, keeps 54% recall.
* **Most of the gain is the operating point, not a better score.** Threshold-free average precision barely moves for `balanced`
  (0.024 -> 0.026) and improves modestly for `low_noise` (0.037). All detectors are ~3-5x a random scorer.
* **Precision is low in absolute terms (5-7%).** Event cells are <1% of all cells, and natural month-to-month noise in lumpy
  categories (Travel, Health, Gifts, Transfers) is comparable in size to the injected 1.5-3x events. In exploratory runs
  (~1,700 randomly sampled configurations across three rule families; the throwaway search scripts are not in the repo) nothing
  broke a ceiling of roughly 5-10% precision at low alert volume, so no more than that is claimed on this data.
* **Dev -> test drop** (7.8% -> 5.8% precision) is the expected cost of choosing presets on dev.
* **A perfect categoriser does not give a consistent gain** (`TEST, perfect categoriser` rows in `docs/overspend_results.md`), so
  label noise is not the bottleneck here; categoriser accuracy and detector quality are separate problems.
* "False" alerts are *unlabeled*, not necessarily wrong: a Rs 40,000 one-off trip is a real spike.
* The pandas `naive` preset reproduces the SQL view `v_overspend_alerts` alert-for-alert (verified in `tests/test_overspend.py`).

## What did not help (exploratory)
Gating on category volatility; requiring a purchase-frequency increase as a hard filter (kept only as a mild one); seasonal
adjustment via same-month-last-year (peer adjustment helps only in `low_noise`).

## Ideas that could move the ceiling
A supervised model (e.g. XGBoost on shift / z / frequency / peer features, trained on dev events, evaluated on test),
aggregating lumpy categories into a quarterly view, and *cumulative-pace alerts* ("you are Rs 6,000 above your usual pace this
month") instead of per-category monthly ones.

## Tests
```bash
python -m pytest tests            # or run the functions directly if pytest is not installed
python -m tests.run_sql_checks    # SQL views vs pandas (SQLite smoke test)
```


---

# Phase 4: Expense Categoriser (`ml/categorizer/`)

Classifies debit transactions into 22 categories (19 expense categories plus EMI, Investments and Self Transfer) and
fills `transactions.predicted_category` / `confidence`, with a calibrated "needs review" queue.

| File | Purpose |
|---|---|
| `features.py` | Description cleaning, CPI-deflated amount / channel / time features, leak-free **payee-history** features |
| `model.py` | Two-stage stack: char + word n-gram TF-IDF -> linear model (out-of-fold probabilities) -> gradient boosting (XGBoost if installed, else scikit-learn `HistGradientBoostingClassifier`) |
| `train.py` | Time-split training, validation-chosen confidence thresholds, test report, ablation, confusion matrix |
| `evaluate.py`, `predict.py` | Metrics / selective-accuracy helpers; inference wrapper and Postgres write-back |

```bash
python -m ml.categorizer.train --data-dir data/raw --out models/categorizer     # ~8 min on 1 CPU; faster with more cores
python -m ml.categorizer.predict --dir models/categorizer                         # label uncategorised debits in Postgres
```

## Design
* **Text** (`raw_description` only; `merchant` / `true_category` are never used): character n-grams survive typos, truncation and
  aggregator prefixes (`RAZORPAY*`, `PTM*`).
* **Stacking without leakage**: stage 2 is trained on 5-fold out-of-fold stage-1 probabilities.
* **Amount / channel / time context**, with amounts **deflated by a CPI index** so trees do not learn absolute thresholds that
  break under inflation (before this fix, accuracy collapsed from ~94% to 54% on the next year's data).
* **Payee history** (strictly earlier rows of the same customer and payee): payment count (ever / 90 days), days since last payment,
  same-amount-as-last-time, ~monthly recurrence, and the customer's *own earlier labels* for that payee. Unit tests assert that
  changing future rows never changes a row's features and that a row's own label never feeds its own memory features.
* **Labels**: trained on the noisy bank/user `category` (8% blank, 3% wrong); evaluated on the clean `true_category`.
* **Split**: train = labeled rows before 2024-10, validation = Q4 2024 (used *only* to pick confidence thresholds), test = 2025.
  A random 20% of customers are never seen in training.

## Making the task realistic (`data_gen/ambiguity.py`)
With every merchant named in its description, text alone scored 99.9%, which is a lookup table, not a categoriser. A post-processing
step now rewrites ~13% of debit descriptions (own RNG; amounts, dates and categories are untouched, verified column by column):
payments to local shops through a **personal UPI id** (stable per customer), **parent-brand collapse** ("AMAZON" for Prime / Fresh /
shopping, "SWIGGY" for delivery / Instamart / Swiggy One), and **generic aggregators** ("PAYTM", "PHONEPE"). Use
`python -m data_gen.generate --no-ambiguity` for the easy version. The ambiguity rates are design choices, so absolute accuracy
describes this simulation, not real bank data.

## Results (2025 test set, 120,694 debit transactions; from `docs/categorizer/metrics.json`)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always predict the most common class | 19.6% | 0.015 |
| Text only (stage 1) | 93.0% | 0.948 |
| **Full stacked model** | **95.6%** | **0.964** |
| &nbsp;&nbsp;customers seen in training | 95.7% | 0.967 |
| &nbsp;&nbsp;customers never seen in training | 94.9% | 0.955 |
| &nbsp;&nbsp;only rows with a blank label (the real use case) | 95.7% | 0.961 |

**What each part adds** (stage 2 re-fit on shared stage-1 outputs, test set):

| Stage-2 inputs | Accuracy | Macro-F1 |
|---|---|---|
| text probabilities only | 93.1% | 0.947 |
| + amount / channel / time | 94.4% | 0.956 |
| + payee history (full model) | 95.6% | 0.964 |

**"Needs review" queue**: thresholds picked on validation, reported on test. The target must exceed the model's overall accuracy,
otherwise almost nothing is flagged (an earlier version used 95% and sent <0.01% of rows to review).

| Target accuracy of auto-accepted rows | Achieved on test | Auto-accepted | Sent to review |
|---|---|---|---|
| 97% | 96.8% | 98.1% | 1.9% |
| **98% (default)** | 98.0% | 95.8% | 4.2% |
| 99% | 99.0% | 92.7% | 7.3% |

**Where it still errs** (largest confusions). These are the ambiguities built into the data:

| True | Predicted | Count |
|---|---|---|
| Groceries | Food Delivery | 621 |
| Food Delivery | Groceries | 343 |
| Groceries | Shopping | 340 |
| Shopping | Groceries | 267 |
| Dining Out | Transfers | 257 |
| Groceries | Transfers | 226 |

Confusion matrix: `docs/categorizer/confusion_matrix.png`. Per-class results (worst first):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Groceries | 0.918 | 0.904 | 0.911 | 16,997 |
| Food Delivery | 0.917 | 0.947 | 0.932 | 11,411 |
| Gifts & Donations | 0.986 | 0.892 | 0.937 | 2,648 |
| Transfers | 0.903 | 0.975 | 0.938 | 8,511 |
| Dining Out | 0.954 | 0.924 | 0.939 | 6,889 |
| Fuel | 0.965 | 0.919 | 0.941 | 2,628 |
| Shopping | 0.939 | 0.949 | 0.944 | 9,209 |
| Personal Care | 0.987 | 0.914 | 0.949 | 3,247 |
| Education | 0.958 | 0.946 | 0.952 | 1,356 |
| Health | 0.991 | 0.916 | 0.952 | 3,375 |
| Entertainment | 0.963 | 0.957 | 0.960 | 4,208 |
| Utilities | 0.946 | 0.983 | 0.964 | 6,475 |
| Mobile & Internet | 0.974 | 0.958 | 0.966 | 3,600 |
| Travel | 0.976 | 0.964 | 0.970 | 1,895 |
| Subscriptions | 0.987 | 0.983 | 0.985 | 6,120 |
| Rent | 0.994 | 0.983 | 0.988 | 1,104 |
| Transport | 0.989 | 0.994 | 0.991 | 23,687 |
| Insurance | 0.998 | 0.993 | 0.995 | 548 |
| Cash Withdrawal | 1.000 | 1.000 | 1.000 | 4,265 |
| EMI | 1.000 | 1.000 | 1.000 | 612 |
| Self Transfer | 1.000 | 1.000 | 1.000 | 246 |
| Investments | 1.000 | 1.000 | 1.000 | 1,663 |

## Honest notes
* One simulated dataset (150 customers, one seed). Not a real-world accuracy claim.
* **Payee history helps largely because shopkeeper aliases are stable per customer in the simulation** (as on real statements, but the
  size of the gain depends on that choice).
* History features use customers' earlier labels, including labels accrued during the test year, as in production where users label over time.
* Development path, for transparency: stage-1 solvers were compared using test metrics, and the stage-2 collapse was diagnosed on test.
  The final stage-2 feature choices (structured + history, no class weights) were selected on the validation quarter. Treat the test
  numbers as good but not a pristine holdout.
* A faster stage 1 trained on unique (text, label) pairs under-converged (85.9% vs 93.1%), so training uses exact row-level SGD.
* XGBoost is supported (`--stage2 xgboost`) but has not been benchmarked; the numbers above use the scikit-learn fallback.
* `predict.py`'s Postgres write-back is not covered by automated tests (the test suite runs the SQL against SQLite).
* Phase 3 results were re-run after this change and are identical (only descriptions changed).


---

# Phase 5: Health Score, Savings Planner, Explainable Insights (`core/`)

| File | Purpose |
|---|---|
| `health_score.py` | 0-100 score from `v_health_features`: per-component points, top drag, band, peer percentile, one-sentence explanation, "score after a savings plan" projection, writer for `health_scores` |
| `savings_planner.py` | "Save Rs X a month": minimum-lifestyle-pain cuts per category, exact LP solution, plain-language plan |
| `insights.py` | Template-driven insights from the SQL queries: largest category, "why did I spend more?", top discretionary expenses, weekday pattern, seasonality |
| `queries.py` | Loads the named SQL blocks (shared with the future API and chatbot tools) |

```python
from core import health_score, savings_planner, insights
scored = health_score.score(features_df, income_type_series)          # features from v_health_features
plan   = savings_planner.plan(candidates_df, target=10_000)            # candidates from the `savings_candidates` query
print(savings_planner.format_plan(plan))
insights.generate_insights(run, "C0001", "2025-06-01")                 # run = insights.make_runner(engine) or any query runner
```

## Health score

| Component | Weight | 0 points at | Full points at |
|---|---|---|---|
| Savings rate (6-month) | 30 | 0% saved | 20% or more |
| Debt ratio (EMI / income) | 20 | 50% of income | 10% or less |
| Emergency fund (cash + savings / monthly spend) | 20 | 0 months | 6 months or more |
| Expense consistency (CV of monthly spend) | 15 | CV 0.50 | CV 0.10 |
| Income stability (CV of monthly income, 80%, plus an income-type prior, 20%) | 15 | CV 0.50 | CV 0.05 |

Every result carries per-component points, the component losing the most points, and a sentence such as
*"Biggest drag: Emergency fund, because you hold 1.8 months of expenses in cash and savings (target 6)"*.

**Results on the simulated population** (from `docs/core_results.md`): 3,100 customer-months scored (full 6-month history). Total score quantiles: p10 = 53, p25 = 76, p50 = 91, p75 = 96, p90 = 99.

Bands: Excellent 72%, Fair 10%, Good 10%, Needs attention 8%. The population is skewed high because simulated customers save generously; do not read the bands as a real-world distribution.

| Persona | Savings | Debt | Emergency | Consistency | Income | Total |
|---|---|---|---|---|---|---|
| student_parttime | 0.6 | 20.0 | 9.2 | 12.2 | 10.5 | 52.5 |
| retiree | 13.0 | 20.0 | 14.4 | 9.9 | 15.0 | 72.3 |
| freelancer | 20.0 | 19.5 | 16.1 | 12.7 | 9.0 | 77.4 |
| family_single_earner | 22.1 | 14.9 | 15.8 | 13.3 | 13.7 | 79.8 |
| govt_stable | 26.4 | 17.6 | 18.4 | 11.9 | 14.9 | 89.2 |
| high_earner_spender | 29.8 | 18.6 | 19.6 | 11.4 | 13.7 | 93.1 |
| young_techie | 29.0 | 19.1 | 19.1 | 13.2 | 13.4 | 93.8 |
| frugal_saver | 28.8 | 19.9 | 19.7 | 12.6 | 13.2 | 94.3 |

**Validity check against injected ground truth.** 4 job-loss events had enough history to score. Mean score in the 3 months before the loss: 97.8; at the end of the loss window and 2 months after: 66.9. The score fell in 4 of 4 events. This is a very small sample, so treat it as directional evidence only.

## Savings planner
Minimises total *lifestyle pain* subject to cuts adding up to the target. With one linear constraint the LP is solved exactly by taking the
cheapest-pain rupees first; `tests/test_core.py` compares the planner's objective with scipy's `linprog` on random inputs. Rules:
essentials such as rent, EMI, insurance, education and health are never cut; groceries, transport, fuel and utilities can only be trimmed 10%;
discretionary categories 15-50% with different pain per rupee; and spending *above the customer's own 6-month normal* is cheapest to cut.
If the target is unreachable the plan says how much is possible and what is left. `score_impact` shows the health score before and after.

## Insights
Each insight is a template filled from a named SQL query, so every number traces back to data. "Why did I spend more?" separates habit changes
(with a one-off purchase detector) from fixed or periodic bills, and hedges its explanations for those bills ("likely a rent revision").
A complete worked example is in `docs/core_results.md`.

## Honest notes
* **The score's anchors and weights are design choices** (taken from the project spec and common rules of thumb), not fitted to or validated
  against real financial outcomes. The job-loss check is a handful of events.
* **The simulated population skews high** (most customers are "Excellent") because simulated customers save generously. The peer percentile
  stays informative when raw scores bunch at the top.
* **Planner caps and pain weights are assumptions**, not measured behaviour. The planner is exact for its own model, which is a different claim
  from "these cuts are easy for real people".
* **Insight heuristics** (e.g. a purchase is "one-off" if it is at least 60% of the increase and 2.5x the typical ticket) are fixed thresholds that were
  not tuned. Explanations for fixed bills are inferences, hence "likely".
* In the worked example the story leads with a one-off Shopping purchase and does not mention the injected Dining Out ramp, because that
  ramp is smaller than other movers that month. That is an accurate reading of the data, not a miss, but it shows the story ranks by rupee impact.
* Automated tests exercise the SQL through the SQLite harness; `insights.make_runner` and the Streamlit app also run against PostgreSQL (Docker).
* Test status for the whole repo is summarised under Phase 6.


---

# Phase 6: Chatbot, Tool Layer and API (`core/service.py`, `llm/`, `api/`)

```
question -> agent loop -> LLM client -> tool call -> validator -> service -> SQL views / models -> JSON -> answer -> grounding check
```

| File | Purpose |
|---|---|
| `core/service.py` | `AdvisorService`: every question as a method (top categories, category total, month comparison, why-spending-changed, alerts, health score, savings plan, insights). Database-agnostic (takes a `run(query, **params)` callable), scoped to one customer |
| `llm/tools.py` | 8 tool definitions (Anthropic tool-use format), strict argument validation, result sanitising, never-raising executor |
| `llm/agent.py` | Vendor-neutral tool-calling loop (step cap, parallel-call cap, error feedback to the model, grounding check on the final answer) |
| `llm/grounding.py` | Flags rupee amounts / percentages in an answer that no tool result supports |
| `llm/rulebased.py` | **Offline demo client (not an LLM)**: routes the spec's example questions to the same tools, no API key or cost |
| `llm/clients.py` | `AnthropicClient` adapter (model via `ADVISOR_MODEL`, default `claude-haiku-4-5-20251001`) |
| `llm/prompts.py`, `llm/testing.py` | System prompt; scripted fake models for tests |
| `api/handlers.py`, `api/main.py`, `api/schemas.py` | Framework-free handlers (tested) + thin FastAPI routing (HTTP tests with `TestClient`) |

```bash
uvicorn api.main:app --port 8000          # needs DATABASE_URL and JWT_SECRET; optional ANTHROPIC_API_KEY
TOKEN=$(curl -s -X POST localhost:8000/auth/demo | python -c "import sys, json; print(json.load(sys.stdin)['access_token'])")   # demo login (DEMO_CUSTOMER_ID)
curl -X POST localhost:8000/chat -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
     -d '{"message": "Why did I spend more this month?"}'
```
Endpoints: `GET /health`; `POST /auth/register | login | demo | verify-email | resend-verification | forgot-password | reset-password`; with a bearer token
(always the caller's own data): `GET /me`, `GET /me/summary | categories | alerts | health-score | insights`, `POST /me/savings-plan`, `POST /chat`, `POST /me/delete`.
There is no customer id in any URL or request body.
Without `ANTHROPIC_API_KEY` the chat uses the offline rule-based client. A full worked transcript is in `docs/chatbot_demo.md`; two examples:

**Where did I spend the most last month?**  
`tools:` `get_top_categories(month='last_month')` | `grounding:` all figures verified  
> Your largest spending category in November 2025 was Rent at Rs 8,342 (39% of the month's spending across 2 transactions). Next: Utilities (Rs 2,653), Cash Withdrawal (Rs 2,500), Groceries (Rs 2,306). Total spending that month was Rs 21,153.

**Why did I spend more this month?**  
`tools:` `explain_spending_change(month='this_month')` | `grounding:` all figures verified  
> You spent Rs 29,969 in December 2025, Rs 12,737 more than your usual Rs 17,233 (74%). The biggest increases in your habits: Travel (+Rs 6,187, mostly one Rs 4,234 purchase at Makemytrip); Groceries (+Rs 2,675, from more or larger purchases than usual); Dining Out (+Rs 2,359, mostly one Rs 1,482 purchase at Burger King). Fixed and periodic bills added another Rs 893 (likely a rent revision; a plan or bill change), which is not a change in your habits.

## Design decisions worth knowing
* **The customer id is never a tool argument.** It is bound to the service when the session starts, so a model (or an instruction hidden in a
  transaction description) cannot ask for someone else's data. Unknown arguments are rejected, and a test asserts no tool schema mentions a customer.
* **Numbers come from tools, and the answer is checked.** The grounding checker compares every rupee amount and percentage in the final answer
  with the tool results (and the user's own message). Unverified figures are listed and a visible note is appended. It is a heuristic: it ignores
  counts and dates, accepts x12 annualised figures, and cannot tell a wrong calculation from a right one.
* **"This month" means the latest month in the data**, reported as `as_of` in every result (the dataset is a simulation ending Dec 2025).
* **"Usual" spending ignores one-off spikes** but is otherwise a plain mean. An earlier version trimmed the largest month unconditionally, which biased the
  baseline downward and exaggerated "more than usual"; it now drops that month only when it is a clear outlier.
* **Fixed or periodic bills are reported separately from habit changes** ("likely a rent revision").
* Errors never reach the model as stack traces: tool exceptions become a generic message.

## What the tests cover
Service numbers cross-checked against independent pandas calculations; customer scoping; argument validation and injection attempts
(`customer_id` argument, control characters, over-long strings); a scripted fake model that hallucinates a number, calls a non-existent tool,
loops forever, or fires six parallel calls; session isolation between customers; and 17 phrasings of the example questions through the offline router.

## Honest limits
* **No real LLM has been run.** Chat uses the offline rule-based client unless `ANTHROPIC_API_KEY` is set. The agent loop is tested with scripted models and the offline client, which
  verifies the *harness* (protocol, validation, limits, grounding), not how well a real model chooses tools or words answers. `AnthropicClient` is untested.
* **The FastAPI app is covered at two levels:** behaviour through the framework-free handlers, and the routes through HTTP-level tests with `TestClient` (e.g. `tests/test_auth_api.py`).
* **Security (Phase 9):** the shared API key was replaced by per-user login (signed tokens, server-side web sessions), per-account and per-IP rate limiting stored in PostgreSQL,
  and optional row-level security. See Phase 9 for what is and is not verified.
* **Prompt-injection defences are partial** (scoping, validation, sanitising, a system-prompt rule). They reduce risk; they do not eliminate it.
* The rule-based client understands only the question shapes it was written for; it is a demo and test fixture, not a chatbot.
* Cost and latency of the real model path were not measured.
* Category labels in the simulation are deliberately noisy, so answers can show oddities such as two rent transactions in a month.


---

# Phase 7: Spending Forecast (`ml/forecasting/`)

Predicts each customer's **total spending next month** with an empirical 80% range, and compares models with a leak-safe rolling-origin backtest.

| File | Purpose |
|---|---|
| `models.py` | Forecasters: naive last month, 3/6-month mean, 6-month median, seasonal naive, simple exponential smoothing, **global gradient boosting** (L1 and L2), plus ARIMA, Prophet and an LSTM (optional) |
| `backtest.py` | Rolling-origin backtest (MAE, WAPE, sMAPE, MASE, win-rate), paired bootstrap CIs over customers, interval-coverage check |
| `compare.py` | CLI that runs everything and writes `docs/forecast_results.md` |
| `run_forecast.py` | Batch job: next-month forecast + calibrated interval per customer -> `forecasts` table (or CSV) |

Design: every model has one method, `forecast(Y, months, meta)`, and only ever receives history up to the forecast origin, so it cannot see the future;
a test corrupts later months and asserts that no earlier prediction changes. The gradient-boosting model is **one model for all customers**, predicting next
month as a ratio of the recent 6-month level from lags, rolling statistics, month of year and income. Pooling customers is what lets a 36-point series
benefit from seasonality seen elsewhere. The chatbot tool `get_spending_forecast` reads the `forecasts` table ("How much will I spend next month?").

## Results (150 simulated customers, 12 rolling origins, seed 42)

| model          |     MAE |   WAPE |   sMAPE |   MASE |   win_vs_naive |   series |   origins |   seconds |
|:---------------|--------:|-------:|--------:|-------:|---------------:|---------:|----------:|----------:|
| gbm_l1         | 11564.5 |  0.138 |   0.131 |  0.816 |          0.657 |      150 |        12 |     3.391 |
| gbm_l2         | 12054.6 |  0.144 |   0.136 |  0.855 |          0.624 |      150 |        12 |     3.465 |
| median6        | 12400.2 |  0.148 |   0.144 |  0.889 |          0.637 |      150 |        12 |     0.002 |
| mean6          | 12509.6 |  0.15  |   0.145 |  0.889 |          0.641 |      150 |        12 |     0.001 |
| ses            | 12936.7 |  0.155 |   0.148 |  0.914 |          0.639 |      150 |        12 |     0.007 |
| mean3          | 13655.1 |  0.163 |   0.156 |  0.957 |          0.609 |      150 |        12 |     0.002 |
| naive_last     | 16465.7 |  0.197 |   0.191 |  1.178 |          0     |      150 |        12 |     0.003 |
| seasonal_naive | 16666.1 |  0.199 |   0.185 |  1.106 |          0.517 |      150 |        12 |     0.001 |

WAPE = total absolute error / total actual spend; MASE below 1 beats last-month-as-forecast. Difference in WAPE against the **strongest simple baseline**
(`median6`), with a 95% bootstrap CI resampling customers (negative is better):

```
         model baseline  wape_diff  ci_low  ci_high  significant
        gbm_l1  median6    -0.0100 -0.0146  -0.0056         True
        gbm_l2  median6    -0.0041 -0.0101   0.0020        False
         mean6  median6     0.0013 -0.0027   0.0057        False
           ses  median6     0.0064  0.0010   0.0126         True
         mean3  median6     0.0150  0.0090   0.0210         True
    naive_last  median6     0.0486  0.0404   0.0575         True
seasonal_naive  median6     0.0510  0.0367   0.0671         True
```

Prediction interval for the best model, built from the first half of the origins and measured on the second half:

```
 model  target  rel_low  rel_high  coverage  n_test
gbm_l1     0.8   -0.169     0.257     0.794     900
```

## Honest reading
* **Machine learning helps, modestly.** Global gradient boosting (L1) beats the best simple baseline by about **1.0 percentage point of WAPE
  (14.8% to 13.8%, roughly a 7% relative improvement)** and the CI excludes zero. Against last-month-as-forecast the gain is larger (19.7% to 13.8%), but that
  baseline is weak here, so comparing only to it would flatter the models.
* **The L2 variant is not significantly better than the median.** Squared loss chases the lumpy months (medical bills, trips); the L1 loss is more robust.
* **Seasonal naive is no better than naive** with only two years of history, since one noisy value from a year ago is a poor forecast.
* **The 80% interval is well calibrated** on held-out origins (79.4% coverage).
* **Limits:** simulated data with a seasonality structure built into the simulator (festival and holiday multipliers); one seed; total spend only, not per category;
  the interval is a pooled relative range, so it does not widen for customers with more volatile spending.
* **ARIMA, Prophet and LSTM are not in the table.** They are implemented but have not been run (optional libraries not installed). A *hypothesis*, not a
  result: with 36 monthly points per customer, per-series ARIMA/Prophet will not beat the median or the pooled GBM, and the LSTM will land near the GBM at much higher cost.
  Running `python -m ml.forecasting.compare --max-series 30 --origins 6` would test it. Note that comparisons are restricted to customers all
  models share, so including the slow per-series models shrinks every row to that subset; run once with `--no-optional` for the full 150-customer table.


---

# Phase 8: Streamlit App (`app/`)

```bash
streamlit run app/streamlit_app.py          # needs the database loaded (see Quickstart); open http://localhost:8501
```

| File | Purpose |
|---|---|
| `streamlit_app.py` | Layout only: login screen, sidebar (account, month) and seven tabs |
| `data_access.py` | Framework-free queries -> clean DataFrames (handles PostgreSQL `Decimal` / `date` types), slider bounds, model-lab file loader |
| `charts.py` | Plotly figures: spending trend with forecast and range, category donut and stack, weekday and seasonal charts, health gauge and components |
| `chat.py` | One chat turn: the same agent, tools and grounding check as the API, called in-process |

**Tabs:** *Overview* (health score, this month vs usual, next-month forecast with range, alert count, trend chart, category donut, "why did spending change?"),
*Spending* (12-month category stack, weekday and seasonal patterns, top discretionary merchants, largest transactions), *Health & savings* (score gauge and components,
interactive savings planner with the plan, per-category cuts and score impact), *Alerts & insights*, *Chat* (sample-question buttons, tools used per answer,
unverified-figure warnings), *Import statement* (upload a CSV bank statement), and *Model lab* (the backtest, detector and categoriser results from `docs/`, including the confusion matrix).

**Resilience:** if a table has not been populated yet (alerts, forecasts, health inputs) the matching card shows `n/a` with the reason instead of crashing, and if
the database is unreachable the app shows the exact commands to start and load it. Both paths are tested.

**Choices worth knowing**
* `streamlit run app/streamlit_app.py` puts `app/`, not the repo root, on `sys.path`; the script adds the root itself so `import core` works.
* Chart width: Streamlit renamed this argument between versions, so the app inspects `st.plotly_chart` and uses whichever argument exists.
* Query results are cached for 10 minutes (`st.cache_data`). After re-running the alerts / forecast jobs, use *Rerun* with *Clear cache* (press `C`) or wait.
* The forecast and savings plan always use the most recent data, whichever month is selected; the savings tab says so.
* Chat uses the offline rule-based client unless `ANTHROPIC_API_KEY` is set (the sidebar shows which).

**How it was verified, and the limits**
* Data functions, charts and chat run under both SQLite types and PostgreSQL-style types (`Decimal`, `date`). The whole script runs end to end against a recording fake of
  Streamlit and Plotly: 7 charts render, metrics are correct, the chat flow and the sample-question button work, missing tables degrade gracefully, and a dead database
  shows the help message. Writing these tests found three real bugs early (a `Timestamp` passed where a string was required, a misleading metric colour, and a slider
  default that could fall outside its range for low incomes).
* **Real Streamlit and Plotly:** the app has been run in a browser with Streamlit 1.58 and Plotly 6.7, and
  `tests/test_app.py::test_real_streamlit_app_starts_and_reports_unreachable_database` uses Streamlit's own `AppTest`.
* **Access:** the app now requires a login (Phase 9). The old customer picker, which listed every customer, was removed.
* No Power BI dashboard yet; the SQL views (`v_monthly_summary`, `v_category_month_grid`, ...) are the natural source for one.


---

# Phase 9: Login, privacy, statement import and deployment (`core/auth.py`, `core/statements.py`, `deploy/`)

Turns the demo into something a real person could sign up for: accounts, email verification, password reset, per-user data isolation, a way to bring your own
bank statement, and a production setup for an Ubuntu server.

**What it adds**
* **Accounts:** sign-up with an 18+ confirmation and a recorded privacy consent, scrypt password hashing, email verification, password reset, a shared demo login for visitors.
* **Sessions:** API users get short-lived signed tokens (HS256, `exp` required). The Streamlit app keeps a server-side session behind a cookie, so a browser refresh keeps you logged in and logout / password reset really end it.
* **Abuse protection:** per-account login lockout and per-IP limits, stored in PostgreSQL so they hold across containers. Chat has per-user hourly limits.
* **Isolation:** the customer always comes from the verified login, never from a URL, request body or chat message. The old customer picker and the `customer_list` query were removed, and a test fails if someone adds an unscoped query, a chat tool with a customer argument, or a route with a customer id.
  Optional PostgreSQL row-level security (`sql/rls.sql`) adds a second wall inside the database.
* **Statement import:** upload a CSV statement from an Indian bank (preamble lines, separate debit / credit columns or Dr/Cr, dd/mm/yy dates, newest-first files). Long account / reference numbers are masked, duplicates are skipped, and a first-pass categoriser fills the categories.
* **Privacy controls:** consent records, withdraw consent, delete imported transactions, delete the whole account (`docs/privacy_notice_template.md` is a template, not legal advice).
* **Deployment:** `docker-compose.prod.yml` (Postgres not exposed, API, Streamlit app, Caddy with automatic HTTPS), backup and nightly-job scripts, server hardening. Step by step in `docs/DEPLOY.md`.

**Security properties and how each is checked**
| Property | How |
|---|---|
| Unknown email and wrong password look identical, including timing | Same message; a dummy hash is verified for unknown emails. Measured ~257 ms vs ~274 ms on the test machine |
| Sign-up does not reveal which emails exist | Same response for new and taken addresses; the owner of a taken address gets a different email (test) |
| Login attempts throttled per account and per IP | 5 failures / 15 min per account, 20 / 15 min per IP (tests) |
| JWT pins HS256 and requires `exp`; forged, expired and `alg: none` tokens are refused | Tests |
| A password reset invalidates old tokens and sessions | Token version checked on every request; web sessions deleted (tests) |
| Verification and reset links are single-use and expire | 256-bit random, only the hash stored, atomic delete when used, 24 h / 30 min (tests) |
| Reset emails do not reveal whether an account exists | Identical response; mail is sent in a background thread (tests) |
| One customer cannot read another customer's data or names | Customer comes from the token / session; static tests; optional row-level security with a test against a real PostgreSQL |

**Tests:** 152 tests across auth, flows, statement import, the HTTP API and isolation pass (`python -m pytest tests/test_auth.py tests/test_auth_flows.py tests/test_statements.py tests/test_auth_api.py tests/test_isolation.py`).
The flow (sign-up, verification, login, refresh, import, duplicate check, lockout, reset, delete) was also click-tested by hand in a real browser against Streamlit 1.58.

**Honest limits**
* Row-level security is optional and off by default; `tests/test_rls_postgres.py` proves it only when run against a real PostgreSQL with the role applied.
* The Docker production stack (Caddy, HTTPS, SMTP email) has not been run end to end yet; email was only tried with the console backend.
* The login cookie is set from the browser (Streamlit cannot set HttpOnly cookies) and is best-effort: if it is blocked, a refresh just asks for a login again.
* Imported categories come from keyword rules, not the ML categoriser (which was trained on synthetic descriptions and may transfer poorly to real statements). Only CSV statements are supported; salary detection is a heuristic.
* Password-reset and verification tokens travel in the link URL, and the throttle checks and records in two steps, so a burst of simultaneous requests can slightly exceed a limit.
* The privacy notice is a template. Have it reviewed (India's DPDP Rules are being phased in) before inviting real users, and keep the public demo on synthetic data.
