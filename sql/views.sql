-- =====================================================================
-- Analytics views (PostgreSQL). Everything downstream reads from these:
-- dashboards, forecasting, alerts, health score, the LLM tools.
--
-- Conventions
--   * "Spending" = debits whose category is_expense (excludes investments,
--     self-transfers, sweeps). EMI is included.
--   * Uncategorised debits are kept as category 'Uncategorized' until the ML
--     categoriser fills predicted_category.  final category =
--     COALESCE(category, predicted_category, 'Uncategorized')
--   * Written to stay portable: no generate_series, no STDDEV in windows
--     (variance is derived from first/second moments), so the same file is
--     smoke-tested under SQLite in tests/.
-- =====================================================================

DROP VIEW IF EXISTS v_health_features;
DROP VIEW IF EXISTS v_overspend_alerts;
DROP VIEW IF EXISTS v_category_baseline;
DROP VIEW IF EXISTS v_monthly_summary;
DROP VIEW IF EXISTS v_category_month_grid;
DROP VIEW IF EXISTS v_monthly_category_spend;
DROP VIEW IF EXISTS v_month_spine;
DROP VIEW IF EXISTS v_expense_txn;

-- 1. Clean expense ledger ---------------------------------------------
CREATE OR REPLACE VIEW v_expense_txn AS
SELECT t.txn_id,
       t.customer_id,
       t.txn_date,
       t.txn_ts,
       date_trunc('month', t.txn_date)::date               AS month,
       t.amount,
       t.merchant,
       t.channel,
       t.raw_description,
       COALESCE(t.category, t.predicted_category, 'Uncategorized') AS category,
       t.true_category,                                     -- evaluation only
       COALESCE(d.is_essential, FALSE)                      AS is_essential
FROM transactions t
LEFT JOIN dim_category d ON d.category = COALESCE(t.category, t.predicted_category)
WHERE t.type = 'debit'
  AND COALESCE(d.is_expense, TRUE);

-- 2. Calendar spine (months present in the data) --------------------------
CREATE OR REPLACE VIEW v_month_spine AS
SELECT DISTINCT date_trunc('month', txn_date)::date AS month
FROM transactions;

-- 3. Spend per customer / month / category ---------------------------------
CREATE OR REPLACE VIEW v_monthly_category_spend AS
SELECT customer_id, month, category,
       SUM(amount) AS spend,
       COUNT(*)    AS txn_count
FROM v_expense_txn
GROUP BY customer_id, month, category;

-- 4. Dense grid: zero-fills months with no spend so rolling stats are honest -----
CREATE OR REPLACE VIEW v_category_month_grid AS
SELECT p.customer_id,
       p.category,
       m.month,
       COALESCE(s.spend, 0)     AS spend,
       COALESCE(s.txn_count, 0) AS txn_count
FROM (SELECT DISTINCT customer_id, category FROM v_monthly_category_spend) p
CROSS JOIN v_month_spine m
LEFT JOIN v_monthly_category_spend s
       ON s.customer_id = p.customer_id AND s.category = p.category AND s.month = m.month;

-- 5. Customer-month summary: income, expenses, balances, reserve --------------
CREATE OR REPLACE VIEW v_monthly_summary AS
WITH income AS (
    SELECT t.customer_id, date_trunc('month', t.txn_date)::date AS month, SUM(t.amount) AS income
    FROM transactions t
    JOIN dim_category d ON d.category = t.category
    WHERE t.type = 'credit' AND d.is_primary_income
    GROUP BY 1, 2
), expense AS (
    SELECT customer_id, month,
           SUM(amount) AS expenses,
           SUM(CASE WHEN category = 'EMI' THEN amount ELSE 0 END) AS emi,
           SUM(CASE WHEN NOT is_essential AND category <> 'Uncategorized' THEN amount ELSE 0 END) AS discretionary
    FROM v_expense_txn
    GROUP BY customer_id, month
), invest AS (
    SELECT customer_id, date_trunc('month', txn_date)::date AS month, SUM(amount) AS investments
    FROM transactions
    WHERE type = 'debit' AND category = 'Investments'
    GROUP BY 1, 2
), xfer AS (   -- net money moved INTO savings reserve (sweeps) minus pulled OUT (top-ups)
    SELECT customer_id, date_trunc('month', txn_date)::date AS month,
           SUM(CASE WHEN type = 'debit' THEN amount ELSE -amount END) AS net_to_reserve
    FROM transactions
    WHERE category = 'Self Transfer'
    GROUP BY 1, 2
), month_end AS (
    SELECT customer_id, month, balance_after AS month_end_balance
    FROM (SELECT customer_id,
                 date_trunc('month', txn_date)::date AS month,
                 balance_after,
                 ROW_NUMBER() OVER (PARTITION BY customer_id, date_trunc('month', txn_date)
                                    ORDER BY txn_ts DESC, txn_id DESC) AS rn
          FROM transactions) x
    WHERE rn = 1
)
SELECT me.customer_id,
       me.month,
       COALESCE(i.income, 0)          AS income,
       COALESCE(e.expenses, 0)        AS expenses,
       COALESCE(e.emi, 0)             AS emi,
       COALESCE(e.discretionary, 0)   AS discretionary,
       COALESCE(v.investments, 0)     AS investments,
       me.month_end_balance,
       a.opening_reserve + SUM(COALESCE(x.net_to_reserve, 0))
           OVER (PARTITION BY me.customer_id ORDER BY me.month) AS reserve_balance,
       CASE WHEN COALESCE(i.income, 0) > 0
            THEN (i.income - COALESCE(e.expenses, 0)) / i.income END AS savings_rate
FROM month_end me
JOIN accounts a ON a.customer_id = me.customer_id
LEFT JOIN income  i ON i.customer_id = me.customer_id AND i.month = me.month
LEFT JOIN expense e ON e.customer_id = me.customer_id AND e.month = me.month
LEFT JOIN invest  v ON v.customer_id = me.customer_id AND v.month = me.month
LEFT JOIN xfer    x ON x.customer_id = me.customer_id AND x.month = me.month;

-- 6. Rolling 6-month baseline per category (EXCLUDES the current month) -------
CREATE OR REPLACE VIEW v_category_baseline AS
WITH w AS (
    SELECT g.customer_id, g.category, g.month, g.spend, g.txn_count,
           COUNT(*)           OVER win AS n_hist,
           AVG(g.spend)       OVER win AS m1,
           AVG(g.spend * g.spend) OVER win AS m2
    FROM v_category_month_grid g
    WINDOW win AS (PARTITION BY g.customer_id, g.category ORDER BY g.month
                   ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING)
)
SELECT customer_id, category, month, spend, txn_count, n_hist,
       m1 AS baseline_mean,
       CASE WHEN n_hist > 1 THEN
            SQRT(CASE WHEN m2 - m1 * m1 > 0 THEN (m2 - m1 * m1) * n_hist / (n_hist - 1) ELSE 0 END)
       END AS baseline_std,
       CASE WHEN m1 > 0 THEN spend / m1 - 1 END AS pct_vs_baseline,
       CASE WHEN n_hist > 1 AND m2 - m1 * m1 > 0
            THEN (spend - m1) / SQRT((m2 - m1 * m1) * n_hist / (n_hist - 1)) END AS z_score
FROM w;

-- 7. Rule-based overspend alerts (benchmark detector; thresholds are tunable) -----
--    needs a full 6-month history, >=25% above baseline, z>=2, and >= Rs 500 materiality
CREATE OR REPLACE VIEW v_overspend_alerts AS
SELECT customer_id, category, month, spend, baseline_mean, baseline_std, pct_vs_baseline, z_score
FROM v_category_baseline
WHERE n_hist = 6
  AND category <> 'Uncategorized'
  AND baseline_mean > 0
  AND pct_vs_baseline >= 0.25
  AND z_score >= 2
  AND spend - baseline_mean >= 500;

-- 8. Health-score inputs (trailing 6-month windows) --------------------------
CREATE OR REPLACE VIEW v_health_features AS
WITH w AS (
    SELECT s.*,
           COUNT(*) OVER win AS n6,
           AVG(income)   OVER win AS inc_m1,  AVG(income * income)     OVER win AS inc_m2,
           AVG(expenses) OVER win AS exp_m1,  AVG(expenses * expenses) OVER win AS exp_m2,
           SUM(income)   OVER win AS inc_sum, SUM(expenses) OVER win AS exp_sum
    FROM v_monthly_summary s
    WINDOW win AS (PARTITION BY s.customer_id ORDER BY s.month ROWS BETWEEN 5 PRECEDING AND CURRENT ROW)
)
SELECT customer_id, month, n6,
       income, expenses, emi, savings_rate,
       month_end_balance, reserve_balance,
       inc_m1 AS avg_income_6m,
       exp_m1 AS avg_expense_6m,
       CASE WHEN inc_sum > 0 THEN (inc_sum - exp_sum) / inc_sum END AS savings_rate_6m,
       CASE WHEN inc_m1 > 0 THEN emi / inc_m1 END                  AS debt_ratio,
       CASE WHEN exp_m1 > 0 THEN (month_end_balance + reserve_balance) / exp_m1 END AS emergency_months,
       CASE WHEN n6 > 1 AND exp_m1 > 0 THEN
            SQRT(CASE WHEN exp_m2 - exp_m1 * exp_m1 > 0 THEN (exp_m2 - exp_m1 * exp_m1) * n6 / (n6 - 1) ELSE 0 END) / exp_m1
       END AS expense_cv_6m,
       CASE WHEN n6 > 1 AND inc_m1 > 0 THEN
            SQRT(CASE WHEN inc_m2 - inc_m1 * inc_m1 > 0 THEN (inc_m2 - inc_m1 * inc_m1) * n6 / (n6 - 1) ELSE 0 END) / inc_m1
       END AS income_cv_6m
FROM w;
