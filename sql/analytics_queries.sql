-- =====================================================================
-- Named, parametrised analytics queries (SQLAlchemy `text()` bind style).
-- Loaded by core/queries.py; reused by FastAPI endpoints and LLM tools.
-- Months are passed as the FIRST day of the month (e.g. 2025-11-01).
-- =====================================================================

-- name: top_categories_month
-- Where did I spend the most in a given month?
SELECT category,
       SUM(amount)  AS spend,
       COUNT(*)     AS txns,
       ROUND(100.0 * SUM(amount) / SUM(SUM(amount)) OVER (), 1) AS share_pct
FROM v_expense_txn
WHERE customer_id = :cid AND month = :month
GROUP BY category
ORDER BY spend DESC;

-- name: category_total
-- How much did I spend on <category> between :start (inclusive) and :end (exclusive)?
SELECT category, SUM(amount) AS spend, COUNT(*) AS txns
FROM v_expense_txn
WHERE customer_id = :cid AND LOWER(category) = LOWER(:category)
  AND txn_date >= :start AND txn_date < :end
GROUP BY category;

-- name: month_compare
-- Category-level difference between two months (m2 minus m1), biggest movers first.
SELECT category,
       SUM(CASE WHEN month = :m1 THEN amount ELSE 0 END) AS m1_spend,
       SUM(CASE WHEN month = :m2 THEN amount ELSE 0 END) AS m2_spend,
       SUM(CASE WHEN month = :m2 THEN amount ELSE 0 END)
     - SUM(CASE WHEN month = :m1 THEN amount ELSE 0 END) AS diff
FROM v_expense_txn
WHERE customer_id = :cid AND month IN (:m1, :m2)
GROUP BY category
ORDER BY ABS(SUM(CASE WHEN month = :m2 THEN amount ELSE 0 END)
           - SUM(CASE WHEN month = :m1 THEN amount ELSE 0 END)) DESC;

-- name: weekday_pattern
-- Spending by weekday (1 = Monday ... 7 = Sunday) over a period.
SELECT EXTRACT(isodow FROM txn_date)  AS isodow,
       SUM(amount)                    AS spend,
       COUNT(*)                       AS txns,
       ROUND(100.0 * SUM(amount) / SUM(SUM(amount)) OVER (), 1) AS share_pct,
       ROUND(AVG(amount), 0)          AS avg_ticket
FROM v_expense_txn
WHERE customer_id = :cid AND txn_date >= :start AND txn_date < :end
  AND category NOT IN ('Rent', 'EMI', 'Insurance', 'Education', 'Mobile & Internet', 'Subscriptions', 'Health')
GROUP BY EXTRACT(isodow FROM txn_date)
ORDER BY isodow;

-- name: seasonal_index
-- Average monthly spend by calendar month vs the customer's overall average (1.0 = normal).
WITH m AS (
    SELECT EXTRACT(month FROM month) AS moy, AVG(expenses) AS avg_expenses, AVG(discretionary) AS avg_discretionary
    FROM v_monthly_summary
    WHERE customer_id = :cid
    GROUP BY EXTRACT(month FROM month)
)
SELECT moy,
       ROUND(avg_expenses, 0)      AS avg_expenses,
       ROUND(avg_discretionary, 0) AS avg_discretionary,
       ROUND(avg_expenses      / AVG(avg_expenses)      OVER (), 2) AS expense_index,
       ROUND(avg_discretionary / AVG(avg_discretionary) OVER (), 2) AS discretionary_index
FROM m
ORDER BY moy;

-- name: top_unnecessary_spend
-- Top merchants inside NON-essential categories (candidate "unnecessary" expenses).
SELECT merchant, category, SUM(amount) AS spend, COUNT(*) AS txns
FROM v_expense_txn
WHERE customer_id = :cid AND NOT is_essential AND category <> 'Uncategorized'
  AND txn_date >= :start AND txn_date < :end
GROUP BY merchant, category
ORDER BY spend DESC
LIMIT :limit;

-- name: merchant_concentration
-- Share of total spend going to the top merchants.
SELECT merchant,
       SUM(amount) AS spend,
       ROUND(100.0 * SUM(amount) / SUM(SUM(amount)) OVER (), 1) AS share_pct
FROM v_expense_txn
WHERE customer_id = :cid AND txn_date >= :start AND txn_date < :end
  AND category NOT IN ('Rent', 'EMI', 'Cash Withdrawal', 'Transfers')
GROUP BY merchant
ORDER BY spend DESC
LIMIT :limit;

-- name: spend_drivers
-- Mean-based baseline (kept for reference). Prefer category_history + insights.robust_drivers, which ignores one-off spikes.
SELECT category, spend, ROUND(baseline_mean, 0) AS baseline_mean,
       ROUND(spend - baseline_mean, 0)  AS delta,
       ROUND(100.0 * pct_vs_baseline, 1) AS pct_vs_baseline,
       ROUND(z_score, 2)                AS z_score
FROM v_category_baseline
WHERE customer_id = :cid AND month = :month AND n_hist >= 3
  AND category <> 'Uncategorized'
ORDER BY (spend - baseline_mean) DESC;

-- name: largest_txns_in_category
-- Drill-down for a driver category: biggest transactions vs the typical ticket (one-off detection).
SELECT t.txn_date, t.merchant, t.amount, t.channel,
       ROUND((SELECT AVG(h.amount) FROM v_expense_txn h
              WHERE h.customer_id = t.customer_id AND h.category = t.category AND h.month < :month), 0) AS typical_ticket
FROM v_expense_txn t
WHERE t.customer_id = :cid AND t.month = :month AND t.category = :category
ORDER BY t.amount DESC
LIMIT :limit;

-- name: savings_candidates
-- Average of the last 3 months per category plus the customer's normal level (6-month baseline): raw material for the savings planner.
WITH ranked AS (
    SELECT category, month, spend,
           ROW_NUMBER() OVER (PARTITION BY category ORDER BY month DESC) AS rn
    FROM v_category_month_grid
    WHERE customer_id = :cid AND month <= :month
), recent AS (
    SELECT category, AVG(spend) AS avg_3m_spend FROM ranked WHERE rn <= 3 GROUP BY category
)
SELECT r.category,
       COALESCE(d.is_essential, FALSE) AS is_essential,
       ROUND(r.avg_3m_spend, 0)        AS avg_3m_spend,
       ROUND(b.baseline_mean, 0)       AS baseline_mean
FROM recent r
LEFT JOIN dim_category d ON d.category = r.category
LEFT JOIN v_category_baseline b ON b.customer_id = :cid AND b.category = r.category AND b.month = :month
WHERE r.category <> 'Uncategorized' AND r.avg_3m_spend > 0
ORDER BY r.avg_3m_spend DESC;

-- name: health_inputs
-- Inputs for the 0-100 financial health score.
SELECT * FROM v_health_features WHERE customer_id = :cid AND month = :month;

-- name: alerts_for_month
SELECT category, spend, ROUND(baseline_mean, 0) AS baseline_mean,
       ROUND(100.0 * pct_vs_baseline, 1) AS pct_vs_baseline, ROUND(z_score, 2) AS z_score
FROM v_overspend_alerts
WHERE customer_id = :cid AND month = :month
ORDER BY (spend - baseline_mean) DESC;

-- name: detector_precision
-- Evaluation vs injected ground truth: of all alerts, how many fall inside a real event window?
SELECT COUNT(*) AS n_alerts,
       SUM(CASE WHEN EXISTS (
            SELECT 1 FROM injected_events e
            WHERE e.customer_id = a.customer_id AND e.category = a.category
              AND e.event_type IN ('category_creep', 'category_spike', 'medical_emergency')
              AND a.month BETWEEN e.start_month AND e.end_month) THEN 1 ELSE 0 END) AS n_true_alerts
FROM v_overspend_alerts a;

-- name: detector_recall
-- Evaluation vs injected ground truth: share of injected events with >=1 alert inside their window.
SELECT e.event_type,
       COUNT(*) AS n_events,
       SUM(CASE WHEN EXISTS (
            SELECT 1 FROM v_overspend_alerts a
            WHERE a.customer_id = e.customer_id AND a.category = e.category
              AND a.month BETWEEN e.start_month AND e.end_month) THEN 1 ELSE 0 END) AS n_detected
FROM injected_events e
WHERE e.category IS NOT NULL
GROUP BY e.event_type
ORDER BY e.event_type;

-- name: false_positive_breakdown
-- Where do the "false" alerts come from? (seasonality / festivals are expected culprits)
SELECT a.category, EXTRACT(month FROM a.month) AS moy, COUNT(*) AS n_alerts
FROM v_overspend_alerts a
WHERE NOT EXISTS (
    SELECT 1 FROM injected_events e
    WHERE e.customer_id = a.customer_id AND e.category = a.category
      AND e.event_type IN ('category_creep', 'category_spike', 'medical_emergency')
      AND a.month BETWEEN e.start_month AND e.end_month)
GROUP BY a.category, EXTRACT(month FROM a.month)
ORDER BY n_alerts DESC
LIMIT 15;

-- name: month_range
-- First and last month present in the data (the chatbot's notion of "now").
SELECT MIN(month) AS first_month, MAX(month) AS last_month FROM v_month_spine;

-- name: expense_categories
SELECT category FROM dim_category WHERE is_expense ORDER BY category;

-- name: customer_profile
SELECT customer_id, name, age, city, city_tier, occupation, income_type, monthly_income, dependents, has_loan
FROM customers WHERE customer_id = :cid;

-- name: alerts_table_for_month
-- Alerts written by the improved detector (ml/anomaly/run_detector.py).
SELECT category, alert_type, current_spend, baseline_mean, pct_change, z_score, message
FROM alerts
WHERE customer_id = :cid AND alert_month = :month
ORDER BY (current_spend - baseline_mean) DESC;

-- name: category_history
-- Monthly spend per category for the 6 months up to and including :month (input to insights.robust_drivers).
SELECT category, month, spend
FROM v_category_month_grid
WHERE customer_id = :cid AND month >= :start AND month <= :month;

-- name: forecast_for_customer
-- Next-month spending forecasts written by ml/forecasting/run_forecast.py (total spend: category IS NULL).
SELECT target_month, model_name, predicted_amount, lower_bound, upper_bound
FROM forecasts
WHERE customer_id = :cid AND category IS NULL
ORDER BY target_month DESC, model_name;

-- name: monthly_expense_history
-- Last 6 monthly expense totals up to and including :month.
SELECT month, expenses FROM v_monthly_summary
WHERE customer_id = :cid AND month <= :month
ORDER BY month DESC
LIMIT 6;

-- name: monthly_summary_history
-- Income / expense totals for the last :limit months up to and including :month (newest first).
SELECT month, income, expenses, emi, discretionary, investments, savings_rate
FROM v_monthly_summary
WHERE customer_id = :cid AND month <= :month
ORDER BY month DESC
LIMIT :limit;

-- name: recent_transactions
-- Largest spending transactions in a month.
SELECT txn_date, merchant, category, amount, channel
FROM v_expense_txn
WHERE customer_id = :cid AND month = :month
ORDER BY amount DESC
LIMIT :limit;
