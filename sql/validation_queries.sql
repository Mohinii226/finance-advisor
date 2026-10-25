-- Sanity checks to run right after loading --------------------------------------

-- 1. Row counts
SELECT 'customers' t, COUNT(*) FROM customers UNION ALL
SELECT 'transactions', COUNT(*) FROM transactions UNION ALL
SELECT 'sms_raw', COUNT(*) FROM sms_raw UNION ALL
SELECT 'injected_events', COUNT(*) FROM injected_events;

-- 2. Share of uncategorised debits (the ML categoriser's target population)
SELECT ROUND(100.0 * COUNT(*) FILTER (WHERE category IS NULL) / COUNT(*), 1) AS pct_unlabelled
FROM transactions WHERE type = 'debit' AND true_category NOT IN ('EMI','Investments','Self Transfer');

-- 3. Monthly spend by category for one customer (excludes transfers/investments)
SELECT date_trunc('month', t.txn_date)::date AS month, t.true_category, SUM(t.amount) AS spend
FROM transactions t JOIN dim_category d ON d.category = t.true_category
WHERE t.customer_id = 'C0001' AND t.type = 'debit' AND d.is_expense
GROUP BY 1, 2 ORDER BY 1, 3 DESC;

-- 4. Weekday spending pattern (0 = Sunday in Postgres)
SELECT EXTRACT(dow FROM txn_date) AS dow, ROUND(SUM(amount)) AS spend
FROM transactions t JOIN dim_category d ON d.category = t.true_category
WHERE t.type = 'debit' AND d.is_expense GROUP BY 1 ORDER BY 1;

-- 5. Ledger integrity: balance should never go below zero
SELECT COUNT(*) AS negative_balances FROM transactions WHERE balance_after < 0;
