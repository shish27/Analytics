-- ============================================================
-- Customer Churn Analysis — SQL (SQLite syntax, легко переносится на Postgres)
-- Таблица customers загружена из telco_churn.csv через pandas.to_sql()
-- ============================================================

-- 1. Общий процент оттока
SELECT
    Churn,
    COUNT(*) AS customers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM customers), 1) AS pct
FROM customers
GROUP BY Churn;

-- 2. Отток по типу контракта — обычно самый сильный фактор
SELECT
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 3. Отток по способу оплаты
SELECT
    PaymentMethod,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 4. Отток по сегментам "стажа" клиента (tenure)
SELECT
    CASE
        WHEN tenure <= 12 THEN '0-12 мес'
        WHEN tenure <= 24 THEN '13-24 мес'
        WHEN tenure <= 48 THEN '25-48 мес'
        ELSE '48+ мес'
    END AS tenure_segment,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY tenure_segment
ORDER BY MIN(tenure);

-- 5. Влияние дополнительных услуг на отток (TechSupport)
SELECT
    TechSupport,
    COUNT(*) AS total_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
GROUP BY TechSupport
ORDER BY churn_rate_pct DESC;

-- 6. Потери в выручке от ушедших клиентов (упущенная ежемесячная выручка)
SELECT
    ROUND(SUM(MonthlyCharges), 2) AS lost_monthly_revenue,
    COUNT(*) AS churned_customers,
    ROUND(SUM(MonthlyCharges) / COUNT(*), 2) AS avg_monthly_charge_of_churned
FROM customers
WHERE Churn = 'Yes';

-- 7. Комбинированный "профиль риска": помесячный контракт + короткий tenure
SELECT
    COUNT(*) AS high_risk_customers,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM customers
WHERE Contract = 'Month-to-month' AND tenure <= 12;
