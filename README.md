# Customer Churn Prediction and Financial Customer Lifetime Value (CLV) Forecasting in Telecommunications

**Prepared by:** SENZELE Benjamine, Applied Data Science and Machine Learning Intern

This project predicts which prepaid telecom customers are likely to churn in the next 90 days, forecasts the revenue each customer will generate over the same period, and combines both into retention priorities: who to contact first, how, and how much revenue is at stake. The results are presented in a Streamlit dashboard.

## Data

Daily records for 3,500 customers from 1 Sep 2025 to 30 Sep 2026: data, voice and SMS revenue, data and voice traffic, and recharges. Customers are scored on 2 Jul 2026; the 1,924 customers with revenue in the 90 days before that date are included.

## Approach

- **Churn:** a customer has churned if they generate no revenue in the 90 days after the scoring date (16.5% of scored customers). A random forest is compared with logistic regression, and its probabilities are calibrated so that they match real churn rates.
- **CLV:** revenue expected over the next 90 days. A random forest is compared with a simple baseline that repeats each customer's last 90 days of revenue, and both are checked on a later period.
- **Retention priorities:** churn risk (low, medium, high) is combined with customer value (low, medium, high) into four groups, each with a recommended action. Value takes the higher of the forecast and spending before the recent drop, because spending falls in the 60 days before churn.
- **Expected revenue at risk:** churn probability × predicted 90-day revenue, used to rank customers within each group.

## Key results

- The churn model separates churners from stayers well (ROC AUC 0.87), and calibrated probabilities match reality (16.6% predicted against 16.5% actual).
- Recency is the strongest warning sign: churn rises from 4% for customers who paid in the last week to 53% after two months without revenue.
- Customers who churned kept steady spending until about 60 days before the scoring date, then cut it sharply.
- The CLV forecast ranks customers better than the baseline (Spearman 0.78 against 0.72), which is what the value categories rely on.

| Retention priority | Rule | Action | Customers |
|---|---|---|---|
| Retain now | High risk, medium or high value | Contact within 7 days | 127 |
| Monitor closely | Medium risk, medium or high value | Targeted loyalty offer | 144 |
| Low-cost action | At risk, low value | Cheap automated offer | 551 |
| Maintain | Low risk | Business as usual | 1,102 |

## Dashboard

The dashboard covers customer behavior, churn, customer value, model performance, retention priorities, campaign lists for each customer group, and business recommendations.

## Repository contents and data privacy

The repository holds the notebook (`BI_Internship_Project.ipynb`, stored without outputs), the dashboard (`app.py`) and aggregated results (`eda_*` files and `feature_importance.csv`, with no customer rows).

The raw dataset and the five customer-level result files are not included because they describe real customers. Most dashboard pages need those five files; they are uploaded in the dashboard sidebar under **Customer files** (`retention_output.csv`, `customer_activity.csv`, `customer_signals.csv`, `churn_test.csv`, `clv_test.csv`), used for that session only and not stored online.
