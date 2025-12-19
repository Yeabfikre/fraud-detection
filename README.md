# Fraud Detection for E-commerce and Banking

## Project Overview
This project develops a robust fraud detection system for Adey Innovations Inc. using machine learning. We analyze two distinct datasets: e-commerce transactions and bank credit card records, addressing the critical challenge of **Class Imbalance**.

## Data Summary
- **Fraud_Data.csv**: E-commerce data requiring IP-to-Country mapping.
- **IpAddress_to_Country.csv**: Mapping table for geolocation analysis.
- **creditcard.csv**: Anonymized bank transactions (PCA-transformed).

## Key Features Engineered
- **Time-based**: Hour of day, Day of week.
- **Velocity**: Transaction frequency per user.
- **Geolocation**: Country mapping from IP addresses.
- **Behavioral**: Time duration between signup and purchase.|

## How to Run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run notebooks in the `notebooks/` folder sequentially.
4. Run tests: `pytest tests/`.

## Explainability (XAI)
We use **SHAP** to interpret model decisions, ensuring that fraud flags are based on logical patterns (e.g., high velocity or suspicious geolocation) to build trust with stakeholders.
