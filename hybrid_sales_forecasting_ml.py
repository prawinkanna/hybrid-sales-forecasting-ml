# ============================================================
# Hybrid Sales Forecasting using Time Series + Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
df = pd.read_csv("data/Walmart_Sales.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Aggregate weekly sales across all stores
ts = (
    df.groupby("Date")["Weekly_Sales"]
    .sum()
    .reset_index()
    .sort_values("Date")
    .reset_index(drop=True)
)

print("Dataset loaded successfully")
print(ts.head())


# ------------------------------------------------------------
# 2. Prophet-like Time Series Component (Trend + Seasonality)
# ------------------------------------------------------------

# ---- Trend (Linear Regression on time index)
time_index = np.arange(len(ts)).reshape(-1, 1)
trend_model = LinearRegression()
trend_model.fit(time_index, ts["Weekly_Sales"])
ts["trend"] = trend_model.predict(time_index)

# ---- Seasonality (weekly pattern)
ts["week"] = ts["Date"].dt.isocalendar().week.astype(int)
seasonality = ts.groupby("week")["Weekly_Sales"].transform("mean")

# ---- Prophet-like prediction
ts["prophet_pred"] = ts["trend"] + (seasonality - seasonality.mean())


# ------------------------------------------------------------
# 3. Machine Learning Component (XGBoost-like using GBM)
# ------------------------------------------------------------

# Feature engineering (lags & rolling mean)
ts["lag_1"] = ts["Weekly_Sales"].shift(1)
ts["lag_2"] = ts["Weekly_Sales"].shift(2)
ts["rolling_3"] = ts["Weekly_Sales"].rolling(3).mean()

# Drop missing values created by lagging
ts = ts.dropna().reset_index(drop=True)

X = ts[["lag_1", "lag_2", "rolling_3"]]
y = ts["Weekly_Sales"]

# Time-based train-test split (80/20)
train_size = int(len(ts) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Predictions
ts.loc[X_test.index, "xgb_pred"] = gb_model.predict(X_test)


# ------------------------------------------------------------
# 4. Hybrid Model (Weighted Combination)
# ------------------------------------------------------------
prophet_pred = ts.loc[X_test.index, "prophet_pred"]
xgb_pred = ts.loc[X_test.index, "xgb_pred"]

# Weighted hybrid (tunable)
hybrid_pred = (0.4 * prophet_pred) + (0.6 * xgb_pred)
ts.loc[X_test.index, "hybrid_pred"] = hybrid_pred


# ------------------------------------------------------------
# 5. Evaluation Metrics
# ------------------------------------------------------------
def mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100


y_true = y_test.values

metrics = {
    "Prophet-like MAE": mean_absolute_error(y_true, prophet_pred),
    "XGB-like MAE": mean_absolute_error(y_true, xgb_pred),
    "Hybrid MAE": mean_absolute_error(y_true, hybrid_pred),
    "Hybrid RMSE": np.sqrt(mean_squared_error(y_true, hybrid_pred)),
    "Hybrid R2": r2_score(y_true, hybrid_pred),
    "Hybrid MAPE (%)": mape(y_true, hybrid_pred),
}

metrics["Hybrid Accuracy (%)"] = 100 - metrics["Hybrid MAPE (%)"]

print("\n--- Model Performance ---")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}")


# ------------------------------------------------------------
# 6. Visualization
# ------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.plot(ts.loc[X_test.index, "Date"], y_true, label="Actual", linewidth=2)
plt.plot(ts.loc[X_test.index, "Date"], prophet_pred, label="Prophet-like")
plt.plot(ts.loc[X_test.index, "Date"], xgb_pred, label="XGB-like")
plt.plot(ts.loc[X_test.index, "Date"], hybrid_pred, label="Hybrid", linewidth=2)

plt.title("Hybrid Sales Forecasting Model")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.legend()
plt.tight_layout()
plt.show()
