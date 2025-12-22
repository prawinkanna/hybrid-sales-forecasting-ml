pip install -r requirements.txt
python sales_forecasting_ml.py


## 📊 Model Comparison – Actual vs Predicted Sales

![Hybrid Sales Forecast Comparison](hybrid-sales-forecast-comparison.png)

**Description:**  
This visualization compares actual weekly sales with predictions from a Prophet-like time series model, an XGBoost-like machine learning model, and a hybrid approach.  
The hybrid model closely follows actual sales trends by combining seasonality, trend, and non-linear learning, demonstrating improved forecasting accuracy.

## 🔮 Future Sales Forecast (52 Weeks)

![52-Week Hybrid Forecast](hybrid-52-week-sales-forecast.png)

**Description:**  
This chart shows historical weekly sales along with a 52-week future forecast generated using a hybrid forecasting model.  
The dashed vertical line marks the start of the forecast period.  
By combining time-series trend and seasonality with machine learning predictions, the hybrid model produces stable and realistic future sales estimates.
## 📊 Power BI Dashboard

A Power BI dashboard was created to visualize and compare forecasting results
from Prophet, XGBoost, and the Hybrid model.

The dashboard highlights key evaluation metrics (RAE, RMSE, MAPE, Accuracy)
and provides business insights for inventory planning and demand forecasting.

![Dashboard Preview](hybrid-retail-sales-forecast-dashboard.png)



📊 Hybrid Sales Forecasting Using Time Series & Machine Learning
🚀 Project Overview

Accurate sales forecasting is essential for effective inventory management, workforce planning, and strategic decision-making in the retail industry.
However, traditional machine learning models ignore time-based patterns, while pure time-series models may fail to capture complex non-linear relationships.

This project addresses both challenges by implementing a Hybrid Sales Forecasting Model that combines:

⏱️ Time Series Modeling (Prophet-like approach) for trend & seasonality

🤖 Machine Learning (Gradient Boosting / XGBoost-style) for non-linear patterns

👉 The hybrid approach provides more accurate and robust forecasts than standalone models.

🎯 Project Objectives

Forecast weekly retail sales accurately

Capture trend and seasonality using time-series methods

Improve predictions using machine learning residual modeling

Compare Time Series vs ML vs Hybrid model performance

📊 Dataset Information

Dataset Name: Walmart Weekly Sales Dataset

Source: Public retail sales dataset

Records: 6,435 weekly observations

Target Variable: Weekly_Sales

🔹 Features Description
Feature	Description
Store	Store identifier
Date	Weekly sales date
Holiday_Flag	Indicates holiday week (0/1)
Temperature	Average weekly temperature
Unemployment	Unemployment rate
Month	Extracted from date
Year	Extracted from date

📁 The dataset is available in the data/ directory as Walmart_Sales.csv.

🧠 Methodology – Hybrid Forecasting Approach
🔸 1. Time Series Component (Prophet-like)

Aggregates sales by date

Captures:

Long-term trend

Weekly/seasonal patterns

🔸 2. Residual Calculation
Residual = Actual Sales − Time Series Prediction

🔸 3. Machine Learning Component

Uses lag features and rolling averages

Trains a Gradient Boosting Regressor (XGBoost-style)

Learns short-term and non-linear patterns

🔸 4. Hybrid Prediction
Final Forecast = Time Series Prediction + ML Residual Prediction


📌 This combination leverages the strengths of both approaches.

🤖 Models Used

Linear Regression (trend modeling)

Gradient Boosting Regressor (machine learning)

Hybrid Model (Time Series + ML)

📈 Evaluation Metrics

Model performance is evaluated using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

MAPE & Accuracy (%)

👉 The Hybrid Model achieves the best overall performance.

📊 Results Visualization

🏆 Key Findings

Time series models effectively capture trend & seasonality

Machine learning models capture non-linear patterns

Hybrid approach outperforms standalone models

Gradient Boosting improves short-term prediction accuracy

▶ How to Run the Project
🔹 Step 1: Install Dependencies
pip install -r requirements.txt

🔹 Step 2: Run the Model
python sales_forecasting_ml.py

📁 Repository Structure
📁 data/
   └── Walmart_Sales.csv
sales_forecasting_ml.py
README.md
requirements.txt
ML Forcast.png
pk mini pro report.pdf

🚀 Future Enhancements

Integrate Facebook Prophet

Use XGBoost / LightGBM

Hyperparameter tuning

Power BI / Tableau dashboard

Deploy as a web application

👤 Author

Thiruprawin Kanna
AI & Machine Learning Enthusiast
📍 India

⭐ If You Find This Useful

⭐ Star the repository

🍴 Fork it

💡 Suggest improvements
