# 🚗 Used Car Price Prediction System
A complete **Machine Learning-based web application** that predicts the price of a used car based on its features. This project was developed as a Graduation Thesis (Topic 17 - Second-hand Price Prediction).

## 📌 Project Overview
This system uses a **LightGBM Regressor** (selected after comparing multiple models) trained on a dataset of **10,000 used car listings**. It includes a fully interactive **Streamlit web application** with data visualization, currency conversion, SHAP explainability, and report download features.

## 🎯 Key Features

- **🔮 Price Prediction:** Predicts the price of a used car based on input features.
- **💵 Multi-Currency Support:** Displays prices in USD ($), BDT (৳), RMB (¥), and EUR (€).
- **🚗 Car Image Display:** Shows the selected car image based on Brand and Color.
- **⚖️ Car Comparison:** Compares two cars side-by-side.
- **📉 Depreciation Calculator:** Estimates future car value over the next few years.
- **📊 Data Visualization:** 
  - Feature Importance Chart
  - Price Distribution Histogram
  - Brand-wise Average Price Chart
  - **SHAP Values** (Explainable AI)
  - Dataset Preview Table
- **📈 Dataset Statistics:** Total cars, average, max, and min prices.
- **📥 Download Report:** Allows users to download the prediction result as a CSV file.
- **🔄 Reset Inputs:** Resets all input fields to default values.
- **🎨 Custom Theme:** Light theme configured via `.streamlit/config.toml`.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| **Programming Language** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Web Framework** | Streamlit |
| **Visualization** | Matplotlib, Seaborn, SHAP |
| **Model Saving** | Pickle, JSON |

## 📂 Project Structure

pythonProject3/
│
├── .streamlit/
│   └── config.toml              # Theme configuration
│
├── car_images/                  # Car images by brand and color
│
├── app.py                       # Main Streamlit web application
├── train_model.py               # Model training script
├── car_price_model.pkl          # Saved trained model
├── feature_names.json           # List of feature names used in training
├── feature_importances.csv      # Feature importance data for charts
├── shap_values.csv              # SHAP values for explainability
├── used_car_price.csv           # Dataset (10,000 rows)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

## 🚀 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/mdmohibullahalmuhib5-cell/used-car-price-prediction-v1.git
cd used-car-price-prediction-v1
```
2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Train the Model (Optional)

```bash
python train_model.py
```

4. Run the Web Application

```bash
streamlit run app.py
```

The app will open in your browser at http://localhost:8501.

---

📊 Model Performance

Model MAE (USD) R² Score
Linear Regression $790.80 87.66%
LightGBM (Best) $815.12 86.87%
Tuned Random Forest $852.21 85.47%
XGBoost $870.89 84.72%

Selected Model: LightGBM Regressor

---

📈 Dataset Information

· Total Records: 10,000
· Target Variable: price_usd
· Feature Engineering: Created car_age feature
· Encoding: One-Hot Encoding applied to categorical columns

---

👨‍💻 Developer

· Name: MD MOHIBULLAH Al MUHIB (莫哈)
· Topic: 17 - Second-hand Price Prediction
· Institution: Wuhu Institute of Technology

---

📄 License

This project is developed for academic purposes as part of a graduation thesis.

---
🙏 Acknowledgements

· Dataset Source: Kaggle
· Libraries: Scikit-learn, XGBoost, LightGBM, Streamlit, Pandas, Matplotlib, Seaborn, SHAP
