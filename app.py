import streamlit as st
import pandas as pd
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Page Configuration
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗", layout="wide")

# --- CUSTOM CSS FOR BETTER UI ---
st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; font-weight: bold; }
    .sub-header { font-size: 1.2rem; color: #555; text-align: center; }
    .prediction-box { background-color: #e8f5e9; padding: 20px; border-radius: 10px; border: 1px solid #c8e6c9; }
    @media (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
        .sub-header { font-size: 1rem; }
    }
    </style>
    """, unsafe_allow_html=True)


# Load the trained model
@st.cache_resource
def load_model():
    with open('car_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model


model = load_model()


# Load the dataset
@st.cache_data
def load_dataset():
    return pd.read_csv('used_car_price.csv')


try:
    df = load_dataset()
except FileNotFoundError:
    st.error("Dataset 'used_car_price.csv' not found. Please ensure it is in the project folder.")
    st.stop()

# Currency Exchange Rates
exchange_rates = {
    "USD ($)": 1.0,
    "BDT (৳)": 110.0,
    "RMB (¥)": 7.2,
    "EUR (€)": 0.92
}

# --- SIDEBAR ---
with st.sidebar:
    sidebar_img_path = "car_images/car_icon.png"
    if os.path.exists(sidebar_img_path):
        st.image(sidebar_img_path, width=100)
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/3097/3097144.png", width=100)

    st.title("About This Project")
    st.info(
        "This application uses a **Machine Learning model** "
        "to predict the price of a used car based on its features. "
        "The model was trained on a dataset of **10,000 car listings**."
    )

    st.markdown("---")
    st.write("**Developer:** MD MOHIBULLAH Al MUHIB (莫哈)")
    st.write("**Topic:** 17 - Second-hand Price Prediction")
    st.write("**Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, LightGBM, Streamlit")

    st.markdown("---")
    st.write("### 🤖 Model Information")
    st.write("- **Algorithm:** LightGBM (Best)")
    st.write("- **R-squared Score:** ~86.87%")
    st.write("- **Mean Absolute Error:** ~$815.12")

# --- MAIN PAGE ---
st.markdown('<p class="main-header">🚗 Used Car Price Predictor</p >', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter the vehicle details below to get an AI-powered price estimate.</p >',
            unsafe_allow_html=True)

st.divider()

# --- DATASET STATISTICS ---
st.subheader("📈 Dataset Statistics")
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
with col_stat1:
    st.metric(label="Total Cars", value=f"{len(df):,}")
with col_stat2:
    st.metric(label="Average Price", value=f"${df['price_usd'].mean():,.2f}")
with col_stat3:
    st.metric(label="Maximum Price", value=f"${df['price_usd'].max():,.2f}")
with col_stat4:
    st.metric(label="Minimum Price", value=f"${df['price_usd'].min():,.2f}")

st.divider()

# --- MULTI-PAGE TABS ---
tab_predict, tab_compare, tab_depreciation, tab_visual, tab_data = st.tabs([
    "🔮 Predict Price",
    "⚖️ Compare Cars",
    "📉 Depreciation Calculator",
    "📊 Data Visualization",
    "📋 Dataset Preview"
])

# ==================== TAB 1: PREDICT PRICE ====================
with tab_predict:
    with st.container(border=True):
        st.subheader("Vehicle Specifications")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Basic Info**")
            make_year = st.slider("Year of Manufacture", 1990, 2026, 2016)
            mileage_kmpl = st.number_input("Mileage (in kmpl)", 1.0, 50.0, 15.0, 0.1)
            owner_count = st.number_input("Owner Count", 1, 5, 1)

            # ➕ Quick Insight Chart to fill empty space
            st.markdown("---")
            st.markdown("**📊 Quick Insight: Top 5 Brands by Avg Price**")
            brand_avg_quick = df.groupby('brand')['price_usd'].mean().sort_values(ascending=False).head(5)
            st.bar_chart(brand_avg_quick)

        with col2:
            st.markdown("**Engine & Brand**")
            brand = st.selectbox("Brand", ["Toyota", "Honda", "BMW", "Hyundai", "Tesla", "Nissan", "Chevrolet", "Kia",
                                           "Volkswagen", "Ford"])
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
            engine_cc = st.number_input("Engine (CC)", 500, 6000, 1500, 100)

        with col3:
            st.markdown("**Other Details**")
            transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
            color = st.selectbox("Color", ["White", "Black", "Red", "Silver", "Gray", "Blue"])
            service_history = st.selectbox("Service History", ["None", "Full", "Partial"])
            insurance_valid = st.selectbox("Insurance Valid", ["Yes", "No"])
            accidents_reported = st.number_input("Accidents Reported", 0, 5, 0)

            # Currency Selector (এখন ডান দিকের কলামে)
            st.markdown("---")
            selected_currency = st.selectbox("💵 Select Currency", list(exchange_rates.keys()))

    if st.button("🔍 Predict Price", use_container_width=True, type="primary"):
        input_data = {
            'make_year': [make_year],
            'car_age': [2026 - make_year],
            'mileage_kmpl': [mileage_kmpl],
            'engine_cc': [engine_cc],
            'owner_count': [owner_count],
            'accidents_reported': [accidents_reported],
            'fuel_type_Diesel': [1 if fuel_type == 'Diesel' else 0],
            'fuel_type_Electric': [1 if fuel_type == 'Electric' else 0],
            'fuel_type_Petrol': [1 if fuel_type == 'Petrol' else 0],
            'brand_BMW': [1 if brand == 'BMW' else 0],
            'brand_Chevrolet': [1 if brand == 'Chevrolet' else 0],
            'brand_Ford': [1 if brand == 'Ford' else 0],
            'brand_Honda': [1 if brand == 'Honda' else 0],
            'brand_Hyundai': [1 if brand == 'Hyundai' else 0],
            'brand_Kia': [1 if brand == 'Kia' else 0],
            'brand_Nissan': [1 if brand == 'Nissan' else 0],
            'brand_Tesla': [1 if brand == 'Tesla' else 0],
            'brand_Toyota': [1 if brand == 'Toyota' else 0],
            'brand_Volkswagen': [1 if brand == 'Volkswagen' else 0],
            'transmission_Manual': [1 if transmission == 'Manual' else 0],
            'color_Blue': [1 if color == 'Blue' else 0],
            'color_Gray': [1 if color == 'Gray' else 0],
            'color_Red': [1 if color == 'Red' else 0],
            'color_Silver': [1 if color == 'Silver' else 0],
            'color_White': [1 if color == 'White' else 0],
            'service_history_Partial': [1 if service_history == 'Partial' else 0],
            'insurance_valid_Yes': [1 if insurance_valid == 'Yes' else 0]
        }

        input_df = pd.DataFrame(input_data)

        try:
            with open('feature_names.json', 'r') as f:
                trained_features = json.load(f)
            input_df = input_df.reindex(columns=trained_features, fill_value=0)
        except FileNotFoundError:
            st.error("Feature names file not found. Please run 'train_model.py' first.")
            st.stop()

        try:
            prediction_usd = model.predict(input_df)[0]
            rate = exchange_rates[selected_currency]
            prediction_converted = prediction_usd * rate

            st.divider()
            col_img, col_res = st.columns([1, 1.5])

            with col_img:
                st.markdown("### 🚗 Selected Car")
                img_filename = f"car_images/{brand.lower()}_{color.lower()}.png"

                if os.path.exists(img_filename):
                    st.image(img_filename, use_container_width=True, caption=f"{color} {brand}")
                else:
                    st.info(
                        f"Image not found for {color} {brand}.\nPlease add '{img_filename}' to the 'car_images' folder.")

            with col_res:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("Estimated Market Value")

                symbols = {"USD ($)": "$", "BDT (৳)": "৳", "RMB (¥)": "¥", "EUR (€)": "€"}
                st.markdown(
                    f"<h1 style='text-align: center; color: #28a745;'>{symbols[selected_currency]}{prediction_converted:,.2f}</h1>",
                    unsafe_allow_html=True)

                st.write(
                    f"Based on a **{make_year} {brand}** with **{mileage_kmpl:.2f} kmpl** ({fuel_type}, {transmission})")

                avg_price = df['price_usd'].mean()
                if prediction_usd > avg_price:
                    st.warning(f"⚠️ This price is higher than the dataset average of ${avg_price:,.2f}.")
                else:
                    st.success(f"✅ This price is lower than the dataset average of ${avg_price:,.2f}.")

                report_df = pd.DataFrame({
                    "Field": ["Make Year", "Car Age", "Brand", "Mileage (kmpl)", "Engine (CC)", "Fuel Type",
                              "Transmission", "Color", "Predicted Price (USD)", "Predicted Price (Selected)"],
                    "Value": [make_year, 2026 - make_year, brand, mileage_kmpl, engine_cc, fuel_type, transmission,
                              color, f"${prediction_usd:,.2f}", f"{prediction_converted:,.2f}"]
                })
                csv = report_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Prediction Report", csv, f"prediction_{brand}_{make_year}.csv",
                                   "text/csv")
                st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check if the model was trained correctly.")

# ==================== TAB 2: COMPARE CARS ====================
with tab_compare:
    st.subheader("⚖️ Compare Two Cars Side-by-Side")
    st.info("Enter details for two cars to compare their estimated prices.")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### 🚗 Car A")
        a_year = st.slider("Year (Car A)", 1990, 2026, 2018, key="a_year")
        a_mileage = st.number_input("Mileage (Car A)", 1.0, 50.0, 15.0, 0.1, key="a_mileage")
        a_brand = st.selectbox("Brand (Car A)", ["Toyota", "BMW", "Honda", "Tesla"], key="a_brand")
        a_engine = st.number_input("Engine (Car A)", 500, 6000, 1500, key="a_engine")
    with colB:
        st.markdown("### 🚙 Car B")
        b_year = st.slider("Year (Car B)", 1990, 2026, 2015, key="b_year")
        b_mileage = st.number_input("Mileage (Car B)", 1.0, 50.0, 20.0, 0.1, key="b_mileage")
        b_brand = st.selectbox("Brand (Car B)", ["Toyota", "BMW", "Honda", "Tesla"], key="b_brand")
        b_engine = st.number_input("Engine (Car B)", 500, 6000, 2000, key="b_engine")

    if st.button("Compare Prices", type="primary"):
        st.info("Comparison feature is a simplified demonstration. Full prediction available in the Predict tab.")
        st.write(f"**Car A:** {a_year} {a_brand} → Estimated price around **${a_year * 300 + a_engine * 2:.2f}**")
        st.write(f"**Car B:** {b_year} {b_brand} → Estimated price around **${b_year * 300 + b_engine * 2:.2f}**")

# ==================== TAB 3: DEPRECIATION CALCULATOR ====================
with tab_depreciation:
    st.subheader("📉 Car Depreciation Calculator")
    st.info("Estimate how much your car's value will decrease over the next few years.")

    dep_year = st.slider("Current Year of Car", 1990, 2026, 2018)
    dep_price = st.number_input("Current Estimated Price (USD)", 1000, 100000, 10000)
    dep_years = st.slider("Years into the Future", 1, 10, 3)

    if st.button("Calculate Depreciation"):
        future_price = dep_price * (0.85 ** dep_years)
        st.success(f"After **{dep_years} years**, the estimated value will be: **${future_price:,.2f}**")
        st.write(f"Total depreciation: **${dep_price - future_price:,.2f}**")

        years = list(range(0, dep_years + 1))
        prices = [dep_price * (0.85 ** y) for y in years]
        chart_df = pd.DataFrame({"Year": [dep_year + y for y in years], "Price": prices})
        st.line_chart(chart_df.set_index("Year"))

# ==================== TAB 4: DATA VISUALIZATION ====================
with tab_visual:
    st.header("📊 Data Visualization & Insights")
    vtab1, vtab2, vtab3, vtab4 = st.tabs(
        ["Feature Importance", "Price Distribution", "Brand-wise Price", "SHAP Values"])

    with vtab1:
        st.subheader("Top Features Affecting Car Price")
        try:
            fi_df = pd.read_csv('feature_importances.csv').head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax)
            ax.set_title('Top 10 Features Affecting Car Price')
            st.pyplot(fig)
            st.info("**Interpretation:** This chart shows which features the model found most important.")
        except FileNotFoundError:
            st.warning("Feature importance data not found. Please run 'train_model.py' first.")

    with vtab2:
        st.subheader("Distribution of Car Prices")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.histplot(df['price_usd'], bins=30, kde=True, color='blue', ax=ax2)
        ax2.set_title('Distribution of Used Car Prices')
        ax2.set_xlabel('Price (USD)')
        st.pyplot(fig2)

    with vtab3:
        st.subheader("Average Price by Brand")
        brand_avg = df.groupby('brand')['price_usd'].mean().sort_values(ascending=False).reset_index()
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        sns.barplot(x='price_usd', y='brand', data=brand_avg, palette='magma', ax=ax3)
        ax3.set_title('Average Price by Brand')
        ax3.set_xlabel('Average Price (USD)')
        st.pyplot(fig3)

    with vtab4:
        st.subheader("SHAP Values (Feature Impact)")
        try:
            shap_df = pd.read_csv('shap_values.csv')
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_df.values, feature_names=shap_df.columns, show=False)
            st.pyplot(fig4)
            st.info("SHAP values show how each feature impacts the prediction (positive or negative).")
        except FileNotFoundError:
            st.warning("SHAP values not found. Please run train_model.py first.")

# ==================== TAB 5: DATASET PREVIEW ====================
with tab_data:
    st.subheader("📋 Dataset Preview (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)