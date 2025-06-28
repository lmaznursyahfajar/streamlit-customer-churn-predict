import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------------
# Load Model & Preprocessing
# -----------------------------
model = joblib.load("best_gradient_boosting_model.pkl")
scaler = joblib.load("scaler.pkl")

# Kolom training (hasil dari X_train.columns.tolist())
with open("columns.pkl", "rb") as f:
    expected_cols = joblib.load(f)

# Kolom numerik yang akan di-scale
scale_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("telco_churn_with_all_feedback.csv")

df = load_data()

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# -----------------------------
# Sidebar Navigasi
# -----------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")
st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman", ["Dashboard", "Prediksi Churn"])

# -----------------------------
# Halaman Dashboard
# -----------------------------
if menu == "Dashboard":
    st.title("📊 Dashboard Pelanggan Telco")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribusi Gender")
        fig1 = px.pie(df, names='gender', title='Distribusi Pelanggan Berdasarkan Gender')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("### Proporsi Senior Citizen")
        senior_df = df['SeniorCitizen'].value_counts().reset_index()
        senior_df.columns = ['SeniorCitizen', 'count']
        fig2 = px.bar(senior_df, 
                      x='SeniorCitizen', y='count',
                      labels={'SeniorCitizen': 'Senior Citizen', 'count': 'Jumlah'},
                      title='Jumlah Senior Citizen vs Non-Senior')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Rata-rata Biaya Bulanan per Jenis Kontrak")
        fig3 = px.bar(df.groupby("Contract")["MonthlyCharges"].mean().reset_index(),
                      x='Contract', y='MonthlyCharges', title='Rata-rata Biaya Bulanan per Kontrak')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if 'Churn' in df.columns:
            st.markdown("### Distribusi Churn")
            fig4 = px.histogram(df, x='Churn', color='Churn', title='Distribusi Status Churn')
            st.plotly_chart(fig4, use_container_width=True)

       # Bersihkan dan konversi kolom numerik
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
    
    # Drop baris yang memiliki NaN setelah konversi
    df_clean = df.dropna(subset=["tenure", "TotalCharges", "PaymentMethod"])
    
    # Visualisasi bar setelah pembersihan
    fig5 = px.bar(df_clean.groupby("PaymentMethod")[["tenure", "TotalCharges"]].mean().reset_index(),
                  x="PaymentMethod", y=["tenure", "TotalCharges"],
                  barmode='group', title="Rata-rata Tenure & Total Charges per Payment Method")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("### Churn berdasarkan Jenis Layanan Internet")
    fig6 = px.histogram(df, x='InternetService', color='Churn', barmode='group', title='Churn berdasarkan Internet Service')
    st.plotly_chart(fig6, use_container_width=True)

# -----------------------------
# Halaman Prediksi
# -----------------------------
elif menu == "Prediksi Churn":
    st.title("🔍 Customer Churn Predict")

    # Form Input (semua kolom kategorikal yang umum di dataset Telco)
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    senior = st.selectbox("Apakah Senior Citizen?", [0, 1])
    partner = st.selectbox("Memiliki Pasangan?", ["Yes", "No"])
    dependents = st.selectbox("Memiliki Tanggungan?", ["Yes", "No"])
    phone_service = st.selectbox("Layanan Telepon?", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security?", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup?", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection?", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support?", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV?", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies?", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
    payment_method = st.selectbox("Metode Pembayaran", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    tenure = st.slider("Lama Berlangganan (bulan)", 0, 72, 12)
    monthly = st.number_input("Biaya Bulanan", min_value=0.0)
    total = st.number_input("Total Tagihan", min_value=0.0)

    # Tampilkan input user
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    st.markdown("#### 📋 Data yang Anda Masukkan")
    st.dataframe(pd.DataFrame([input_dict]))

    if st.button("🔮 Prediksi Sekarang"):
        try:
            # Buat DataFrame input
            input_df = pd.DataFrame([input_dict])

            # One-hot encoding
            input_encoded = pd.get_dummies(input_df, drop_first=True)

            # Reindex agar sesuai dengan kolom saat training
            input_encoded = input_encoded.reindex(columns=expected_cols, fill_value=0)

            # Scaling kolom numerik
            input_encoded[scale_cols] = scaler.transform(input_encoded[scale_cols])

            # Prediksi
            pred = model.predict(input_encoded)[0]
            proba = model.predict_proba(input_encoded)[0][1]

            # Output
            if pred == 1:
                st.error(f"⚠️ Pelanggan kemungkinan akan **CHURN** (Probabilitas: {proba:.2%})")
            else:
                st.success(f"✅ Pelanggan kemungkinan **BERTAHAN** (Probabilitas: {proba:.2%})")

        except Exception as e:
            st.warning(f"❌ Terjadi error saat prediksi:\n\n{e}")
