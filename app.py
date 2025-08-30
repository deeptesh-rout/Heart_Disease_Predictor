import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from fpdf import FPDF

# 🌸 Calming Visuals
st.set_page_config(page_title="Heart Disease Report", layout="centered")
st.markdown("""
    <style>
    body { background-color: #f4f6f9; font-family: 'Georgia', serif; }
    .stApp { background-color: #f4f6f9; }
    h1, h2, h3 { color: #4b6e7d; }
    .reportview-container .main .block-container {
        padding-top: 2rem; padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🏥 Heart Disease Prediction & Poetic Report")
st.write("Upload your patient data or enter it manually to receive a hospital-style report with a poetic touch.")

# Load model and scaler
model = joblib.load('heart_rf_model.pkl')
scaler = joblib.load('heart_scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# 📝 Manual Input Option
with st.expander("📝 Or enter patient details manually"):
    with st.form("manual_input"):
        age = st.number_input("Age", min_value=1, max_value=120)
        trestbps = st.number_input("Resting Blood Pressure")
        chol = st.number_input("Cholesterol")
        thalch = st.number_input("Max Heart Rate Achieved")
        oldpeak = st.number_input("ST Depression")
        ca = st.number_input("Number of Major Vessels", min_value=0, max_value=3)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        manual_df = pd.DataFrame([{
            'age': age, 'trestbps': trestbps, 'chol': chol, 'thalch': thalch,
            'oldpeak': oldpeak, 'ca': ca, 'sex': sex, 'cp': cp, 'thal': thal
        }])
        manual_df = pd.get_dummies(manual_df)
        manual_df = manual_df.reindex(columns=feature_columns, fill_value=0)
        scaled_manual = scaler.transform(manual_df)
        pred = model.predict(scaled_manual)[0]

        poetic = "💔 A heart that whispers warnings in silence." if pred == 1 else "💖 A rhythm steady, untouched by storm."
        note = ("Patient shows signs of cardiac risk. Immediate consultation recommended."
                if pred == 1 else
                "No immediate cardiac risk detected. Maintain healthy lifestyle and regular checkups.")
        companion = ("🌧️ You are not alone. Let your heart be heard. Seek care, seek comfort."
                     if pred == 1 else
                     "🌸 Your heart sings steady. Keep nurturing its rhythm with kindness.")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result_df = pd.DataFrame([{
            'Heart_Disease_Prediction': pred,
            'Poetic_Summary': poetic,
            'Doctor_Note': note,
            'Companion_Message': companion,
            'Report_Generated_At': timestamp
        }])
        st.subheader("🩺 Full Report")
        st.dataframe(result_df)

        # 📄 Poetic PDF Export
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, f"""
        🏥 Heart Disease Report

        Age: {age}
        Blood Pressure: {trestbps}
        Cholesterol: {chol}
        Max Heart Rate: {thalch}
        ST Depression: {oldpeak}
        Major Vessels: {ca}
        Sex: {sex}
        Chest Pain Type: {cp}
        Thalassemia: {thal}

        Prediction: {'Heart Disease Detected' if pred == 1 else 'No Disease Detected'}

        Poetic Summary:
        {poetic}

        Doctor's Note:
        {note}

        Companion Message:
        {companion}

        Report Generated At: {timestamp}
        """)
        pdf.output("poetic_report.pdf")

        with open("poetic_report.pdf", "rb") as f:
            st.download_button("📄 Download Poetic PDF Report", f.read(), "poetic_report.pdf", "application/pdf")

# 📄 CSV Upload Option
uploaded_file = st.file_uploader("📄 Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    bool_cols = df.select_dtypes(include='bool').columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
    for col in bool_cols:
        df[col] = df[col].astype(int)

    df_encoded = pd.get_dummies(df, columns=cat_cols)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    scaled = scaler.transform(df_encoded)
    preds = model.predict(scaled)

    df['Heart_Disease_Prediction'] = preds
    df['Poetic_Summary'] = df['Heart_Disease_Prediction'].apply(lambda x:
        "💔 A heart that whispers warnings in silence." if x == 1 else
        "💖 A rhythm steady, untouched by storm.")
    df['Doctor_Note'] = df['Heart_Disease_Prediction'].apply(lambda x:
        "Patient shows signs of cardiac risk. Immediate consultation recommended." if x == 1 else
        "No immediate cardiac risk detected. Maintain healthy lifestyle and regular checkups.")
    df['Companion_Message'] = df['Heart_Disease_Prediction'].apply(lambda x:
        "🌧️ You are not alone. Let your heart be heard. Seek care, seek comfort." if x == 1 else
        "🌸 Your heart sings steady. Keep nurturing its rhythm with kindness.")
    df['Report_Generated_At'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.subheader("🩺 Full Report")
    st.dataframe(df)

    st.download_button("📥 Download Hospital-Style Report", df.to_csv(index=False), "heart_poetic_report.csv", "text/csv")

