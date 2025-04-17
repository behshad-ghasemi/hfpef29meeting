import streamlit as st
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO


# Load models and pipeline
log_model = joblib.load("Logistic_model.joblib")
rf_model = joblib.load("randomforest_model.joblib")
xgb_model = joblib.load("xgboost_model.joblib")
pipeline = joblib.load("data_with_pca.joblib")

FEATURES = ["BSA (non toccare)", 'Epicardial fat thickness (mm)', 'LVEDD (mm)', 'LVESD (mm)', 'LV mass (g)',
            'LV mass i (g/m2) Calcolo automatico', 'LAD (mm)', 'LAV (ml)', 'iLAV (mL/m2) Calcolo automatico',
            'LVEF (%)', 'MR 0=no, 1=min, 2=lieve, 3=mod, 4=sev', 'E/A', 'E/e avg', 'VCI diameter (mm)',
            'sPAP (mmHg) = RV-RA + RAP (3-8-15)', 'Probabilità IP a riposo (1, bassa; 2, intermedia; 3, alta)', 'eta',
            'BSA', 'IAS (0, assente; 1, presente)', 'Dislipidemia (0, assente; 1, presente, 2 s. metabolica)',
            'Diabete mellito (0, assente; 1, presente, 2 diabete complicato)', 'Beta bloccante 1 si, 0 no',
            'MRA 1 si, 0 no', 'ACE/ARB 1 si, 0 no',
            'Terapia antiscompenso assente (0), sub-ottimale (1-2), ottimale (3)',
            "Insufficienza renale lieve (si 1; no 0)", "Insufficienza renale severa (si 1; no 0)",
            'classe funzionale WHO-FC',
            'segni clinici di insufficienza cardiaca sinistra, astenia, dispnea (0, assenti; 1, presenti)',
            'Segni clinici di insufficienz acardiaca destra, edemi (0, assenti; 1, presenti)',
            'Qualsiasi segno di insufficienza cardiaca',
            'PAS (mmHg)', 'PAD (mmHg)', 'NT-pro-BNP (pg/mL)',
            'Fibrillazione atriale parossistica (0, assente; 1, presente)',
            'Fibrillazione atriale cronica (0, assente; 1, presente)', 'Fumo (0 no; 1 si)', 'sesso']

st.title("🫀 Heart Failure (HFpEF) Probability Prediction 🫀")
st.markdown("Insert the patient's clinical data below to estimate the probability of HFpEF. )
st.markdown(👩‍💻Good Prediction,Behshad 🥰")

user_input = {}
for feature in FEATURES:
    if feature == "sesso":
        user_input[feature] = st.selectbox("Sesso", options=["0", "1"])
    elif any(x in feature for x in ["(0", "1", "2", "presente", "assente", "no", "si"]):
        user_input[feature] = st.number_input(f"{feature}", min_value=None, max_value=None, step=0.001)
    else:
        user_input[feature] = st.number_input(f"{feature}", step=0.1)

if st.button("Estimate"):
    input_df = pd.DataFrame([user_input])
    input_df['sesso'] = input_df['sesso'].astype('category')
    transformed_input = pipeline.transform(input_df)

    # Predictions
    prob_log = log_model.predict_proba(transformed_input)[:, 1][0]
    prob_rf = rf_model.predict_proba(transformed_input)[:, 1][0]
    prob_gb = xgb_model.predict_proba(transformed_input)[:, 1][0]

    # Show predictions
    st.subheader("Prediction Probabilities")
    st.write(f"🔹 **Logistic Regression**: {prob_log:.4f}")
    st.write(f"🔹 **Random Forest**: {prob_rf:.4f}")
    st.write(f"🔹 **XG Boosting**: {prob_gb:.4f}")

    if prob_gb > 0.6:
        st.error("🚨 OH MY GOD, High Risk of HFpEF Detected!😭😔")
    else:
        st.success("💃🥳YOHOOOOOOOOOO, Low Risk of HFpEF 🥳💃")

    # Bar chart for comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    models = ["Logistic Regression", "Random Forest", "XG Boosting"]
    probabilities = [prob_log, prob_rf, prob_gb]
    sns.barplot(x=models, y=probabilities, palette='mako', ax=ax)
    ax.set_title("Model Probability Comparison")
    ax.set_ylabel("HFpEF Probability")
    st.pyplot(fig)
