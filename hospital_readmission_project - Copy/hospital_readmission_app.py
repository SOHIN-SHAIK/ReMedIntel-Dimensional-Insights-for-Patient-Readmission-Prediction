import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf

# Load Data using fixed paths
df = pd.read_csv(r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\data\patient_data.csv")
df_reduced = pd.read_csv(r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\data\reduced_data_v2.csv")

dl_model = tf.keras.models.load_model(r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\models\dl_model.h5")
lin_model = joblib.load(r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\models\linear_model.pkl")

st.title("ReMedIntel: Accurate Patient Readmission Prediction")

option = st.radio("Choose Prediction Mode", ["Predict by Patient Name", "Predict by Time Period"])

# --- Predict by Patient Name ---
def predict_by_patient_name(df, df_reduced, dl_model):
    st.subheader("Predict Readmission Days by Patient Name")
    name = st.selectbox("Select patient", df['Name'].unique())

    patient_row = df[df['Name'] == name].iloc[0]
    discharge_status = str(patient_row['Discharge']).lower()
    st.write("Patient Info:", patient_row)

    if "under observation" in discharge_status:
        st.warning("🟡 Patient is still under observation. Cannot predict readmission days yet.")
        return
    elif "deceased" in discharge_status:
        st.error("❌ Patient is marked as deceased. Readmission prediction is not applicable.")
        return

    idx = df_reduced[df_reduced['Name'] == name].index[0]
    input_data = df_reduced.loc[idx, ['PCA1', 'PCA2', 'PCA3']].values.reshape(1, -1)

    if st.button("Predict Readmission Days"):
        pred_days = dl_model.predict(input_data)[0][0]
        predicted_days = max(1, round(pred_days))
        st.success(f"✅ Predicted Readmission Days: {predicted_days} days")


# --- Predict by Time Period ---
def predict_by_time_period(df, df_reduced, dl_model):
    st.subheader("Predict Patients by Readmission Window")
    days = st.slider("Select max days to readmission", 1, 10)

    predictions = []
    for i, row in df_reduced.iterrows():
        name = row['Name']
        full_record = df[df['Name'] == name].iloc[0]
        discharge_status = str(full_record['Discharge']).lower()

        if "under observation" in discharge_status or "deceased" in discharge_status:
            continue  # skip invalid cases

        input_data = row[['PCA1', 'PCA2', 'PCA3']].values.reshape(1, -1)
        pred_days = dl_model.predict(input_data)[0][0]
        rounded_days = max(1, round(pred_days))

        if rounded_days <= days:
            predictions.append((name, rounded_days))

    if predictions:
        st.success(f"✅ Patients likely to be readmitted in {days} days:")
        for name, pred in predictions:
            st.markdown(f"• **{name}** → {pred} days")
    else:
        st.info("ℹ️ No patients predicted to be readmitted in this time frame.")

# --- Main Logic ---
if option == "Predict by Patient Name":
    predict_by_patient_name(df, df_reduced, dl_model)

elif option == "Predict by Time Period":
    predict_by_time_period(df, df_reduced, dl_model)
