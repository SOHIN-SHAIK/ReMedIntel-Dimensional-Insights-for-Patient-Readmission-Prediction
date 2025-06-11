import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

def preprocess_data():
    df = pd.read_csv(r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\data\patient_data.csv")
    print("Columns:", df.columns)
    print("Sample Discharge values:", df['Discharge'].head(10).tolist())

    # Encode 'Discharge' as categorical label
    le_status = LabelEncoder()
    df['Discharge_Status'] = le_status.fit_transform(df['Discharge'].astype(str))

    le_diag = LabelEncoder()
    le_med = LabelEncoder()
    df['Diagnosis'] = le_diag.fit_transform(df['Diagnosis'].astype(str))
    df['Medicine'] = le_med.fit_transform(df['Medicine'].astype(str))

    df['Discharge_Weekday'] = 0  # placeholder, no date info

    if 'Readmitted_Days' not in df.columns:
        df['Readmitted_Days'] = (df['LabResult'] % 30) + 1  # dummy readmission days

    if 'Days_Admitted' not in df.columns:
        df['Days_Admitted'] = df['LabResult'] // 10

    features = df[['Diagnosis', 'LabResult', 'Days_Admitted', 'Discharge_Weekday', 'Medicine', 'Discharge_Status']]

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(features)
    reduced_df = pd.DataFrame(reduced, columns=['PCA1', 'PCA2', 'PCA3'])

    final_df = pd.concat([df[['PatientID', 'Name', 'Readmitted_Days']].reset_index(drop=True), reduced_df], axis=1)
    final_df.to_csv(r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\data\reduced_data_v2.csv", index=False)
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    preprocess_data()