import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("patient_data.csv")

# Encode categorical columns
le = LabelEncoder()
df['Name'] = le.fit_transform(df['Name'])
df['Gender'] = le.fit_transform(df['Gender'])
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])
df['Medicine'] = le.fit_transform(df['Medicine'])

# Drop non-numeric or irrelevant columns
df_features = df.drop(columns=['PatientID', 'Discharge'])

# Standardize
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_features)

# Apply PCA
pca = PCA(n_components=3)
pca_features = pca.fit_transform(scaled_features)

# Save reduced dataset
pca_df = pd.DataFrame(pca_features, columns=['PCA1', 'PCA2', 'PCA3'])
pca_df.to_csv("patient_data_pca.csv", index=False)
