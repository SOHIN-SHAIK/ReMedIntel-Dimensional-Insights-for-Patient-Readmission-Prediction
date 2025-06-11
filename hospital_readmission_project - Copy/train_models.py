import os
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

os.makedirs(r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\models", exist_ok=True)

def train_models():
    df = pd.read_csv(r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\data\reduced_data_v2.csv")
    X = df[['PCA1', 'PCA2', 'PCA3']]
    y = df['Readmitted_Days']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    joblib.dump(lin_model, r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\models\linear_model.pkl")

    # Decision Tree
    tree_model = DecisionTreeRegressor()
    tree_model.fit(X_train, y_train)
    joblib.dump(tree_model, r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\models\tree_model.pkl")

    # Deep Learning Model
    dl_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(3,)),
        tf.keras.layers.Dense(1)
    ])
    dl_model.compile(optimizer='adam', loss='mae')
    dl_model.fit(X_train, y_train, epochs=30, verbose=0)
    dl_model.save(r"C:\Users\Shaik Sohin\OneDrive\Desktop\hospital_readmission_project - Copy\models\dl_model.h5")

    print("Models trained and saved successfully.")

if __name__ == "__main__":
    train_models()
