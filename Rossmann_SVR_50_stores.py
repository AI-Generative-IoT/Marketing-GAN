""" 
Titel: Rossmann_SVR_50_stores.py
Autor: Hubert Sellmer
Organisation: IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt
Datum: 31. Januar 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Dateien einlesen
train_file = "train_50_stores.csv"  # Originaldaten
store_file = "store.csv"

# Explizite Datentypen festlegen
dtype_dict = {
    "StateHoliday": str,  # Spalte mit gemischten Typen als String interpretieren
}

train_data = pd.read_csv(train_file, dtype=dtype_dict, low_memory=False)
store_data = pd.read_csv(store_file)

# 2. Daten vorbereiten
def preprocess_data(train_data, store_data):
    # Merge Store-Informationen
    data = pd.merge(train_data, store_data, on="Store", how="left")

    # Datum in Jahr, Monat, Tag umwandeln
    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day

    # Fehlende Werte behandeln
    data["CompetitionDistance"] = data["CompetitionDistance"].fillna(data["CompetitionDistance"].median())
    data.fillna(0, inplace=True)

    # Kategorische Variablen kodieren
    label_encoders = {}
    categorical_cols = ["StoreType", "Assortment", "StateHoliday"]

    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    return data

# 3. Daten vorverarbeiten
train_data = preprocess_data(train_data, store_data)

# 4. Features und Zielvariable definieren
features = [
    "Store", "DayOfWeek", "Promo", "Year", "Month", "Day", "StateHoliday",
    "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance"
]
X = train_data[features]
y = train_data["Sales"]

# 5. Datenaufteilung in Training und Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Daten normalisieren
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. SVM-Modell trainieren
svr = SVR(kernel="rbf", C=1000, gamma=0.1, epsilon=0.01)
svr.fit(X_train_scaled, y_train)

# 8. Vorhersagen und Bewertung
y_pred = svr.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# 9. Visualisierung der Vorhersagen
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual Sales", alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted Sales", alpha=0.6)
plt.title("Actual vs Predicted Sales")
plt.xlabel("Sample Index")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.show()