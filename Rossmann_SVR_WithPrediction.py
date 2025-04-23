""" 
Titel: Rossmann_SVR_WithPrediction.py
Autor: Hubert Sellmer
Organisation: IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt
Datum: 31. Januar 2025
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
#import matplotlib.pyplot as plt

# 1. Dateien einlesen
train_file = "train_50_stores.csv"  # Gefilterte Trainingsdaten
store_file = "store.csv"  # Zusätzliche Store-Daten

train_data = pd.read_csv(train_file)
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

train_data = preprocess_data(train_data, store_data)

# 3. Features und Zielvariable definieren
features = [
    "Store", "DayOfWeek", "Promo", "Year", "Month", "Day", "StateHoliday",
    "SchoolHoliday", "StoreType", "Assortment", "CompetitionDistance"
]
X = train_data[features]
y = train_data["Sales"]

# 4. Daten normalisieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. SVR-Modell trainieren
svr = SVR(kernel="rbf", C=1000, gamma=0.1, epsilon=0.01)
svr.fit(X_scaled, y)

# 6. Prognosefunktion erstellen
def predict_sales(store_id, date, promo, state_holiday, school_holiday):
    # Datum aufteilen in Jahr, Monat, Tag
    year = pd.to_datetime(date).year
    month = pd.to_datetime(date).month
    day = pd.to_datetime(date).day

    # Holen der Store-Informationen
    store_info = store_data[store_data["Store"] == store_id]
    if store_info.empty:
        raise ValueError(f"Store-ID {store_id} ist im Datensatz nicht vorhanden.")
    
    # Input-Daten erstellen
    input_data = pd.DataFrame({
        "Store": [store_id],
        "DayOfWeek": [pd.to_datetime(date).dayofweek + 1],
        "Promo": [promo],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "StateHoliday": [state_holiday],
        "SchoolHoliday": [school_holiday],
        "StoreType": [store_info["StoreType"].values[0]],
        "Assortment": [store_info["Assortment"].values[0]],
        "CompetitionDistance": [store_info["CompetitionDistance"].values[0]]
    })

    # Kodierung sicherstellen
    input_data["StoreType"] = LabelEncoder().fit_transform(input_data["StoreType"].astype(str))
    input_data["Assortment"] = LabelEncoder().fit_transform(input_data["Assortment"].astype(str))

    # Normalisieren
    input_scaled = scaler.transform(input_data)

    # Prognose erstellen
    predicted_sales = svr.predict(input_scaled)[0]
    return predicted_sales

# 7. Prognose für Promo-Szenarien
def promo_effect(store_id, date, state_holiday, school_holiday):
    no_promo_sales = predict_sales(store_id, date, promo=0, state_holiday=state_holiday, school_holiday=school_holiday)
    promo_sales = predict_sales(store_id, date, promo=1, state_holiday=state_holiday, school_holiday=school_holiday)
    effect = promo_sales - no_promo_sales
    return no_promo_sales, promo_sales, effect

# 8. Interaktive Eingabe für zukünftige Prognosen
while True:  # Die Schleife läuft, bis der Benutzer sie beendet
    # Eingaben für die Prognose
    store_id = int(input("Geben Sie die Store-ID ein: "))
    date = input("Geben Sie ein Datum ein (YYYY-MM-DD): ")
    promo = int(0) # Eingabe umgehen und stattdessen 0 festlegen
    state_holiday = int(input("State Holiday? (0 = Kein Feiertag, 1 = Feiertag): "))
    school_holiday = int(input("School Holiday? (0 = Nein, 1 = Ja): "))

    # Prognose berechnen
    try:
        predicted_sales = predict_sales(store_id, date, promo, state_holiday, school_holiday)
        print(f"Die prognostizierten Verkäufe ohne Promo für Store {store_id} am {date} sind: {predicted_sales:.2f}")
        # Promo-Wirkung berechnen
        no_promo, with_promo, promo_effect_value = promo_effect(store_id, date, state_holiday, school_holiday)
        print(f"Mit Promo: {with_promo:.2f}, Wirkung der Promo: {promo_effect_value:.2f}")
    except ValueError as e:
        print(f"Fehler: {e}")

    # Möchte der Benutzer eine weitere Prognose durchführen?
    repeat = input("Möchten Sie eine weitere Prognose durchführen? (ja/nein): ").strip().lower()
    if repeat != "ja":
        print("Programm beendet.")
        break  # Schleife beenden, wenn der Benutzer 'nein' eingibt
