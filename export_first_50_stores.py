""" 
Titel: export_first_50_stores.py
Autor: Hubert Sellmer
Organisation: IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt
Datum: 31. Januar 2025
"""


import pandas as pd

# Originaldatei einlesen
input_file = "train.csv"  # Pfad zur Originaldatei
output_file = "train_50_stores.csv"  # Pfad zur neuen Datei

# Originaldaten einlesen
train_data = pd.read_csv(input_file)

# Filter: Nur die ersten 50 Stores
filtered_data = train_data[train_data["Store"] <= 50]

# Neuen Datensatz speichern
filtered_data.to_csv(output_file, index=False)

print(f"Der gefilterte Datensatz wurde gespeichert unter: {output_file}")
print(f"Anzahl der Zeilen im gefilterten Datensatz: {len(filtered_data)}")