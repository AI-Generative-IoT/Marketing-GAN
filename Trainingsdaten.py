import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Funktion zur Generierung realistischer Verkaufsdaten
def generate_sales_data(start_date, num_days, base_sales=1000, trend=0.1, 
                        seasonality=100, noise_level=50):
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Trend
    trend_component = np.arange(num_days) * trend
    
    # Saisonalität (wöchentlich)
    seasonality_component = np.sin(np.arange(num_days) * (2 * np.pi / 7)) * seasonality
    
    # Zufälliges Rauschen
    noise = np.random.normal(0, noise_level, num_days)
    
    # Kombinieren aller Komponenten
    sales = base_sales + trend_component + seasonality_component + noise
    
    # Sicherstellen, dass alle Verkaufszahlen positiv sind
    sales = np.maximum(sales, 0)
    
    return pd.DataFrame({'Date': dates, 'Sales': sales})

# Generieren der Daten
start_date = datetime(2024, 1, 1)
num_days = 1825  # 5 Jahre an Daten
#num_days = 365  # 1 Jahr an Daten
sales_data = generate_sales_data(start_date, num_days)

# Speichern der Daten als CSV
csv_filename = 'realistic_sales_data.csv'
sales_data.to_csv(csv_filename, index=False)

print(f"Daten wurden in {csv_filename} gespeichert.")
print(sales_data.head())
