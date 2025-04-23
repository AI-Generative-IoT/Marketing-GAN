##########################################################################################################################################
#                                                                                                                                        #
#   Autor: Onurcan Cesmeci and Dr. Alexej Schelle. Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt  #
#                                                                                                                                        #
##########################################################################################################################################

### Generic Python Code developed in assistance with ChatGPT 3.5 ###

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("adidas_dataset_new.csv", delimiter=';', header=4)
df.info()

df['Operating Margin'] = df['Operating Margin'].str.replace(',', '.')
df['Operating Margin'] = df['Operating Margin'].astype(float)
df['Total Sales'] = df['Total Sales'].astype(float)
df['Total Sales'] = df['Units Sold'] * df['Price per Unit']
df['Operating Profit'] = df['Total Sales'] * df['Operating Margin']
df['Invoice Date'] = pd.to_datetime('1900-01-01') + pd.to_timedelta(df['Invoice Date'] - 2, unit='d')


# Verteilung der Gesamtverkäufe
plt.figure(figsize=(8, 6))
sns.histplot(df["Total Sales"], bins=10, kde=True, color="blue")
plt.title("Verteilung der Gesamtverkäufe", fontsize=16)
plt.xlabel("Verkäufe", fontsize=12)
plt.ylabel("Häufigkeit", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Gesamtumsatz nach Regionen
region_sales = df.groupby("Region")["Total Sales"].sum()

plt.figure(figsize=(8, 6))
region_sales.plot(kind='bar', color='orange', edgecolor='black')
plt.title("Gesamtumsatz nach Regionen", fontsize=16)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Gesamtumsatz", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Gesamtumsatz nach Produktkategorie
product_sales = df.groupby("Product")["Total Sales"].sum()

plt.figure(figsize=(8, 6))
product_sales.plot(kind='bar', color='green', edgecolor='black')
plt.title("Gesamtumsatz nach Produktkategorie", fontsize=16)
plt.xlabel("Produkt", fontsize=12)
plt.ylabel("Gesamtumsatz", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# One-Hot_Encoding
df = pd.get_dummies(df, columns=["Sales Method"], prefix="SalesMethod")
df = pd.get_dummies(df, columns=["Retailer"], prefix="Retailer")
df = pd.get_dummies(df, columns=["Product"], prefix="Product")

#Frequenzcodierung
for column in ["Region", "State", "City"]:
    df[f"{column}_encoded"] = df[column].map(df[column].value_counts())

# Input Frame

X = df.drop(columns=['Retailer ID','Region','State','City','Units Sold','Total Sales','Operating Profit'])
X.info()

# Output Frame

y = df[['Units Sold', 'Total Sales', 'Operating Profit']]
y.info()
