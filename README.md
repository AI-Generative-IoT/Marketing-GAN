# Marketing-GAN
Ziel dieses Projektes, welches im Rahmen der Vorlesungsreihe 'Datenmodellierung und Datenbanksysteme' stattfindet, ist es ein Kozept für ein GAN (Generative Adverserial Network) zu entwickeln, welches in der Lage ist, Verkaufszahlen für bestimmte Verkaufsperioden vorherzusagen. Die Files in diesem Repository stellen eine Sammlung der grundlegenden Routinen in der Programmiersprache Python dar, welche dazu benötigt werden.

# Installation unter Linux und Mac Betriebssystemen
*********************************************************************************************************************
    git clone https://github.com/alexej-schelle/Marketing-GAN.git and start the software with python filename.py
*********************************************************************************************************************

# Installation unter Windows Betriebssystemen
*********************************************************************************************************************
    Download files at https://github.com/alexej-schelle/Marketing-GAN.git and start the software with filename.py
*********************************************************************************************************************

# Dokumentation unter Linux, Mac und Windows Betriebssystemen
*******************************************************************************************************************************
    git clone https://github.com/alexej-schelle/Marketing-GAN.git and read docs/README.txt
*******************************************************************************************************************************

# Dokumentierte Anwendungsszenarien
*******************************************************************************************************************************

    - Modellierung von Verkaufszahlen mit statistischer Gleichverteilung um die originalen Daten
    - Vorhersage von Verkaufszahlen für den Datensatz der Firma Adidas (CC0 1.0 Universal License)
    - Vorhersage von Verkaufszahlen für den Datensatz der Firma Rossmann (CC0 1.0 Universal License)
    - Visualisierung von statistischen Kennzahlen und Korrelationen zum Datensatz der Firma Adidas (Quelle: siehe Kaggle.com)
    
*******************************************************************************************************************************

# Übersicht und Beschreibung der Komponenten
*******************************************************************************************************************************

    - GAN.py modelliert Verkaufszahlen durch den Vergleich von Verkaufszahlen aus zufällig generierten Daten und
      originalen Daten durch ein GAN. Generator und Diskriminator bilden zwei Komponenten im GAN, jede Komponente
      wird dabei durch ein künstliches neuronales Netzwerk modelliert.
    - Rossmann*.py modelliert Verkaufszahlen durch ein SVM Modell.
    - SyntheticDataGenerator.py modelliert künstliche Verkaufszahlen nach einer Gleichverteilung
    - VisualizeAdidasSalesData.py wertet den Datensatz adidas_dataset_new.csv der Firma Adidas aus (Quelle: siehe Kaggle.com)
 
********************************************************************************************************************************

# Autoren
*********************************************************************************************************************

- Onurcan Cesmeci
- Elena Gardea Harder
- Hubert Sellmer
- FH-Doz. Dr. Alexej Schelle

*********************************************************************************************************************

# Supported by
*********************************************************************************************************************
- NLP Model ChatGPT 3.5
*********************************************************************************************************************


