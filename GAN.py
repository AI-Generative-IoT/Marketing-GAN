import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Pfad-Angabe nicht nötig, weil im gleichen Verzeichnis
data = pd.read_csv('realistic_sales_data.csv')

# Generator-Modell
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')  # Verkaufszahlen-Output
    ])
    return model

# Diskriminator-Modell
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Echtheit-Wahrscheinlichkeit
    ])
    return model

# GAN-Modell
class SalesGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(SalesGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(SalesGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_sales):
        batch_size = tf.shape(real_sales)[0]
        noise = tf.random.normal([batch_size, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_sales = self.generator(noise, training=True)

            real_output = self.discriminator(real_sales, training=True)
            fake_output = self.discriminator(generated_sales, training=True)

            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            disc_loss = self.loss_fn(tf.ones_like(real_output), real_output) + \
                        self.loss_fn(tf.zeros_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return {"g_loss": gen_loss, "d_loss": disc_loss}

# Modell initialisieren und kompilieren
generator = build_generator()
discriminator = build_discriminator()
gan = SalesGAN(generator, discriminator)
gan.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=tf.keras.losses.BinaryCrossentropy()
)

# Trainingsdaten vorbereiten (Beispiel)
real_sales_data = data['Sales'].values.reshape(-1, 1)

# Modell trainieren
gan.fit(real_sales_data, epochs=1000, batch_size=32)

# Neue Verkaufszahlen generieren
noise = tf.random.normal([1000, 100])
generated_sales = generator(noise).numpy()

# Visualisierung hinzufügen
plt.figure(figsize=(10, 6))
plt.hist(real_sales_data, bins=50, alpha=0.5, label='Echte Daten')
plt.hist(generated_sales, bins=50, alpha=0.5, label='Generierte Daten')
plt.xlabel('Verkaufszahlen')
plt.ylabel('Häufigkeit')
plt.title('Vergleich von echten und generierten Verkaufszahlen')
plt.legend()
plt.grid(True)
plt.show()

# Die ersten 20 generierten Verkaufszahlen ausgeben
print("\nDie ersten 20 generierten Verkaufszahlen:")
for i, sale in enumerate(generated_sales[:20]):
    print(f"Verkauf {i+1}: {sale[0]:.2f}")
    
 # Erstellen eines DataFrames mit den generierten Daten
generated_df = pd.DataFrame(generated_sales, columns=['Generated_Sales'])

# Hinzufügen einer Datumsspalte, beginnend mit dem letzten Datum aus den echten Daten
last_date = pd.to_datetime(data['Date'].iloc[-1])
date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(generated_sales))
generated_df['Date'] = date_range

# Neuanordnung der Spalten
generated_df = generated_df[['Date', 'Generated_Sales']]

# Speichern der generierten Daten als CSV-Datei
generated_df.to_csv('generierte_daten.csv', index=False)

print("Generierte Daten wurden in 'generierte_daten.csv' gespeichert.")