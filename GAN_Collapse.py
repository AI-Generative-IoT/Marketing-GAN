import numpy as np
from typing import List, Tuple
import pandas as pd

class Generator:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)

    def forward(self, z: np.ndarray) -> np.ndarray:
        h = np.tanh(z @ self.w1)
        return h @ self.w2

    def backward(self, z: np.ndarray, d_out: np.ndarray, lr: float) -> None:
        h = np.tanh(z @ self.w1)
        d_h = d_out @ self.w2.T * (1 - h**2)
        self.w2 -= lr * h.T @ d_out
        self.w1 -= lr * z.T @ d_h

class Discriminator:
    def __init__(self, input_size: int, hidden_size: int):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, 1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1)
        return 1 / (1 + np.exp(-h @ self.w2))

    def backward(self, x: np.ndarray, d_out: np.ndarray, lr: float) -> None:
        h = np.tanh(x @ self.w1)
        d_h = d_out @ self.w2.T * (1 - h**2)
        self.w2 -= lr * h.T @ d_out
        self.w1 -= lr * x.T @ d_h

class SalesGAN:
    def __init__(self, noise_dim: int, hidden_dim: int):
        self.generator = Generator(noise_dim, hidden_dim, 1)
        self.discriminator = Discriminator(1, hidden_dim)

    def train(self, real_data: np.ndarray, epochs: int, batch_size: int, lr: float) -> List[Tuple[float, float]]:
        losses = []
        for epoch in range(epochs):
            for _ in range(len(real_data) // batch_size):
                # Train Discriminator
                real_batch = real_data[np.random.choice(len(real_data), batch_size)]
                z = np.random.randn(batch_size, self.generator.w1.shape[0])
                fake_batch = self.generator.forward(z)

                real_output = self.discriminator.forward(real_batch)
                fake_output = self.discriminator.forward(fake_batch)

                d_loss = -np.mean(np.log(real_output) + np.log(1 - fake_output))

                d_real_out = -(1 - real_output)
                d_fake_out = fake_output
                self.discriminator.backward(real_batch, d_real_out, lr)
                self.discriminator.backward(fake_batch, d_fake_out, lr)

                # Train Generator
                z = np.random.randn(batch_size, self.generator.w1.shape[0])
                fake_batch = self.generator.forward(z)
                fake_output = self.discriminator.forward(fake_batch)

                g_loss = -np.mean(np.log(fake_output))

                d_fake_out = -(1 - fake_output)
                d_out = self.discriminator.w2.T @ d_fake_out * (1 - fake_output**2)
                self.generator.backward(z, d_out, lr)

            if epoch % 10 == 0:
                losses.append((d_loss, g_loss))
                print(f"Epoch {epoch}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")

        return losses

    def generate_sales(self, num_samples: int) -> np.ndarray:
        z = np.random.randn(num_samples, self.generator.w1.shape[0])
        return self.generator.forward(z)

# Beispielverwendung
if __name__ == "__main__":
    # Erstellen von synthetischen "echten" Daten
    real_sales_data = np.random.normal(1000, 200, (10000, 1))

    # Initialisieren und Trainieren des GANs
    gan = SalesGAN(noise_dim=10, hidden_dim=64)
    gan.train(real_sales_data, epochs=1000, batch_size=64, lr=0.0001)

    # Generieren neuer Verkaufszahlen
    generated_sales = gan.generate_sales(1000)

    print("Beispiel generierter Verkaufszahlen:")
    print(generated_sales[:20])
