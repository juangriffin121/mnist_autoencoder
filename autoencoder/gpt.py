import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import matplotlib.pyplot as plt


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1
            ),  # Output: 32 x 14 x 14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 7 x 7
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                16, 8, kernel_size=3, stride=2, padding=1
            ),  # Output: 8 x 4 x 4 (bottleneck)
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                8, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: 16 x 7 x 7
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: 32 x 14 x 14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Output: 1 x 28 x 28
            nn.Tanh(),  # Use Tanh for [-1, 1] normalization
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(
#                 1, 16, kernel_size=3, stride=2, padding=1
#             ),  # Output: 16 x 14 x 14
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 4, kernel_size=3, stride=2, padding=1),  # Output: 4 x 7 x 7
#             nn.BatchNorm2d(4),
#             nn.ReLU(),
#         )
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(
#                 4, 16, kernel_size=3, stride=2, padding=1, output_padding=1
#             ),  # Output: 16 x 14 x 14
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.ConvTranspose2d(
#                 16, 1, kernel_size=3, stride=2, padding=1, output_padding=1
#             ),  # Output: 1 x 28 x 28
#             nn.Sigmoid(),  # Output values between 0 and 1
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    ]
)

dataset_full = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
subset_size = 1000
indices = np.random.choice(len(dataset_full), subset_size, replace=False)

# Create subset of the dataset
train_dataset = Subset(dataset_full, indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

model = ConvAutoencoder()  # Move the model to GPU if available

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data

        # Forward pass
        output = model(img)
        loss = criterion(output, img)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


def show_images(original, reconstructed):
    original = original.detach().numpy()
    reconstructed = reconstructed.detach().numpy()

    plt.figure(figsize=(8, 4))
    for i in range(8):
        # Original images
        ax = plt.subplot(2, 8, i + 1)
        plt.imshow(original[i][0], cmap="gray")
        plt.axis("off")

        # Reconstructed images
        ax = plt.subplot(2, 8, i + 1 + 8)
        plt.imshow(reconstructed[i][0], cmap="gray")
        plt.axis("off")
    plt.savefig("img.png")


# Get some test images
test_images, _ = next(iter(train_loader))
with torch.no_grad():
    reconstructed_images = model(test_images)

# Show original and reconstructed images
show_images(test_images, reconstructed_images)
