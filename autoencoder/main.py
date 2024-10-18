import torch
from torchvision import datasets, transforms
from torch import nn
from pytorch_symbolic import Input, SymbolicModel

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Mean and std for MNIST
    ]
)

# train_dataset = datasets.MNIST(
#     root="./data", train=True, download=True, transform=transforms.ToTensor()
# )

from torch.utils.data import Subset
import numpy as np

# Load full MNIST dataset
dataset_full = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

# Define subset size (e.g., 1000 samples)
subset_size = 1000
indices = np.random.choice(len(dataset_full), subset_size, replace=False)

# Create subset of the dataset
train_dataset = Subset(dataset_full, indices)


test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=8, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=8, shuffle=True
)

for image, label in train_loader:
    print(image.shape, label.shape)
    break

input_ = Input(shape=(1, 28, 28))


def encoder(x):
    x = nn.Conv2d(x.C, 4, 3, stride=2, padding=1)(x)
    x = nn.ReLU()(x)

    x = nn.Conv2d(x.C, 8, 3, stride=2, padding=1)(x)
    x = nn.ReLU()(x)

    x = nn.Conv2d(x.C, 16, 3, stride=2, padding=1)(x)
    x = nn.ReLU()(x)

    x = nn.Conv2d(x.C, 32, 3, stride=1, padding=1)(x)
    x = nn.ReLU()(x)

    return x


def decoder(x):
    x = nn.ConvTranspose2d(x.C, 16, 3, stride=1, padding=1)(x)
    x = nn.BatchNorm2d(16)(x)
    x = nn.ReLU()(x)

    x = nn.ConvTranspose2d(x.C, 8, 3, stride=2, padding=1, output_padding=0)(x)
    x = nn.BatchNorm2d(8)(x)
    x = nn.ReLU()(x)

    x = nn.ConvTranspose2d(x.C, 4, 3, stride=2, padding=1, output_padding=1)(x)
    x = nn.BatchNorm2d(4)(x)
    x = nn.ReLU()(x)

    x = nn.ConvTranspose2d(x.C, 1, 3, stride=2, padding=1, output_padding=1)(x)
    x = nn.ReLU()(x)
    return x


x = input_
x = encoder(x)
x = decoder(x)
model = SymbolicModel(inputs=input_, outputs=x)

model.summary()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
num_epochs = 1000

loss = 0
num = 0
for data in train_loader:
    images, _ = data

    outputs = model(images)

    loss += criterion(outputs, images).item()
    num += 1

print(f"Initial loss: {loss/num:.4f}")

for epoch in range(num_epochs):
    for data in train_loader:
        images, _ = data

        outputs = model(images)

        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
