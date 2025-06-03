import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from create_bags import creating_bags
from load_dataset import MILDataset
from evaluation import evaluate_model
from plot_targets_initial import plot_save_targets_distribution
from plt_loss_values import plot_loss
from actual_model import MILModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load training dataset (MNIST Train)
mnist_train = datasets.MNIST(root='./data', train=True, download=True)
filtered_train = [(img, label) for img, label in zip(mnist_train.data, mnist_train.targets) if label in [0, 7]]

# Convert to tensors
images_train, labels_train = zip(*filtered_train)
images_train = torch.stack(images_train).unsqueeze(1).float() / 255.0  # Normalize
labels_train = torch.tensor(labels_train)

# Create bags for training
bags_train, targets_train = creating_bags(images_train, labels_train, len(filtered_train) // 100)

# Create dataset for training
dataset_train = MILDataset(bags_train, targets_train)

# Split into Train and Validation
train_size = int(0.85 * len(dataset_train))  # 70% for training
val_size = len(dataset_train) - train_size  # 30% for validation
train_dataset, val_dataset = random_split(dataset_train, [train_size, val_size])

# Load test dataset (MNIST Test)
mnist_test = datasets.MNIST(root='./data', train=False, download=True)
filtered_test = [(img, label) for img, label in zip(mnist_test.data, mnist_test.targets) if label in [0, 7]]

# Convert test data to tensors
images_test, labels_test = zip(*filtered_test)
images_test = torch.stack(images_test).unsqueeze(1).float() / 255.0  # Normalize
labels_test = torch.tensor(labels_test)

# Create bags for testing
bags_test, targets_test = creating_bags(images_test, labels_test, len(filtered_test) // 100)

# Create dataset for testing
test_dataset = MILDataset(bags_test, targets_test)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Save initial target distribution plots
plot_save_targets_distribution(train_loader, "train_targets.png")
plot_save_targets_distribution(val_loader, "val_targets.png")
plot_save_targets_distribution(test_loader, "test_targets.png")

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        total_train_loss = 0

        # Training Loop
        for bags, targets in train_loader:
            targets = targets.float().to(device)
            bags = bags.to(device)

            optimizer.zero_grad()
            outputs = model(bags)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for bags, targets in val_loader:
                targets = targets.float().to(device)
                bags = bags.to(device)
                outputs = model(bags)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        model.train()

    # Plot Loss Curves
    plot_loss(train_losses, "Train_Loss.png")
    plot_loss(val_losses, "Val_Loss.png")

# Initialize and train model
model = MILModel().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)

# Save the trained model
torch.save(model.state_dict(), "trained_MIL.pth")

# Evaluate the model
evaluate_model(model, train_loader, device, test_or_train = True)  # Train evaluation
evaluate_model(model, test_loader, device, test_or_train = False)   # Test evaluation
