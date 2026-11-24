
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import pandas as pd

from models import CustomCNN
from data_loader import train_loader, val_loader


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

NUM_EPOCHS = 30
LEARNING_RATE = 0.001

model = CustomCNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# --------------------------------------------------
# 2. Output directory
# --------------------------------------------------

MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

best_model_path = MODEL_DIR / "custom_cnn_best.pth"


# --------------------------------------------------
# 3. Training and validation functions
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate_one_epoch(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# --------------------------------------------------
# 4. Training loop
# --------------------------------------------------

history = []

best_val_loss = float("inf")

print(f"Training on: {device}")
print(f"Number of epochs: {NUM_EPOCHS}")
print("-" * 65)

for epoch in range(NUM_EPOCHS):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_accuracy = validate_one_epoch(
        model,
        val_loader,
        criterion,
        device
    )

    # Save metrics
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    # Save best model based on validation loss
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        # Move saved weights to CPU
        state_dict = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }

        torch.save(state_dict, best_model_path)

        best_marker = " <-- Best model saved"

    else:
        best_marker = ""

    print(
        f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
        f"{best_marker}"
    )


# --------------------------------------------------
# 5. Save training history
# --------------------------------------------------

history_df = pd.DataFrame(history)

history_df.to_csv(
    MODEL_DIR / "custom_cnn_history.csv",
    index=False
)

print("-" * 65)
print("Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best model saved at: {best_model_path}")
