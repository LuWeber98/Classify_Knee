
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from model_resnet import ResNet50Transfer
from data_loader import train_loader_resnet, val_loader_resnet

# Device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Training on:", device)

# Model
model = ResNet50Transfer(num_classes=5).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.resnet.fc.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

# Training settings
NUM_EPOCHS = 30
best_val_loss = float("inf")

history = []

best_model_path = "../models/resnet50_best.pth"
history_path = "../models/resnet50_history.csv"


# Training loop
for epoch in range(NUM_EPOCHS):

    # -------------------------
    # Training
    # -------------------------
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader_resnet:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= train_total
    train_accuracy = train_correct / train_total


    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader_resnet:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    # -------------------------
    # Update learning rate
    # -------------------------
    scheduler.step(val_loss)

    # Optional: print current learning rate(s)
    current_lrs = [group["lr"] for group in optimizer.param_groups]
    print(f"Learning rate(s): {current_lrs}")


    # -------------------------
    # Save history
    # -------------------------
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    print(
        f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # -------------------------
    # Save best model
    # -------------------------
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        state_dict = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }

        torch.save(state_dict, best_model_path)

        print("Best model saved!")


# Save training history
pd.DataFrame(history).to_csv(history_path, index=False)

print("\nTraining completed!")
print("Best validation loss:", best_val_loss)
print("Model saved to:", best_model_path)
print("History saved to:", history_path)
