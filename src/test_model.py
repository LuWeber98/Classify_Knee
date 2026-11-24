
import torch
from model_resnet import ResNet50Transfer
from data_loader import train_loader_resnet


# Device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Initialize model
model = ResNet50Transfer(num_classes=5).to(device)

# Load one batch
images, labels = next(iter(train_loader_resnet))

images = images.to(device)

# Forward pass
model.eval()

with torch.no_grad():
    output = model(images)

print("Device:", device)
print("Input shape:", images.shape)
print("Output shape:", output.shape)

# Count trainable parameters
trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("Trainable parameters:", trainable_params)
print("Total parameters:", total_params)
