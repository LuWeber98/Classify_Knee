
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

from model_resnet import ResNet50Transfer
from data_loader import test_loader_resnet


# Device
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Load model
model = ResNet50Transfer(num_classes=5)

checkpoint = torch.load(
    "../models/resnet50_best.pth",
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()


# Predictions
all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader_resnet:

        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())


# Metrics
accuracy = accuracy_score(all_labels, all_predictions)
balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
macro_f1 = f1_score(
    all_labels, all_predictions, average="macro"
)

print("\n--- ResNet50 Test Results ---")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=[
            "KL Grade 0",
            "KL Grade 1",
            "KL Grade 2",
            "KL Grade 3",
            "KL Grade 4"
        ],
        zero_division=0
    )
)


# Confusion Matrix
cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3, 4]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1, 2, 3, 4]
)

disp.plot(cmap="Blues", values_format="d")
plt.title("ResNet50 - Test Confusion Matrix")
plt.tight_layout()
plt.savefig("../models/resnet50_confusion_matrix.png", dpi=300)
plt.show()
