import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "../Raw_Images"

image_paths = []
labels = []

classes = ["0", "1", "2", "3", "4"]

for label in classes:

    class_path = os.path.join(DATASET_PATH, label)

    for image in os.listdir(class_path):

        if image.lower().endswith(".png"):

            image_path = os.path.join(class_path, image)

            image_paths.append(image_path)
            labels.append(int(label))

df = pd.DataFrame({
    "image_path": image_paths,
    "label": labels
})

print(df.head())
print("\nGesamtanzahl Bilder:", len(df))

print("\nKlassenverteilung:")
print(df["label"].value_counts().sort_index())

# 70 % Training, 30 % Rest
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# 15 % Validation, 15 % Test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

os.makedirs("../Splits", exist_ok=True)

train_df.to_csv("../Splits/train.csv", index=False)
val_df.to_csv("../Splits/validation.csv", index=False)
test_df.to_csv("../Splits/test.csv", index=False)