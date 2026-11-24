
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class KneeArthritisDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_path = self.data.iloc[index]["image_path"]
        label = int(self.data.iloc[index]["label"])

        # Load Image and convert to RGB
        image = Image.open(image_path).convert("RGB")

        # Transform
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label
