from torch.utils.data import DataLoader

from dataset import KneeArthritisDataset
from preprocessing import (
    train_transform,
    val_transform,
    test_transform
)

# Einstellungen
BATCH_SIZE = 32
NUM_WORKERS = 0

# Dataset erstellen
train_dataset = KneeArthritisDataset(
    csv_file="../Splits/train.csv",
    transform=train_transform
)

val_dataset = KneeArthritisDataset(
    csv_file="../Splits/validation.csv",
    transform=val_transform
)

test_dataset = KneeArthritisDataset(
    csv_file="../Splits/test.csv",
    transform=test_transform
)

# DataLoader erstellen
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)



from torch.utils.data import DataLoader

from dataset import KneeArthritisDataset

from preprocessing import (
    train_transform_resnet,
    val_transform_resnet,
    test_transform_resnet
)


BATCH_SIZE = 32
NUM_WORKERS = 0

train_dataset_resnet = KneeArthritisDataset(
    csv_file="../Splits/train.csv",
    transform=train_transform_resnet
)

val_dataset_resnet = KneeArthritisDataset(
    csv_file="../Splits/validation.csv",
    transform=val_transform_resnet
)

test_dataset_resnet = KneeArthritisDataset(
    csv_file="../Splits/test.csv",
    transform=test_transform_resnet
)


train_loader_resnet = DataLoader(
    train_dataset_resnet,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader_resnet = DataLoader(
    val_dataset_resnet,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader_resnet = DataLoader(
    test_dataset_resnet,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

