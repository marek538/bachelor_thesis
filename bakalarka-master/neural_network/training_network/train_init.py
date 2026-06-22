import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
import collections
import transform_to_tensor
import TRAINING_CONSTANTS

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "labels.csv"
IMAGES_PATH = BASE_DIR / "data" / "images"


# Wrapper class to apply transforms AFTER the dataset has been split
class DatasetWrapper(Dataset):
    """
    A wrapper for a dataset subset that applies a specified transform to each image sample.

    Parameters:
        subset (torch.utils.data.Dataset): The dataset subset (e.g., from random_split).
        transform (callable, optional): Transform to apply to each image (e.g., augmentation or normalization).

    Methods:
        __getitem__(index):
            Returns the transformed image and its label at the given index.

        __len__():
            Returns the number of samples in the subset.
    """

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # Extract the raw PIL image and label from the subset
        image, label = self.subset[index]

        # Apply the corresponding transform (augmentation for train, basic for val)
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.subset)


def initialize_training():
    """
        Initializes the training and validation data loaders, including dataset splitting, transforms, and class
        balancing.

        Loads the dataset, splits it into training and validation subsets, applies appropriate transforms,
        computes class weights for balanced sampling, and returns the data loaders and device.

        Parameters:
            None

        Returns:
            tuple:
                - device (torch.device): The device to use for training (CPU or CUDA).
                - train_loader (DataLoader): DataLoader for the training set with weighted sampling.
                - val_loader (DataLoader): DataLoader for the validation set.
                - train_dataset (Dataset): The wrapped training dataset.
    """

    # 1. Load the raw dataset WITHOUT any transforms (returns raw PIL images)
    full_dataset = transform_to_tensor.TransformToTensor(
        csv_file=CSV_PATH, root_dir=IMAGES_PATH, transform=None
    )

    if len(full_dataset) == 0:
        raise ValueError("Dataset is empty.")

    # 2. Split the raw dataset first (Ensures zero data leakage)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    raw_train_subset, raw_val_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # SAMPLING
    train_labels = [label.item() for _, label in raw_train_subset]

    # Count how many samples belong to class 0 and class 1
    class_counts = collections.Counter(train_labels)

    # 2.2 Calculate the weight for each class (Inverse frequency: 1 / count)
    # The class with fewer samples will have a higher weight
    class_weights = {
        class_id: 1.0 / count for class_id, count in class_counts.items()
    }

    # 2.3 Assign a weight to EVERY INDIVIDUAL SAMPLE in the training set based on its class
    sample_weights = [class_weights[label] for label in train_labels]

    # 2.4 Create the sampler
    # replacement=True allows the rare samples to be drawn multiple times per epoch
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # 3. Define specific transforms for training (with augmentation) and validation (clean)
    train_transforms = transforms.Compose([
        transforms.Resize((TRAINING_CONSTANTS.CELL_WIDTH, TRAINING_CONSTANTS.CELL_HEIGHT)),
        # transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.7, 1.3)),
        transforms.RandomRotation(degrees=5, fill=255),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.11), fill=255),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((TRAINING_CONSTANTS.CELL_WIDTH, TRAINING_CONSTANTS.CELL_HEIGHT)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 4. Wrap the subsets to apply the defined transforms dynamically
    train_dataset = DatasetWrapper(raw_train_subset, transform=train_transforms)
    val_dataset = DatasetWrapper(raw_val_subset, transform=val_transforms)

    # 5. Create DataLoaders
    batch_size = 64

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # sampler loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, train_loader, val_loader, train_dataset
