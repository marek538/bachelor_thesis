import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class TransformToTensor(Dataset):
    """
        Initializes the training and validation data loaders, including dataset splitting, transforms, and class balancing.

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

    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = self.root_dir / self.labels_frame.iloc[idx, 0]

        image = Image.open(img_name).convert('RGB')
        label = int(self.labels_frame.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
