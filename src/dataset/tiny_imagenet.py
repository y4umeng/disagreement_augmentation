from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class TinyImageNetDataset(Dataset):
    """Custom PyTorch Dataset for Tiny ImageNet loaded from Hugging Face."""
    def __init__(self, hf_dataset, transform=None, train=False):
        self.dataset = hf_dataset
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load a single example
        sample = self.dataset[idx]
        img = sample['image']  # Image object
        label = sample['label']  # Corresponding label
        # print(np.array(img).shape, idx, label)

        # Convert grayscale to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Apply transformation if specified
        if self.transform:
            img = self.transform(img)

        if self.train:
            return img, label, idx
        return img, label


def get_tinyimagenet_dataloader(batch_size, val_batch_size, num_workers):
    """Data Loader for Tiny ImageNet downloaded via Hugging Face."""
    # Download the dataset using the Hugging Face datasets library
    dataset = load_dataset("zh-plus/tiny-imagenet", split=["train", "valid"])
    train_dataset = dataset[0]
    val_dataset = dataset[1]

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    # Wrap the Hugging Face dataset with PyTorch's Dataset class
    train_set = TinyImageNetDataset(train_dataset, transform=train_transform, train=True)
    val_set = TinyImageNetDataset(val_dataset, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers
    )

    # Return loaders and dataset size
    return train_loader, val_loader, len(train_set)
