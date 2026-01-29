import os
import logging
from typing import Tuple, List, Optional, Dict, Callable
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
import numpy as np

logger = logging.getLogger(__name__)

class LC25000Dataset(Dataset):
    """
    Custom Dataset class for the LC25000 Lung Cancer Histopathology dataset.
    """
    def __init__(self, root_dir: Path, class_names: List[str], transform: Optional[Callable] = None, mode: str = "train"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.classes = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.samples = self._load_samples()

        if len(self.samples) == 0:
            logger.warning(f"No images found in {self.root_dir}. Check paths.")
        else:
            logger.info(f"Initialized {self.mode} dataset with {len(self.samples)} images.")

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                for file_path in sorted(class_dir.glob(ext)):
                    samples.append((str(file_path), class_idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if img.size[0] < 10 or img.size[1] < 10:
                    raise ValueError(f"Image too small: {img.size}")

                if self.transform:
                    img = self.transform(img)
                return img, target
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            raise e

class TransformSubset(Dataset):
    """
    A Dataset wrapper that applies a specific transform to a Subset.
    Moved to global scope to allow pickling on Windows.
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        # Store samples reference for weighted sampler compatibility
        self.samples = subset.dataset.samples
        
    def __len__(self):
        return len(self.subset)
        
    def __getitem__(self, idx):
        # Retrieve the original image path and label using the subset indices
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(mode: str, img_size: int, mean: List[float], std: List[float]) -> transforms.Compose:
    normalize = transforms.Normalize(mean=mean, std=std)

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

def get_data_loaders(
    data_dir: Path,
    class_names: List[str],
    img_size: int,
    mean: List[float],
    std: List[float],
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    
    # 1. Initialize full dataset
    full_dataset = LC25000Dataset(root_dir=data_dir, class_names=class_names, transform=None)
    
    # 2. Stratified split per class
    train_indices = []
    val_indices = []
    
    labels = [s[1] for s in full_dataset.samples]
    labels = np.array(labels)
    
    for class_idx in range(len(class_names)):
        cls_indices = np.where(labels == class_idx)[0]
        n_class = len(cls_indices)
        split = int(np.floor(val_split * n_class))
        
        # Take last part as validation
        val_indices.extend(cls_indices[-split:])
        train_indices.extend(cls_indices[:-split])

    np.random.seed(seed)
    np.random.shuffle(train_indices)
    
    logger.info(f"Train indices: {len(train_indices)}, Val indices: {len(val_indices)}")
    
    # 3. Create subsets with transforms
    train_transform = get_transforms('train', img_size, mean, std)
    val_transform = get_transforms('val', img_size, mean, std)
    
    train_subset_base = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset_base = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_subset = TransformSubset(train_subset_base, train_transform)
    val_subset = TransformSubset(val_subset_base, val_transform)
    
    # 4. Handle Class Imbalance (optional)
    sampler = None
    shuffle = True
    if use_weighted_sampler:
        train_targets = [full_dataset.samples[i][1] for i in train_indices]
        class_counts = np.bincount(train_targets)
        class_counts = np.maximum(class_counts, 1)
        class_weights = 1. / class_counts
        sample_weights = [class_weights[t] for t in train_targets]
        
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"DataLoaders created. Train size: {len(train_subset)}, Val size: {len(val_subset)}")
    
    return train_loader, val_loader