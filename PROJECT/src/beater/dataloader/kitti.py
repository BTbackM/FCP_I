from __future__ import annotations
from os import path, listdir
from PIL import Image
from torch.utils.data import Dataset

# Nodule beater imports
from beater.utils.disk import DATASET_PATH, ABS_PATH

class KITTI(Dataset):
    def __init__(self, root_dir = path.join(DATASET_PATH, 'kitti'), split = 'train', transform = None) -> None:
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Get the image and label directories
        self.image_dir = path.join(root_dir, split, 'image_2')
        self.label_dir = path.join(root_dir, split, 'semantic_rgb')

        # Get the image and label names
        self.image_names = listdir(self.image_dir)
        self.label_names = listdir(self.label_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get the image and label names
        image_name = self.image_names[idx]
        label_name = self.label_names[idx]

        # Get the image and label paths
        image_path = path.join(self.image_dir, image_name)
        label_path = path.join(self.label_dir, label_name)

        # Open the image and label
        image = Image.open(image_path)
        label = Image.open(label_path)

        # Apply the transform
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
