import os
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from tqdm import tqdm


class CelebADataset(Dataset):
    def __init__(self, image_dir, bbox_file, sizes_file, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            bbox_file (str): Path to the file containing bounding box annotations.
            transform (callable, optional): Optional transform to be applied to both images and bounding boxes.
        """
        self.image_dir = image_dir
        self.bbox_file = bbox_file
        self.sizes_file = sizes_file
        self.transform = transform

        self.bboxes = []
        self.image_names = []
        self.image_sizes = []
        with open(bbox_file, 'r') as f:
            for i, line in enumerate(f):
                if i == 0 or i == 1:
                    continue
                parts = line.strip().split()
                image_name = parts[0]
                bbox = parts[1:]
                self.bboxes.append([int(i) for i in bbox])
                self.image_names.append(image_name)

        with open(sizes_file, 'r') as f:
            for i, line in enumerate(f):
                if i == 0 or i == 1:
                    continue
                parts = line.strip().split()
                sizes = parts[1:]
                self.image_sizes.append([int(i) for i in sizes])


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")

        bbox = self.bboxes[idx].copy()
        
        #size = self.image_sizes[idx]

        # Apply transformations if any
        if self.transform:
            image, bbox = self.transform(image, bbox)

        return image, bbox


class ToTensorWithBBox:
    def __init__(self):
        self.image_transform = transforms.ToTensor()

    def __call__(self, image, bbox):
        image = self.image_transform(image)
        bbox = torch.tensor(bbox, dtype=torch.float32)
        return image, bbox


class ResizeWithBBox:
    def __init__(self, size):
        self.size = size
        self.image_transform = transforms.Resize(self.size)

    def __call__(self, image, bbox):
        x_strech = image.size[0]/self.size[0]
        y_strech = image.size[1]/self.size[1]

        image = self.image_transform(image)
        bbox[0] = bbox[0]/x_strech
        bbox[1] = bbox[1]/y_strech
        bbox[2] = bbox[2]/x_strech
        bbox[3] = bbox[3]/y_strech

        return image, bbox


class NormaliseWithBBox:
    def __init__(self, mean, std):
        self.image_transform = torchvision.transforms.Normalize(mean, std)

    def __call__(self, image, bbox):
        image = self.image_transform(image)
        return image, bbox


class RandomHorizontalFlipWithBBox:
    def __init__(self, prob):
        self.image_transform = torchvision.transforms.RandomHorizontalFlip(p=1)
        self.prob = prob
        
    def __call__(self, image, bbox):
        if random.random() < self.prob:
            image = torchvision.transforms.v2.functional.horizontal_flip(image)
            bbox[0] = image.size[0] - bbox[0] - bbox[2]
        return image, bbox


class RandomPadWithBBox:
    def __init__(self, padding_range):
        self.padding_range = padding_range

    def __call__(self, image, bbox):
        padding = random.randint(self.padding_range[0], self.padding_range[1]) * min(image.size) // 100

        image = torchvision.transforms.functional.pad(image, padding, fill=0, padding_mode='edge')

        bbox[0] += padding  # Shift x_min by padding amount
        bbox[1] += padding  # Shift y_min by padding amount

        return image, bbox