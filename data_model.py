import os
import torch
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

        # Load bounding boxes (assuming the bbox file is in the format of: image_id xmin ymin xmax ymax)
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

        # Get bounding box
        bbox = self.bboxes[idx]

        # Get size of image
        size = self.image_sizes[idx]

        # Apply transformations if any
        if self.transform:
            # To apply the same transformation to both image and bbox, we pass them together as a dictionary
            image, bbox = self.transform(image, bbox)

        return image, bbox


# Define a custom transform to handle both image and bbox
class ToTensorWithBBox:
    def __call__(self, image, bbox):
        # Convert the image to a tensor
        image = transforms.ToTensor()(image)
        # Bboxes should be converted to float
        bbox = torch.tensor(bbox, dtype=torch.float32)
        return image, bbox

class ResizeWithBBox:
    def __init__(self, size):
        self.size = size
    def __call__(self, image, bbox):
        dimensions = image.size
        x_strech = dimensions[0]/self.size[0]
        y_strech = dimensions[1]/self.size[1]

        image = transforms.Resize(self.size)(image)
        bbox[0] = bbox[0]/x_strech
        bbox[1] = bbox[1]/y_strech
        bbox[2] = bbox[2]/x_strech
        bbox[3] = bbox[3]/y_strech

        return image, bbox