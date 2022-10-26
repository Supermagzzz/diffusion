import os
from torch.utils.data import Dataset
import torch
import pickle


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.images = []
        for i in os.listdir(img_dir):
            if i.endswith('.pkl'):
                img_path = os.path.join(img_dir, i)
                data = pickle.load(open(img_path, 'rb'))
                self.images.append(data)
                break
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
        img_path = os.path.join(self.img_dir, self.images[idx])
        data = pickle.load(open(img_path, 'rb'))
        if self.transform:
            data = self.transform(data)
        return data
