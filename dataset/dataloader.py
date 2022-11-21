import os
from torch.utils.data import Dataset
import torch
import pickle


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, train=True):
        self.images = []
        for i in os.listdir(img_dir):
            if i.endswith('.pkl'):
                img_path = os.path.join(img_dir, i)
                data = pickle.load(open(img_path, 'rb'))
                self.images.append((img_path, data))
        self.images.sort(key=lambda path, im: hash(path))

        self.train = self.images[:int(len(self.images) * 0.9)]
        self.test = self.images[int(len(self.images) * 0.9):]
        self.images = self.train if train else self.test

        self.img_dir = img_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
