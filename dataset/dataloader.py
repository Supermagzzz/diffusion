import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import pickle


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.images = os.listdir(img_dir)[:1]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        data = pickle.load(open(img_path, 'rb'))
        if self.transform:
            data = self.transform(data)
        return data
