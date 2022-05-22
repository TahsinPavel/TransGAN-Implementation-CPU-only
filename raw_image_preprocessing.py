import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

"""class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir = './data', transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label"""

data_dir = './data'

dataset = ImageFolder(data_dir+'/train', transform=ToTensor())
test_data = ImageFolder(data_dir+'/test', transform=ToTensor())

batch_size = 10

train_dl = DataLoader(dataset, batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size*2)

