from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class ImageDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):

        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.df.iloc[idx,0])
        image = Image.open(img_path).convert("RGB")

        label = self.df.iloc[idx,1]

        if self.transform:
            image = self.transform(image)

        return image, label