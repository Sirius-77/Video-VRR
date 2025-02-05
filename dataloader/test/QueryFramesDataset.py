import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class QueryFramesDataset(Dataset):
    def __init__(self, root_dir, query_frames, input_transform=None):
        self.query_frames = query_frames
        self.root_dir = root_dir
        self.input_transform = input_transform

    def __getitem__(self, index):
        img = Image.open(self.root_dir + self.query_frames[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.query_frames)