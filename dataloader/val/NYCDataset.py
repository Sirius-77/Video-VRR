import os

from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import sys

sys.path.append("../../")
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent.parent.parent / 'DS' / 'NYCDataset' / 'streetview'
GT_ROOT = Path(__file__).resolve().parent.parent.parent / 'dataset' / 'NYC'


class NYCDataset(Dataset):
    def __init__(self, input_transform=None):
        self.input_transform = input_transform
        self.dbImages = np.load(os.path.join(GT_ROOT, 'NYC_dbImages.npy'))
        self.images = self.dbImages

    def __getitem__(self, index):
        img = Image.open(os.path.join(DATASET_ROOT,self.images[index]))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    cropped_sv_path = "../../DS/NYCDataset/streetview/ny_persp"
    dbImages = []
    paths = os.listdir(cropped_sv_path)
    for path in paths:
        p = "ny_persp/" + path
        print(p)
        dbImages.append(p)
    dbImages = np.array(dbImages)
    np.save(f"{GT_ROOT}NYC_dbImages.npy", dbImages)
