import torch
import torchvision.transforms as T
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloader.val.NYCDataset import NYCDataset
from model.SemVGNet import SemVGNet
from util import parser

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
args = parser.parse_arguments()


def input_transform(image_size=args.resize):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def extract_features(
        batch_size=64,
        ckpt_path='../ckpt/semvgl_aggr_cate_best_model.pth'):
    # Load checkpoint
    checkpoint = torch.load(ckpt_path)

    # Initialize SemVG Model
    model = SemVGNet(args)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Initialize Dataset
    dataset = NYCDataset(input_transform=input_transform())
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = model.eval()
    descs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, 'Extracting features...'):
            imgs, labels = batch
            output = model(imgs.to(device)).cpu()
            descs.append(output)
    descs = torch.cat(descs).numpy()
    print(descs.shape)
    np.save("feature_db/nyc_db_features.npy", descs)
    return


if __name__ == '__main__':
    extract_features()