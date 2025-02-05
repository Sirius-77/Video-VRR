import time

import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib import image as mpimg
from PIL import Image
import torchvision.transforms as T
import faiss

import util.parser as parser
from dataloader.val.NYCDataset import NYCDataset
from model.SemVGNet import SemVGNet

localargs = parser.parse_arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt_path = '../ckpt/semvgl_aggr_cate_best_model.pth'
feature_db_file_path = "../demo/feature_db/nyc_db_features.npy"


def plot_util(img_path):
    img = mpimg.imread(img_path)

    fig, ax = plt.subplots()

    ax.imshow(img)

    ax.grid(color='white', linestyle='-', linewidth=1)

    ax.set_xticks(range(0, img.shape[1], 14))
    ax.set_yticks(range(0, img.shape[0], 14))

    plt.show()
    return


def feature_retrieval(query_fea, feature_db_path, topK):
    embed_size = query_fea.shape[1]
    print("embedding size: ", embed_size)
    xq = np.array(query_fea)
    xb = np.load(feature_db_path)
    print("xq.shape: ", xq.shape)
    print("xb.shape: ", xb.shape)

    index = faiss.index_factory(embed_size, 'Flat', faiss.METRIC_L2)
    index.train(xb)
    index.add(xb)

    D, I = index.search(xq, topK)

    result = []
    ds = NYCDataset(input_transform=input_transform())
    b_index = ds.dbImages
    for k in range(0, topK):
        result.append(b_index[I[0][k]].split("/")[1].split(".")[0])
    return result


def input_transform():
    return T.Compose([
        T.Resize([322, 322], interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def eval_img(query_img_path):
    t_s = time.time()
    # Load checkpoint
    checkpoint = torch.load(ckpt_path)

    # Initialize SemVG Model
    model = SemVGNet(localargs)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Load query img
    query_img = Image.open(query_img_path)
    tf = input_transform()
    query_img = tf(query_img)

    # Model Eval
    model = model.eval()
    with torch.no_grad():
        query_img = query_img.to(device).unsqueeze(dim=0)
        # Make image patchable (14, 14 patches)
        b, c, h, w = query_img.shape
        h_new, w_new = (h // 14) * 14, (w // 14) * 14
        query_img = T.CenterCrop((h_new, w_new))(query_img)
        query_fea = model(query_img).cpu()
    print("Time for extraction: {}ms".format(1000 * (time.time() - t_s)))
    # Feature Retrieval
    t_s = time.time()
    result = feature_retrieval(query_fea, feature_db_file_path, 5)
    print("Time for searching: {}ms".format(1000 * (time.time() - t_s)))
    return result