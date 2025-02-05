import os
import random

import numpy as np
from pytorch_metric_learning.distances import DotProductSimilarity
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

import util.parser as parser
from os.path import join
from dataloader.train.GSVCitiesDataloader import GSVCitiesDataModule
from dataloader.val.PittsburghDataset import PittsburghDataset
from model.SemVGNet import SemVGNet
from pytorch_metric_learning import losses, miners
from util.validation import get_validation_recalls

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get Training Parameters
args = parser.parse_arguments()


def save_checkpoints(args, state, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)


def resume_train(args, model, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters"""
    print(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume)
    if "epoch_num" in checkpoint:
        start_epoch_num = checkpoint["epoch_num"] + 1
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, start_epoch_num


"""Validation at Epoch End"""


def input_transform(image_size=args.resize):
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dataloader(val_dataset_name):
    if val_dataset_name == "Pitts30k_test":
        ds = PittsburghDataset(which_ds='pitts30k_test', input_transform=input_transform())
    elif val_dataset_name == "Pitts250k_test":
        ds = PittsburghDataset(which_ds='pitts250k_test', input_transform=input_transform())
    return ds


def get_val_descriptors(model, dataloader, device):
    model = model.eval()
    descriptors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, 'Calculating descritptors of val dataset...'):
            imgs, labels = batch
            output = model(imgs.to(device)).cpu()
            descriptors.append(output)

    return torch.cat(descriptors)


def validation_epoch_end(model, val_dataset_name):
    val_dataset = get_dataloader(val_dataset_name)
    num_ref = val_dataset.num_references
    num_query = val_dataset.num_queries
    ground_truth = val_dataset.ground_truth

    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=180)
    val_descriptors = get_val_descriptors(model, val_loader, device)
    # split to ref and queries
    r_list = val_descriptors[: num_ref]
    q_list = val_descriptors[num_ref:]

    recalls_dict, predictions = get_validation_recalls(r_list=r_list,
                                                       q_list=q_list,
                                                       k_values=[1, 5, 10, 15, 20, 25],
                                                       gt=ground_truth,
                                                       print_results=True,
                                                       dataset_name=val_dataset_name,
                                                       faiss_gpu=False
                                                       )
    del r_list, q_list, val_descriptors, num_ref, ground_truth

    return recalls_dict


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def train():
    seed = 42
    seed_everything(seed)
    """Load Training Dataset"""
    datamodule = GSVCitiesDataModule(
        batch_size=100,
        img_per_place=4,
        min_img_per_place=4,
        image_size=args.resize,
        shuffle_all=False,
        random_sample_from_each_place=True,
        num_workers=8,
        show_data_stats=True,
    )
    train_dataloader = datamodule.train_dataloader()

    """Initialize SemVG Model"""
    model = SemVGNet(args)
    model = model.to(device)

    for name, param in model.named_parameters():
        if 'semantic_extractor' in name:
            param.requires_grad = False

    # Initialize Triplet Loss Function and Optimizer
    miner = miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3, distance=DotProductSimilarity())
    loss_fn = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=DotProductSimilarity())
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0007, momentum=0.9,
                                weight_decay=0.0001)

    """Training Process"""
    start_epoch_num = 0
    best_r1 = 0
    # Resume model, optimizer
    if args.resume:
        model, optimizer, start_epoch_num = resume_train(args, model, optimizer)
        # Evaluate on val dataset
        validation_epoch_end(model, "Pitts30k_test")
    for epoch_num in range(start_epoch_num, args.epochs_num):
        model = model.train()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)
        for batch in tqdm(train_dataloader, f'Epoch:{epoch_num + 1} / {args.epochs_num}'):
            places, labels = batch
            BS, N, ch, h, w = places.shape

            # Reshape places and labels
            images = places.view(BS * N, ch, h, w)
            labels = labels.view(-1)

            # Forward SemVGNet
            descriptors = model(images.to(device))
            # Compute Loss within a batch.
            hard_pairs = miner(descriptors, labels)
            loss_triplet = loss_fn(descriptors, labels, hard_pairs)
            # Backward propagation
            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()
            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_triplet
        print(f"Epoch[{epoch_num:02d}]: " +
              f"average epoch loss = {epoch_losses.mean():.4f}")

        # Evaluate on val dataset
        recalls = validation_epoch_end(model, "Pitts30k_test")
        print(recalls)
        print(recalls[1])

        # Save Last Model
        save_checkpoints(args, {"epoch_num": epoch_num,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict()
                                }, filename="last_model.pth")

        # Save Best Model
        is_best = recalls[1] > best_r1
        print(f"Previous best R@1 = {best_r1:.4f}")
        if is_best:
            print(f"Improved: previous best R@1 = {best_r1:.4f}, current R@1 = {recalls[1]:.4f}")
            best_r1 = recalls[1]
            save_checkpoints(args, {"epoch_num": epoch_num,
                                    "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict()
                                    }, filename="best_model.pth")

    return


if __name__ == "__main__":
    train()
