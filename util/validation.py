import os
import time
import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable

from dataloader.val.PittsburghDataset import PittsburghDataset
from util import parser
import torchvision.transforms as T

args = parser.parse_arguments()


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


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False,
                           dataset_name='dataset without name ?', save_results=False):
    embed_size = r_list.shape[1]
    print('Database feature:', r_list.cpu().numpy().shape, r_list.cpu().numpy().dtype, r_list.cpu().numpy().nbytes)
    print('Memory:', r_list.cpu().numpy().nbytes / (1024 ** 3), 'GB')
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    # build index
    else:
        faiss_index = faiss.IndexFlatL2(embed_size)

    # add references
    faiss_index.add(r_list)

    t_s = time.time()
    # search for queries in the index
    _, predictions = faiss_index.search(q_list, max(k_values))
    print("Retrieval time for each: {}ms".format(1000 * ((time.time() - t_s) / q_list.shape[0])))

    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print('\n')  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Performance on {dataset_name}"))

    return d, predictions
