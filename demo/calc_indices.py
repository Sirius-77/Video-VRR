import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm
import subprocess

from traj_reconstruction_kde import readJSON

# The directory where the CSIM Class is located (the location after compilation).
JAVA_CLASS_PATH = 'E:/C-SIM'


def csim_eval(gt_route_path, pred_route_path, buffer_distance):
    """Calculate C-SIM indices"""
    result = subprocess.run(
        ['java', '-cp', JAVA_CLASS_PATH, 'CSIM', gt_route_path, pred_route_path, str(buffer_distance)],
        capture_output=True,
        text=True
    )
    standard_output = result.stdout
    print(standard_output)
    csim = float(standard_output.split("\n")[1].split(": ")[1])
    return csim


def calc_localization_error(gt_traj, pred_traj):
    len_pred_traj = pred_traj.shape[0]
    len_gt_traj = gt_traj.shape[0]
    length = min(len_pred_traj, len_gt_traj)
    error_dis_list = []
    for j in range(0, length):
        pt_loc = (pred_traj.iloc[j, 0], pred_traj.iloc[j, 1])
        gt_pt_loc = (gt_traj.iloc[j, 0], gt_traj.iloc[j, 1])
        error_dis = geodesic(pt_loc, gt_pt_loc).m
        error_dis_list.append(error_dis)
    localization_error = np.mean(error_dis_list)
    return localization_error


if __name__ == "__main__":
    Dis_Tolerance = [25, 50, 100]
    gt_traj_rootpath = "video_gt_traj"
    pred_traj_rootpath = "video_pred_traj_top5_vtoler35_vclip10_radius100"
    traj_list = os.listdir(gt_traj_rootpath)
    result_path = "eval_result_video_pred_traj_top5_vtoler35_vclip10_radius100.csv"
    result = []
    for i in tqdm(range(len(traj_list))):
        traj_name = traj_list[i]
        gt_traj_path = f"{gt_traj_rootpath}/{traj_name}"
        pred_traj_path = f"{pred_traj_rootpath}/{traj_name}"

        gt_traj = pd.read_csv(gt_traj_path, sep=" ")
        pred_traj = pd.read_csv(pred_traj_path, sep=" ")

        localization_error = calc_localization_error(gt_traj, pred_traj)

        csim_list = []
        for d_tol in Dis_Tolerance:
            csim = csim_eval(gt_traj_path, pred_traj_path, d_tol)
            csim_list.append(csim)

        # Read video label
        label_rootpath = "../DS/NYCDataset/bdd100k/labels"
        video_name = traj_name.split(".")[0]
        label_path = os.path.join(label_rootpath, "%s.json" % video_name)
        label = readJSON(label_path)
        label = label["attributes"]
        result.append({
            "Name": video_name,
            "Localization_Error": localization_error,
            "CSIM@25": csim_list[0],
            "CSIM@50": csim_list[1],
            "CSIM@100": csim_list[2],
            "Weather": label["weather"],
            "Scene": label["scene"],
            "TimeofDay": label["timeofday"]
        })

    result = pd.DataFrame(result)
    result.to_csv(result_path, index=None)

    result = pd.read_csv("eval_result_video_pred_traj_top5_vtoler35_vclip10_radius100.csv")
    print(f"CSIM@25: {result['CSIM@25'].mean()}")
    print(f"CSIM@50: {result['CSIM@50'].mean()}")
    print(f"CSIM@100: {result['CSIM@100'].mean()}")
    print(f"Average Localization Error: {result['Localization_Error'].mean()}")
    print(f"Median Localization Error: {result['Localization_Error'].median()}")




