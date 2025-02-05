import os
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm

from util.getSVInfo import readJSON


def getSVInfoByPanoId(panoId, svInfo):
    info = svInfo.loc[svInfo['panoId'] == panoId]
    sv_loc = [float(info.iloc[0, 2]), float(info.iloc[0, 1])]
    return sv_loc


def getGTTraj(video_info):
    loc_info = video_info["locations"]
    gt_traj = []
    for l in loc_info:
        gt_traj.append({
            "Lat": float(l['latitude']),
            "Lon": float(l["longitude"])
        })
    gt_traj = pd.DataFrame(gt_traj)
    return gt_traj


def evaluate_precision_and_errordis(traj, gt_traj, tolerance=30):
    print(traj.shape, gt_traj.shape)
    len_traj = traj.shape[0]
    len_gt_traj = gt_traj.shape[0]
    length = min(len_traj, len_gt_traj)
    error_dis_list = []
    recall = []
    for i in range(0, length):
        pt_loc = (traj.iloc[i, 0], traj.iloc[i, 1])
        gt_pt_loc = (gt_traj.iloc[i, 0], gt_traj.iloc[i, 1])
        error_dis = geodesic(pt_loc, gt_pt_loc).m
        error_dis_list.append(error_dis)
        if error_dis <= tolerance:
            recall.append(1)
        else:
            recall.append(0)
    recall = np.array(recall)
    average_error_dis = np.mean(error_dis_list)
    total_recall = np.count_nonzero(recall) / len(recall)
    return average_error_dis, total_recall


def evaluate_precision(search_result, gt_traj, tolerance=30, topk=10, svInfo=None):
    len_retrieval = search_result.shape[0]
    len_gt_traj = gt_traj.shape[0]
    min_length = min(len_retrieval, len_gt_traj)
    total_error_dis_list = []
    total_recall_list = []
    for i in range(0, min_length):
        gt_loc = (gt_traj.iloc[i, 0], gt_traj.iloc[i, 1])
        error_dis_list = []
        flag = 0
        for k in range(0, topk):
            r_loc = getSVInfoByPanoId(search_result[i, k].split("____")[0], svInfo)
            error_dis = geodesic(gt_loc, r_loc).m
            if error_dis < tolerance:
                flag = 1
            error_dis_list.append(error_dis)
        total_recall_list.append(flag)
        total_error_dis_list.append(error_dis_list)
    recall = np.array(total_recall_list)
    total_recall = np.count_nonzero(recall) / len(recall)

    total_error_dis_list = np.array(total_error_dis_list)
    average_error_dis = np.min(total_error_dis_list, axis=1)
    average_error_dis = np.mean(average_error_dis)
    return average_error_dis, total_recall
