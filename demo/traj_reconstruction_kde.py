import numpy as np
import pandas as pd
import os
import json
import math
from geopy.distance import geodesic
from shapely.geometry import LineString
import folium
from tqdm import tqdm

from demo.evaluation import evaluate_precision_and_errordis


def readJSON(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content


def getSVInfoByPanoId(panoId, svInfo):
    info = svInfo.loc[svInfo['panoId'] == panoId]
    sv_loc = [float(info.iloc[0, 2]), float(info.iloc[0, 1])]
    return sv_loc


def split_and_merge_dataframe(df, subset_length):
    subsets = [df.iloc[i:i + subset_length] for i in range(0, len(df), subset_length)]
    if len(subsets[-1]) < subset_length:
        subsets[-2] = pd.concat([subsets[-2], subsets[-1]])
        subsets.pop()
    return subsets


def calculate_distance(coords):
    n = len(coords)
    distances = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            # 使用geodesic函数计算两个经纬度坐标之间的距离，返回结果单位为米
            dist = geodesic(coords[i], coords[j]).m
            distances[i][j] = dist
            distances[j][i] = dist
    return np.array(distances)


def lesser_than_threshold(lst, threshold):
    return [index for index, value in enumerate(lst) if value < threshold]


# define a function to plot the trajectory
def plot_traj(traj, m, color):
    # 2. Add the trajectory
    folium.PolyLine(
        traj[['Lat', 'Lng']].values.tolist(),  # Trajectory coordinates
        color=color,  # Trajectory color
        weight=2.5,  # Trajectory width
        opacity=1).add_to(m)  # Trajectory opacity, add to the map after creating the trajectory
    # 3. Add trajectory points
    for i in range(len(traj)):
        folium.CircleMarker(
            location=[traj['Lat'].iloc[i], traj['Lng'].iloc[i]],  # Trajectory point coordinates
            radius=3,  # Trajectory point radius
            color=color,  # Trajectory point color
            popup=str(i)
        ).add_to(m)  # Fill opacity, add to the map after creating the trajectory point
    # 4. Add start and end markers
    folium.Marker([traj['Lat'].iloc[0], traj['Lng'].iloc[0]],  # Start point coordinates
                  popup='Start',  # Start marker label
                  icon=folium.Icon(color='green')).add_to(m)  # Start marker color
    folium.Marker([traj['Lat'].iloc[-1], traj['Lng'].iloc[-1]],
                  popup='End',
                  icon=folium.Icon(color='red')).add_to(m)
    # 5. Display the map, directly in Jupyter Notebook
    return m


def getGTTraj(video_info):
    loc_info = video_info["locations"]
    gt_traj = []
    for l in loc_info:
        gt_traj.append({
            "Lat": float(l['latitude']),
            "Lng": float(l["longitude"])
        })
    gt_traj = pd.DataFrame(gt_traj)
    return gt_traj


def getVideoLabels(video_label_path):
    video_label = readJSON(video_label_path)


def segment_array(arr):
    seg_array = []
    flag = -1
    seg = []
    for index, num in enumerate(arr):
        if num * flag < 0 and seg:
            seg_array.append(seg)
            seg = []
            seg.append(num)
        else:
            seg.append(num)
        flag = arr[index]
    seg_array.append(seg)
    return seg_array


def interpolate_coordinates(lat1, lon1, lat2, lon2, num_points):
    """
    Interpolate coordinates between two points on the Earth's surface.

    Args:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.
        num_points (int): Number of points to interpolate (including the endpoints).

    Returns:
        List of tuples: List of (latitude, longitude) tuples for the interpolated points.
    """
    line = LineString([(lon1, lat1), (lon2, lat2)])
    interpolated_points = [line.interpolate((i + 1) / (num_points + 1), normalized=True) for i in range(num_points)]
    return [[point.y, point.x] for point in interpolated_points]


def calculate_cosine_between_points(a_coord, b_coord, c_coord):
    """
    计算三点之间的角度余弦
    """
    dist_ab = geodesic(a_coord, b_coord).m
    dist_bc = geodesic(b_coord, c_coord).m
    dist_ac = geodesic(a_coord, c_coord).m
    cos_angle = (dist_ab ** 2 + dist_bc ** 2 - dist_ac ** 2) / (2 * dist_ab * dist_bc)
    return cos_angle


def kde_estimation(search_result, vclip_num=8, radius=150, topk=10, svInfo=None, search_result_distance=None):
    num_vframe = search_result.shape[0]
    subset_length = len(search_result) // vclip_num
    search_result = pd.DataFrame(search_result)
    search_result_distance = pd.DataFrame(search_result_distance)
    v_clips = split_and_merge_dataframe(search_result, subset_length)
    sim_clips = split_and_merge_dataframe(search_result_distance, subset_length)
    v_clip_kde_max = []
    v_clip_kde_max_index = []
    for idx, v_clip in enumerate(v_clips):
        _v_clip = v_clip.stack().to_frame().T
        _v_clip.reset_index(drop=True, inplace=True)
        _v_sim_clip = sim_clips[idx].stack().to_frame().T
        _v_sim_clip.reset_index(drop=True, inplace=True)
        locations = []
        for i in range(_v_clip.shape[1]):
            pano_id = _v_clip.iloc[0, i].split("____")[0]
            locations.append(getSVInfoByPanoId(pano_id, svInfo))
        # generate distance table
        dist_table = calculate_distance(locations)
        # calculate kde value
        kde_list = []
        for dist_list in dist_table:
            range_index = lesser_than_threshold(dist_list, threshold=radius)
            kde_value = 0
            for i in range(len(range_index)):
                sim = float(_v_sim_clip.iloc[0, range_index[i]])
                kde_value = kde_value + (3 / math.pi) * sim * math.pow(
                    1 - math.pow((dist_list[range_index[i]] / radius), 2), 2)
            kde_value = (1 / math.pow(radius, 2)) * kde_value
            kde_list.append(kde_value)
        max_index = kde_list.index(max(kde_list))
        v_clip_kde_max.append(kde_list[max_index])
        v_clip_kde_max_index.append(max_index)
    _max_index = v_clip_kde_max.index(max(v_clip_kde_max))
    k = _max_index * subset_length + v_clip_kde_max_index[_max_index] // topk
    r = v_clip_kde_max_index[_max_index] % topk + 1
    return k, r


def smooth_with_interpolation(search_result, signals, svInfo):
    seg_array = segment_array(signals)
    print(seg_array)
    smooth_traj = []
    s_index = -1
    for index, seg in enumerate(seg_array):
        if seg[0] < 0:
            # 首
            if index - 1 < 0:
                cali_frame_index = len(seg)
                for s in seg:
                    corr_loc = getSVInfoByPanoId(
                        search_result[cali_frame_index, signals[cali_frame_index] - 1].split("____")[0], svInfo)
                    smooth_traj.append(corr_loc)
                s_index = s_index + len(seg)
            # 尾
            elif index == len(seg_array) - 1:
                cali_frame_index = len(signals) - len(seg) - 1
                for s in seg:
                    corr_loc = getSVInfoByPanoId(
                        search_result[cali_frame_index, signals[cali_frame_index] - 1].split("____")[0], svInfo)
                    smooth_traj.append(corr_loc)
                s_index = s_index + len(seg)
            # 中间插值
            else:
                pre_cali_index = s_index
                post_cali_index = s_index + len(seg) + 1
                pre_geoloc = getSVInfoByPanoId(
                    search_result[pre_cali_index, signals[pre_cali_index] - 1].split("____")[0], svInfo)
                post_geoloc = getSVInfoByPanoId(
                    search_result[post_cali_index, signals[post_cali_index] - 1].split("____")[0], svInfo)
                interpolate_pts = interpolate_coordinates(pre_geoloc[0], pre_geoloc[1], post_geoloc[0], post_geoloc[1], num_points=len(seg))
                smooth_traj = smooth_traj + interpolate_pts
                s_index = s_index + len(seg)
        elif seg[0] > 0:
            for s in seg:
                s_index = s_index + 1
                geo_loc = getSVInfoByPanoId(search_result[s_index, signals[s_index] - 1].split("____")[0], svInfo)
                smooth_traj.append(geo_loc)
    return smooth_traj


def traj_reconstruction(vframe_re, tolerance=20, svInfo=None, vframe_sim=None, radius=150, topK=10, vclip_num=8, m=None):
    search_result = vframe_re
    if topK > 1:
        calibrate_frame, retrieved_rank = kde_estimation(
            search_result,
            svInfo=svInfo,
            search_result_distance=vframe_sim,
            radius=radius,
            topk=topK,
            vclip_num=vclip_num
        )
        signals = np.zeros(search_result.shape[0], dtype=int)
        signals[calibrate_frame] = retrieved_rank
    else:
        signals = np.ones(search_result.shape[0], dtype=int)

    sure = np.where(signals > 0)[0]
    unsure = np.where(signals == 0)[0]
    while len(unsure) != 0:
        differ = []
        for num_unsure in unsure:
            diffs = [np.abs(num_sure - num_unsure) for num_sure in sure]
            differ.append(diffs)
        differ = np.array(differ)
        check_indices = list(set(np.argmin(differ, axis=0)))
        traverse_order = unsure[check_indices]

        map_sure = differ[check_indices]
        map_sure = sure[np.argmin(map_sure, axis=1)]
        map_sure_re = signals[map_sure]

        for index, value in enumerate(traverse_order):
            sure_re = search_result[map_sure[index], map_sure_re[index] - 1]
            s_panoid = sure_re.split("____")[0]
            s_geoloc = getSVInfoByPanoId(s_panoid, svInfo)

            t_interv = np.abs(map_sure[index] - value)

            unsure_re_list = search_result[value, :]
            u_topk = []
            for j, u in enumerate(unsure_re_list):
                u_panoid = u.split("____")[0]
                u_geoloc = getSVInfoByPanoId(u_panoid, svInfo)
                su_dis = geodesic(s_geoloc, u_geoloc).m
                if su_dis <= tolerance * t_interv:
                    u_topk.append(j + 1)
            if len(u_topk) == 0:
                signals[value] = -1
            else:
                sig = u_topk[np.argmin(u_topk)]
                signals[value] = sig
        sure = np.where(signals > 0)[0]
        unsure = np.where(signals == 0)[0]
    # Smoothen Traj
    print("Signals:")
    print(signals)
    smooth_traj = smooth_with_interpolation(search_result, signals, svInfo)
    smooth_traj = pd.DataFrame(smooth_traj, columns=["Lat", "Lng"])
    return smooth_traj


def get_video_label(result_without_label):
    label_rootpath = "D:/VGProgram/Dataset/studyarea_0to8/labels"
    result_with_label = []
    for i in range(0, result_without_label.shape[0]):
        video_name = result_without_label.iloc[i, 0]
        label_path = os.path.join(label_rootpath, "%s.json" % video_name)
        label = readJSON(label_path)
        label = label["attributes"]
        result_with_label.append({
            "Name": result_without_label.iloc[i, 0],
            "Average_Error_Dis": result_without_label.iloc[i, 1],
            "Recall@30": result_without_label.iloc[i, 2],
            "Recall@50": result_without_label.iloc[i, 3],
            "Recall@100": result_without_label.iloc[i, 4],
            "Weather": label["weather"],
            "Scene": label["scene"],
            "TimeofDay": label["timeofday"]
        })
    return result_with_label


if __name__ == "__main__":
    METHODS = ["semvgl"]
    for method in METHODS:
        svInfo = readJSON("D:/VGProgram/Dataset/studyarea_0to8/sv_info_0to8.json")
        svInfo = pd.DataFrame(svInfo)
        video_rootpath = "D:/VGProgram/Dataset/studyarea_0to8/videos"
        video_list = os.listdir(video_rootpath)
        result_path = "D:/SemVG/demo/traj_reconstruct_result/%s_reconstruct_result.csv" % method
        result = []
        csim_eval_path = "D:/SemVG/demo/csim_evaluation"
        for v in tqdm(video_list):
            video_name = v.split(".")[0]
            vframe_retrieved_path = os.path.join(
                "D:/SemVG/demo/feature_db/nyc_search_result/search_result_%s_top10_0to8" % method,
                "%s.json" % video_name)
            vframe_sim_path = os.path.join(
                "D:/SemVG/demo/feature_db/nyc_search_result/sim_%s_top10_0to8" % method,
                "%s.json" % video_name)

            # Read v-frames search result
            vframe_re = readJSON(vframe_retrieved_path)
            vframe_re = np.asarray(list(vframe_re.values()))

            vframe_sim = readJSON(vframe_sim_path)
            vframe_sim = np.asarray(list(vframe_sim.values()))

            # Read video info and get GT-Traj
            video_info_rootpath = "D:/VGProgram/Dataset/studyarea_0to8/info"
            info_path = os.path.join(video_info_rootpath, "%s.json" % video_name)
            video_info = readJSON(info_path)
            gt_traj = getGTTraj(video_info)

            # Video Geo-Temporal Traj-Reconstruction
            traj = traj_reconstruction(vframe_re, svInfo=svInfo, vframe_sim=vframe_sim)

            # Saving predicted trajectory for csim evaluation
            csim_save_path = os.path.join(csim_eval_path, video_name)
            if not os.path.exists(csim_save_path):
                os.makedirs(csim_save_path)
            print("*******************Predicted Trajectory**********************")
            traj.to_csv(os.path.join(csim_save_path, "pred_traj.txt"), index=None, header=None, sep=" ", columns=["Lat", "Lng"])
            print("***********************GT Trajectory*************************")
            gt_traj.to_csv(os.path.join(csim_save_path, "gt_traj.txt"), index=None, header=None, sep=" ", columns=["Lat", "Lng"])

            # Evaluation
            average_error_dis, total_recall_30 = evaluate_precision_and_errordis(traj, gt_traj, 30)
            average_error_dis, total_recall_50 = evaluate_precision_and_errordis(traj, gt_traj, 50)
            average_error_dis, total_recall_100 = evaluate_precision_and_errordis(traj, gt_traj, 100)
            print("Evaluation Result:", average_error_dis, total_recall_30, total_recall_50, total_recall_100)
            result.append({
                "Name": video_name,
                "Average_Error_Dis": average_error_dis,
                "Recall@30": total_recall_30,
                "Recall@50": total_recall_50,
                "Recall@100": total_recall_100
            })
        result = pd.DataFrame(result)
        result_with_label = get_video_label(result)
        result_with_label = pd.DataFrame(result_with_label)
        result_with_label.to_csv(result_path, index=None)

