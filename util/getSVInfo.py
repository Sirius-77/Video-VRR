import os
import json

import pandas as pd
from tqdm import tqdm


def readJSON(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content


def getSVInfo(sv_root_path, result_type, save_dir):
    sv_list = os.listdir(sv_root_path)
    result = []

    if result_type == "json":
        for sv in sv_list:
            if sv.endswith(".json"):
                sv_path = os.path.join(sv_root_path, sv)
                sv_info = readJSON(sv_path)
                panoId = sv_info["panoId"]
                lat = sv_info["lat"]
                lng = sv_info["lng"]
                result.append({
                    "panoId": panoId,
                    "lng": lng,
                    "lat": lat
                })
        with open(os.path.join(save_dir, "ny_sv_info.json"), "w", encoding="utf-8") as f:
            json.dump(result, f)
    elif result_type == "txt":
        for i in tqdm(range(len(sv_list))):
            sv = sv_list[i]
            if sv.endswith(".json"):
                sv_path = os.path.join(sv_root_path, sv)
                sv_info = readJSON(sv_path)
                panoId = sv_info["panoId"]
                lat = sv_info["lat"]
                lng = sv_info["lng"]
                heading = sv_info["rotation"]
                result.append({
                    "heading": heading,
                    "pitch": heading,
                    "roll": heading,
                    "lat": lat,
                    "lng": lng,
                    "panoId": panoId
                })
        result = pd.DataFrame(result)
        result['panoId'] = result['panoId'].str.strip()
        result.to_csv(path_or_buf=os.path.join(save_dir, "ny_sv_info.txt"), index=None, header=None, sep=",", lineterminator="\n")
    return
