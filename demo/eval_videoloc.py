import os
import time

import cv2
import faiss
import numpy as np
import torch
import util.parser as parser
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader.test.QueryFramesDataset import QueryFramesDataset
from dataloader.val.NYCDataset import NYCDataset
from traj_reconstruction_kde import getGTTraj, traj_reconstruction, readJSON
from model.SemVGNet import SemVGNet

localargs = parser.parse_arguments()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ckpt_path = '../ckpt/semvgl_aggr_cate_best_model.pth'
feature_db_file_path = "feature_db/nyc_db_features.npy"
svInfo = readJSON("../DS/NYCDataset/streetview/ny_sv_info.json")
svInfo = pd.DataFrame(svInfo)
xb = np.load(feature_db_file_path)

db_index = faiss.index_factory(2048, 'Flat', faiss.METRIC_INNER_PRODUCT)
db_index.train(xb)
db_index.add(xb)


def getKeyFrameImg(video_path, img_dir):
    """
        由视频逐帧输出图片
        :param video_path: 视频文件路径
        :param img_dir: 图片保存目录路径，路径不支持中文
        :return:
    """
    # 创建关键帧存放文件夹
    img_dir = img_dir + video_path.split("/")[-1].split(".")[0]
    os.makedirs(img_dir, exist_ok=True)
    # 视频获取
    cap = cv2.VideoCapture(video_path)
    # 判断是否正常打开
    ret = cap.isOpened()
    # 输出视频的帧率和帧数信息
    print("帧率：", cap.get(5))
    print("帧数：", cap.get(7))
    fps = cap.get(5)
    n = 0
    # 循环读取视频帧
    while ret:
        n += 1
        # 获取当前视频帧位置
        now_fps = cap.get(1)  # cv2.VideoCapture.get(1) 当前帧，基于以0开始的被捕获或解码的帧索引
        # 设置每1s保存一次
        if now_fps % int(fps) != 0:
            # 跳帧
            ret = cap.grab()
            continue
        # 保存图片
        ret, frame = cap.read()
        img_name = "{0}.jpg".format(n)
        img_file_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_file_path):
            cv2.imwrite(img_file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])


def input_transform():
    return T.Compose([
        T.Resize([322, 322], interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def feature_retrieval(query_ds, query_fea, feature_db_path, topK):
    embed_size = query_fea.shape[1]
    print("embedding size: ", embed_size)
    xq = query_fea

    D, I = db_index.search(xq, topK)

    result = []
    q_ds = query_ds
    q_index = q_ds.query_frames
    db_ds = NYCDataset(input_transform=input_transform())
    b_index = db_ds.dbImages
    result = {}
    distance = {}
    for i in range(0, len(q_index)):
        result[q_index[i]] = []
        distance[q_index[i]] = []
        for j in range(0, topK):
            result[q_index[i]].append(b_index[I[i][j]].split(".")[0].split("/")[-1])
            distance[q_index[i]].append(str(D[i][j]))
    print(result)
    return result, distance


def eval_video_frames(query_video_frames_path, topK, v_toler, vclip_num, radius, video_id):
    # Load checkpoint
    checkpoint = torch.load(ckpt_path)

    # Initialize SemVG Model
    model = SemVGNet(localargs)
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # Load query video frames
    root_dir = query_video_frames_path
    query_frames = os.listdir(query_video_frames_path)
    query_frames = sorted(query_frames, key=lambda x: int(x.split(".")[0]))
    dataset = QueryFramesDataset(
        root_dir=root_dir,
        query_frames=query_frames,
        input_transform=input_transform()
    )
    dataloader = DataLoader(dataset, batch_size=128)

    # Model Eval
    t_s = time.time()
    model = model.eval()
    descs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, 'Extracting features...'):
            imgs, labels = batch
            output = model(imgs.to(device)).cpu()
            descs.append(output)
    descs = torch.cat(descs).numpy()

    # Feature Retrieval
    result, distance = feature_retrieval(query_ds=dataset, query_fea=descs, feature_db_path=feature_db_file_path, topK=topK)

    # Spatio-temporal trajectory reconstruction
    vframe_re = np.asarray(list(result.values()))
    vframe_sim = np.asarray(list(distance.values()))

    # Read video info and get GT-Traj
    video_info_rootpath = "../DS/NYCDataset/bdd100k/info"
    info_path = os.path.join(video_info_rootpath, "%s.json" % video_id)
    video_info = readJSON(info_path)
    gt_traj = getGTTraj(video_info)

    # Video Geo-Temporal Traj-Reconstruction
    traj = traj_reconstruction(vframe_re=vframe_re,
                               svInfo=svInfo,
                               vframe_sim=vframe_sim,
                               tolerance=v_toler,
                               vclip_num=vclip_num,
                               radius=radius,
                               topK=topK
                               )
    print("Latency time for each: {}ms".format(1000 * (time.time() - t_s)))
    gt_traj = pd.DataFrame(np.array(gt_traj), columns=['Lat', 'Lng'])
    traj = pd.DataFrame(np.array(traj), columns=['Lat', 'Lng'])
    return traj, gt_traj


if __name__ == '__main__':
    """Parameter Settings"""
    # Top K
    topK_list = [5]
    # Velocity Tolerance(m/s)
    v_toler_list = [35]
    # Frame Num in Video Clips
    vclip_num_list = [10]
    # Spatial Kernel Analysis Radius
    radius_list = [100]

    """Path Settings"""
    # Videos Rootpath
    video_rootpath = "../DS/NYCDataset/bdd100k/videos"
    # Video Frame Rootpath
    if not os.path.exists("video_frame"):
        os.mkdir("video_frame")
    # GT-Traj Rootpath
    if not os.path.exists("video_gt_traj"):
        os.mkdir("video_gt_traj")
    # Pred-Traj Rootpath
    for topK in topK_list:
        for v_toler in v_toler_list:
            for vclip_num in vclip_num_list:
                for radius in radius_list:
                    if not os.path.exists(f"video_pred_traj_top{topK}_vtoler{v_toler}_vclip{vclip_num}_radius{radius}"):
                        os.mkdir(f"video_pred_traj_top{topK}_vtoler{v_toler}_vclip{vclip_num}_radius{radius}")

                    """Execute Evaluate Process"""
                    video_list = os.listdir(video_rootpath)
                    for i in tqdm(range(len(video_list))):
                        video_name = video_list[i]
                        video_id = video_name.split(".")[0]
                        video_path = f"{video_rootpath}/{video_name}"
                        getKeyFrameImg(
                            video_path=video_path,
                            img_dir="video_frame/"
                        )
                        video_frame_path = f"video_frame/{video_id}/"
                        traj, gt_traj = eval_video_frames(video_frame_path, topK, v_toler, vclip_num, radius, video_id)
                        print("*******************Predicted Trajectory**********************")

                        traj.to_csv(f"video_pred_traj_top{topK}_vtoler{v_toler}_vclip{vclip_num}_radius{radius}/{video_id}.txt",
                            index=None, header=None, sep=" ",
                            columns=["Lat", "Lng"])
                        print("***********************GT Trajectory*************************")
                        gt_traj.to_csv(f"video_gt_traj/{video_id}.txt", index=None, header=None, sep=" ",
                                       columns=["Lat", "Lng"])
