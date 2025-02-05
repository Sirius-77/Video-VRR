import os
import folium
from folium.plugins import FloatImage

def read_gps_points(file_path):
    """
    读取轨迹文件中的GPS点信息，每个文件中每行是一个GPS点（纬度 经度）
    返回点的列表 [(lat, lon), (lat, lon), ...]
    """
    points = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            lat, lon = map(float, line.strip().split())
            points.append((lat, lon))
    return points

def plot_trajectory_on_map(trajectory_id, real_points, predicted_points, output_folder):
    """
    使用folium绘制真实轨迹（绿线）和预测轨迹（红线）的地图，添加图例
    """
    # 创建地图，初始中心为真实轨迹的第一个点
    center_lat, center_lon = real_points[0]
    map_ = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB Positron')

    # 绘制真实轨迹（绿线）
    real_route = folium.PolyLine(real_points, color='green', weight=3, opacity=1, name="Ground Truth Route").add_to(map_)

    # 绘制预测轨迹（红线）
    predicted_route = folium.PolyLine(predicted_points, color='red', weight=3, opacity=1, name="Predicted Route").add_to(map_)

    # 添加中文图例
    legend_html_zh = """
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:18px;
     background-color:white; opacity: 0.85;">
     &nbsp;<b>图例</b><br>
     &nbsp;<i style="color:green;">●</i> 真实轨迹<br>
     &nbsp;<i style="color:red;">●</i> 预测轨迹
     </div>
     """

    # 添加英文图例
    legend_html_en = """
         <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 200px; height: 90px; 
         border:2px solid grey; z-index:9999; font-size:18px;
         background-color:white; opacity: 0.85;">
         &nbsp;<b>Legend</b><br>
         &nbsp;<i style="color:green;">●</i> Ground Truth Route<br>
         &nbsp;<i style="color:red;">●</i> Predicted Route
         </div>
         """

    map_.get_root().html.add_child(folium.Element(legend_html_en))

    # 保存地图
    map_file_path = os.path.join(output_folder, f'{trajectory_id}_map.html')
    map_.save(map_file_path)
    print(f'地图已保存: {map_file_path}')

def main(real_folder, predicted_folder, output_folder):
    """
    读取真实和预测轨迹文件夹，并为每个轨迹ID生成地图
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历预测文件夹中的所有轨迹文件
    for pred_filename in os.listdir(predicted_folder):
        trajectory_id = os.path.splitext(pred_filename)[0]
        pred_file_path = os.path.join(predicted_folder, pred_filename)
        real_file_path = os.path.join(real_folder, pred_filename)

        if os.path.exists(real_file_path):
            # 读取预测和真实轨迹点
            predicted_points = read_gps_points(pred_file_path)
            real_points = read_gps_points(real_file_path)

            # 绘制地图
            plot_trajectory_on_map(trajectory_id, real_points, predicted_points, output_folder)
        else:
            print(f'未找到匹配的真实轨迹文件: {pred_filename}')


if __name__ == '__main__':
    main(
        real_folder="video_gt_traj/",
        predicted_folder="ablation_study/video_pred_traj_top5_vtoler35_vclip10_radius100",
        output_folder="figure_10/semvg_ny_map_en"
    )