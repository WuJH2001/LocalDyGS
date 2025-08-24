import open3d as o3d
import sys
import os
import glob

import numpy as np


def statistical_filter(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    统计滤波：移除离群点
    :param pcd: 输入点云
    :param nb_neighbors: 每个点考虑的邻居数量
    :param std_ratio: 标准差比例，用于判断异常值
    :return: 滤波后的点云
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)

def radius_filter(pcd, radius=0.05, min_points=20):
    """
    半径滤波：移除局部密度低的点
    :param pcd: 输入点云
    :param radius: 邻域半径
    :param min_points: 邻域内最小点数
    :return: 滤波后的点云
    """
    cl, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    return pcd.select_by_index(ind)


def process_ply_file(input_file, output_file):

    pcd = o3d.geometry.PointCloud()
    pcd_paths = glob.glob(os.path.join(input_file, "fused_*.ply"))
    for i in range(len(pcd_paths)):
        cur_pcd = o3d.io.read_point_cloud(pcd_paths[i])
        pcd += cur_pcd

    print(f"Total points: {len(pcd.points)}")
    voxel_size=0.02
    while len(pcd.points) > 90000:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled points: {len(pcd.points)}, voxel size: {voxel_size}")
        voxel_size+=0.01

    o3d.io.write_point_cloud(output_file, pcd)

def sort_by_image_name( image):
    return int(image.split('.')[0].split('_')[2])

def sort_by_image_name2( image):
    return int(image.split('/')[-1].split('.')[0])



def process_ply_file2(input_file, output_file, start, end, filter_type="radius", **filter_params):

    pcd = o3d.geometry.PointCloud()
    pcd_paths = glob.glob(os.path.join(input_file, "*.ply"))
    pcd_paths.sort(key=sort_by_image_name2)

    # 加载第一个点云
    pcd += o3d.io.read_point_cloud(os.path.join(input_file, "1000.ply"))
    print(f"Total points before filtering: {len(pcd.points)}")
    # 滤波
    if filter_type == "statistical":
        pcd = statistical_filter(pcd, **filter_params)
    elif filter_type == "radius":
        pcd = radius_filter(pcd, **filter_params)
    else:
        print("Unknown filter type. No filtering applied.")
    print(f"Total points after filtering: {len(pcd.points)}")

    voxel_size = 0.01
    while len(pcd.points) > 300000:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        voxel_size += 0.001


    for i, pcd_path in enumerate(pcd_paths[start // 2: end // 2]):
        cur_pcd = o3d.io.read_point_cloud(pcd_path)
        voxel_size = 0.01

        print(f"Total points before filtering: {len(cur_pcd.points)}")
        # 滤波
        if filter_type == "statistical":
            cur_pcd = statistical_filter(cur_pcd, **filter_params)
        elif filter_type == "radius":
            cur_pcd = radius_filter(cur_pcd, **filter_params)
        else:
            print("Unknown filter type. No filtering applied.")
        print(f"Total points after filtering: {len(cur_pcd.points)}")

        pcd += cur_pcd

    
    # 再次下采样
    voxel_size = 0.005
    while voxel_size < 0.045 and len(pcd.points) > 400000 :
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled points: {len(pcd.points)}, voxel size: {voxel_size}")
        voxel_size += 0.001

    # 保存点云
    o3d.io.write_point_cloud(output_file, pcd)



process_ply_file(sys.argv[1], sys.argv[2])     



# process_ply_file2(sys.argv[1], sys.argv[2])    
# for i in range(0,1):
#     start = i * 16 
#     end = i * 16 + 20
#     output_file = f"/data1/wjh/dataset/basketball/png/360/pcds/downsample_{start}_{end}.ply"
#     process_ply_file2("/data1/wjh/dataset/basketball/png/360/onlyPlayer/", output_file, start, end )  


