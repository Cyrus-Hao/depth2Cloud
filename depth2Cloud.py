import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def write_point_cloud(ply_filename, points):
    formatted_points = []
    for point in points:
        formatted_points.append("%f %f %f %d %d %d 0\n" % (point[0], point[1], point[2], point[3], point[4], point[5]))

    out_file = open(ply_filename, "w")
    out_file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar blue
    property uchar green
    property uchar red
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(formatted_points)))
    out_file.close()


def adjust_intrinsics_for_resolution(K_original, original_size, target_size):
    """
    根据分辨率调整内参矩阵
    Args:
        K_original: 原始内参矩阵 (3x3)
        original_size: 原始分辨率 (height, width)
        target_size: 目标分辨率 (height, width)
    Returns:
        K_adjusted: 调整后的内参矩阵
    """
    scale_x = target_size[1] / original_size[1]  # width
    scale_y = target_size[0] / original_size[0]  # height
    
    K_adjusted = K_original.copy()
    K_adjusted[0, 0] *= scale_x  # fx
    K_adjusted[1, 1] *= scale_y  # fy
    K_adjusted[0, 2] *= scale_x  # cx
    K_adjusted[1, 2] *= scale_y  # cy
    
    return K_adjusted


def depth_image_to_point_cloud(rgb, depth, scale, K, pose, rgb_size=None):
    """
    将深度图转换为点云
    Args:
        rgb: RGB图像
        depth: 深度图
        scale: 深度尺度因子
        K: 内参矩阵
        pose: 相机位姿矩阵
        rgb_size: RGB图像的原始尺寸 (height, width)，用于调整内参
    """
    # 如果提供了RGB原始尺寸，调整内参矩阵以匹配深度图分辨率
    if rgb_size is not None and rgb.shape[:2] != depth.shape[:2]:
        K_adjusted = adjust_intrinsics_for_resolution(K, rgb_size, depth.shape[:2])
        print(f"调整内参矩阵以匹配深度图分辨率: {depth.shape[:2]}")
        K = K_adjusted
        
        # 调整RGB图像尺寸以匹配深度图
        rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        print(f"调整RGB图像尺寸以匹配深度图: {rgb.shape[:2]}")
    
    # 使用深度图分辨率进行反投影
    u = range(0, depth.shape[1])
    v = range(0, depth.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    # OpenCV读取的是BGR格式，需要转换为RGB
    B = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    R = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()

    return points


# image_files: XXXXXX.png (RGB, 24-bit, PNG)
# depth_files: XXXXXX.png (16-bit, PNG)
# poses: camera-to-world, 4×4 matrix in homogeneous coordinates
def build_point_cloud(dataset_path, scale, view_ply_in_world_coordinate):
    K = np.fromfile(os.path.join(dataset_path, "K.txt"), dtype=float, sep="\n ")
    K = np.reshape(K, (3, 3))
    image_files = sorted(Path(os.path.join(dataset_path, "images")).glob('*.png'))
    depth_files = sorted(Path(os.path.join(dataset_path, "depth_maps")).glob('*.png'))

    if view_ply_in_world_coordinate:
        poses = np.fromfile(os.path.join(dataset_path, "poses.txt"), dtype=float, sep="\n ")
        poses = np.reshape(poses, (-1, 4, 4))
    else:
        poses = np.eye(4)

    # 获取RGB图像的原始尺寸（用于调整内参矩阵）
    first_rgb = cv2.imread(str(image_files[0]))
    rgb_original_size = first_rgb.shape[:2]  # (height, width)
    print(f"RGB图像原始尺寸: {rgb_original_size}")
    
    for i in tqdm(range(0, len(image_files))):
        image_file = image_files[i]
        depth_file = depth_files[i]

        rgb = cv2.imread(image_file)
        depth = cv2.imread(depth_file, -1).astype(np.uint16)

        if view_ply_in_world_coordinate:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses[i], rgb_size=rgb_original_size)
        else:
            current_points_3D = depth_image_to_point_cloud(rgb, depth, scale=scale, K=K, pose=poses, rgb_size=rgb_original_size)
        save_ply_name = os.path.basename(os.path.splitext(image_files[i])[0]) + ".ply"
        save_ply_path = os.path.join(dataset_path, "point_clouds")

        if not os.path.exists(save_ply_path):  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.mkdir(save_ply_path)
        write_point_cloud(os.path.join(save_ply_path, save_ply_name), current_points_3D)


if __name__ == '__main__':
    dataset_folder = Path("dataset")
    scene = Path("promptda/137")
    # 如果view_ply_in_world_coordinate为True,那么点云的坐标就是在world坐标系下的坐标，否则就是在当前帧下的坐标
    view_ply_in_world_coordinate = True  # 使用世界坐标系
    # 深度图对应的尺度因子，即深度图中存储的值与真实深度（单位为m）的比例, depth_map_value / real depth = scale_factor
    # 不同数据集对应的尺度因子不同，比如TUM的scale_factor为5000， hololens的数据的scale_factor为1000, Apollo Scape数据的scale_factor为200
    scale_factor = 1000.0  # Stray Scanner: 毫米转米
    build_point_cloud(os.path.join(dataset_folder, scene), scale_factor, view_ply_in_world_coordinate)
