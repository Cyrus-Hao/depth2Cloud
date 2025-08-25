# python convert_K_format.py "dataset/promptda" -d -t odometry -f 60
#python convert_K_format.py "dataset/promptda" -d -t csv

import os
import argparse
import csv
import numpy as np
from pathlib import Path


def read_csv_matrix(csv_path):
    """
    从CSV文件读取矩阵数据
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        list: 3x3矩阵数据
    """
    matrix = []
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # 跳过空行
                    # 跳过标题行（如果第一行包含非数字）
                    try:
                        # 尝试转换为浮点数
                        row_data = [float(x.strip()) for x in row if x.strip()]
                        if len(row_data) == 3:  # 确保是3列
                            matrix.append(row_data)
                    except ValueError:
                        # 如果转换失败，可能是标题行，跳过
                        continue
    except Exception as e:
        print(f"❌ 读取CSV文件失败: {e}")
        return None
    
    # 检查矩阵维度
    if len(matrix) != 3:
        print(f"❌ 错误: CSV文件应该包含3x3矩阵，当前维度: {len(matrix)}x{len(matrix[0]) if matrix else 0}")
        return None
    
    return matrix


def read_txt_matrix(txt_path):
    """
    从TXT文件读取矩阵数据
    
    Args:
        txt_path: TXT文件路径
        
    Returns:
        list: 3x3矩阵数据
    """
    matrix = []
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if line.strip():  # 跳过空行
                # 分割制表符或空格
                row = line.strip().split()
                matrix.append([float(x) for x in row])
    except Exception as e:
        print(f"❌ 读取TXT文件失败: {e}")
        return None
    
    # 检查矩阵维度
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        print(f"❌ 错误: TXT文件应该包含3x3矩阵，当前维度: {len(matrix)}x{len(matrix[0]) if matrix else 0}")
        return None
    
    return matrix


def convert_matrix_to_K_format(matrix, output_path):
    """
    将矩阵转换为K.txt格式并保存
    
    Args:
        matrix: 3x3矩阵数据
        output_path: 输出文件路径
    """
    if not matrix:
        return False
    
    # 转换为科学计数法格式
    converted_lines = []
    for row in matrix:
        formatted_row = []
        for value in row:
            # 转换为科学计数法，保持精度
            formatted_value = f"{value:.16e}"
            formatted_row.append(formatted_value)
        converted_lines.append(" ".join(formatted_row) + "\n")
    
    # 写入转换后的文件
    try:
        with open(output_path, 'w') as f:
            f.writelines(converted_lines)
        
        print(f"✅ 转换完成: -> {output_path}")
        print("转换后的格式:")
        for line in converted_lines:
            print(f"  {line.strip()}")
        return True
    except Exception as e:
        print(f"❌ 写入文件失败: {e}")
        return False


def convert_K_format(input_path, output_path=None):
    """
    转换K.txt文件格式（保持原有功能）
    
    Args:
        input_path: 输入K.txt文件路径
        output_path: 输出K.txt文件路径，如果为None则覆盖原文件
    """
    # 读取原始文件
    matrix = read_txt_matrix(input_path)
    if not matrix:
        return False
    
    # 确定输出路径
    if output_path is None:
        output_path = input_path
    
    return convert_matrix_to_K_format(matrix, output_path)


def convert_csv_to_K(input_path, output_path):
    """
    将CSV文件转换为K.txt格式
    
    Args:
        input_path: 输入CSV文件路径
        output_path: 输出K.txt文件路径
    """
    # 读取CSV文件
    matrix = read_csv_matrix(input_path)
    if not matrix:
        return False
    
    return convert_matrix_to_K_format(matrix, output_path)


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    将四元数转换为旋转矩阵
    
    Args:
        qx, qy, qz, qw: 四元数分量
        
    Returns:
        3x3旋转矩阵
    """
    # 归一化四元数
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # 计算旋转矩阵
    R = np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
    ])
    
    return R


def convert_odometry_to_poses(input_path, output_path, frame_skip=1):
    """
    将odometry.csv文件转换为poses.txt格式，支持抽帧
    
    Args:
        input_path: 输入odometry.csv文件路径
        output_path: 输出poses.txt文件路径
        frame_skip: 抽帧比例，每隔frame_skip个位姿选择1个（默认1表示不抽帧）
    """
    poses = []
    all_poses = []  # 存储所有位姿，用于抽帧
    
    try:
        with open(input_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # 检查列名
            fieldnames = reader.fieldnames
            print(f"📋 检测到的列名: {fieldnames}")
            
            # 清理列名（去除空格）
            cleaned_fieldnames = [col.strip() if col else col for col in fieldnames]
            print(f"🧹 清理后的列名: {cleaned_fieldnames}")
            
            # 验证必需的列是否存在（考虑可能的空格）
            required_columns = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
            missing_columns = []
            
            # 检查原始列名和清理后的列名
            for col in required_columns:
                if col not in fieldnames and col not in cleaned_fieldnames:
                    missing_columns.append(col)
            
            if missing_columns:
                print(f"❌ 缺少必需的列: {missing_columns}")
                return False
            
            for row in reader:
                try:
                    # 尝试从原始列名或清理后的列名获取数据
                    def get_value(row, col_name):
                        # 尝试原始列名
                        if col_name in row:
                            return row[col_name]
                        # 尝试带空格的列名
                        spaced_col = f" {col_name}"
                        if spaced_col in row:
                            return row[spaced_col]
                        raise KeyError(f"找不到列: {col_name}")
                    
                    # 提取位置和四元数
                    x = float(get_value(row, 'x').strip() if isinstance(get_value(row, 'x'), str) else get_value(row, 'x'))
                    y = float(get_value(row, 'y').strip() if isinstance(get_value(row, 'y'), str) else get_value(row, 'y'))
                    z = float(get_value(row, 'z').strip() if isinstance(get_value(row, 'z'), str) else get_value(row, 'z'))
                    qx = float(get_value(row, 'qx').strip() if isinstance(get_value(row, 'qx'), str) else get_value(row, 'qx'))
                    qy = float(get_value(row, 'qy').strip() if isinstance(get_value(row, 'qy'), str) else get_value(row, 'qy'))
                    qz = float(get_value(row, 'qz').strip() if isinstance(get_value(row, 'qz'), str) else get_value(row, 'qz'))
                    qw = float(get_value(row, 'qw').strip() if isinstance(get_value(row, 'qw'), str) else get_value(row, 'qw'))
                    
                    # 转换为旋转矩阵
                    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
                    
                    # 构建4x4变换矩阵
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = [x, y, z]
                    
                    # 展平为一行
                    pose_flat = pose.flatten()
                    all_poses.append(pose_flat)
                    
                except (ValueError, KeyError) as e:
                    print(f"⚠️  跳过无效行: {e}")
                    continue
        
        # 抽帧处理
        if frame_skip > 1:
            poses = all_poses[::frame_skip]  # 每隔frame_skip个选择1个
            print(f"🔄 抽帧处理: 从 {len(all_poses)} 个位姿中每隔 {frame_skip} 个选择1个，共选择 {len(poses)} 个位姿")
        else:
            poses = all_poses
        
        # 写入poses.txt文件
        with open(output_path, 'w') as f:
            for pose in poses:
                # 转换为科学计数法格式
                pose_str = " ".join([f"{val:.16e}" for val in pose])
                f.write(pose_str + "\n")
        
        print(f"✅ 转换完成: {input_path} -> {output_path}")
        print(f"📊 转换了 {len(poses)} 个位姿")
        return True
        
    except Exception as e:
        print(f"❌ 转换odometry文件失败: {e}")
        return False


def convert_directory(input_dir, output_dir=None, file_type="txt", frame_skip=1):
    """
    转换目录中所有的文件
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径，如果为None则覆盖原文件
        file_type: 文件类型，"txt"、"csv"或"odometry"
        frame_skip: 抽帧比例，每隔frame_skip个位姿选择1个（仅对odometry文件有效）
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ 错误: 输入目录不存在: {input_dir}")
        return
    
    # 智能识别文件类型
    if file_type == "csv":
        # 分别处理camera_matrix.csv和odometry.csv
        camera_files = list(input_path.rglob("camera_matrix.csv"))
        odometry_files = list(input_path.rglob("odometry.csv"))
        
        print(f"📁 找到 {len(camera_files)} 个camera_matrix.csv文件:")
        for file in camera_files:
            print(f"  - {file}")
        
        print(f"📁 找到 {len(odometry_files)} 个odometry.csv文件:")
        for file in odometry_files:
            print(f"  - {file}")
        
        # 转换camera_matrix.csv文件（内参）
        for file in camera_files:
            if output_dir:
                rel_path = file.relative_to(input_path)
                output_file = Path(output_dir) / rel_path.parent / "K.txt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                convert_csv_to_K(str(file), str(output_file))
            else:
                output_file = file.parent / "K.txt"
                convert_csv_to_K(str(file), str(output_file))
        
        # 转换odometry.csv文件（外参）
        for file in odometry_files:
            if output_dir:
                rel_path = file.relative_to(input_path)
                output_file = Path(output_dir) / rel_path.parent / "poses.txt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                convert_odometry_to_poses(str(file), str(output_file), frame_skip)
            else:
                output_file = file.parent / "poses.txt"
                convert_odometry_to_poses(str(file), str(output_file), frame_skip)
                
    elif file_type == "odometry":
        # 只处理odometry.csv文件
        pattern = "odometry.csv"
        files = list(input_path.rglob(pattern))
        
        if not files:
            print(f"❌ 错误: 在目录 {input_dir} 中未找到{pattern}文件")
            return
        
        print(f"📁 找到 {len(files)} 个{pattern}文件:")
        for file in files:
            print(f"  - {file}")
        
        for file in files:
            if output_dir:
                rel_path = file.relative_to(input_path)
                output_file = Path(output_dir) / rel_path.parent / "poses.txt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                convert_odometry_to_poses(str(file), str(output_file), frame_skip)
            else:
                output_file = file.parent / "poses.txt"
                convert_odometry_to_poses(str(file), str(output_file), frame_skip)
    else:
        # 处理K.txt文件
        pattern = "K.txt"
        files = list(input_path.rglob(pattern))
        
        if not files:
            print(f"❌ 错误: 在目录 {input_dir} 中未找到{pattern}文件")
            return
        
        print(f"📁 找到 {len(files)} 个{pattern}文件:")
        for file in files:
            print(f"  - {file}")
        
        for file in files:
            if output_dir:
                rel_path = file.relative_to(input_path)
                output_file = Path(output_dir) / rel_path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                convert_K_format(str(file), str(output_file))
            else:
                convert_K_format(str(file))


def main():
    parser = argparse.ArgumentParser(description="转换相机内参矩阵和位姿数据格式")
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("-o", "--output", help="输出文件或目录路径（可选，默认覆盖原文件）")
    parser.add_argument("-d", "--directory", action="store_true", 
                       help="指定输入为目录，将转换目录中所有文件")
    parser.add_argument("-t", "--type", choices=["txt", "csv", "odometry"], default="txt",
                       help="文件类型：txt（K.txt文件）、csv（camera_matrix.csv文件）或odometry（odometry.csv文件）")
    parser.add_argument("-f", "--frame_skip", type=int, default=1,
                       help="抽帧比例，每隔frame_skip个位姿选择1个（仅对odometry文件有效，默认1表示不抽帧）")
    
    args = parser.parse_args()
    
    if args.directory:
        # 转换目录
        convert_directory(args.input, args.output, args.type, args.frame_skip)
    else:
        # 转换单个文件
        if not os.path.exists(args.input):
            print(f"❌ 错误: 输入文件不存在: {args.input}")
            return
        
        if args.type == "csv":
            # CSV文件转换（camera_matrix.csv -> K.txt）
            if args.output is None:
                # 默认输出到同目录的K.txt
                output_path = str(Path(args.input).parent / "K.txt")
            else:
                output_path = args.output
            convert_csv_to_K(args.input, output_path)
        elif args.type == "odometry":
            # odometry.csv文件转换（odometry.csv -> poses.txt）
            if args.output is None:
                # 默认输出到同目录的poses.txt
                output_path = str(Path(args.input).parent / "poses.txt")
            else:
                output_path = args.output
            convert_odometry_to_poses(args.input, output_path, args.frame_skip)
        else:
            # TXT文件转换
            convert_K_format(args.input, args.output)


if __name__ == "__main__":
    # 如果没有命令行参数，提供交互式界面
    import sys
    if len(sys.argv) == 1:
        print("🔧 相机内参矩阵格式转换工具")
        print("=" * 50)
        
        # 获取输入路径
        input_path = input("请输入文件或目录路径: ").strip()
        
        if not input_path:
            print("❌ 未输入路径，退出程序")
            sys.exit(1)
        
        # 获取文件类型
        print("请选择文件类型:")
        print("1. TXT文件 (K.txt)")
        print("2. CSV文件 (camera_matrix.csv)")
        print("3. Odometry文件 (odometry.csv)")
        choice = input("请输入选择 (1/2/3): ").strip()
        
        if choice == "2":
            file_type = "csv"
        elif choice == "3":
            file_type = "odometry"
        else:
            file_type = "txt"
        
        # 获取抽帧参数（仅对odometry文件有效）
        frame_skip = 1
        if file_type == "odometry":
            frame_skip_input = input("请输入抽帧比例（每隔多少个位姿选择1个，直接回车表示不抽帧）: ").strip()
            if frame_skip_input:
                try:
                    frame_skip = int(frame_skip_input)
                    if frame_skip < 1:
                        print("❌ 抽帧比例必须大于0，使用默认值1")
                        frame_skip = 1
                except ValueError:
                    print("❌ 输入无效，使用默认值1")
                    frame_skip = 1
        
        # 判断是文件还是目录
        if os.path.isfile(input_path):
            # 单个文件
            if file_type == "csv":
                output_path = input("请输入输出K.txt文件路径（直接回车使用默认路径）: ").strip()
                if not output_path:
                    output_path = str(Path(input_path).parent / "K.txt")
                convert_csv_to_K(input_path, output_path)
            elif file_type == "odometry":
                output_path = input("请输入输出poses.txt文件路径（直接回车使用默认路径）: ").strip()
                if not output_path:
                    output_path = str(Path(input_path).parent / "poses.txt")
                convert_odometry_to_poses(input_path, output_path, frame_skip)
            else:
                output_path = input("请输入输出文件路径（直接回车覆盖原文件）: ").strip()
                if not output_path:
                    output_path = None
                convert_K_format(input_path, output_path)
                
        elif os.path.isdir(input_path):
            # 目录
            output_dir = input("请输入输出目录路径（直接回车覆盖原文件）: ").strip()
            if not output_dir:
                output_dir = None
            convert_directory(input_path, output_dir, file_type, frame_skip)
            
        else:
            print(f"❌ 错误: 路径不存在: {input_path}")
    else:
        # 使用命令行参数
        main()
