import os
import shutil

def move_txt_files(source_folder, destination_folder):
    # 检查源文件夹是否存在
    if not os.path.exists(source_folder):
        print(f"源文件夹 '{source_folder}' 不存在")
        return

    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹中的所有文件
    files = os.listdir(source_folder)

    # 遍历文件列表
    files_to_move = []
    for file in files:
        # 检查文件是否是以 '.txt' 结尾
        if file.lower().endswith('.txt'):
            files_to_move.append(file)
            # 移动文件
    print(files_to_move)
    for file in files_to_move:
        source_file = os.path.join(source_folder, file)
        destination_file = os.path.join(destination_folder, file)
        shutil.move(source_file, destination_file)
        print(f"移动文件 '{source_file}' 到 '{destination_file}'")

# 指定源文件夹和目标文件夹的路径
source_folder = 'Label'
destination_folder = 'dst1'

# 调用函数进行文件移动
move_txt_files(source_folder, destination_folder)
