import os

def process_txt_files(folder_path):
    # 获取文件夹中的所有文件
    all_files = os.listdir(folder_path)
    
    # 遍历所有文件
    for file in all_files:
        file_path = os.path.join(folder_path, file)
        
        # 检查是否为txt文件
        if file.endswith('.txt'):
            # 提取txt文件名（不包括扩展名）
            txt_filename = os.path.splitext(file)[0]
            
            # 构建对应的图像文件路径
            image_file = os.path.join(folder_path, txt_filename + '.jpg')
            
            # 检查图像文件是否存在，不存在则删除txt文件
            if not os.path.isfile(image_file):
                os.remove(file_path)

# 测试示例
folder_path = '/media/rangeronmars/1903QF2/Dataset/Data/HITSZ-SJTU-Emergency'  # 替换为实际的文件夹路径
process_txt_files(folder_path)

