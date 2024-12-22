import os

def rename_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 筛选出图片文件（假设是jpg, png, jpeg 格式的文件）
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 对图片文件按名称排序，以确保重命名是按文件名顺序进行
    image_files.sort()
    image_files.reverse() # 反转列表，确保最新的文件在最前面
    
    # 遍历图片文件并重命名
    for index, file_name in enumerate(image_files, start=1):
        # 获取文件的原始路径
        old_path = os.path.join(folder_path, file_name)
        index=index
        
        # 构造新的文件名和路径（保留原始扩展名）
        new_name = f"{index}{os.path.splitext(file_name)[1]}"
        new_path = os.path.join(folder_path, new_name)
        
        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed '{file_name}' to '{new_name}'")

# 使用方法
folder_path = "/home/data/shizh/real_or_fake/idCard/3_real"  # 替换为实际文件夹路径
rename_images(folder_path)
