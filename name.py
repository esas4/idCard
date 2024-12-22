import os

def rename_files(folder_path, prefix):
    # 获取文件夹下的所有文件
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # 过滤出图片文件（根据需要可以调整扩展名）
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_files = [f for f in files if f.lower().endswith(image_extensions)]
    
    # 重命名文件
    for index, file in enumerate(image_files):
        new_name = f"{prefix}_{index}.jpg"  # 你可以根据需要调整扩展名
        old_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_name)
        
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} to {new_file_path}")

# 修改为你的文件夹路径
boy_folder = '/home/data/shizh/real_or_fake/idCard/pics/boy'
girl_folder = '/home/data/shizh/real_or_fake/idCard/pics/girl'

rename_files(boy_folder, 'boy')
rename_files(girl_folder, 'girl')
