import os
import csv

# 定义真实图片文件夹及其标签
real_folder = ["2_test_real_cropped"]
real_label = 0

# 定义假图片文件夹列表及其标签
fake_folders = ["3_fake_number_cutout_cropped"]
fake_label = 1

output_csv = 'test.csv'

# 初始化列表用于存储文件路径和标签
data = []
            
# 处理真图片
for folder in real_folder:
    for i in range(1, 51):  # train的真图片编号从 1 到 90
        filename = f"{i}.jpg"
        file_path = os.path.join(folder, filename)
        
        # 确认文件存在
        if os.path.isfile(file_path):
            # 使用绝对路径
            abs_path = os.path.abspath(file_path)
            data.append([abs_path, real_label])

# # 处理真实图片
# for i in range(41, 51):      # 图片编号从 1 到 30
#     for j in range(1, 4):   # 每个编号下有 1 到 3 个 _aug 图片
#         filename = f"{i}_aug_{j}.jpg"
#         file_path = os.path.join(real_folder, filename)
        
#         # 确认文件存在
#         if os.path.isfile(file_path):
#             # 使用绝对路径
#             abs_path = os.path.abspath(file_path)
#             data.append([abs_path, real_label])

# 处理假图片
for folder in fake_folders:
    for i in range(1,21):  # 假图片编号从 1 到 30
        filename = f"{i}.png"
        file_path = os.path.join(folder, filename)
        
        # 确认文件存在
        if os.path.isfile(file_path):
            # 使用绝对路径
            abs_path = os.path.abspath(file_path)
            data.append([abs_path, fake_label])

# 将结果写入 CSV 文件
with open(output_csv, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(['filepath', 'label'])  # 写入标题行
    writer.writerows(data)

print(f"所有文件夹的图片路径和标签已成功写入 {output_csv}")