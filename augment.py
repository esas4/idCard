import os
import random
from PIL import Image

def augment_image(image):
    """
    对一张图片进行裁剪、旋转和翻转的增强操作
    """
    # 随机选择增强操作，确保只保留像素特征一致的增强方式
    transformations = [
        random_rotate,
        random_crop,
        random_flip
    ]
    
    # 随机应用 1 到 2 个增强操作，避免过度增强
    num_transformations = random.randint(1, 2)
    transformations_to_apply = random.sample(transformations, num_transformations)
    
    # 应用每个增强操作
    for transform in transformations_to_apply:
        image = transform(image)
    
    return image

def random_rotate(image):
    """随机旋转图片，仅保留旋转后裁剪的中心区域，保持原图像素特征"""
    angle = random.choice([90, 180])  # 随机选择 90、180 或 270 度旋转
    rotated_image = image.rotate(angle)
    # 裁剪出旋转后的中心区域，保证与原图大小一致
    # width, height = image.size
    # left = (rotated_image.width - width) // 2
    # top = (rotated_image.height - height) // 2
    return rotated_image

def random_crop(image):
    """随机裁剪图片的中心区域，保留 90% 的图像"""
    width, height = image.size
    crop_fraction = 0.9  # 裁剪出 90% 的中心区域
    new_width = int(width * crop_fraction)
    new_height = int(height * crop_fraction)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    return image.crop((left, top, left + new_width, top + new_height)).resize((width, height))

def random_flip(image):
    """随机水平或垂直翻转图片"""
    if random.random() > 0.5:
        return image.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM)  # 垂直翻转

def augment_images_in_directory(input_dir, output_dir, num_augmented_copies=5):
    """
    对目录中的所有图片进行数据增强，并将增强后的图片保存到输出目录中
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            image = Image.open(image_path)

            for i in range(num_augmented_copies):
                augmented_image = augment_image(image)
                new_filename = f"{os.path.splitext(filename)[0]}_aug_{i+1}.jpg"
                augmented_image.save(os.path.join(output_dir, new_filename))

    print(f"Data augmentation completed. Augmented images saved in '{output_dir}'.")

# 使用示例
input_dir = '/home/data/shizh/real_or_fake/idCard/real_part'       # 原始图片目录
output_dir = '/home/data/shizh/real_or_fake/idCard/real_augment_old'     # 增强后图片保存目录
num_augmented_copies = 3                # 每张图片生成 5 个增强图片

augment_images_in_directory(input_dir, output_dir, num_augmented_copies)
