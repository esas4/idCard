import pandas as pd
from fastai.vision.all import *
from pathlib import Path
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

def main():
    # 定义文件路径
    test_csv_path = '/home/data/shizh/real_or_fake/idCard/test.csv'  # 测试集 CSV 文件路径
    model_path = '/home/data/shizh/real_or_fake/idCard/res/res_models/20_all_model_epoch'    # 训练好的模型路径

    # 加载测试数据
    df = pd.read_csv(test_csv_path, delim_whitespace=True, header=None, names=['image_path', 'label'])
    df['image_path'] = df['image_path'].apply(lambda x: Path(x))  # 转换为 Path 对象

    # 检查是否有图像文件不存在
    for img_path in df['image_path']:
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

    # 创建 DataLoader
    dls = ImageDataLoaders.from_df(
        df,                  # 数据框
        fn_col='image_path', # 图像文件列
        label_col='label',   # 标签列
        valid_pct=0.0,       # 测试集不需要划分验证集
        # item_tfms=Resize(224),  # 调整图像大小
        bs=1                # 批量大小
    )
    
    # 由于训练时使用save来保存模型，此处重建learner并加载权重
    learn=vision_learner(dls, resnet34)
    learn.load(model_path)   

    # 加载训练好的模型
    # learn = load_learner(model_path)
    print("Model loaded successfully.")

    # 获取预测值和真实标签
    preds, targs = learn.get_preds(dl=dls.train)
    
    # # 打印所有图像的路径和标签，确保顺序一致
    # for i, (img, label) in enumerate(dls.train):
    #     img_path = df.iloc[i]['image_path']  # 获取第 i 个图像的路径
    #     true_label = df.iloc[i]['label']  # 获取第 i 个图像的标签
    #     print(f"Index {i}: Image Path: {img_path}, True Label: {true_label}")
    # quit()
    
    # for i in range(len(preds)):
    #     pred_class = preds.argmax(dim=1)[i].item()  # 预测的类别索引
    #     pred_confidence = preds.max(dim=1).values[i].item() # 预测的置信度
    #     if i<=19:
    #         print(f"Sample real_{i+1}: Predicted class = {pred_class}, Confidence = {pred_confidence}")
    #     else:
    #         print(f"Sample fake_{i-19}: Predicted class = {pred_class}, Confidence = {pred_confidence}")

    # 计算准确率
    accuracy = (preds.argmax(dim=1) == targs).float().mean().item()
    print(f"Number of predictions: {len(preds)}, Number of targets: {len(targs)}")
    print(f"Test Accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    main()
