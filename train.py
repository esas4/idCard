from fastai.vision.all import *
import pandas as pd
import matplotlib.pyplot as plt
# from fastai.callback.schedule import ReduceLROnPlateau

# 1. 加载数据
train_df = pd.read_csv('train.csv', sep=' ')  # 假设用空格分隔
val_df = pd.read_csv('val.csv', sep=' ')

# print(train_df.head())
# print(val_df.head())
# quit()

# train_df[1] = train_df[1].astype(int)  # 将标签转换为整数
# val_df[1] = val_df[1].astype(int)

# 2. 准备DataLoaders
def get_dls(train_df, val_df, img_size=224, bs=64):
    # 转换DataFrame为DataLoaders
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),  # 图像和类别块
        get_x=ColReader(0),  # 第一列是图片路径
        get_y=ColReader(1),  # 第二列是图片标签
        splitter=IndexSplitter(val_df.index),  # 使用验证集索引
        # item_tfms=Resize(img_size),  # 调整图像大小
        # item_tfms=[Resize(img_size), ToTensor()], # 调整图像大小
        # batch_tfms=aug_transforms(mult=0.5),  # 数据增强
    )
    return dblock.dataloaders(pd.concat([train_df, val_df]), bs=bs)

# 创建DataLoaders
dls = get_dls(train_df, val_df, bs=16)

# 3. 定义模型和训练过程
def train_model(dls, lr=1e-3, epochs=10, save_path='model_epoch', train_losses=[], val_losses=[], accuracies=[]):
    learn = vision_learner(dls, resnet34, metrics=accuracy)  # 使用ResNet34
    # learn.add_cb(ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=5))
    learn.model_dir = Path('.')  # 设置模型保存目录
    
    min_train_loss=100

    # 训练并保存每个epoch的模型
    for epoch in range(epochs):
        print(f"Training Epoch {epoch+1}/{epochs}")
        learn.fit_one_cycle(1, lr)
        if learn.recorder.values[0][0]<min_train_loss and learn.recorder.values[0][2]>0.999999:
            model_save_path = f"{save_path}"
            learn.save(model_save_path)
            learn.export(model_save_path+'.pkl')
            min_train_loss=learn.recorder.values[0][0]
            print(f"In Epoch {epoch}, saved model to: {model_save_path}")
        train_losses.append(learn.recorder.values[0][0])
        val_losses.append(learn.recorder.values[0][1])
        accuracies.append(learn.recorder.values[0][2])
        # print(f"train_losses: {train_losses}")
        # print(f"val_losses: {val_losses}")
        # print(f"accuracies: {accuracies}")

    return learn

# 4. 可视化训练过程
def plot_metrics(train_losses, val_losses, accuracies):
    # metrics = learn.recorder.values
    # epochs = range(1, len(metrics) + 1)
    # losses = [m[0] for m in metrics]
    # accuracies = [m[1] for m in metrics]
    epochs=range(1, len(train_losses)+1)
    t_losses=train_losses
    v_losses=val_losses
    accuracies=accuracies

    # 绘制train损失值的图表
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(t_losses)), t_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss over Epochs')

    # 标出最大值和最小值
    min_loss = min(t_losses)
    max_loss = max(t_losses)
    min_loss_epoch = t_losses.index(min_loss)
    max_loss_epoch = t_losses.index(max_loss)

    plt.plot(min_loss_epoch, min_loss, 'ro')  # 标出最小值
    plt.plot(max_loss_epoch, max_loss, 'bo')  # 标出最大值
    plt.annotate(f'Min: {min_loss:.2f}', (min_loss_epoch, min_loss), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'Max: {max_loss:.2f}', (max_loss_epoch, max_loss), textcoords="offset points", xytext=(0,-15), ha='center')

    plt.legend()
    plt.savefig('res/training_loss.png')
    plt.close()
    
    # 绘制val损失值的图表
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(v_losses)), v_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss over Epochs')

    # 标出最大值和最小值
    min_loss = min(v_losses)
    max_loss = max(v_losses)
    min_loss_epoch = v_losses.index(min_loss)
    max_loss_epoch = v_losses.index(max_loss)

    plt.plot(min_loss_epoch, min_loss, 'ro')  # 标出最小值
    plt.plot(max_loss_epoch, max_loss, 'bo')  # 标出最大值
    plt.annotate(f'Min: {min_loss:.2f}', (min_loss_epoch, min_loss), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'Max: {max_loss:.2f}', (max_loss_epoch, max_loss), textcoords="offset points", xytext=(0,-15), ha='center')

    plt.legend()
    plt.savefig('res/validation_loss.png')
    plt.close()

    # 绘制准确率的图表
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(accuracies)), accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')

    # 标出最大值和最小值
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    min_accuracy_epoch = accuracies.index(min_accuracy)
    max_accuracy_epoch = accuracies.index(max_accuracy)

    plt.plot(min_accuracy_epoch, min_accuracy, 'ro')  # 标出最小值
    plt.plot(max_accuracy_epoch, max_accuracy, 'bo')  # 标出最大值
    plt.annotate(f'Min: {min_accuracy:.2f}', (min_accuracy_epoch, min_accuracy), textcoords="offset points", xytext=(0,10), ha='center')
    plt.annotate(f'Max: {max_accuracy:.2f}', (max_accuracy_epoch, max_accuracy), textcoords="offset points", xytext=(0,-15), ha='center')

    plt.legend()
    plt.savefig('res/validation_accuracy.png')
    plt.close()

# 5. 执行训练并可视化
if __name__ == "__main__":
    train_losses=[]
    val_losses=[] 
    accuracies=[]
    learn = train_model(dls, lr=1e-3, epochs=20, save_path='/home/data/shizh/real_or_fake/idCard/res/res_models/model_epoch', train_losses=train_losses, val_losses=val_losses, accuracies=accuracies)
    # print(learn.recorder.values)    
    # quit()
    plot_metrics(train_losses, val_losses, accuracies)