import pandas as pd
from fastai.vision.all import *
from pathlib import Path
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import torch

def generate_heatmap(model, image_tensor, cam_extractor):
    """Generates a heatmap for a given image using SmoothGradCAMpp."""
    device = next(model.parameters()).device  # Get the device of the model
    image_tensor = image_tensor.to(device)    # Move image tensor to the same device as the model

    # Ensure image is float32 and requires gradients
    image_tensor = image_tensor.float()  # Ensure the tensor is float32
    image_tensor.requires_grad_()  # Enable gradient computation for this tensor

    # Forward pass on the model to get the outputs
    model.eval()  # Set model to evaluation mode
    output = model(image_tensor.unsqueeze(0))  # Add batch dimension

    # Get the class with the maximum output
    class_idx = output.argmax(dim=1).item()

    # Generate the CAM
    activation_map = cam_extractor(class_idx=class_idx, scores=output)
    heatmap = activation_map[0]  # Get the first map (batch size 1)

    # Remove the batch dimension and make sure it's 2D (H, W)
    heatmap = heatmap.squeeze(0)  # Remove the batch dimension
    heatmap = heatmap.cpu().numpy()  # Convert to NumPy for plotting

    return heatmap, class_idx, output

def plot_heatmap(image_path, heatmap, predicted_label, true_label, confidence, output_path=None):
    """Plots and optionally saves the heatmap superimposed on the original image."""
    # Convert image tensor to PIL image
    # image = to_pil_image(image_tensor)
    image = plt.imread(image_path)
    # plt.imsave('image.png', image)
    # quit()

    # Plot the image and the heatmap
    plt.figure(figsize=(8, 8))
    
    plt.imshow(heatmap, cmap='jet', alpha=1)  # Heatmap overlay
    # plt.imshow(image, alpha=1)  # Original image
    plt.axis('off')
    
    label_text = f"Pred: {predicted_label}, True: {true_label}"
    if predicted_label != true_label:
        label_text += " (Error)"
        
    confidence_text = f"cred:{confidence:.4f}"
    
    plt.text(10, 10, label_text, fontsize=12, color='white', backgroundcolor='black')
    plt.text(10, 30, confidence_text, fontsize=12, color='white', backgroundcolor='black')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def main():
    # Define file paths
    test_csv_path = 'test.csv'  # Test CSV file path
    model_path = '/home/data/shizh/real_or_fake/idCard/res/res_models/20_all_model_epoch'  # Model path
    csvfilename = 'test_result_data.csv'

    # Load test data
    df = pd.read_csv(test_csv_path, delim_whitespace=True, header=None, names=['image_path', 'label'])
    df['image_path'] = df['image_path'].apply(lambda x: Path(x))  # Convert to Path object

    # Check if all image files exist
    for img_path in df['image_path']:
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

    # Create DataLoader
    dls = ImageDataLoaders.from_df(
        df,                  # DataFrame
        fn_col='image_path', # Image path column
        label_col='label',   # Label column
        valid_pct=0.0,       # No validation split for test set
        # item_tfms=Resize(224),  # Resize images
        bs=1,                 # Batch size (one at a time for CAM visualization)
        shuffle=False
    )

    # Load model and move it to the appropriate device (GPU if available)
    learn = vision_learner(dls, resnet34)
    learn.load(model_path)
    # learn = load_learner(model_path)
    learn.model = learn.model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print("Model loaded successfully.")
    
    preds, targs = learn.get_preds(dl=dls.train)
    
    # for i in range(len(preds)):
    #     predicted_label = learn.dls.vocab[preds.argmax(dim=1)[i].item()]
    #     true_label = targs[i].item()
    #     print(f"Index {i}: Predicted: {predicted_label}, True: {true_label}")
    
    # quit()
    
    accuracy = (preds.argmax(dim=1) == targs).float().mean().item()

    # Integrate CAM
    cam_extractor = SmoothGradCAMpp(learn.model)
    
    # 打印所有图像的路径和标签，确保顺序一致
    # for i, (img, label) in enumerate(dls.train):
    #     img_path = df.iloc[i]['image_path']  # 获取第 i 个图像的路径
    #     true_label = df.iloc[i]['label']  # 获取第 i 个图像的标签
    #     print(f"Index {i}: Image Path: {img_path}, True Label: {true_label}")
    # quit()
    
    # original_order = dls.train.items.reset_index()
    # print(f'original_order:', original_order)   
    # print(f'dls.train.items:', dls.train.items)
    # print(f'dls.train', dls.train)
    # quit()
    # error_idx=[]
    csv_predictions=[]

    # Generate predictions and heatmaps
    for i, (img, label) in enumerate(dls.train):
        
        print(f"Processing image {i+1}/{len(dls.train)}")
        
        # image_path_from_dataloader = dls.train.items.iloc[i]['image_path']
        # print(F"image_path_from_ dls.train", image_path_from_dataloader)
        # print(f"image Path from Dataloader:{dls.train.items}", original_order)
                       
        # original_idx = original_order.iloc[i]['index']
        # print(f"original_idx", original_idx)
        image_path = dls.train.items.iloc[i]['image_path']
        image_path_str = str(image_path)
        parts = image_path_str.split('/')
        filename = parts[-1].split('.')[0]
        file_type = parts[0].split('_')[1]
        # print(f"file_type", file_type)
        # quit()
        result = f"{file_type}_{filename}"
        print(result)
        print(f"image_path", image_path)
        # quit()

        # Predict and get heatmap
        heatmap, predicted_class_idx, output = generate_heatmap(learn.model, img[0], cam_extractor)
        
        predicted_label = learn.dls.vocab[predicted_class_idx]
        true_label = learn.dls.vocab[label.item()]
        
        probs = F.softmax(output, dim = 1) 
        confidence = probs[0, predicted_class_idx].item()
              
        if predicted_label == true_label:
            # error_idx.append((result, predicted_label, true_label))
            print(f"For pic {result}: Predicted: {predicted_label}, True: {true_label}, Credence: {confidence:.4f}")
        else:
            print(f"For pic {result}: Predicted: {predicted_label}, True: {true_label}, Credence: {confidence:.4f}, Error!!!")
            # quit()
        
        # 写进csv文件中
        error = 'error!' if predicted_label != true_label else ''
        csv_predictions.append([result, predicted_label, true_label, confidence, error])
            
        with open(csvfilename, mode="w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image', 'prediction', 'lable', 'confidence', 'error'])
            for csv_prediction in csv_predictions:
                writer.writerow(csv_prediction)         

        # Plot and save heatmap
        output_path = f"/home/data/shizh/real_or_fake/idCard/cam/test_all_20_cam/heatmap_{result}.png"
        plot_heatmap(image_path_str, heatmap, predicted_label, true_label, confidence, output_path)
        print(f"Saved heatmap to {output_path}")
        
    print(f"Number of predictions: {len(preds)}, Number of targets: {len(targs)}")
    print(f"Test Accuracy: {accuracy:.2%}")
    
    # for i in range(len(error_idx)): 
    #     (result, predicted_label, true_label) = error_idx[i]
    #     print(F"Prediction error for image {result}: predicted {predicted_label}, true {true_label}")

if __name__ == '__main__':
    main()