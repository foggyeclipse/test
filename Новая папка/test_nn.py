import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

def load_model():
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize(512),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

def visualize_segmentation(image_path, output_tensor):
    output = output_tensor['out'].squeeze().detach().cpu().numpy()
    output = np.argmax(output, axis=0)

    num_classes = 21
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    colors[1] = [184, 223, 245]  # Установите цвета для нужных классов

    mask = colors[output]
    mask = np.stack([mask[..., 0], mask[..., 1], mask[..., 2]], axis=-1)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    gray_mask = cv2.cvtColor(mask_resized.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    image_with_boxes = image_rgb.copy()
    for contour in contours:
        if len(contour) >= 5:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.show()

def main(image_path):
    model = load_model()
    image = preprocess_image(image_path)
    
    with torch.no_grad():
        output = model(image)
    
    visualize_segmentation(image_path, output)

if __name__ == "__main__":
    main('167415538315872121-2.jpg')
