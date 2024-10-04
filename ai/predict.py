import os
import numpy as np
from ai.unet_ai import model
from skimage.transform import resize
from skimage.io import imread, imsave


class_colors = {       
    1: [0, 255, 0],     # Зеленый для деревьев
    3: [255, 255, 0],   # Желтый для полей
    2: [128, 128, 128], # Серый для дорог
    4: [0, 0, 255],     # Синий для водоемов
    0: [0, 0, 0]     # Черный для фона
}

def apply_colors_to_mask(masks, class_colors):
    """
    Преобразование многоканальной маски в RGB-изображение с цветами для каждого класса.
    
    masks: массив размером (h, w, num_classes) - Предсказанные каналы масок
    class_colors: dict - Словарь с цветами для каждого класса
    
    Возвращает: RGB-изображение (h, w, 3)
    """
    h, w, num_classes = masks.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx in range(num_classes):
        class_mask = masks[:, :, class_idx]
        color = class_colors[class_idx]
        
        for i in range(3):
            color_mask[:, :, i] += (class_mask * color[i]).astype(np.uint8)
    
    return color_mask



model.load_weights('ai/weights/model.weights.h5')

SAMPLE_SIZE = (256, 256)

def predict_place(frames):
    frames = frames

    for filename in frames:
        frame = imread(filename)
        print(f"Original frame shape: {frame.shape}")
        
        sample = resize(frame, SAMPLE_SIZE)
        print(f"Resized sample shape: {sample.shape}")
        
        if sample.shape[2] == 4:
            sample = sample[:, :, :3]  # Удаляем альфа-канал, если он присутствует
        
        predict = model.predict(sample.reshape((1,) +  SAMPLE_SIZE + (3,)))
        predict = predict.reshape(SAMPLE_SIZE + (5,))  # 5 каналов для каждого класса
        
        color_mask = apply_colors_to_mask(predict, class_colors)
        
        color_mask_resized = resize(color_mask, frame.shape[:2], preserve_range=True).astype(np.uint8)
        
        imsave(f'./masked_{os.path.basename(filename)}', color_mask_resized)
