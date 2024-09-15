import numpy as np
import matplotlib.pyplot as plt
from unet_ai import model, dice_bce_mc_loss, dice_mc_metric
from train import dataset, CLASSES
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage import measure
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk
from skimage.draw import polygon_perimeter


model.load_weights('weights/model.weights.h5')

class_colors = {       # Черный для фона
    1: [0, 255, 0],     # Зеленый для деревьев
    3: [255, 255, 0],   # Желтый для полей
    2: [128, 128, 128], # Серый для дорог
    4: [0, 0, 255],
    0: [0, 0, 0]     # Синий для водоемов
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
    
    # Проходим по каждому классу и закрашиваем соответствующим цветом
    for class_idx in range(num_classes):
        class_mask = masks[:, :, class_idx]
        color = class_colors[class_idx]
        
        # Закрашиваем пиксели, относящиеся к данному классу
        for i in range(3):  # Проходим по каждому цветному каналу (RGB)
            color_mask[:, :, i] += (class_mask * color[i]).astype(np.uint8)
    
    return color_mask



model.load_weights('weights/model.weights.h5')

SAMPLE_SIZE = (256, 256)

# frames = sorted(glob.glob('data/raw_imgs/*.png'))
frames = ['data/test.png']

for filename in frames:
    frame = imread(filename)
    print(f"Original frame shape: {frame.shape}")
    
    # Изменение размера изображения для предсказания
    sample = resize(frame, SAMPLE_SIZE)
    print(f"Resized sample shape: {sample.shape}")
    
    if sample.shape[2] == 4:
        sample = sample[:, :, :3]  # Удаляем альфа-канал, если он присутствует
    
    # Предсказание маски
    predict = model.predict(sample.reshape((1,) +  SAMPLE_SIZE + (3,)))
    predict = predict.reshape(SAMPLE_SIZE + (5,))  # 5 каналов для каждого класса
    
    # Применение цветов к маске
    color_mask = apply_colors_to_mask(predict, class_colors)
    
    # Изменение размера цветной маски до исходного разрешения кадра
    color_mask_resized = resize(color_mask, frame.shape[:2], preserve_range=True).astype(np.uint8)
    
    # Выводим или сохраняем изображение с наложенной цветной маской
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(frame)
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted Color Mask")
    plt.imshow(color_mask_resized)
    
    plt.show()

    # Сохранение результата
    imsave(f'data/masked_{os.path.basename(filename)}', color_mask_resized)