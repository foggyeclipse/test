import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_model():
    # Загрузка предобученной модели DeepLabV3+ из TensorFlow
    model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(None, None, 3))
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((513, 513))  # Размер 513x513
    image_array = np.array(image)
    image_array = tf.image.convert_image_dtype(image_array, dtype=tf.float32)
    image_array = tf.expand_dims(image_array, 0)  # Добавляем batch dimension
    return image_array

def postprocess_and_display(prediction, original_image):
    prediction = tf.argmax(prediction, axis=-1).numpy()
    prediction = np.squeeze(prediction)
    
    # Создание цветовой карты для классов (проверьте правильность значений)
    colormap = plt.get_cmap('tab20')  # Используем tab20 или другую цветовую карту
    color_mask = colormap(prediction / np.max(prediction))[:, :, :3]
    color_mask = (color_mask * 255).astype(np.uint8)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(original_image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image')
    plt.imshow(color_mask)
    plt.axis('off')
    
    plt.show()

def main(image_path):
    model = load_model()
    image_array = preprocess_image(image_path)
    predictions = model.predict(image_array)
    original_image = Image.open(image_path).resize((513, 513))
    postprocess_and_display(predictions, original_image)

if __name__ == '__main__':
    image_path = 'map-3.png'
    main(image_path)