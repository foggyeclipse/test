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

print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.config.list_physical_devices("GPU") else "OFF" }')

CLASSES = 5  # У вас 4 класса: Дорога, Деревья, Поле и Вода

SAMPLE_SIZE = (256, 256)

OUTPUT_SIZE = (1080, 1920)

# Обновленные цвета
COLORS = ['black', '#00FF00', '#808080', '#FFFF00', '#0000FF']  # Серый, Зелёный, Жёлтый, Синий

def load_images(image, mask):
    image = tf.io.read_file(image)
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize(image, OUTPUT_SIZE)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = image / 255.0
    
    mask = tf.io.read_file(mask)
    mask = tf.io.decode_png(mask, channels=3)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize(mask, OUTPUT_SIZE)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    
    masks = []
    
    for i in range(CLASSES):
        masks.append(tf.where(tf.equal(mask, float(i)), 1.0, 0.0))
    
    masks = tf.stack(masks, axis=2)
    masks = tf.reshape(masks, OUTPUT_SIZE + (CLASSES,))

    return image, masks

def augmentate_images(image, masks):   
    random_crop = tf.random.uniform((), 0.3, 1)
    image = tf.image.central_crop(image, random_crop)
    masks = tf.image.central_crop(masks, random_crop)
    
    random_flip = tf.random.uniform((), 0, 1)    
    if random_flip >= 0.5:
        image = tf.image.flip_left_right(image)
        masks = tf.image.flip_left_right(masks)
    
    image = tf.image.resize(image, SAMPLE_SIZE)
    masks = tf.image.resize(masks, SAMPLE_SIZE)
    
    return image, masks


images = sorted(glob.glob('./data/raw_imgs/*.png'))
masks = sorted(glob.glob('./data/masks/*.png'))

images_dataset = tf.data.Dataset.from_tensor_slices(images)
masks_dataset = tf.data.Dataset.from_tensor_slices(masks)

dataset = tf.data.Dataset.zip((images_dataset, masks_dataset))

dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.repeat(100)
dataset = dataset.map(augmentate_images, num_parallel_calls=tf.data.AUTOTUNE)

# images_and_masks = list(dataset.take(5))

# fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize=(16, 6))

# for i, (image, masks) in enumerate(images_and_masks):
#     ax[0, i].set_title('Image')
#     ax[0, i].set_axis_off()
#     ax[0, i].imshow(image)
        
#     ax[1, i].set_title('Mask')
#     ax[1, i].set_axis_off()    
#     ax[1, i].imshow(image/1.5)
   
#     for channel in range(CLASSES):
#         contours = measure.find_contours(np.array(masks[:,:,channel]))
#         for contour in contours:
#             ax[1, i].plot(contour[:, 1], contour[:, 0], linewidth=1, color=COLORS[channel])

# plt.show()
# plt.close()

# train_dataset = dataset.take(2000).cache()
# test_dataset = dataset.skip(2000).take(100).cache()
 
# train_dataset = train_dataset.batch(8)
# test_dataset = test_dataset.batch(8)