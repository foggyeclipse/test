import glob
import tensorflow as tf
from unet_ai import model, dice_mc_metric, dice_bce_mc_loss

print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.config.list_physical_devices("GPU") else "OFF" }')


CLASSES = 5  # 5 классов: Дорога, Деревья, Поле, Вода, Фон

SAMPLE_SIZE = (256, 256)

OUTPUT_SIZE = (1080, 1920)

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

train_dataset = dataset.take(2000).cache()
test_dataset = dataset.skip(2000).take(100).cache()
 
train_dataset = train_dataset.batch(8)
test_dataset = test_dataset.batch(8)

model.load_weights('ai/weights/model.weights.h5')

# Компиляция модели
model.compile(optimizer='adam', loss=[dice_bce_mc_loss], metrics=[dice_mc_metric])
history_dice = model.fit(train_dataset, validation_data=test_dataset, epochs=25, initial_epoch=0)

model.save_weights('weights/model.weights.h5')