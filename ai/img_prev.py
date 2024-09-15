import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.io import imread
import glob
from skimage import measure

images = sorted(glob.glob('./data/raw_imgs/*.png'))

# for img in images:
#     img = imread(img)
#     print(img.shape)
#     plt.imshow(img[:,:,3])
#     plt.show()
# Обновленные цвета
COLORS = ['black', '#808080', '#00FF00', '#FFFF00', '#0000FF']  # Серый, Зелёный, Жёлтый, Синий

fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize=(16, 6))

for i, image in enumerate(images):
    # ax[0, i].set_title('Image')
    # ax[0, i].set_axis_off()
    # ax[0, i].imshow(image)
    image = imread(image)
    ax[1, i].set_title('Mask')
    ax[1, i].set_axis_off()    
    ax[1, i].imshow(image/1.5)
   
    for channel in range(5):
        contours = measure.find_contours(np.array(image[:,:,channel]))
        for contour in contours:
            ax[1, i].plot(contour[:, 1], contour[:, 0], linewidth=1, color=COLORS[channel])


# print(img.shape)
# cv2.imshow('a',img)
# cv2.waitKey(0)