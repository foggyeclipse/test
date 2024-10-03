import glob
import numpy as np
from skimage import measure
from skimage.io import imread
import matplotlib.pyplot as plt

images = sorted(glob.glob('./data/raw_imgs/*.png'))

COLORS = ['black', '#808080', '#00FF00', '#FFFF00', '#0000FF']  # Чёрный, Серый, Зелёный, Жёлтый, Синий

fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize=(16, 6))

for i, image in enumerate(images):
    image = imread(image)
    ax[1, i].set_title('Mask')
    ax[1, i].set_axis_off()    
    ax[1, i].imshow(image/1.5)
   
    for channel in range(5):
        contours = measure.find_contours(np.array(image[:,:,channel]))
        for contour in contours:
            ax[1, i].plot(contour[:, 1], contour[:, 0], linewidth=1, color=COLORS[channel])
