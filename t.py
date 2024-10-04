import cv2
import numpy as np

import cv2
import numpy as np

# Загрузка изображения
result = cv2.imread('masked_map_image0.png')

# Получение одного канала (например, красного)
result = result[:, :, 0]

# Расширение границ изображения
# result = cv2.dilate(result, None, iterations=3)  # Толщина контура
kernel = np.ones((5, 5), np.uint8)  
    # Применение эрозии перед дилатацией для уменьшения артефактов
result = cv2.erode(result, kernel, iterations=2)
result = cv2.dilate(result, kernel, iterations=3)  # Увеличение толщины контура
result = cv2.Canny(result, 100, 200)


mask1 = np.zeros((result.shape[0]+2,result.shape[1]+2), np.uint8)
cv2.floodFill(result, mask1, (result.shape[1]//2,result.shape[0]//2), 255)
# # Нахождение контуров
# contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # #
# print(contours[0][:,0,:])
# mask1 = np.zeros((result.shape[0], result.shape[1]), np.uint8)
# cv2.drawContours(mask1, contours, -1, 255, 1)
# # Проверкаly(mask1, contours, 255)  # Заполнение контуров белым цветом (255)

# Отображение изображений
cv2.imshow('Edges', result)  # Показать изображение с границами
cv2.waitKey(0)
cv2.imshow('Mask', mask1)  # Показать маску
cv2.waitKey(0)

cv2.destroyAllWindows()
