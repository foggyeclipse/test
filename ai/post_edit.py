import cv2
import numpy as np

def post_edit(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    # Преобразование изображения в цветовую модель HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([30, 255, 255])
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([180, 50, 200])
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Создание маски для выделения объектов
    mask_tree = cv2.inRange(hsv, lower_green, upper_green)
    mask_field = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_road = cv2.inRange(hsv, lower_gray, upper_gray)
    mask_water = cv2.inRange(hsv, lower_blue, upper_blue)

    # Применение морфологических операций для улучшения маски
    kernel = np.ones((7, 7), np.uint8)
    kernel_f = np.ones((10, 10), np.uint8)
    kernel_r = np.ones((10, 10), np.uint8)
    kernel_w = np.ones((10, 10), np.uint8)
    mask_tree = cv2.morphologyEx(mask_tree, cv2.MORPH_OPEN, kernel)
    mask_tree = cv2.morphologyEx(mask_tree, cv2.MORPH_CLOSE, kernel)
    mask_field = cv2.morphologyEx(mask_field, cv2.MORPH_OPEN, kernel_f)
    mask_field = cv2.morphologyEx(mask_field, cv2.MORPH_CLOSE, kernel_f)
    mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, kernel_r)
    mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_CLOSE, kernel_r)
    mask_water = cv2.morphologyEx(mask_water, cv2.MORPH_OPEN, kernel_w)
    mask_water = cv2.morphologyEx(mask_water, cv2.MORPH_CLOSE, kernel_w)
    mask_field = cv2.add(mask_field, cv2.subtract(cv2.bitwise_not(mask_field), mask_tree))


    # Расширение маски для создания утолщенного контура
    dilated_mask_t = cv2.dilate(mask_tree, kernel, iterations=1)
    contour_mask_t = cv2.bitwise_or(dilated_mask_t, mask_tree)
    dilated_mask_f = cv2.dilate(mask_field, kernel_f, iterations=1)
    contour_mask_f = cv2.bitwise_or(dilated_mask_f, mask_field)
    dilated_mask_r = cv2.dilate(mask_road, kernel_r, iterations=1)
    contour_mask_r = cv2.bitwise_or(dilated_mask_r, mask_road)
    dilated_mask_w = cv2.dilate(mask_water, kernel_w, iterations=1)
    contour_mask_w = cv2.bitwise_or(dilated_mask_w, mask_water)

    mask_tree = cv2.add(mask_tree, cv2.subtract(cv2.bitwise_not(mask_tree), mask_field))

    # Поиск внешних контуров на заполненной маске
    contours_tree, _ = cv2.findContours(contour_mask_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_field, _ = cv2.findContours(contour_mask_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_road, _ = cv2.findContours(contour_mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_water, _ = cv2.findContours(contour_mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Отрисовка контуров на оригинальном изображении
    cv2.drawContours(image, contours_tree, -1, (30, 100,  33), thickness=cv2.FILLED)
    cv2.drawContours(image, contours_field, -1, (255, 255, 255), thickness=cv2.FILLED)
    cv2.drawContours(image, contours_road, -1, (128, 128, 128), thickness=cv2.FILLED)
    cv2.drawContours(image, contours_water, -1, (255, 0, 0), thickness=cv2.FILLED)

    cv2.imwrite(path, image)
