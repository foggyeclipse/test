import cv2
import numpy as np
import math
from collections import deque

# Коэффициенты скорости в зависимости от цвета
speed_map = {
    (30, 100,  33): 0.5,  # Зеленый
    (255, 255, 255): 1.0,  # Белый
    (243, 235, 0): 0.0  # Голубой
}

# Функция для получения скорости в зависимости от цвета пикселя
def get_speed_from_color(color):
    color_tuple = tuple(color)  # преобразуем цвет в кортеж
    return speed_map.get(color_tuple, 1.0)  # если цвет не найден, использовать скорость 1.0

# Функция для вычисления Евклидова расстояния
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def make_radius_mask(path, radius_in_pixel):
    # Параметры движения
    total_time = radius_in_pixel - 21  # общее время движения (в секундах)
    initial_speed = 1  # начальная скорость
    cell_size = 1  # шаг по пикселям
    dt = 8  # шаг времени в секундах

    # Радиус для построения круга
    radius = 1000  # Радиус круга
    # Загрузка изображения
    image = cv2.imread(path)

    # Определение центра изображения
    h, w, _ = image.shape
    center_x, center_y = w // 2, h // 2

    # Маска для отслеживания пути
    mask = np.zeros((h, w), dtype=np.uint8)

    # Очередь для BFS-алгоритма (ширина первого поиска)
    queue = deque([(center_x, center_y, 0)])  # (x, y, время)
    mask[center_y, center_x] = 255  # Центр — начальная точка

    # Соседи по 8 направлениям (включая диагонали)
    directions =  [(8, 0), (0, 8), (0, -8), (-8, 0), 
    (-7, 4), (-6, 5), (-5,6), (-4, 7), 
    (7, 4), (6, 5), (5,6), (4, 7), 
    (-7, -4), (-6, -5), (-5,-6), (-4, -7), 
    (7, -4), (6, -5), (5,-6), (4, -7)]
    # [(0, -1), (-1, 0),(0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Пока есть элементы в очереди
    while queue:
        x, y, current_time = queue.popleft()

        # Проверяем, можем ли мы продолжать движение (не превысили время)
        if current_time >= total_time:
            continue

        # Проверяем расстояние от центра, если оно больше радиуса, прекращаем
        distance = euclidean_distance(x, y, center_x, center_y)
        if distance > radius:
            continue

        # Проходим по всем соседним пикселям (включая диагонали)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            # Проверяем границы изображения
            if 0 <= nx < w and 0 <= ny < h:
                # Если этот пиксель ещё не был посещён и находится в пределах круга
                if mask[ny, nx] == 0 and euclidean_distance(nx, ny, center_x, center_y) <= radius:
                    # Определяем цвет и соответствующую скорость
                    color = image[ny, nx]
                    speed = initial_speed * get_speed_from_color(color)

                    # Если скорость больше 0 (пиксель проходим), продолжаем движение
                    if speed > 0:
                        # Если движение по диагонали, увеличиваем время в 1.414 (sqrt(2))
                        if dx != 0 and dy != 0:
                            new_time = current_time + dt / speed 
                        else:
                            new_time = current_time + (dt / (speed)) #* math.sqrt(2)))

                        # Если хватает времени, добавляем пиксель в маску и очередь
                        if new_time < total_time:
                            mask[ny, nx] = 255
                            queue.append((nx, ny, new_time))
    # Сохранение маски как изображения
    cv2.imwrite(path, mask)
    return mask, (center_x, center_y)

# make_radius_mask("masked_map.jpg", 300)