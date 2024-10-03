import cv2
import math
import numpy as np
from collections import deque

speed_map = {
    (30, 100,  33): 0.5,  # Зеленый
    (255, 255, 255): 1.0,  # Белый
    (255, 0, 0): 0.0  # Голубой
}

def get_speed_from_color(color):
    color_tuple = tuple(color)
    return speed_map.get(color_tuple, 1.0)

# Функция для вычисления Евклидова расстояния
def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def make_radius_mask(path, radius_in_pixel):
    total_time = radius_in_pixel - 21
    initial_speed = 1
    dt = 8 
    radius = radius_in_pixel

    image = cv2.imread(path)

    h, w, _ = image.shape
    center_x, center_y = w // 2, h // 2

    mask = np.zeros((h, w), dtype=np.uint8)

    # Очередь для BFS-алгоритма (ширина первого поиска)
    queue = deque([(center_x, center_y, 0)])  # (x, y, время)
    mask[center_y, center_x] = 255

    # Соседи по 8 направлениям (включая диагонали)
    directions =  [(8, 0), (0, 8), (0, -8), (-8, 0), 
    (-7, 4), (-6, 5), (-5,6), (-4, 7), 
    (7, 4), (6, 5), (5,6), (4, 7), 
    (-7, -4), (-6, -5), (-5,-6), (-4, -7), 
    (7, -4), (6, -5), (5,-6), (4, -7)]

    while queue:
        x, y, current_time = queue.popleft()

        if current_time >= total_time:
            continue

        distance = euclidean_distance(x, y, center_x, center_y)
        if distance > radius:
            continue

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < w and 0 <= ny < h:
                if mask[ny, nx] == 0 and euclidean_distance(nx, ny, center_x, center_y) <= radius:
                    color = image[ny, nx]
                    speed = initial_speed * get_speed_from_color(color)

                    if speed > 0:
                        if dx != 0 and dy != 0:
                            new_time = current_time + dt / speed 
                        else:
                            new_time = current_time + dt / speed

                        if new_time < total_time:
                            mask[ny, nx] = 255
                            queue.append((nx, ny, new_time))
    
    cv2.imwrite(path, mask)
    return mask, (center_x, center_y)
