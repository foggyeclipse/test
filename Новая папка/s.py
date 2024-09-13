import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('Снимок экрана 2024-08-18 в 20.18.28.png')

# Определяем цветовые диапазоны для различных классов объектов
color_ranges = {
    'tree': ([34, 100, 30], [90, 255, 150]),   # Зеленый для деревьев
    'lake': ([90, 100, 100], [140, 255, 255]), # Голубой для озер
    'road': ([100, 100, 100], [180, 180, 180]) # Серый для дорог
}

# Конвертация изображения в HSV цветовую модель
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Нахождение центра изображения и задание радиуса для центрального круга
height, width = hsv_image.shape[:2]
center = (width // 2, height // 2)
central_radius = width // 4

# Создаем изображение с альфа-каналом
rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
rgba_image[..., :3] = image  # Копируем исходное изображение в RGB каналы
rgba_image[..., 3] = 0  # Начально делаем альфа-канал полностью прозрачным

# Создаем маску для центрального круга
circle_mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(circle_mask, center, central_radius, 255, thickness=-1)

# Обработка объектов и создание масок
for obj_class, (lower, upper) in color_ranges.items():
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    # Создаем маску для текущего класса
    class_mask = cv2.inRange(hsv_image, lower, upper)
    
    # Находим контуры объектов
    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if obj_class == 'tree':
            # Найти центр масс контура (центр зоны деревьев)
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                center_of_mass = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            else:
                center_of_mass = (0, 0)
            
            # Найти самую дальнюю и самую ближнюю точки от центра исходного радиуса
            max_distance = 0
            min_distance = float('inf')
            farthest_point = None
            closest_point = None
            
            for point in contour[:, 0, :]:
                distance = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
                if distance > max_distance:
                    max_distance = distance
                    farthest_point = point
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
            
            # Нарисовать самую дальнюю точку
            if farthest_point is not None:
                # Нарисовать радиус вокруг самой дальней точки
                radius = int(max_distance / 2)
                tree_circle_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(tree_circle_mask, center, radius, 255, thickness=-1)
                
                # Создаем маску для границы оранжевого радиуса
                contour_tree_circle_mask = cv2.dilate(tree_circle_mask, None, iterations=2)  # Толщина контура
                cv2.imshow('Processed Image1',tree_circle_mask)
                # Создаем маску для объекта
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
                
                # Найти пересечение между маской объекта и маской радиуса
                intersection_mask = cv2.bitwise_and(mask, contour_tree_circle_mask)
                
                # Наносим фиолетовый цвет на пересечение
                rgba_image[intersection_mask > 0, :3] = (255, 0, 255)  # Фиолетовый цвет по пересечению
                rgba_image[intersection_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)
                
            # Нарисовать самую ближнюю точку
            if closest_point is not None:
                cv2.circle(rgba_image, tuple(closest_point), 5, (255, 255, 0), -1)  # Желтая точка

# Сохраняем результат
cv2.imwrite('result_with_intersections.png', rgba_image)

# Показать изображение
cv2.imshow('Processed Image', rgba_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
