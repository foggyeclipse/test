import cv2
import numpy as np

# Загрузка изображения
image = cv2.imread('Снимок экрана 2024-08-18 в 20.18.28.png')

# Определяем цветовые диапазоны для различных классов объектов
color_ranges = {
    'tree': ([34, 100, 30], [90, 255, 150]),   # Зеленый для деревьев
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

# Нарисовать исходный центральный радиус (красный)
cv2.circle(rgba_image, center, central_radius, (0, 0, 255), 2)  # Красный радиус

# Обработка объектов "деревьев"
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
            
            # Проверка, все ли точки внутри красного радиуса
            all_points_inside = True
            for point in contour[:, 0, :]:
                distance = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
                if distance > central_radius:
                    all_points_inside = False
                    break
            
            if all_points_inside:
                # Нарисовать оранжевый радиус для дерева
                cv2.circle(rgba_image, center_of_mass, 10, (0, 165, 255), 2)  # Оранжевый радиус
            else:
                # Определить самую ближнюю и самую дальнюю точки контура относительно центра красного радиуса
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
                
                if farthest_point is not None and closest_point is not None:
                    # Найти середину между синей и желтой точками
                    mid_point = ((farthest_point[0] + closest_point[0]) // 2, (farthest_point[1] + closest_point[1]) // 2)
                    
                    # Нарисовать розовый радиус через эту середину
                    radius = int(np.sqrt((mid_point[0] - center[0]) ** 2 + (mid_point[1] - center[1]) ** 2))
                    cv2.circle(rgba_image, center, radius, (255, 105, 180), 2)  # Розовый радиус

# Сохраняем результат
cv2.imwrite('result_with_all_details_and_black_center.png', rgba_image)

# Показать изображение
cv2.imshow('Processed Image', rgba_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
