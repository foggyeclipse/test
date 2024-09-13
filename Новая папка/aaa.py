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

# Нахождение центра изображения и задание радиуса
height, width = hsv_image.shape[:2]
center = (width // 2, height // 2)
radius = width // 4

# Создание маски для круга
circle_mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(circle_mask, center, radius, 255, thickness=-1)

# Создание маски для объектов внутри круга
object_mask = np.zeros((height, width), dtype=np.uint8)

# Обработка объектов
for obj_class, (lower, upper) in color_ranges.items():
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    # Создаем маску для текущего класса
    class_mask = cv2.inRange(hsv_image, lower, upper)
    
    # Находим контуры объектов
    contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Проверяем, находится ли хотя бы одна точка контура внутри круга
        for point in contour[:, 0, :]:
            distance = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
            if distance <= radius:
                cv2.drawContours(object_mask, [contour], -1, 255, thickness=-1)
                break

# Создаем маску для зоны радиуса без объектов
radius_background_mask = cv2.bitwise_and(circle_mask, cv2.bitwise_not(object_mask))

# Создаем изображение с альфа-каналом
rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
rgba_image[..., :3] = image  # Копируем исходное изображение в RGB каналы
rgba_image[..., 3] = 0  # Начально делаем альфа-канал полностью прозрачным

# Нарисовать красный контур на альфа-изображении
contour_circle_mask = cv2.dilate(radius_background_mask, None, iterations=2)  # Увеличиваем толщину контура
contour_circle = cv2.Canny(contour_circle_mask, 100, 200)
rgba_image[contour_circle > 0, :3] = (0, 0, 255)  # Красный цвет по контуру
rgba_image[contour_circle > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

# Сохраняем результат
cv2.imwrite('result_with_thicker_circle.png', rgba_image)

# Показать изображение
cv2.imshow('Processed Image', rgba_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
