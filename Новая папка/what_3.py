import cv2
import numpy as np

# Функция для вычисления расстояния между двумя точками
def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Функция для вычисления угла между двумя векторами
# def angle_between_vectors(v1, v2):
#     dot_product = np.dot(v1, v2)
#     norm_v1 = np.linalg.norm(v1)
#     norm_v2 = np.linalg.norm(v2)
#     cos_angle = dot_product / (norm_v1 * norm_v2)
#     angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Ограничение для числовой стабильности
#     return np.degrees(angle)

# def convert_to_degrees(vectors):
#     angles = []
#     for vector in vectors:
#         angle = np.degrees(np.arctan2(vector[1], vector[0]))
#         if angle < 0:
#             angle += 360
#         angles.append(angle)
#     return angles

# def is_point_in_arc(point, center, radius, start_angle, end_angle):
#     vector_point = np.array(point) - np.array(center)
#     distance_to_center = np.linalg.norm(vector_point)
    
#     if distance_to_center > radius:
#         return False

#     angle_point = np.degrees(np.arctan2(vector_point[1], vector_point[0]))
#     if angle_point < 0:
#         angle_point += 360

#     if start_angle <= end_angle:
#         return start_angle <= angle_point <= end_angle
#     else:
#         return angle_point >= start_angle or angle_point <= end_angle

# def is_point_in_arc(point, center, radius, start_angle, end_angle):
#     vector_point = np.array(point) - np.array(center)
#     distance_to_center = np.linalg.norm(vector_point)
    
#     if distance_to_center > radius:
#         print(f"Точка {point} находится за пределами радиуса.")
#         return False

#     angle_point = np.degrees(np.arctan2(vector_point[1], vector_point[0]))
#     if angle_point < 0:
#         angle_point += 360

#     print(f"Точка {point} имеет угол {angle_point:.2f} градусов.")

#     # Проверка, если сектор пересекает 0 градусов
#     if start_angle > end_angle:
#         in_arc = angle_point >= start_angle or angle_point <= end_angle
#         print(f"Сектор пересекает 0 градусов. Точка в секторе: {in_arc}.")
#         return in_arc
#     else:
#         in_arc = start_angle <= angle_point <= end_angle
#         print(f"Сектор не пересекает 0 градусов. Точка в секторе: {in_arc}.")


# def add_arc_to_mask_if_contains_red_point(center, radius, start_angle, end_angle, red_points, mask, width, height):
#     contains_red_point = False
#     for red_point in red_points:
#         if is_point_in_arc(red_point, center, radius, start_angle, end_angle):
#             contains_red_point = True
#             break
    
#     if contains_red_point:
#         print(f"Добавляем сектор с центром {center}, радиусом {radius}, углами {start_angle}° до {end_angle}° на маску.")
#         # Если красная точка находится в секторе, рисуем сектор на маске
#         arc_mask = np.zeros((height, width), dtype=np.uint8)
#         cv2.ellipse(arc_mask, center, (radius, radius), 0, start_angle, end_angle, 255, thickness=-1)
#         mask |= arc_mask
#     else:
#         print(f"Сектор с центром {center}, радиусом {radius}, углами {start_angle}° до {end_angle}° не содержит красных точек.")


def line_circle_intersection(line_point1, line_point2, circle_center, circle_radius):
    # Вектор линии
    line_vector = np.array(line_point2) - np.array(line_point1)
    a = np.dot(line_vector, line_vector)
    b = 2 * np.dot(line_vector, np.array(line_point1) - np.array(circle_center))
    c = np.dot(np.array(line_point1) - np.array(circle_center), np.array(line_point1) - np.array(circle_center)) - circle_radius**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        return []
    
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)
    
    intersection1 = line_point1 + t1 * line_vector
    intersection2 = line_point1 + t2 * line_vector
    
    return [tuple(intersection1.astype(int)), tuple(intersection2.astype(int))]

image = cv2.imread('/Users/katerina/Desktop/TestMain/Снимок экрана 2024-09-03 в 19.27.24.png')

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
red_circle_mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(red_circle_mask, center, central_radius, 255, thickness=-1)

contour_red_circle_mask = cv2.dilate(red_circle_mask, None, iterations=2)  # Толщина контура
contour_red_circle_mask = cv2.Canny(contour_red_circle_mask, 100, 200)

# Создаем маску для объектов внутри круга
object_mask = np.zeros((height, width), dtype=np.uint8)

# Обработка объектов и создание масок
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
            if distance <= central_radius:
                cv2.drawContours(object_mask, [contour], -1, 255, thickness=-1)
                break

# Маска для зоны радиуса без объектов
radius_background_mask = cv2.bitwise_and(red_circle_mask, cv2.bitwise_not(object_mask))

# Наносим фиолетовый цвет на пересечение
rgba_image[radius_background_mask > 0, :3] = (255, 0, 255)  # Фиолетовый цвет по пересечению
rgba_image[radius_background_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

# Нарисовать исходный центральный радиус
contour_radius_background_mask = cv2.dilate(radius_background_mask, None, iterations=2)  # Толщина контура
contour_radius_background_mask = cv2.Canny(contour_radius_background_mask, 100, 200)

rgba_image[contour_radius_background_mask > 0, :3] = (0, 0, 255)  # Красный цвет по контуру
rgba_image[contour_radius_background_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

# Нарисовать черную точку в центре красного радиуса
cv2.circle(rgba_image, center, 5, (0, 0, 0), -1)  # Черная точка в центре радиуса

# Маска для фиолетовых линий
purple_line_mask = np.zeros((height, width), dtype=np.uint8)
green_line_mask = np.zeros((height, width), dtype=np.uint8)

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

            # Проверка, выходит ли объект на 50% и более за красный радиус
            mask_tree = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask_tree, [contour], -1, 255, thickness=-1)

            # Найти центр масс контура (центр зоны деревьев)
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                center_of_mass = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            else:
                center_of_mass = (0, 0)
            
            # Нарисовать центр зоны деревьев
            cv2.circle(rgba_image, center_of_mass, 5, (255, 0, 0), -1)  # Синяя точка для центра зоны

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
                cv2.circle(rgba_image, tuple(farthest_point), 5, (0, 255, 0), -1)  # Зеленая точка

                # Преобразование в кортеж и добавление в список
                farthest_point_tuple = tuple(farthest_point)
                
                # Нарисовать радиус вокруг самой дальней точки
                orange_radius = int(max_distance / 2)
                orange_circle_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(orange_circle_mask, center, orange_radius, 255, thickness=-1)

                # Создаем контур радиуса для дерева
                contour_orange_circle_mask = cv2.dilate(orange_circle_mask, None, iterations=2)  # Толщина контура
                contour_orange_circle_mask = cv2.Canny(contour_orange_circle_mask, 100, 200)
                
                # Отмечаем оранжевый контур на альфа-изображении
                rgba_image[contour_orange_circle_mask > 0, :3] = (0, 165, 255)  # Оранжевый цвет по контуру
                rgba_image[contour_orange_circle_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)
                
                # Создаем маску для границы зоны деревьев
                tree_contour_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(tree_contour_mask, [contour], -1, 255, thickness=2)  # Граница зоны деревьев

                # Найти пересечение между маской границы оранжевого/фиолетового радиуса и маской границы зоны деревьев
                tree_intersection_mask = cv2.bitwise_and(tree_contour_mask, contour_orange_circle_mask)
                
                # Найти контуры пересечения и нарисовать фиолетовые точки
                contours_intersection, _ = cv2.findContours(tree_intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_distance = 5  # Минимальное расстояние между точками
                unique_points = []
                
                # Рисуем линии от центра до фиолетовой точки и продолжения линии до красного радиуса
                for contour_inter in contours_intersection:
                    for point in contour_inter[:, 0, :]:
                        if all(int(point_distance(point, dp)) > min_distance for dp in unique_points):
                            unique_points.append(point)
                            # cv2.circle(rgba_image, tuple(point), 5, (255, 0, 255), -1)  # Фиолетовая точка

                            # Провести линию от черной точки до фиолетовой точки
                            # cv2.line(rgba_image, center, tuple(point), (255, 0, 255), 2)  # Линия фиолетового цвета

                            # Добавить фиолетовую линию на маску
                            cv2.line(purple_line_mask, center, tuple(point), 255, 2)  # Линия на маске

                            # Нарисовать черную линию от синей точки до фиолетовой точки
                            # cv2.line(rgba_image, center_of_mass, tuple(point), (0, 0, 0), 2)  # Черная линия

                            # Вычисление перпендикулярного вектора
                            direction = np.array(point) - np.array(center_of_mass)
                            perp_direction = np.array([-direction[1], direction[0]])
                            perp_direction = perp_direction / np.linalg.norm(perp_direction)  # Нормализация

                            # Определение точек хорды
                            half_chord_length = 30  # Задаем длину половины хорды (можно изменить при необходимости)
                            chord_point1 = np.array(point) + perp_direction * half_chord_length
                            chord_point2 = np.array(point) - perp_direction * half_chord_length
                            chord_point1 = tuple(chord_point1.astype(int))
                            chord_point2 = tuple(chord_point2.astype(int))

                            # Продлить хорды до пересечения с красным радиусом
                            circle_center = center
                            circle_radius = central_radius

                            # Найти точки пересечения
                            intersection_points = line_circle_intersection(chord_point1, chord_point2, circle_center, circle_radius)

                            if len(intersection_points) == 2:
                                # Находим ближайшую к фиолетовой точке точку пересечения
                                distances = [np.linalg.norm(np.array(point) - np.array(ip)) for ip in intersection_points]
                                nearest_intersection_index = np.argmin(distances)
                                nearest_intersection_point = intersection_points[nearest_intersection_index]

                                # Рисуем линию от фиолетовой точки до ближайшей точки пересечения
                                cv2.line(rgba_image, tuple(point), tuple(nearest_intersection_point), (0, 255, 0), 2)  # Зеленая линия

                                # Добавляем зелёную линию на маску
                                cv2.line(green_line_mask, tuple(point), tuple(nearest_intersection_point), 255, 2)  # Линия на маске

                                # Рисование точек пересечения с красным радиусом
                                # cv2.circle(rgba_image, intersection_points[0], 5, (0, 200, 0), -1)  # Темно-зеленая точка
                                # cv2.circle(rgba_image, intersection_points[1], 5, (0, 200, 0), -1)  # Темно-зеленая точка


            # Нарисовать самую ближнюю точку
            if closest_point is not None:
                cv2.circle(rgba_image, tuple(closest_point), 5, (255, 255, 0), -1)  # Голубая точка

            overlap_with_red_radius = cv2.bitwise_and(mask_tree, red_circle_mask)
            area_within_red_radius = cv2.countNonZero(overlap_with_red_radius)
            area_object = cv2.countNonZero(mask_tree)

            # Отображаем объединенную маску дуги
            # cv2.imshow('Combined Arc Mask', overlap_with_red_radius)
            # cv2.waitKey(0)

            # aaa = cv2.bitwise_and(mask_tree, contour_red_circle_mask)


            # aaa = cv2.bitwise_and(contour_radius_background_mask, contour_red_circle_mask)
            # bbb = cv2.bitwise_or(aaa, green_line_mask)
            # ccc = cv2.bitwise_or(bbb, contour_radius_background_mask)
            # cv2.imshow('Combined Arc Mask', ccc)

            # cv2.waitKey(0)

            # q = cv2.bitwise_or(contour_orange_circle_mask, green_line_mask)

            # cv2.imshow('Combined Arc Mask', q)

            # cv2.waitKey(0)

        
            if area_within_red_radius / area_object < 0.5:
                # Если объект хотя бы на 50% выходит за пределы красного радиуса
                if center_of_mass != (0, 0) and closest_point is not None:
                    # Вычисляем середину между центром зоны деревьев и самой близкой точкой
                    midpoint = ((center_of_mass[0] + closest_point[0]) // 2, (center_of_mass[1] + closest_point[1]) // 2)

                    # # Нарисовать розовый радиус
                    pink_radius = int(np.sqrt((farthest_point[0] - center[0]) ** 2 + (farthest_point[1] - center[1]) ** 2) / 2)
                    pink_circle_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.circle(pink_circle_mask, center, pink_radius, 255, thickness=-1)
                    
                    # Создаем контур радиуса для дерева
                    contour_pink_circle_mask = cv2.dilate(pink_circle_mask, None, iterations=2)  # Толщина контура
                    contour_pink_circle_mask = cv2.Canny(contour_pink_circle_mask, 100, 200)
                    
                    # Отмечаем розовый контур на альфа-изображении
                    rgba_image[contour_pink_circle_mask > 0, :3] = (255, 105, 180)  # Розовый цвет по контуру
                    rgba_image[contour_pink_circle_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

        # Обработка пересечений фиолетовых линий с красным радиусом
        contours_intersection_orange, _ = cv2.findContours(purple_line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_distance = 5  # Минимальное расстояние между точками
        orange_points = []
        unique_orange_points = []

        # Вычисляем середину между двумя точками
        midpoint = ((center_of_mass[0] + closest_point[0]) // 2, (center_of_mass[1] + closest_point[1]) // 2)

        # Рисуем красную точку между синей и голубой точками
        cv2.circle(rgba_image, midpoint, 5, (0, 0, 255, 255), -1)  # Красная точка

        # Проходим по каждому контуру пересечения
        for contour_inter in contours_intersection_orange:
            for point in contour_inter[:, 0, :]:
                if all(int(point_distance(point, dp)) > min_distance for dp in orange_points):
                    orange_points.append(point)

                    # Проверка на пересечение с красным радиусом
                    distance_to_center = point_distance(point, center)
                    if abs(distance_to_center - central_radius) <= 5:  # Увеличено значение для более точного попадания
                        cv2.circle(rgba_image, tuple(point), 5, (0, 165, 255), -1)  # Оранжевая точка
                        unique_orange_points.append(point)

tree_intersection_mask = cv2.bitwise_or(contour_radius_background_mask, green_line_mask)

# Отображаем объединенную маску дуги
# cv2.imshow('Combined Arc Mask', contour_red_circle_mask)
# cv2.waitKey(0)

# aaa = cv2.bitwise_and(mask_tree, contour_red_circle_mask)
# cv2.imshow('Combined Arc Mask', aaa)

# aaa = cv2.bitwise_and(mask_tree, contour_red_circle_mask)
# cv2.imshow('Combined Arc Mask', aaa)

# cv2.imshow('Combined Arc Mask', q)


# Показать и сохранить результат
cv2.imshow('Processed Image', rgba_image)
cv2.waitKey(0)
cv2.destroyAllWindows()