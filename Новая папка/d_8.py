import cv2
import numpy as np

# Функция для вычисления расстояния между двумя точками
def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Функция для вычисления угла между двумя векторами
def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # Ограничение для числовой стабильности
    return np.degrees(angle)

def convert_to_degrees(vectors):
    angles = []
    for vector in vectors:
        angle = np.degrees(np.arctan2(vector[1], vector[0]))
        if angle < 0:
            angle += 360
        angles.append(angle)
    return angles

def is_point_in_arc(point, center, radius, start_angle, end_angle):
    vector_point = np.array(point) - np.array(center)
    distance_to_center = np.linalg.norm(vector_point)
    
    if distance_to_center > radius:
        return False

    angle_point = np.degrees(np.arctan2(vector_point[1], vector_point[0]))
    if angle_point < 0:
        angle_point += 360

    if start_angle <= end_angle:
        return start_angle <= angle_point <= end_angle
    else:
        return angle_point >= start_angle or angle_point <= end_angle

def is_point_in_arc(point, center, radius, start_angle, end_angle):
    vector_point = np.array(point) - np.array(center)
    distance_to_center = np.linalg.norm(vector_point)
    
    if distance_to_center > radius:
        print(f"Точка {point} находится за пределами радиуса.")
        return False

    angle_point = np.degrees(np.arctan2(vector_point[1], vector_point[0]))
    if angle_point < 0:
        angle_point += 360

    print(f"Точка {point} имеет угол {angle_point:.2f} градусов.")

    # Проверка, если сектор пересекает 0 градусов
    if start_angle > end_angle:
        in_arc = angle_point >= start_angle or angle_point <= end_angle
        print(f"Сектор пересекает 0 градусов. Точка в секторе: {in_arc}.")
        return in_arc
    else:
        in_arc = start_angle <= angle_point <= end_angle
        print(f"Сектор не пересекает 0 градусов. Точка в секторе: {in_arc}.")


def add_arc_to_mask_if_contains_red_point(center, radius, start_angle, end_angle, red_points, mask, width, height):
    contains_red_point = False
    for red_point in red_points:
        if is_point_in_arc(red_point, center, radius, start_angle, end_angle):
            contains_red_point = True
            break
    
    if contains_red_point:
        print(f"Добавляем сектор с центром {center}, радиусом {radius}, углами {start_angle}° до {end_angle}° на маску.")
        # Если красная точка находится в секторе, рисуем сектор на маске
        arc_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(arc_mask, center, (radius, radius), 0, start_angle, end_angle, 255, thickness=-1)
        mask |= arc_mask
    else:
        print(f"Сектор с центром {center}, радиусом {radius}, углами {start_angle}° до {end_angle}° не содержит красных точек.")

      
# Загрузка изображения
# image = cv2.imread('Снимок экрана 2024-09-03 в 19.27.24.png')
# Снимок экрана 2024-08-18 в 20.18.28.png
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

# Создаем маску для объектов внутри круга
object_mask = np.zeros((height, width), dtype=np.uint8)

# Создаем маску для дуги
combined_arc_mask = np.zeros((height, width), dtype=np.uint8)

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
radius_background_mask = cv2.bitwise_and(circle_mask, cv2.bitwise_not(object_mask))

# Наносим фиолетовый цвет на пересечение
rgba_image[radius_background_mask > 0, :3] = (255, 0, 255)  # Фиолетовый цвет по пересечению
rgba_image[radius_background_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

# Нарисовать исходный центральный радиус
contour_circle_mask = cv2.dilate(radius_background_mask, None, iterations=2)  # Толщина контура
contour_circle = cv2.Canny(contour_circle_mask, 100, 200)
rgba_image[contour_circle > 0, :3] = (0, 0, 255)  # Красный цвет по контуру
rgba_image[contour_circle > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

# Нарисовать черную точку в центре красного радиуса
cv2.circle(rgba_image, center, 5, (0, 0, 0), -1)  # Черная точка в центре радиуса

# Маска для фиолетовых линий
purple_line_mask = np.zeros((height, width), dtype=np.uint8)

combined_arc_mask = np.zeros((height, width), dtype=np.uint8)

blue_or_green_points = []

# or_unique_points = []

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
            
            # Нарисовать центр зоны деревьев
            cv2.circle(rgba_image, center_of_mass, 5, (255, 0, 0), -1)  # Синяя точка для центра зоны

            # Преобразование в кортеж и добавление в список
            center_of_mass_tuple = tuple(center_of_mass)
            blue_or_green_points.append(center_of_mass_tuple)

            # Добавляем точку в список
            blue_or_green_points.append(center_of_mass)
            
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
                blue_or_green_points.append(farthest_point)

                # Добавляем точку в список
                blue_or_green_points.append(farthest_point)
                
                # Нарисовать радиус вокруг самой дальней точки
                radius = int(max_distance / 2)
                tree_circle_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(tree_circle_mask, center, radius, 255, thickness=-1)
                
                # Создаем контур радиуса для дерева
                contour_tree_circle_mask = cv2.dilate(tree_circle_mask, None, iterations=2)  # Толщина контура
                contour_tree_circle = cv2.Canny(contour_tree_circle_mask, 100, 200)
                
                # Отмечаем оранжевый контур на альфа-изображении
                rgba_image[contour_tree_circle > 0, :3] = (0, 165, 255)  # Оранжевый цвет по контуру
                rgba_image[contour_tree_circle > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

                # Создаем маску для границы оранжевого радиуса
                tree_circle_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(tree_circle_mask, center, radius, 255, thickness=2)  # Граница радиуса
                contour_tree_circle_mask = cv2.Canny(tree_circle_mask, 100, 200)
                
                # Создаем маску для границы зоны деревьев
                tree_contour_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(tree_contour_mask, [contour], -1, 255, thickness=2)  # Граница зоны деревьев
                
                # Создаем маску для объекта
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

                # Найти пересечение между маской границы оранжевого радиуса и маской границы зоны деревьев
                intersection_mask = cv2.bitwise_and(tree_contour_mask, contour_tree_circle_mask)
                
                # Отметить пересечение фиолетовым цветом
                rgba_image[intersection_mask > 0, :3] = (255, 0, 255)  # Фиолетовый цвет по пересечению
                rgba_image[intersection_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

                # Найти контуры пересечения и нарисовать фиолетовые точки
                contours_intersection, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_distance = 5  # Минимальное расстояние между точками
                unique_points = []

                # Рисуем линии от центра до фиолетовой точки и продолжения линии до красного радиуса
                for contour_inter in contours_intersection:
                    for point in contour_inter[:, 0, :]:
                        if all(int(point_distance(point, dp)) > min_distance for dp in unique_points):
                            unique_points.append(point)
                            cv2.circle(rgba_image, tuple(point), 5, (255, 0, 255), -1)  # Фиолетовая точка

                            # Провести линию от черной точки до фиолетовой точки
                            cv2.line(rgba_image, center, tuple(point), (255, 0, 255), 2)  # Линия фиолетового цвета
                            
                            # Добавить линию на маску
                            cv2.line(purple_line_mask, center, tuple(point), 255, 2)  # Линия на маске
                            
                            # Вычислить направление линии и продолжить ее до красного радиуса
                            direction = np.array(point) - np.array(center)
                            norm_direction = direction / np.linalg.norm(direction)
                            extended_point = np.array(center) + norm_direction * central_radius
                            extended_point = tuple(extended_point.astype(int))
                            
                            # Нарисовать продолжение линии до пересечения с красным радиусом
                            cv2.line(rgba_image, tuple(point), extended_point, (255, 0, 255), 2)
                            
                            # Добавить продолжение линии на маску
                            cv2.line(purple_line_mask, tuple(point), extended_point, 255, 2)  # Линия на маске

            # Нарисовать самую ближнюю точку
            if closest_point is not None:
                cv2.circle(rgba_image, tuple(closest_point), 5, (255, 255, 0), -1)  # Голубая точка

                # # Преобразование в кортеж и добавление в список
                # closest_point_tuple = tuple(closest_point)
                # blue_or_green_points.append(closest_point_tuple)

                # # Добавляем точку в список
                # blue_or_green_points.append(closest_point_tuple)
            
            # Проверка, выходит ли объект на 50% и более за красный радиус
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
            overlap_with_red_radius = cv2.bitwise_and(mask, circle_mask)
            area_within_red_radius = cv2.countNonZero(overlap_with_red_radius)
            area_object = cv2.countNonZero(mask)
            
            if area_within_red_radius / area_object < 0.5:
                # Если объект хотя бы на 50% выходит за пределы красного радиуса
                if center_of_mass != (0, 0) and closest_point is not None:
                    # Вычисляем середину между центром зоны деревьев и самой близкой точкой
                    midpoint = ((center_of_mass[0] + closest_point[0]) // 2, (center_of_mass[1] + closest_point[1]) // 2)

                    # Нарисовать розовый радиус
                    pink_radius = int(np.sqrt((farthest_point[0] - center[0]) ** 2 + (farthest_point[1] - center[1]) ** 2) / 2)
                    pink_circle_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.circle(pink_circle_mask, center, pink_radius, 255, thickness=-1)
                    
                    # Создаем контур радиуса для дерева
                    contour_pink_circle_mask = cv2.dilate(pink_circle_mask, None, iterations=2)  # Толщина контура
                    contour_pink_circle = cv2.Canny(contour_pink_circle_mask, 100, 200)
                    
                    # Отмечаем розовый контур на альфа-изображении
                    rgba_image[contour_pink_circle > 0, :3] = (255, 105, 180)  # Розовый цвет по контуру
                    rgba_image[contour_pink_circle > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

        # Инициализация переменной arc_mask
        arc_mask = np.zeros((height, width), dtype=np.uint8)

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

        for i in range(len(unique_orange_points)):
            for j in range(i + 1, len(unique_orange_points)):
                point_A = unique_orange_points[i]
                point_B = unique_orange_points[j]

                vector_A = np.array(point_A) - np.array(center)
                vector_B = np.array(point_B) - np.array(center)

                angles = convert_to_degrees([vector_A, vector_B])
                start_angle, end_angle = angles[0], angles[1]

                print(f"Точки: {point_A}, {point_B}")
                print(f"Углы: {start_angle}°, {end_angle}°")

                if end_angle < start_angle:
                    end_angle += 360

                if (end_angle - start_angle) or (start_angle - end_angle) > 180:
                    start_angle, end_angle = end_angle, start_angle + 360

                start_angle %= 360
                end_angle %= 360

                # Обработка случая, когда end_angle выходит за пределы 360 градусов
                if end_angle >= 360:
                    end_angle -= 360


                print(f"Углы после коррекции: {start_angle}°, {end_angle}°")

                arc_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.ellipse(arc_mask, center, (central_radius, central_radius), 0, start_angle, end_angle, 255, thickness=-1)

                if is_point_in_arc(midpoint, center, central_radius, start_angle, end_angle):
                    combined_arc_mask |= arc_mask
                    print(f"Добавляем сектор в маску.")
                else:
                    print(f"Сектор не содержит красной точки.")

                cv2.imshow('Arc Mask', arc_mask)
                # cv2.waitKey(0)
        # Отладочная визуализация
        # cv2.imshow('Arc Mask', arc_mask)
        # cv2.waitKey(900)

        # Основная логика работы с секторами
        # for i in range(len(unique_orange_points)):
        #     for j in range(i + 1, len(unique_orange_points)):
        #         point_A = unique_orange_points[i]
        #         point_B = unique_orange_points[j]

        #         vector_A = np.array(point_A) - np.array(center)
        #         vector_B = np.array(point_B) - np.array(center)

        #         angles = convert_to_degrees([vector_A, vector_B])
        #         start_angle, end_angle = angles[0], angles[1]

        #         if end_angle < start_angle:
        #             end_angle += 360

        #         if end_angle - start_angle > 180:
        #             start_angle, end_angle = end_angle, start_angle

        #         # # Преобразование всех точек в кортежи
        #         # blue_or_green_points = [tuple(point) if isinstance(point, np.ndarray) else point for point in blue_or_green_points]
        #         # unique_blue_or_green_points = list(set(blue_or_green_points))

        #         # arc_has_required_point = False
        #         # for point in unique_blue_or_green_points:
        #         #     if is_point_in_arc(point, center, central_radius, start_angle, end_angle):
        #         #         arc_has_required_point = True
        #         #         break

        #         # if arc_has_required_point:
        #             arc_mask = np.zeros((height, width), dtype=np.uint8)
        #             cv2.ellipse(arc_mask, center, (central_radius, central_radius), 0, start_angle, end_angle, 255, thickness=-1)

        #             # Наложение сектора на маску фиолетовых линий
        #             intersect_mask = cv2.bitwise_and(arc_mask, purple_line_mask)

        #             # Проверяем, есть ли пересечение сектора с фиолетовыми линиями
        #             if np.any(intersect_mask):
        #                 combined_arc_mask |= arc_mask

        #             cv2.imshow('Combined Arc Mask', arc_mask)
        #             cv2.waitKey(900)

# Отображаем объединенную маску дуги
# cv2.imshow('Combined Arc Mask', tree_contour_mask)
# cv2.waitKey(0)

# Показать и сохранить результат
cv2.imshow('Processed Image', rgba_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
