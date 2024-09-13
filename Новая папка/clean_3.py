import math
import cv2
import numpy as np

# Функция для вычисления расстояния между двумя точками
def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

combined_mask_1 = None

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

chords = []
green_line_masks = []
red_line_masks = []
combined_masks_red_green = []
orange_tree_masks = []
pink_tree_masks = []

def process_intersections(contours_intersection, min_distance, center, center_of_mass, purple_line_mask, circle_center, circle_radius):
    global combined_mask_1
    unique_points = []
    chord_ends = []  # Список для хранения концов хорд (начало и конец зелёной линии)
    chord_starts = []
    green_line_mask = np.zeros((height, width), dtype=np.uint8)
    red_line_mask = np.zeros((height, width), dtype=np.uint8)
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
                cv2.line(rgba_image, center_of_mass, tuple(point), (0, 0, 0), 2)  # Черная линия

                # Вычисление перпендикулярного вектора
                direction = np.array(point) - np.array(center_of_mass)
                perp_direction = np.array([-direction[1], direction[0]])
                perp_direction = perp_direction / np.linalg.norm(perp_direction)  # Нормализация

                # Определение точек хорды
                half_chord_length = 30  # Длина половины хорды
                chord_point1 = np.array(point) + perp_direction * half_chord_length
                chord_point2 = np.array(point) - perp_direction * half_chord_length
                chord_point1 = tuple(chord_point1.astype(int))
                chord_point2 = tuple(chord_point2.astype(int))

                # Продлить хорды до пересечения с красным радиусом
                intersection_points = line_circle_intersection(chord_point1, chord_point2, circle_center, circle_radius)

                if len(intersection_points) == 2:
                    # Находим ближайшую к фиолетовой точке точку пересечения
                    distances = [np.linalg.norm(np.array(point) - np.array(ip)) for ip in intersection_points]
                    nearest_intersection_index = np.argmin(distances)
                    nearest_intersection_point = intersection_points[nearest_intersection_index]

                    # Рисуем линию от фиолетовой точки до ближайшей точки пересечения
                    cv2.line(rgba_image, tuple(point), tuple(nearest_intersection_point), (0, 255, 0), 2)  # Зеленая линия

                    # Рисуем зелёную линию от фиолетовой точки до ближайшей точки пересечения
                    cv2.line(green_line_mask, tuple(point), tuple(nearest_intersection_point), 255, 2)  # Линия на маске

                    # Сохраняем конец хорды для последующего рисования красной линии
                    chord_ends.append(tuple(point))
                    chord_starts.append(tuple(nearest_intersection_point))

    # Рисуем красную линию между концами зеленых хорд, если их два
    if len(chord_ends) == 2:
        cv2.line(rgba_image, chord_ends[0], chord_ends[1], (0, 0, 255), 2)  # Красная линия

        # Добавляем красную линию на маску
        cv2.line(red_line_mask, chord_ends[0], chord_ends[1], 255, 2)  # Линия на маске

        # # Создаем маску для дуги красного радиуса
        red_arc_mask = draw_red_arc_mask(contour_red_circle_mask, chord_starts[1], chord_starts[0], circle_center, circle_radius)
        # red_arc_mask = contour_red_circle_mask
        
        # # Наложение дуги на изображение
        # rgba_image[red_arc_mask > 0, :3] = (0, 0, 255)  # Красный цвет для дуги
        # rgba_image[red_arc_mask > 0, 3] = 255  # Установка альфа-канала в 255 (непрозрачный)

        chords.append([chord_ends[0],chord_ends[1]])

        # 1. Объединяем маски зеленых и красных линий
        combined_mask = cv2.bitwise_or(green_line_mask, red_line_mask)

        combined_mask = cv2.bitwise_or(combined_mask, red_arc_mask)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        combined_mask = cv2.drawContours(combined_mask, contours, -1, 255, thickness=cv2.FILLED)

        # # 1. Объединяем маски зеленых и красных линий
        # combined_mask_2 = cv2.bitwise_or(combined_mask, contour_red_circle_mask)

        # # 2. Находим контуры в объединенной маске
        # contours, _ = cv2.findContours(combined_mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # # 3. Создаем пустую маску для заполнения
        # filled_mask = np.zeros_like(combined_mask_2)

        # combined_mask_1 = combined_mask_2
        # # 4. Заполняем внутренние области замкнутых контуров белым цветом
        # cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)

        green_line_masks.append(green_line_mask)
        red_line_masks.append(red_line_mask)
        combined_masks_red_green.append(combined_mask)


def draw_red_arc_mask(red_circle_mask, start_point, end_point, circle_center, circle_radius):
    # Создаем пустую маску для дуги
    arc_mask = np.zeros((height, width), dtype=np.uint8)
    # cv2.line(arc_mask, start_point, end_point, 255, thickness=3)
    
    # Найти угол начала и конца дуги
    start_angle = np.degrees(np.arctan2((start_point[1] - circle_center[1]), (start_point[0] - circle_center[0])))
    end_angle = np.degrees(np.arctan2((end_point[1] - circle_center[1]), (end_point[0] - circle_center[0])))
    print(start_angle,end_angle)
    # Убедитесь, что угол начальной точки меньше угла конечной точки
    if abs(start_angle) > 90:
        start_angle += 360
    # if abs(end_angle) > 90:
    #     end_angle += 360
    # start_angle = start_angle % 360
    # end_angle = end_angle % 360

    # В случае, если end_angle меньше start_angle, добавляем 360 к end_angle
    # if end_angle > 90:
    #     end_angle += 360

    # print(start_angle,end_angle)
    # Нарисовать дугу на маске
    cv2.ellipse(arc_mask, circle_center, (circle_radius, circle_radius), 0, start_angle, end_angle, 255, thickness=3)
    # cv2.imshow('a', arc_mask) 
    # cv2.waitKey(0)
    return arc_mask

def make_pink_radius(distance_to_closest_point):
# distance_to_closest_point = np.sqrt((closest_point[0] - center[0]) ** 2 + (closest_point[1] - center[1]) ** 2)
    blue_radius = abs(central_radius) - abs(distance_to_closest_point)
    print(distance_to_closest_point)
    # cv2.line(rgba_image, center, closest_point, 255, thickness=2)
    # Вычисляем розовый радиус как половину разности между красным радиусом и расстоянием до голубой точки
    pink_radius = int(distance_to_closest_point + (blue_radius / 2))
    print(pink_radius)
    # pink_radius = blue_radius / 2

    # Создание маски розового круга
    pink_circle_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(pink_circle_mask, center, pink_radius, 255, thickness=-1)
                
    # Создаем контур радиуса для дерева
    contour_pink_circle_mask = cv2.dilate(pink_circle_mask, None, iterations=2)  # Толщина контура
    contour_pink_circle_mask = cv2.Canny(contour_pink_circle_mask, 100, 200)
    
    # Отмечаем розовый контур на альфа-изображении
    rgba_image[contour_pink_circle_mask > 0, :3] = (255, 105, 180)  # Розовый цвет по контуру
    rgba_image[contour_pink_circle_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

    # Найти пересечение между маской границы оранжевого/фиолетового радиуса и маской границы зоны деревьев
    tree_intersection_mask = cv2.bitwise_and(tree_contour_mask, contour_pink_circle_mask)
    intersection_mask = cv2.bitwise_and(mask_tree, pink_circle_mask)
    pink_tree_masks.append(intersection_mask)

    # # Найти контуры пересечения и нарисовать фиолетовые точки
    contours_intersection, _ = cv2.findContours(tree_intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_distance = 5  # Минимальное расстояние между точками

    # Вызов функции для обработки пересечений
    process_intersections(contours_intersection, min_distance, center, center_of_mass, purple_line_mask, center, central_radius)
               


image = cv2.imread('Снимок экрана 2024-09-10 в 23.31.51.png')
# image = cv2.imread('test_clouds.png')

# Определяем цветовые диапазоны для различных классов объектов
color_ranges = {
    # 'tree': ([40, 40, 40], [80, 255, 255]),
    # 'lake': ([90, 50, 50], [130, 255, 255])
    # lower_green = np.array([40, 40, 40]) 
    # upper_green = np.array([80, 255, 255]) 
    # lower_yellow = np.array([20, 50, 50]) 
    # upper_yellow = np.array([30, 255, 255]) 
    # lower_gray = np.array([0, 0, 50]) 
    # upper_gray = np.array([180, 50, 200]) 
    # lower_blue = np.array([90, 50, 50]) 
    # upper_blue = np.array([130, 255, 255])
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
print(central_radius)

# Создаем изображение с альфа-каналом
rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
rgba_image[..., :3] = image  # Копируем исходное изображение в RGB каналы
rgba_image[..., 3] = 0  # Начально делаем альфа-канал полностью прозрачным

# Создаем маску для центрального круга
red_circle_mask = np.zeros((height, width), dtype=np.uint8)
cv2.circle(red_circle_mask, center, central_radius, 255, thickness=-1)

contour_red_circle_mask = cv2.dilate(red_circle_mask, None, iterations=3)  # Толщина контура
contour_red_circle_mask = cv2.Canny(contour_red_circle_mask, 100, 200)

# Создаем маску для объектов внутри круга
object_mask = np.zeros((height, width), dtype=np.uint8)
water_mask = np.zeros((height, width), dtype=np.uint8)

# Обработка объектов и создание масок
for obj_class, (lower, upper) in color_ranges.items():
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    
    # Создаем маску для текущего класса
    class_mask = cv2.inRange(hsv_image, lower, upper)
    
    # Находим контуры объектов
    # contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.dilate(class_mask, None, iterations=2)  # Толщина контура
    # contours = cv2.Canny(contours, 100, 200)
    object_mask = cv2.add(object_mask, contours)
    if obj_class == 'lake':
        water_mask = cv2.add(water_mask, contours)
    # cv2.imshow('a', contours)
    # cv2.waitKey(0)
    # cv2.drawContours(object_mask, [contours], -1, 255, thickness=-1)
    # if obj_class == 'lake':
    #     cv2.drawContours(water_mask, [contours], -1, 255, thickness=-1)
    # for contour in contours:
    #     # Проверяем, находится ли хотя бы одна точка контура внутри круга
    #     for point in contour[:, 0, :]:
    #         # distance = np.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
    #         # if distance <= central_radius:
    #         cv2.drawContours(object_mask, [contour], -1, 255, thickness=-1)
    #         cv2.imshow('a', object_mask)
    #         cv2.waitKey(0)

    #         if obj_class == 'lake':
    #             cv2.drawContours(water_mask, [contour], -1, 255, thickness=-1)
    #         break

# Маска для зоны радиуса без объектов
# object_mask_1 = cv2.bitwise_and(red_circle_mask, object_mask)
radius_background_mask = cv2.bitwise_and(red_circle_mask, cv2.bitwise_not(object_mask))
cv2.imshow('a', object_mask)
cv2.waitKey(0)
# object_mask_1 = cv2.bitwise_and(red_circle_mask, object_mask)

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
main_purple_line_mask = np.zeros((height, width), dtype=np.uint8)
purple_line_mask = np.zeros((height, width), dtype=np.uint8)


# blue_radius = np.zeros((height, width), dtype=np.uint8)
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
                
                overlap_with_red_radius = cv2.bitwise_and(mask_tree, red_circle_mask)
                area_within_red_radius = cv2.countNonZero(overlap_with_red_radius)
                area_object = cv2.countNonZero(mask_tree)

            # Создаем маску для границы зоны деревьев
            tree_contour_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(tree_contour_mask, [contour], -1, 255, thickness=2)  # Граница зоны деревьев

            # Нарисовать самую ближнюю точку
            if closest_point is not None:
                cv2.circle(rgba_image, tuple(closest_point), 5, (255, 255, 0), -1)  # Голубая точка


            # Вычисляем середину между двумя точками
            midpoint = ((center_of_mass[0] + closest_point[0]) // 2, (center_of_mass[1] + closest_point[1]) // 2)
            if area_within_red_radius / area_object < 0.5:
            # Если объект хотя бы на 50% выходит за пределы красного радиуса
                if center_of_mass != (0, 0) and closest_point is not None:
                # if closest_point is not None:
                #     cv2.circle(rgba_image, tuple(closest_point), 5, (255, 255, 0), -1)  # Голубая точка
                # Вычисляем середину между центром зоны деревьев и самой близкой точкой
                # midpoint = ((center_of_mass[0] + closest_point[0]) // 2, (center_of_mass[1] + closest_point[1]) // 2)
                # Проверяем, находится ли самая ближняя точка внутри красного радиуса
                    distance_to_closest_point = np.sqrt((closest_point[0] - center[0]) ** 2 + (closest_point[1] - center[1]) ** 2)
                    if distance_to_closest_point <= central_radius:
                        make_pink_radius(distance_to_closest_point)
                
                        # # distance_to_closest_point = np.sqrt((closest_point[0] - center[0]) ** 2 + (closest_point[1] - center[1]) ** 2)
                        # blue_radius = abs(central_radius) - abs(distance_to_closest_point)
                        # print(distance_to_closest_point)
                        # # cv2.line(rgba_image, center, closest_point, 255, thickness=2)
                        # # Вычисляем розовый радиус как половину разности между красным радиусом и расстоянием до голубой точки
                        # pink_radius = int(distance_to_closest_point + (blue_radius / 2))
                        # print(pink_radius)
                        # # pink_radius = blue_radius / 2

                        # # Создание маски розового круга
                        # pink_circle_mask = np.zeros((height, width), dtype=np.uint8)
                        # cv2.circle(pink_circle_mask, center, pink_radius, 255, thickness=-1)
                                    
                        # # Создаем контур радиуса для дерева
                        # contour_pink_circle_mask = cv2.dilate(pink_circle_mask, None, iterations=2)  # Толщина контура
                        # contour_pink_circle_mask = cv2.Canny(contour_pink_circle_mask, 100, 200)
                        
                        # # Отмечаем розовый контур на альфа-изображении
                        # rgba_image[contour_pink_circle_mask > 0, :3] = (255, 105, 180)  # Розовый цвет по контуру
                        # rgba_image[contour_pink_circle_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

                        # # Найти пересечение между маской границы оранжевого/фиолетового радиуса и маской границы зоны деревьев
                        # tree_intersection_mask = cv2.bitwise_and(tree_contour_mask, contour_pink_circle_mask)
                        # intersection_mask = cv2.bitwise_and(mask_tree, pink_circle_mask)
                        # pink_tree_masks.append(intersection_mask)

                        # # # Найти контуры пересечения и нарисовать фиолетовые точки
                        # contours_intersection, _ = cv2.findContours(tree_intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # min_distance = 5  # Минимальное расстояние между точками

                        # # Вызов функции для обработки пересечений
                        # process_intersections(contours_intersection, min_distance, center, center_of_mass, purple_line_mask, center, central_radius)
                    else:
                        make_pink_radius(0)
            else:
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

                # Найти пересечение между маской границы оранжевого/фиолетового радиуса и маской границы зоны деревьев
                tree_intersection_mask = cv2.bitwise_and(tree_contour_mask, contour_orange_circle_mask)

                intersection_mask = cv2.bitwise_and(mask_tree, orange_circle_mask)
                orange_tree_masks.append(intersection_mask)
                # Найти контуры пересечения и нарисовать фиолетовые точки
                contours_intersection, _ = cv2.findContours(tree_intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_distance = 5  # Минимальное расстояние между точками
                
                # Вызов функции для обработки пересечений
                process_intersections(contours_intersection, min_distance, center, center_of_mass, purple_line_mask, center, central_radius)
            
            # # Нарисовать самую ближнюю точку
            # if closest_point is not None:
            #     cv2.circle(rgba_image, tuple(closest_point), 5, (255, 255, 0), -1)  # Голубая точка

        # # Вычисляем середину между двумя точками
        # midpoint = ((center_of_mass[0] + closest_point[0]) // 2, (center_of_mass[1] + closest_point[1]) // 2)

        # Рисуем красную точку между синей и голубой точками
        cv2.circle(rgba_image, midpoint, 5, (0, 0, 255, 255), -1)  # Красная точка


# tree_intersection_mask = cv2.bitwise_or(contour_radius_background_mask, green_line_mask)
print(chords)
# cv2.imshow('Combined Arcx Mask', combined_masks_red_green[0])
# cv2.waitKey(0)

# cv2.imshow('Combined Arc Mask', red_circle_mask)
# cv2.waitKey(0)

result = radius_background_mask
object_mask = cv2.bitwise_and(red_circle_mask, object_mask)
for i in combined_masks_red_green:
    result = cv2.subtract(result, i)
    object_mask = cv2.subtract(object_mask, i)
    result = cv2.add(result, object_mask)



# cv2.imshow('Processed Image', object_mask)
# cv2.waitKey(0)

# result = cv2.add(result, object_mask)
result = cv2.subtract(result, water_mask)

for i in orange_tree_masks:
    result = cv2.add(result, i)
print(len(orange_tree_masks))


for i in pink_tree_masks:
    result = cv2.add(result, i)
# Наносим фиолетовый цвет на пересечение
# rgba_image[result > 0, :3] = (255, 0, 255)  # Фиолетовый цвет по пересечению
# rgba_image[result > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)



# Показать и сохранить результат
cv2.imshow('Processed Image', rgba_image)
cv2.waitKey(0)

cv2.imshow('Combined Arc Mask', result)

# cv2.imshow('Combined Arc Mask', red_circle_mask)
cv2.waitKey(0)

# cv2.imshow('Combined Arc Mask', tree_contour_mask)
# cv2.waitKey(0)
cv2.destroyAllWindows()