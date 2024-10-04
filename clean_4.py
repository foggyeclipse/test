import math
import cv2
import numpy as np

# Функция для вычисления расстояния между двумя точками
def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


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
pinks_radius = []

def process_intersections(contours_intersection, min_distance, center, center_of_mass, purple_line_mask, circle_center, circle_radius, height, width, rgba_image, contour_red_circle_mask):
    unique_points = []
    chord_ends = []  # Список для хранения концов хорд (начало и конец зелёной линии)
    chord_starts = []
    green_line_mask = np.zeros((height, width), dtype=np.uint8)
    green_line_mask1 = np.zeros((height, width), dtype=np.uint8)
    red_line_mask = np.zeros((height, width), dtype=np.uint8)
    # if contours_intersection:
    #     pass
    # else:
    #     return None
    for i, contour_inter in enumerate([contours_intersection[0], contours_intersection[-1]]):
        # cv2.drawContours(green_line_mask1, contour_inter, -1, 255, -1)
        # print(contours_intersection[0],"ssss")
        # cv2.circle(green_line_mask1,  np.max(contour_inter[:, 0, :], axis=0), 1, 255, 1)
        # cv2.imshow('a',green_line_mask1)
        # cv2.waitKey(0)
        if(i==0):
            a = np.max(contour_inter[:, 0, :], axis=0)
        else:
            a = np.min(contour_inter[:, 0, :], axis=0)
        for point in [a]:
            # if all(int(point_distance(point, dp)) > min_distance for dp in unique_points):
            #     unique_points.append(point)
                

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
        red_arc_mask = draw_red_arc_mask(contour_red_circle_mask, chord_starts[1], chord_starts[0], circle_center, circle_radius, height, width)
        # red_arc_mask = contour_red_circle_mask
        # cv2.imshow('a', red_arc_mask)
        # cv2.waitKey(0)
        # # Наложение дуги на изображение
        # rgba_image[red_arc_mask > 0, :3] = (0, 0, 255)  # Красный цвет для дуги
        # rgba_image[red_arc_mask > 0, 3] = 255  # Установка альфа-канала в 255 (непрозрачный)

        chords.append([chord_ends[0],chord_ends[1]])

        # 1. Объединяем маски зеленых и красных линий
        combined_mask = cv2.bitwise_or(green_line_mask, red_line_mask)

        combined_mask = cv2.bitwise_or(combined_mask, red_arc_mask)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        combined_mask = cv2.drawContours(combined_mask, contours, -1, 255, thickness=cv2.FILLED)

        green_line_masks.append(green_line_mask)
        red_line_masks.append(red_line_mask)
        combined_masks_red_green.append(combined_mask)
        # cv2.imshow('a', combined_masks_red_green[0])
        # cv2.waitKey(0)

def draw_red_arc_mask(red_circle_mask, start_point, end_point, circle_center, circle_radius, height, width):
    # Создаем пустую маску для дуги
    arc_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(arc_mask, start_point, 1,  255, thickness=3)
    cv2.circle(arc_mask, end_point, 1,  255, thickness=3)
    # Найти угол начала и конца дуги
    start_angle = np.degrees(np.arctan2((start_point[1] - circle_center[1]), (start_point[0] - circle_center[0])))
    end_angle = np.degrees(np.arctan2((end_point[1] - circle_center[1]), (end_point[0] - circle_center[0])))
    # print(start_angle,end_angle)
    # Убедитесь, что угол начальной точки меньше угла конечной точки
    # if start_angle <  end_angle:
    #     start_angle, end_angle = end_angle, start_angle
    if end_angle < 0:
        end_angle += 360
    if start_angle < 0:
        start_angle += 360

    angle_diff = abs(end_angle - start_angle)
    
    # Если разница углов больше 180, нужно рисовать другую дугу
    if angle_diff > 180:
        if start_angle < end_angle:
            start_angle = start_angle + 360  # Увеличиваем угол для замкнутой дуги
        else:
            end_angle = end_angle + 360
    # start_angle = start_angle % 360
    # end_angle = end_angle % 360

    # В случае, если end_angle меньше start_angle, добавляем 360 к end_angle
    # if end_angle > 90:
    #     end_angle += 360

    print(start_angle,end_angle)
    # Нарисовать дугу на маске
    cv2.ellipse(arc_mask, circle_center, (circle_radius, circle_radius), 0, start_angle, end_angle, 255, thickness=2)
    # cv2.imshow('a', arc_mask) 
    # cv2.waitKey(0)
    return arc_mask

def make_pink_radius(distance_to_closest_point, central_radius, height, width, center, rgba_image, tree_contour_mask, mask_tree, center_of_mass, purple_line_mask, contour_red_circle_mask,  need_intersection=True):
# distance_to_closest_point = np.sqrt((closest_point[0] - center[0]) ** 2 + (closest_point[1] - center[1]) ** 2)
    blue_radius = abs(central_radius) - abs(distance_to_closest_point)
    print(distance_to_closest_point)
    # cv2.line(rgba_image, center, closest_point, 255, thickness=2)
    # Вычисляем розовый радиус как половину разности между красным радиусом и расстоянием до голубой точки
    if need_intersection:
        pink_radius = int(distance_to_closest_point + (blue_radius / 2))
    else:
        pink_radius = int(distance_to_closest_point + (blue_radius * 2))
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
    # rgba_image[contour_pink_circle_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

    # Найти пересечение между маской границы оранжевого/фиолетового радиуса и маской границы зоны деревьев
    tree_intersection_mask = cv2.bitwise_and(tree_contour_mask, contour_pink_circle_mask)
    intersection_mask = cv2.bitwise_and(mask_tree, pink_circle_mask)
    pink_tree_masks.append(intersection_mask)
    pinks_radius.append(pink_circle_mask)

    # Найти контуры пересечения и нарисовать фиолетовые точки
    contours_intersection, _ = cv2.findContours(tree_intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_distance = 6  # Минимальное расстояние между точками

    # Вызов функции для обработки пересечений
    if need_intersection:
        process_intersections(contours_intersection, min_distance, center, center_of_mass, purple_line_mask, center, central_radius, height, width, rgba_image, contour_red_circle_mask)
               


# image = cv2.imread('/Users/katerina/Desktop/TestMain/test_green.png')
# image = cv2.imread('test_clouds.png')
def make_mask_of_radius(path, radius_in_pixel):
    image = cv2.imread(path)
    center_in_field = True
    # chords = []
    # green_line_masks = []
    # red_line_masks = []
    # combined_masks_red_green = []
    # orange_tree_masks = []
    # pink_tree_masks = []
    # pinks_radius = []
    # Определяем цветовые диапазоны для различных классов объектов
    color_ranges = {
        # 'tree': ([40, 40, 40], [80, 255, 255]),
        # 'lake': ([90, 50, 50], [130, 255, 255])
        'tree': ([34, 100, 30], [90, 255, 150]),   # Зеленый для деревьев
        'lake': ([90, 100, 100], [140, 255, 255]), # Голубой для озер
        'road': ([100, 100, 100], [180, 180, 180]) # Серый для дорог
    }

    color_ranges_tree_main = {
        # 'tree': ([40, 40, 40], [80, 255, 255]),
        # 'lake': ([90, 50, 50], [130, 255, 255])
        # 'field': ([0, 0, 200], [180, 55, 255]),   # Зеленый для деревьев
        'field': ([0, 0, 200], [180, 30, 255]),
        'lake': ([90, 100, 100], [140, 255, 255]), # Голубой для озер
        'road': ([100, 100, 100], [180, 180, 180]) # Серый для дорог
    }

    # Конвертация изображения в HSV цветовую модель
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow("a",hsv_image)
    # cv2.waitKey(0)
    # Нахождение центра изображения и задание радиуса для центрального круга
    height, width = hsv_image.shape[:2]
    center = (width // 2, height // 2)
    # print(image[center[1],center[0],:3],'!!!!!!!!!!!!!')
    central_radius = radius_in_pixel
    # central_radius = 290
    print(central_radius)

    # Создаем изображение с альфа-каналом
    rgba_image = np.zeros((height, width, 3), dtype=np.uint8)
    rgba_image[..., :3] = image  # Копируем исходное изображение в RGB каналы
    # rgba_image[..., 3] = 0  # Начально делаем альфа-канал полностью прозрачным


    if np.all(hsv_image[center[1],center[0]] <= color_ranges['tree'][1]) and np.all(hsv_image[center[1],center[0]]  >= color_ranges['tree'][0]):
        print('aaaaaa')
        center_in_field = False
        central_radius = central_radius//2
        # Создаем маску для центрального круга
        red_circle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(red_circle_mask, center, central_radius, 255, thickness=-1)

        contour_red_circle_mask = cv2.dilate(red_circle_mask, None, iterations=3)  # Толщина контура
        contour_red_circle_mask = cv2.Canny(contour_red_circle_mask, 100, 200)

        # Создаем маску для объектов внутри круга
        object_mask = np.zeros((height, width), dtype=np.uint8)
        water_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Обработка объектов и создание масок
        for obj_class, (lower, upper) in color_ranges_tree_main.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            
            # Создаем маску для текущего класса
            class_mask = cv2.inRange(hsv_image, lower, upper)
            
            # Находим контуры объектов
            # contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = cv2.dilate(class_mask, None, iterations=0)  # Толщина контура
            # contours = cv2.Canny(contours, 100, 200)
            object_mask = cv2.add(object_mask, contours)
            if obj_class == 'lake':
                water_mask = cv2.add(water_mask, contours)

        # Маска для зоны радиуса без объектов
        # object_mask_1 = cv2.bitwise_and(red_circle_mask, object_mask)

        radius_background_mask = cv2.bitwise_and(red_circle_mask, cv2.bitwise_not(object_mask))
        # Маска для объектов (зеленых) внутри радиуса
        object_inside_radius_mask = cv2.bitwise_and(red_circle_mask, object_mask)

        # Вычисление площадей
        area_background = cv2.countNonZero(radius_background_mask)
        area_objects = cv2.countNonZero(object_inside_radius_mask)

        # Сравнение площадей и вывод результата
        if area_background > area_objects:
            print("В пределах красного радиуса больше белого фона.")
        else:
            print("В пределах красного радиуса больше зеленых объектов.")
        # cv2.imshow('a', object_mask)
        # cv2.waitKey(0)
        # object_mask_1 = cv2.bitwise_and(red_circle_mask, object_mask)

        # Наносим фиолетовый цвет на пересечение
        rgba_image[radius_background_mask > 0, :3] = (255, 0, 255)  # Фиолетовый цвет по пересечению
        # rgba_image[radius_background_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

        # Нарисовать исходный центральный радиус
        contour_radius_background_mask = cv2.dilate(radius_background_mask, None, iterations=0)  # Толщина контура
        contour_radius_background_mask = cv2.Canny(contour_radius_background_mask, 100, 200)

        rgba_image[contour_radius_background_mask > 0, :3] = (0, 0, 255)  # Красный цвет по контуру
        # rgba_image[contour_radius_background_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

        # Нарисовать черную точку в центре красного радиуса
        cv2.circle(rgba_image, center, 5, (0, 0, 0), -1)  # Черная точка в центре радиуса

        # Маска для фиолетовых линий
        main_purple_line_mask = np.zeros((height, width), dtype=np.uint8)
        purple_line_mask = np.zeros((height, width), dtype=np.uint8)


        # blue_radius = np.zeros((height, width), dtype=np.uint8)
        # Обработка объектов "деревьев"
        for obj_class, (lower, upper) in color_ranges_tree_main.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            
            # Создаем маску для текущего класса
            class_mask = cv2.inRange(hsv_image, lower, upper)
            
            # Находим контуры объектов
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if obj_class == 'field':

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
                    cv2.drawContours(tree_contour_mask, [contour], -1, 255, thickness=3)  # Граница зоны деревьев

                    # Нарисовать самую ближнюю точку
                    if closest_point is not None:
                        cv2.circle(rgba_image, tuple(closest_point), 5, (255, 255, 0), -1)  # Голубая точка


                    # Вычисляем середину между двумя точками
                    midpoint = ((center_of_mass[0] + closest_point[0]) // 2, (center_of_mass[1] + closest_point[1]) // 2)
                    if area_within_red_radius / area_object < 0.5:
                    # Если объект хотя бы на 50% выходит за пределы красного радиуса
                        if center_of_mass != (0, 0) and closest_point is not None:
                            distance_to_closest_point = np.sqrt((closest_point[0] - center[0]) ** 2 + (closest_point[1] - center[1]) ** 2)
                            if distance_to_closest_point <= central_radius:
                                make_pink_radius(distance_to_closest_point, central_radius, height, width, center, rgba_image, tree_contour_mask, mask_tree, center_of_mass, purple_line_mask, contour_red_circle_mask, False)
                            # else:
                            #     make_pink_radius(0, False)
                    else:
                        # Нарисовать радиус вокруг самой дальней точки
                        orange_radius = int(max_distance)
                        orange_circle_mask = np.zeros((height, width), dtype=np.uint8)
                        cv2.circle(orange_circle_mask, center, orange_radius, 255, thickness=-1)

                        # Создаем контур радиуса для дерева
                        contour_orange_circle_mask = cv2.dilate(orange_circle_mask, None, iterations=2)  # Толщина контура
                        contour_orange_circle_mask = cv2.Canny(contour_orange_circle_mask, 100, 200)
                        
                        # Отмечаем оранжевый контур на альфа-изображении
                        rgba_image[contour_orange_circle_mask > 0, :3] = (0, 165, 255)  # Оранжевый цвет по контуру
                        # rgba_image[contour_orange_circle_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

                        # Найти пересечение между маской границы оранжевого/фиолетового радиуса и маской границы зоны деревьев
                        tree_intersection_mask = cv2.bitwise_and(tree_contour_mask, contour_orange_circle_mask)

                        intersection_mask = cv2.bitwise_and(mask_tree, orange_circle_mask)
                        orange_tree_masks.append(intersection_mask)
                        # Найти контуры пересечения и нарисовать фиолетовые точки
                        contours_intersection, _ = cv2.findContours(tree_intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        min_distance = 6  # Минимальное расстояние между точками
                        
                        # Вызов функции для обработки пересечений
                        # process_intersections(contours_intersection, min_distance, center, center_of_mass, purple_line_mask, center, central_radius)
                    
                    # # Нарисовать самую ближнюю точку
                    # if closest_point is not None:
                    #     cv2.circle(rgba_image, tuple(closest_point), 5, (255, 255, 0), -1)  # Голубая точка

                # # Вычисляем середину между двумя точками
                # midpoint = ((center_of_mass[0] + closest_point[0]) // 2, (center_of_mass[1] + closest_point[1]) // 2)

                # Рисуем красную точку между синей и голубой точками
                cv2.circle(rgba_image, midpoint, 5, (0, 0, 255, 255), -1)  # Красная точка
    else:
        print('aaaaaa?')
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

        # Маска для зоны радиуса без объектов
        # object_mask_1 = cv2.bitwise_and(red_circle_mask, object_mask)

        radius_background_mask = cv2.bitwise_and(red_circle_mask, cv2.bitwise_not(object_mask))
        # Маска для объектов (зеленых) внутри радиуса
        object_inside_radius_mask = cv2.bitwise_and(red_circle_mask, object_mask)

        # Вычисление площадей
        area_background = cv2.countNonZero(radius_background_mask)
        area_objects = cv2.countNonZero(object_inside_radius_mask)

        # Сравнение площадей и вывод результата
        if area_background > area_objects:
            print("В пределах красного радиуса больше белого фона.")
        else:
            print("В пределах красного радиуса больше зеленых объектов.")
        # cv2.imshow('a', object_mask)
        # cv2.waitKey(0)
        # object_mask_1 = cv2.bitwise_and(red_circle_mask, object_mask)

        # Наносим фиолетовый цвет на пересечение
        rgba_image[radius_background_mask > 0, :3] = (255, 0, 255)  # Фиолетовый цвет по пересечению
        # rgba_image[radius_background_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

        # Нарисовать исходный центральный радиус
        contour_radius_background_mask = cv2.dilate(radius_background_mask, None, iterations=2)  # Толщина контура
        contour_radius_background_mask = cv2.Canny(contour_radius_background_mask, 100, 200)

        rgba_image[contour_radius_background_mask > 0, :3] = (0, 0, 255)  # Красный цвет по контуру
        # rgba_image[contour_radius_background_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

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
                    cv2.drawContours(tree_contour_mask, [contour], -1, 255, thickness=3)  # Граница зоны деревьев

                    # Нарисовать самую ближнюю точку
                    if closest_point is not None:
                        cv2.circle(rgba_image, tuple(closest_point), 5, (255, 255, 0), -1)  # Голубая точка


                    # Вычисляем середину между двумя точками
                    midpoint = ((center_of_mass[0] + closest_point[0]) // 2, (center_of_mass[1] + closest_point[1]) // 2)
                    if area_within_red_radius / area_object < 0.5:
                    # Если объект хотя бы на 50% выходит за пределы красного радиуса
                        if center_of_mass != (0, 0) and closest_point is not None:
                            distance_to_closest_point = np.sqrt((closest_point[0] - center[0]) ** 2 + (closest_point[1] - center[1]) ** 2)
                            if distance_to_closest_point <= central_radius:
                                make_pink_radius(distance_to_closest_point, central_radius, height, width, center, rgba_image, tree_contour_mask, mask_tree, center_of_mass, purple_line_mask, contour_red_circle_mask)
                            # else:
                            #     make_pink_radius(0)
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
                        # rgba_image[contour_orange_circle_mask > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

                        # Найти пересечение между маской границы оранжевого/фиолетового радиуса и маской границы зоны деревьев
                        tree_intersection_mask = cv2.bitwise_and(tree_contour_mask, contour_orange_circle_mask)

                        intersection_mask = cv2.bitwise_and(mask_tree, orange_circle_mask)
                        orange_tree_masks.append(intersection_mask)
                        # Найти контуры пересечения и нарисовать фиолетовые точки
                        contours_intersection, _ = cv2.findContours(tree_intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        min_distance = 5  # Минимальное расстояние между точками
                        
                        # Вызов функции для обработки пересечений
                        process_intersections(contours_intersection, min_distance, center, center_of_mass, purple_line_mask, center, central_radius, height, width, rgba_image, contour_red_circle_mask)
                    
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
    for n, i in enumerate(combined_masks_red_green):
        # cv2.imshow('a', i)
        # cv2.waitKey(0)
        # cv2.imshow("Processed Image", i)
        # cv2.waitKey(0)
        result = cv2.subtract(result, i)
        object_mask = cv2.subtract(object_mask, i)
        result = cv2.add(result, object_mask)
        if len(pinks_radius) > n:
            j = cv2.bitwise_and(i,pinks_radius[n])
            # cv2.imshow("Processed Image", j)
            # cv2.waitKey(0)
            result = cv2.add(result,j)

    # cv2.imshow("Processed Image", result)
    # cv2.waitKey(0)

    # cv2.imshow('Processed Image', object_mask)
    # cv2.waitKey(0)

    # result = cv2.add(result, object_mask)
    # cv2.imshow("Processed Image", result)
    # # cv2.imshow('Combined Arc Mask', red_circle_mask)
    # cv2.waitKey(0)
    if center_in_field:
        for i in orange_tree_masks:
            result = cv2.add(result, i)
        print(len(orange_tree_masks))
        for i in pink_tree_masks:
            result = cv2.add(result, i)
    else:
        for i in orange_tree_masks:
            result = cv2.add(result, i)
        # cv2.imshow("Processed Image", pink_tree_masks[0])
        # cv2.waitKey(0)
        for i in pink_tree_masks:
            result = cv2.add(result, i)

    result = cv2.subtract(result, water_mask)
    print('Mask has been maked!!!!!!!!!!!!!!')
    cv2.imshow("Processed Image", result)
    # cv2.imshow('Combined Arc Mask', red_circle_mask)
    cv2.waitKey(0)
    return result, center
# Наносим фиолетовый цвет на пересечение
# rgba_image[result > 0, :3] = (255, 0, 255)  # Фиолетовый цвет по пересечению
# rgba_image[result > 0, 3] = 255  # Устанавливаем альфа-канал в 255 (непрозрачный)

# make_mask_of_radius("masked_map_image0.png", 230)
#     # width и height — размер изображения
# cv2.imshow("Processed Image", rgba_image)
# # cv2.imshow("Processed Image", water_mask)
# cv2.waitKey(0)
#     # width и height — размер изображения
# cv2.imshow("Processed Image", result)
# # cv2.imshow('Combined Arc Mask', red_circle_mask)
# cv2.waitKey(0)

# # cv2.imshow('Combined Arc Mask', tree_contour_mask)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()
def pixel_to_latlng(pixel_coords, img_size, map_center, scale_factor, scale_factor_y):
    img_width, img_height = img_size
    latlng_coords = []
    
    for x, y in pixel_coords:
        # Инвертируем Y, потому что система координат карты обратная
        inverted_y = img_height - y
        
        # Преобразуем пиксельные координаты в географические с учётом масштабирования
        lat = map_center[0] + (inverted_y - img_height / 2) * scale_factor_y
        lng = map_center[1] + (x - img_width / 2) * scale_factor
        
        latlng_coords.append([lat, lng])
    
    return latlng_coords
def make_txt_mask_of_radius(p, coords_psr, radius, result, center):
    kernel = np.ones((5, 5), np.uint8)  
    # Применение эрозии перед дилатацией для уменьшения артефактов
    result = cv2.erode(result, kernel, iterations=2)
    result = cv2.dilate(result, kernel, iterations=3)  # Увеличение толщины контура
    result = cv2.Canny(result, 100, 200)
    mask1 = np.zeros((result.shape[0]+2, result.shape[1]+2), np.uint8)
    cv2.floodFill(result, mask1, center, 255)
    # cv2.imshow('A', result)
    # cv2.waitKey(0)
    p = p.split('.')[0]
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Вывести первый контур (пример для одной фигуры)
    contour = contours[0]

    # Пример конвертации в удобный формат для передачи (возможно нужно масштабировать координаты)
    coords = contour[:, 0, :].tolist()

    # def pixel_to_latlng(pixel_coords, img_center, map_center, scale_factor):
    #     latlng_coords = []
        
    #     # Преобразуем пиксельные координаты в географические с учётом масштаба
    #     for x, y in pixel_coords:
    #         lat = map_center[0] + (y - img_center[1]) * scale_factor
    #         lng = map_center[1] + (x - img_center[0]) * scale_factor
    #         latlng_coords.append([lat, lng])
        
    #     return latlng_coords



    img_center = center
    img_size = (result.shape[1], result.shape[0])

    # Координаты центра карты (широта и долгота)
    map_center = coords_psr
    radius = radius
    # Пример масштаба (нужно подбирать в зависимости от разрешения изображения и масштаба карты)
    scale_factor = 0.000894 * 805/img_size[0] * radius/10# настройте значение для вашей карты
    scale_factor_y = 0.00045 * 486/img_size[1] * radius/10

    # Преобразуем пиксельные координаты маски в географические
    latlng_coords = pixel_to_latlng(coords, img_size, map_center, scale_factor, scale_factor_y)

    with open(f'.\\templates\\{p}.txt', "w") as file:
        file.write(str(latlng_coords))

# make_txt_mask_of_radius("masked_map_image0", (33,33), 23, cv2.imread("masked_map_image0.png"), (500,400) )