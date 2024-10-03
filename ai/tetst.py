import cv2
import math

image = cv2.imread("map_image1.png")
height, width = image.shape[:2]
print(height,width)
center = (width // 2, height // 2)



def km_to_pixels(radius_km, zoom_level, latitude, image_width, image_height):
    """
    Рассчитывает радиус круга в пикселях на изображении OpenCV с учетом размеров изображения.
    
    :param radius_km: Радиус круга в километрах.
    :param zoom_level: Уровень зума карты.
    :param latitude: Широта в градусах для расчета метра на пиксель.
    :param image_width: Ширина изображения в пикселях.
    :param image_height: Высота изображения в пикселях.
    :return: Радиус круга в пикселях.
    """
    
    # Конвертация зума в метры на пиксель с учетом широты
    meters_per_pixel = 156543.04 * math.cos(latitude * math.pi / 180) / (2 ** zoom_level)
    
    # Перевод радиуса из километров в метры
    radius_meters = radius_km * 1000
    
    # Конвертация радиуса в пиксели
    radius_pixels = radius_meters / meters_per_pixel

    # Убедимся, что радиус не превышает половину меньшей стороны изображения
    max_radius_pixels = min(image_width, image_height) / 2 - 1  # Отступ в 1 пиксель
    
    if radius_pixels > max_radius_pixels:
        radius_pixels = max_radius_pixels  # Ограничиваем радиус
    
    return radius_pixels

# Пример использования
image_width = 1083   # Ширина изображения в пикселях
image_height = 793   # Высота изображения в пикселях
radius_km = 60       # Радиус в километрах
zoom_level = 8     # Уровень зума
latitude = 55.7558  # Широта центра карты (например, Москва)

# 11=км 
# 12-2км
# 1km = 0,94cm
# 1см - 61,95пк

def calculate_pixels_per_centimeter(resolution_x, resolution_y, screen_diagonal_inch):
    # Вычисляем разрешение по диагонали
    diagonal_resolution = (resolution_x ** 2 + resolution_y ** 2) ** 0.5
    
    # Вычисляем PPI (pixels per inch)
    ppi = diagonal_resolution / screen_diagonal_inch
    
    # Конвертируем PPI в пиксели на сантиметр
    pixels_per_centimeter = ppi / 2.54
    
    return pixels_per_centimeter

# Пример использования
resolution_x = 1920  # Разрешение по горизонтали в пикселях
resolution_y = 1080  # Разрешение по вертикали в пикселях
screen_diagonal_inch = 14  # Диагональ экрана в дюймах

pixels_per_cm = calculate_pixels_per_centimeter(resolution_x, resolution_y, screen_diagonal_inch)
print(f"Количество пикселей в сантиметре: {pixels_per_cm:.2f}")

radius_pixels = km_to_pixels(radius_km, zoom_level, latitude, image_width, image_height)
print(f"Радиус круга в пикселях: {radius_pixels:.2f}")


# print(int(9.9),math.floor(9.9))
a14 = 3.76
a13 = 1.88
a12 = 0.94
a11 = 0.47
a10 = 0.235
a9 = 0.1175
a8 = 0.05875
a7 = 0.029375

radius = int(radius_km * a8 * pixels_per_cm)
# cv2.circle(image, center, radius, (255, 255, 0), 3)
# cv2.imshow('a',image)
# cv2.waitKey(0)