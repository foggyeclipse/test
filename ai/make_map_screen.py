import folium
from selenium import webdriver
from PIL import Image
import time
import threading
import math
from http.server import SimpleHTTPRequestHandler
import socketserver

def start_server(port):
    """Функция для запуска HTTP-сервера."""
    handler = SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()

def crop_image_by_1cm(image_path, output_path, pixels_per_cm=38):
    """
    Обрезает края изображения на 1 см (примерно 38 пикселей по умолчанию).
    
    :param image_path: Путь к исходному изображению.
    :param output_path: Путь для сохранения обрезанного изображения.
    :param pixels_per_cm: Количество пикселей в 1 см (по умолчанию 38 пикселей для 96 PPI).
    """
    # Открываем изображение
    image = Image.open(image_path)
    
    # Получаем размер изображения
    width, height = image.size
    
    # Обрезаем изображение: убираем по 1 см с каждой стороны
    left = pixels_per_cm*2
    top = pixels_per_cm//3 + 5
    right = width - pixels_per_cm*2
    bottom = height - pixels_per_cm//3 - 5
    
    # Обрезаем изображение
    cropped_image = image.crop((left, top, right, bottom))
    
    # Сохраняем обрезанное изображение
    cropped_image.save(output_path)
    print(f"Изображение обрезано и сохранено как {output_path}")

def calculate_bounds(center_coords, radius_km):
    """
    Рассчитывает границы карты для полного отображения круга на основе центра и радиуса.
    
    :param center_coords: Координаты центра карты (широта, долгота)
    :param radius_km: Радиус в километрах
    :return: Границы карты в формате [[lat_min, lon_min], [lat_max, lon_max]]
    """
    earth_radius_km = 6371  # Радиус Земли в километрах

    lat, lon = center_coords
    lat_change = (radius_km / earth_radius_km) * (180 / math.pi)
    lon_change = (radius_km / earth_radius_km) * (180 / math.pi) / math.cos(lat * math.pi/180)
    print(lat_change, lon_change, "!!!!")
    return [[lat - lat_change, lon - lon_change], [lat + lat_change, lon + lon_change]]

def get_zoom_level(radius_km):
    """
    Определяет уровень приближения в зависимости от радиуса.
    Это просто пример, вы можете настроить функцию под свои нужды.
    """
    if radius_km <= 1:
        return 14  # Более близкий уровень приближения
    elif radius_km <= 3:
        return 13  # Уровень приближения для 5 км
    elif radius_km <= 6:
        return 12  # Уровень приближения для 10 км
    elif radius_km <= 13:
        return 11
    elif radius_km <= 26:
        return 10
    elif radius_km <= 53:
        return 9
    elif radius_km <= 106:
        return 8
    else:
        return 7

def save_map_image(radius_km, center_coords, output_image_path="map_image.png"):
    print("AAAAAAAAAAAAAAAAAAAAA")
    """
    Создаёт спутниковую карту с заданным центром и радиусом, сохраняет изображение карты с высотой и шириной, равной диаметру круга.
    
    :param radius_km: Радиус области в километрах
    :param center_coords: Координаты центра карты (широта, долгота) - tuple (latitude, longitude)
    :param output_image_path: Путь к выходному изображению, которое будет сохранено
    """
    # Создание карты с центром в указанных координатах
    map_center = center_coords
    print(radius_km)
    try:
        m = folium.Map(location=map_center, zoom_start=get_zoom_level(radius_km))
        print("FFFFFFFFFFFFFFF")
    except Exception as e:
        print(e)
    # Добавляем спутниковый слой ESRI с атрибуцией
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        name='ESRI Satellite',
        overlay=False,
        control=False
    ).add_to(m)

    # Добавляем круг с указанным радиусом
    folium.Circle(
        location=map_center,
        radius=radius_km * 1000,  # Переводим радиус из километров в метры
        color='blue',
        fill=True,
        fill_color='#add8e6',
        fill_opacity=0.5
    )
    # .add_to(m)

    # Рассчитываем границы карты, чтобы весь круг был в пределах
    # bounds = calculate_bounds(center_coords, radius_km)
    # m.fit_bounds(bounds)

    # Сохраняем карту как HTML
    map_path = "map.html"
    m.save(map_path)

    # Определяем порт для сервера
    PORT = 8080

    # Запускаем HTTP-сервер в отдельном потоке
    server_thread = threading.Thread(target=start_server, args=(PORT,))
    server_thread.daemon = True
    server_thread.start()

    time.sleep(5)
    # Открытие карты через Selenium
    # options = webdriver.EdgeOptions()
    # options.add_argument('--headless')  # Фоновый режим, без открытия окна браузера
    # options.add_argument('--no-sandbox')
    # options.add_argument('--disable-dev-shm-usage')

    # Запускаем драйвер браузера
    driver = webdriver.Edge()
    # time.sleep(20)
    # Открываем карту через локальный сервер по адресу http://localhost:8000/map.html
    driver.get(f"http://localhost:{PORT}/map.html")

    # Вычисляем размер скриншота в пикселях на основе радиуса круга
    # pixels_per_km = 1000  # Например, 100 пикселей на километр
    # diameter_px = int(radius_km * 2 * pixels_per_km)  # Диаметр круга в пикселях

    # # Устанавливаем размер окна браузера в соответствии с диаметром
    # driver.set_window_size(diameter_px, diameter_px)

    # Небольшая задержка для загрузки карты
    time.sleep(3)

    # Сделаем скриншот всей страницы (карты)
    screenshot_path = "screenshot.png"
    driver.save_screenshot(screenshot_path)

    # Закрываем драйвер
    driver.quit()

    # Обрезка скриншота (если требуется)
    screenshot = Image.open(screenshot_path)
    
    # Сохранение итогового изображения
    screenshot.save(output_image_path)
    print(f"Изображение карты сохранено как {output_image_path}")
    crop_image_by_1cm(output_image_path, output_image_path)

# Пример использования функции
radius_km = 60  # Радиус в километрах
center_coords = (55.7558, 37.6176)  # Центр Москва
output_image_path = "map_image1.png"  # Имя выходного изображения

# save_map_image(radius_km, center_coords, output_image_path)




# Пример использования
# image_path = "map_image.png"
# output_path = "map_image_cropped.png"

# crop_image_by_1cm(image_path, output_path)
