import time
import folium
import threading
import socketserver
from PIL import Image
from selenium import webdriver
from http.server import SimpleHTTPRequestHandler

def start_server(port):
    """Функция для запуска HTTP-сервера."""
    handler = SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at port {port}")
        httpd.serve_forever()

def crop_image_by_1cm(image_path, output_path, pixels_per_cm=38):
    image = Image.open(image_path)
    
    width, height = image.size

    left = pixels_per_cm * 2
    top = pixels_per_cm // 3 + 5
    right = width - pixels_per_cm * 2
    bottom = height - pixels_per_cm // 3 - 5
    
    cropped_image = image.crop((left, top, right, bottom))
    
    cropped_image.save(output_path)
    print(f"Изображение обрезано и сохранено как {output_path}")

def get_zoom_level(radius_km):
    """
    Определяет уровень приближения в зависимости от радиуса (в километрах).
    """
    zoom_levels = {1: 14, 3: 13, 6: 12, 13: 11, 26: 10, 53: 9, 106: 8}

    for max_radius, zoom in zoom_levels.items():
        if radius_km <= max_radius:
            return zoom
    return 7

def save_map_image(radius_km, center_coords, output_image_path="map_image.png"):
    """
    Создаёт спутниковую карту с заданным центром и радиусом, сохраняет изображение карты с высотой и шириной, равной диаметру круга.
    
    :param radius_km: Радиус области в километрах
    :param center_coords: Координаты центра карты (широта, долгота) - tuple (latitude, longitude)
    :param output_image_path: Путь к выходному изображению, которое будет сохранено
    """

    map_center = center_coords

    try:
        m = folium.Map(location=map_center, zoom_start=get_zoom_level(radius_km))
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

    folium.Circle(
        location=map_center,
        radius=radius_km * 1000,  # Переводим радиус из километров в метры
        color='blue',
        fill=True,
        fill_color='#add8e6',
        fill_opacity=0.5
    )

    map_path = "map.html"
    m.save(map_path)

    PORT = 8080

    # Запускаем HTTP-сервер в отдельном потоке
    server_thread = threading.Thread(target=start_server, args=(PORT,))
    server_thread.daemon = True
    server_thread.start()

    driver = webdriver.Edge()
    driver.get(f"http://localhost:{PORT}/map.html")

    # Небольшая задержка для загрузки карты
    time.sleep(3)

    screenshot_path = "screenshot.png"
    driver.save_screenshot(screenshot_path)

    driver.quit()

    crop_image_by_1cm(output_image_path, output_image_path)
