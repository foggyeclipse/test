import folium
import math

def calculate_bounds(center_coords, radius_km, buffer_km=0.01):
    """
    Рассчитывает границы для круга с заданным радиусом и добавляет отступ.
    
    :param center_coords: Координаты центра карты (широта, долгота)
    :param radius_km: Радиус в километрах
    :param buffer_km: Дополнительный отступ в километрах (1 см)
    :return: Границы карты в формате [[lat_min, lon_min], [lat_max, lon_max]]
    """
    earth_radius_km = 6371.0  # Радиус Земли в километрах
    lat, lon = center_coords
    total_radius_km = radius_km + buffer_km  # Добавляем 1 см отступ

    lat_change = (total_radius_km / earth_radius_km) * (180 / math.pi)
    lon_change = (total_radius_km / earth_radius_km) * (180 / math.pi) / math.cos(lat * math.pi / 180)

    return [[lat - lat_change, lon - lon_change], [lat + lat_change, lon + lon_change]]

# def calculate_zoom_level(radius_km):
#     """
#     Рассчитывает уровень зума для отображения радиуса с 1 см отступом сверху и снизу.
    
#     :param radius_km: Радиус в километрах
#     :return: Уровень зума
#     """
#     # Примерный расчет, возможно, потребуется скорректировать
#     return 12 - (radius_km // 2)

def save_map_image(radius_km, center_coords, output_image_path, i):
    # Создание карты
    map_center = center_coords
    m = folium.Map(location=map_center)

    # Рассчитываем границы с отступами
    bounds = calculate_bounds(center_coords, radius_km)

    # Устанавливаем уровень приближения в зависимости от радиуса
    zoom_level = 14

    # Устанавливаем уровень зума
    m = folium.Map(location=center_coords, zoom_start=zoom_level)

    # Добавляем круг с указанным радиусом
    for j in range(60):
        folium.Circle(
            location=map_center,
            radius=(j+1) * 1000,  # Переводим радиус из километров в метры
            color='red',
            # fill=False,
            # fill_color='#add8e6',
            fill_opacity=0
        ).add_to(m)
        print(zoom_level)

    # Сохраняем карту как HTML
    map_path = f"map{i}.html"
    m.save(map_path)

    print(f"Map saved to {map_path} with zoom level {zoom_level}")

# Пример использования функции
radius_km_6 = 6  # Радиус в километрах
radius_km_5 = 5  # Другой радиус
center_coords = (55.7558, 37.6176)  # Центр Москва
output_image_path_6 = "map_image_6_km.png"  # Имя выходного изображения для 6 км
output_image_path_5 = "map_image_5_km.png"  # Имя выходного изображения для 5 км

# Сохраняем карту для радиуса 6 км
save_map_image(radius_km_6, center_coords, output_image_path_6,1)

# Сохраняем карту для радиуса 5 км
save_map_image(radius_km_5, center_coords, output_image_path_5,2)
