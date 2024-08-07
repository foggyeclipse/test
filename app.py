from flask import Flask, jsonify, render_template, request
import pandas as pd
import math
from math import radians, cos, sin, sqrt, atan2

app = Flask(__name__)

# Загрузка данных из Excel
file_path = '/Users/katerina/Desktop/СourseWork/EXTR_ORIGINAL.xlsx'
df = pd.read_excel(file_path, sheet_name=1, skiprows=[1])

# Переименование столбцов для удобства
df.rename(columns={
    'Координ.ц.ПСР': 'Координ.ц.ПСР X',
    df.columns[14]: 'Координ.ц.ПСР Y',
    'Коорд.нах.': 'Коорд.нах. X',
    df.columns[57]: 'Коорд.нах. Y',
}, inplace=True)

# Преобразование строк с координатами в числовые значения
df['Координ.ц.ПСР X'] = df['Координ.ц.ПСР X'].str.replace(
    ',', '.').astype(float)
df['Координ.ц.ПСР Y'] = df[df.columns[14]].str.replace(',', '.').astype(float)
df['Коорд.нах. X'] = df['Коорд.нах. X'].str.replace(',', '.').astype(float)
df['Коорд.нах. Y'] = df[df.columns[57]].str.replace(',', '.').astype(float)

# Функция для обработки дат
def parse_dates(df, column_name):
    df[column_name] = pd.to_datetime(
        df[column_name], format='%d.%m.%Y %H:%M:%S', errors='coerce')
    df[column_name] = df[column_name].fillna(pd.to_datetime(
        df[column_name], format='%d.%m.%Y', errors='coerce'))


parse_dates(df, 'Дата ПСР')
parse_dates(df, 'Дата завершения')

# Функция расчета расстояния
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Радиус Земли в километрах
    dlat = radians(lat2 - lat1)
    dlon = radians(lon1 - lon2)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance * 1000  # Возвращаем расстояние в метрах

# Функция расчета радиуса поиска
def get_radius(hours_elapsed, terrain_passability=None, path_curvature=None, slope_degree=None, fatigue_level=None, time_of_day=None, weather_conditions=None, group_factor=None):
    normal_speed = 5  # Средняя скорость движения по асфальту в км/ч

    # Коэффициенты понижения
    terrain_passability_coefficient = terrain_passability if terrain_passability is not None else 1.0
    path_curvature_coefficient = path_curvature if path_curvature is not None else 1.0
    slope_degree_coefficient = slope_degree if slope_degree is not None else 1.0
    fatigue_level_coefficient = fatigue_level if fatigue_level is not None else 1.0
    time_of_day_coefficient = time_of_day if time_of_day is not None else 1.0
    weather_conditions_coefficient = weather_conditions if weather_conditions is not None else 1.0
    group_factor_coefficient = group_factor if group_factor is not None else 1.0

    # Индекс скорости движения
    speed_index = (
        terrain_passability_coefficient
        * path_curvature_coefficient
        * slope_degree_coefficient
        * fatigue_level_coefficient
        * time_of_day_coefficient
        * weather_conditions_coefficient
        * group_factor_coefficient
    )

    # Радиус поиска
    search_radius = (
        hours_elapsed
        * normal_speed
        * speed_index
    )
    return search_radius

@app.route('/')
def index():
    return render_template('base.html')

def is_valid_coordinate(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def calculate_distance(lat1, lon1, lat2, lon2):
    # Радиус Земли в километрах
    R = 6371.0
    # Преобразуем градусы в радианы
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

@app.route('/radius', methods=['POST'])
def radius():
    data = request.get_json()
    try:
        coords_psr = data.get('coords_psr')
        coords_finding = data.get('coords_finding')
        if not coords_psr or not coords_finding:
            raise ValueError('Недостаточно данных')

        psr_lat = float(coords_psr.get('latitude'))
        psr_lon = float(coords_psr.get('longitude'))
        finding_lat = float(coords_finding.get('latitude'))
        finding_lon = float(coords_finding.get('longitude'))

        distance = calculate_distance(
            psr_lat, psr_lon, finding_lat, finding_lon)
        return jsonify({
            'status': 'success',
            'distance': distance,
            'coords_psr': coords_psr,
            'coords_finding': coords_finding
        })
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Не удалось обработать запрос'}), 500

if __name__ == '__main__':
    app.run(debug=True)