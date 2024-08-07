from flask import Flask, jsonify, render_template, request
import pandas as pd
import math
from math import radians, cos, sin, sqrt, atan2
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime

app = Flask(__name__)

# Загрузка данных из Excel
# file_path = '/Users/katerina/Desktop/СourseWork/EXTR_ORIGINAL.xlsx'
# df = pd.read_excel(file_path, sheet_name=1, skiprows=[1])

# # Переименование столбцов для удобства
# df.rename(columns={
#     'Координ.ц.ПСР': 'Координ.ц.ПСР X',
#     df.columns[14]: 'Координ.ц.ПСР Y',
#     'Коорд.нах.': 'Коорд.нах. X',
#     df.columns[57]: 'Коорд.нах. Y',
# }, inplace=True)

# # Преобразование строк с координатами в числовые значения
# df['Координ.ц.ПСР X'] = df['Координ.ц.ПСР X'].str.replace(',', '.').astype(float)
# df['Координ.ц.ПСР Y'] = df[df.columns[14]].str.replace(',', '.').astype(float)
# df['Коорд.нах. X'] = df['Коорд.нах. X'].str.replace(',', '.').astype(float)
# df['Коорд.нах. Y'] = df[df.columns[57]].str.replace(',', '.').astype(float)

# Функция для обработки дат
# def parse_dates(df, column_name):
#     df[column_name] = pd.to_datetime(df[column_name], format='%d.%m.%Y %H:%M:%S', errors='coerce')
#     df[column_name] = df[column_name].fillna(pd.to_datetime(df[column_name], format='%d.%m.%Y', errors='coerce'))

# parse_dates(df, 'Дата ПСР')
# parse_dates(df, 'Дата завершения')

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
def get_radius(age, behavior_data, hours_elapsed, terrain_passability=None, path_curvature=None, slope_degree=None, fatigue_level=None, time_of_day=None, weather_conditions=None, group_factor=None):
    normal_speed = 5  # Средняя скорость движения по асфальту в км/ч
    behavior_coef = 1

    if age < 18:
        normal_speed = 4
    elif age >= 60:
        normal_speed = 3

    # Используем регулярное выражение для поиска всех процентов в строке
    percentages = re.findall(r"(\d+\.\d+)%", behavior_data)

    # Преобразуем найденные проценты в числа
    percentages = [float(p) for p in percentages]

    # Найдем максимальный процент
    max_percentage = max(percentages)

    # Поиск текста с максимальным процентом
    # Используем регулярное выражение для нахождения текста с максимальным процентом
    pattern = rf"([^\d]+)\s*{max_percentage:.2f}%"

    # Ищем текст с максимальным процентом
    match = re.search(pattern, behavior_data)
    print(match)

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

    if match == "остаться на месте":
        behavior_coef = 0
    elif match == "искать укрытие":
        behavior_coef = 0.2

    # Радиус поиска
    search_radius = (
        hours_elapsed
        * normal_speed
        * speed_index
        * behavior_coef
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

        distance = calculate_distance(psr_lat, psr_lon, finding_lat, finding_lon)


        #крот2
        data_beh = {
            'Возраст': int(data.get('age')),
            'Пол': str(data.get('gender')),
            'Физическое состояние': str(data.get('physical_condition')),
            'Психическое состояние': str(data.get('mental_condition')),
            'Опыт нахождения в дикой природе': str(data.get('experience')),
            'Знание местности': str(data.get('local_knowledge')),
            'Погодные условия': "unknown",
            'Наличие телефона': str(data.get('phone')),
            'Время суток': "unknown",
            'Моральные обязательства': "unknown",
            'Внешние сигналы': "unknown"
        }

        behavior = predict_behavior(data_beh)

        date_of_loss = datetime.strptime(data.get('date_of_loss'), '%d.%m.%Y')
        date_of_finding = datetime.strptime(data.get('date_of_finding'), '%d.%m.%Y')   
        date_difference = (date_of_finding - date_of_loss)
        days_difference = date_difference.days
        print(days_difference)

        radius = get_radius(int(data.get('age')), behavior, int(days_difference))
    
        return jsonify({
            'status': 'success',
            'distance': distance,
            'radius': radius,
            'coords_psr': coords_psr,
            'coords_finding': coords_finding,
            'behavior': behavior
        })
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Не удалось обработать запрос'}), 500
    

#крот2

def get_weather_data(date:str):
    day,month,year = date.split('.')
    url = f"https://www.gismeteo.ru/diary/4079/{year}/{month}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.find_all("tr", align="center")

        for row in rows:
            cells = row.find_all("td")

            if cells:
                day_number = cells[0].text.strip()

                if day_number == day:
                    rain_icon = any("rain" in img["src"] for img in row.find_all("img"))

                    if rain_icon:
                        print(f"Дата: {day_number} - осадки: дождь")
                    else:
                        print(f"Дата: {day_number} - осадков нет")
                    break
        else:
            print("Информация о погоде на 4 число не найдена.")
    else:
        print("Ошибка при выполнении запроса. Код состояния:", response.status_code)

def calculate_probability(data):
    probabilities = {
        'остаться на месте': 0.0,
        'двигаться с ориентированием': 0.0,
        'двигаться без ориентирования': 0.0,
        'искать укрытие': 0.0
    }

    age = data.get('Возраст', 30)
    gender = data.get('Пол', 'Неизвестно')
    physical_condition = data.get('Физическое состояние', 'Здоров')
    psychological_condition = data.get('Психическое состояние', 'Устойчив')
    experience = data.get('Опыт нахождения в дикой природе', 'Низкий')
    location = data.get('Знание местности', 'Нет')
    weather = data.get('Погодные условия', 'Хорошие')
    has_phone = data.get('Наличие телефона', 'Нет')
    time_of_day = data.get('Время суток', 'День')
    moral_obligations = data.get('Моральные обязательства', 'Слабые')
    external_signals = data.get('Внешние сигналы', 'Нет')

    # Корректируем вероятности в зависимости от условий
    if physical_condition in ['injury', 'health_deterioration']:
        probabilities['остаться на месте'] += 0.4
    if psychological_condition == 'unstable':
        probabilities['двигаться без ориентирования'] += 0.3
    if location == 'no' or experience == 'low':
        probabilities['двигаться без ориентирования'] += 0.3
    if weather == 'bad':
        probabilities['искать укрытие'] += 0.3
    if has_phone == 'yes':
        probabilities['остаться на месте'] += 0.5
    if time_of_day in ['evening', 'night']:
        probabilities['искать укрытие'] += 0.4
    if psychological_condition == 'stable' and moral_obligations == 'strong':
        probabilities['двигаться с ориентированием'] += 0.4
    if external_signals == 'yes':
        probabilities['двигаться с ориентированием'] += 0.3
    if moral_obligations == 'strong':
        probabilities['остаться на месте'] += 0.3

    # Корекция относительно возраста
    if age < 12:
        probabilities['остаться на месте'] += 0.5
        probabilities['искать укрытие'] += 0.2
    elif 12 <= age < 18:
        probabilities['двигаться без ориентирования'] += 0.4
    elif age >= 60:
        probabilities['остаться на месте'] += 0.3
        probabilities['искать укрытие'] += 0.3

    # Корекция относительно гендера
    if gender == 'female':
        probabilities['искать укрытие'] += 0.2
        probabilities['остаться на месте'] += 0.2
    elif gender == 'male':
        probabilities['двигаться с ориентированием'] += 0.2
        probabilities['двигаться без ориентирования'] += 0.2

    # Нормализуем вероятности
    total_probability = sum(probabilities.values())
    for key in probabilities:
        probabilities[key] /= total_probability
    
    return probabilities

def predict_behavior(data):
    probabilities = calculate_probability(data)
    # Формируем строку с процентами вероятности
    probabilities_str = "\n".join([f"{behavior}: {prob * 100:.2f}%" for behavior, prob in probabilities.items()])
    return probabilities_str

def prompt_user(message, options):
    print(message)
    for index, option in enumerate(options, start=1):
        print(f"{index}: {option}")
    
    while True:
        try:
            choice_index = int(input()) - 1
            if 0 <= choice_index < len(options):
                return options[choice_index]
            else:
                print("Некорректный выбор. Пожалуйста, введите номер из предложенных вариантов.")
        except ValueError:
            print("Некорректный выбор. Пожалуйста, введите число.")

# def main():
#     print("Это приложение для предсказания поведения человека, потерявшегося в лесу.")
    
#     # Запрос данных у пользователя
#     age = input("Введите возраст (оставьте пустым, если неизвестно): ")
#     age = int(age) if age else None
    
#     gender = prompt_user("Введите пол (М/Ж, оставьте пустым, если неизвестно): ", ["М", "Ж", "Неизвестно"])
    
#     physical_condition = prompt_user("Введите физическое состояние: ", ["Здоров", "Хронические заболевания", "Травма", "Ухудшение здоровья", "Неизвестно"])
    
#     psychological_condition = prompt_user("Введите психическое состояние: ", ["Устойчив", "Неустойчив", "Неизвестно"])
    
#     experience = prompt_user("Введите опыт нахождения в дикой природе: ", ["Низкий", "Средний", "Высокий", "Неизвестно"])
    
#     location = prompt_user("Знание местности: ", ["Да", "Нет", "Неизвестно"])
    
#     weather = prompt_user("Погодные условия: ", ["Хорошие", "Плохие", "Неизвестно"])
    
#     has_phone = prompt_user("Наличие телефона: ", ["Да", "Нет", "Неизвестно"])
    
#     time_of_day = prompt_user("Время суток: ", ["Утро", "День", "Вечер", "Ночь", "Неизвестно"])

#     moral_obligations = prompt_user("Моральные обязательства: ", ["Сильные", "Слабые", "Неизвестно"])

#     external_signals = prompt_user("Внешние сигналы спасения: ", ["Да", "Нет", "Неизвестно"])

#     # Создание словаря с данными
#     data = {
#         'Возраст': age,
#         'Пол': gender,
#         'Физическое состояние': physical_condition,
#         'Психическое состояние': psychological_condition,
#         'Опыт нахождения в дикой природе': experience,
#         'Знание местности': location,
#         'Погодные условия': weather,
#         'Наличие телефона': has_phone,
#         'Время суток': time_of_day,
#         'Моральные обязательства': moral_obligations,
#         'Внешние сигналы': external_signals
#     }
    
#     # Предсказание поведения
#     behavior = predict_behavior(data)
#     print("Предсказание поведения: ", behavior)

if __name__ == '__main__':
    app.run(debug=True)