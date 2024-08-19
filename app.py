from flask import Flask, jsonify, render_template, request
import pandas as pd
import math
from math import radians, cos, sin, sqrt, atan2
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

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

def get_behavior_coef(behavior_data):
    behavior_coef = 1

    # Используем регулярное выражение для поиска всех процентов в строке
    percentages = re.findall(r"(\d+\.\d+)%", behavior_data)

    # Преобразуем найденные проценты в числа
    percentages = [float(p) for p in percentages]

    # Найдем максимальный процент
    max_percentage = max(percentages)
    # Поиск текста с максимальным процентом
    # Используем регулярное выражение для нахождения текста с максимальным процентом
    pattern = rf"([^\d]+)\s*{max_percentage:.2f}%"
    # print(pattern)
    # Ищем текст с максимальным процентом
    match = re.search(pattern, behavior_data)
    result_text = match.group(1).strip()
    # print(result_text)
    if result_text!="остаться на месте:":
        result_text = result_text.split('%')[1]

    if result_text == "остаться на месте:":
        behavior_coef = 0
    elif result_text == "искать укрытие:":
        behavior_coef = 0.2
    return behavior_coef, result_text + ' ' + str(max_percentage) + '%'

def get_behavior_data(data, current_time, current_date):
    if 5 <= current_time < 10:
        time = "morning"
    elif 10 <= current_time < 17:
        time = "day"
    elif 17 <= current_time < 21:
        time = "evening"
    else:
        time = "night"
    data_beh = {
            'Возраст': int(data.get('age')),
            'Пол': str(data.get('gender')),
            'Физическое состояние': str(data.get('physical_condition')),
            'Психическое состояние': str(data.get('mental_condition')),
            'Опыт нахождения в дикой природе': str(data.get('experience')),
            'Знание местности': str(data.get('local_knowledge')),
            'Наличие телефона': str(data.get('phone')),
            'Время суток': time,
            'Моральные обязательства': "unknown",
            'Внешние сигналы': "unknown",
            'Дата': current_date
        }
    return data_beh, time


# Функция расчета радиуса поиска
def get_radius(data, age, hours_elapsed, terrain_passability=None, path_curvature=None, slope_degree=None, fatigue_level=None, time_of_day=None, weather_conditions=None, group_factor=None):
    normal_speed = 5  # Средняя скорость движения по асфальту в км/ч

    if age < 18:
        normal_speed = 4
    elif age >= 60:
        normal_speed = 3

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

    # Сумма радиусов для каждых 6 часов
    list_of_radius = ''
    total_radius = 0
    current_date = data.get('date_of_loss')
    time_passed = 0
    day = 1
    prev_radius = []

    for i in range(0, hours_elapsed, 6):
        if(time_passed%24==0 and time_passed!=0):
            current_date = (datetime.strptime(current_date, '%d.%m.%Y') + timedelta(days=1)).strftime('%d.%m.%Y')
            time_passed = 0
            day +=1
            prev_radius.append(total_radius)
        data_beh, time = get_behavior_data(data, time_passed, current_date)
        # print(data_beh)

        behavior_data, weather = predict_behavior(data_beh)
        # print(behavior_data)

        behavior_coef, beh_main = get_behavior_coef(behavior_data)
        print(behavior_coef)
        # print(interval_hours)

        # Рассчитываем радиус для каждых 6 часов
        interval_hours = min(6, hours_elapsed - i)  # Учитываем оставшиеся часы в последнем интервале
        print(interval_hours)
        interval_radius = (
            interval_hours
            * normal_speed
            * speed_index
            * behavior_coef
        )
        total_radius += interval_radius
        print(interval_radius)

        if(time_passed==0):
            list_of_radius += f'День {day}: '
        list_of_radius +=' '.join([str(interval_radius), str(beh_main), str(weather), str(time)]) + '  '
        time_passed += 6
    print(total_radius)
    return total_radius, list_of_radius, prev_radius

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

# @app.route('/radius', methods=['POST'])
# def radius():
#     data = request.get_json()
#     try:
#         coords_psr = data.get('coords_psr')
#         # coords_finding = data.get('coords_finding')
#         # if not coords_psr or not coords_finding:
#         #     raise ValueError('Недостаточно данных')

#         psr_lat = float(coords_psr.get('latitude'))
#         psr_lon = float(coords_psr.get('longitude'))
#         # finding_lat = float(coords_finding.get('latitude'))
#         # finding_lon = float(coords_finding.get('longitude'))

#         # distance = calculate_distance(psr_lat, psr_lon, finding_lat, finding_lon)


#         #крот2
#         data_beh = {
#             'Возраст': int(data.get('age')),
#             'Пол': str(data.get('gender')),
#             'Физическое состояние': str(data.get('physical_condition')),
#             'Психическое состояние': str(data.get('mental_condition')),
#             'Опыт нахождения в дикой природе': str(data.get('experience')),
#             'Знание местности': str(data.get('local_knowledge')),
#             'Наличие телефона': str(data.get('phone')),
#             'Время суток': "unknown",
#             'Моральные обязательства': "unknown",
#             'Внешние сигналы': "unknown",
#             'Дата': data.get('date_of_loss')
#         }

#         behavior, _ = predict_behavior(data_beh)

#         date_of_loss = datetime.strptime(data.get('date_of_loss'), '%d.%m.%Y')
#         date_of_finding = datetime.strptime(data.get('date_of_finding'), '%d.%m.%Y')   
#         date_difference = (date_of_finding - date_of_loss)
#         days_difference = date_difference.days*24

#         radius, extra_info, prev_radius = get_radius(data, int(data.get('age')), int(days_difference))

#         return jsonify({
#             'status': 'success',
#             # 'distance': distance,
#             'radius': radius,
#             'coords_psr': coords_psr,
#             # 'coords_finding': coords_finding,
#             'behavior': behavior,
#             'extra_info': extra_info,
#             'prev_radius': prev_radius
#         })
#     except ValueError as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 400
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': 'Не удалось обработать запрос'}), 500

@app.route('/radius', methods=['POST'])
def radius():
    data = request.get_json()
    try:
        coords_psr = data.get('coords_psr')
        coords_finding = data.get('coords_finding')

        # psr_lat = float(coords_psr.get('latitude'))
        # psr_lon = float(coords_psr.get('longitude'))

        # Обработка полей даты и времени
        date_of_loss_str = data.get('date_of_loss')
        time_of_loss_str = data.get('time_of_loss', '00:00')  # Значение по умолчанию

        date_of_finding_str = data.get('date_of_finding')
        time_of_finding_str = data.get('time_of_finding', '00:00')  # Значение по умолчанию

        # Конкатенация даты и времени в одну строку
        date_time_of_loss_str = f"{date_of_loss_str} {time_of_loss_str}"
        date_time_of_finding_str = f"{date_of_finding_str} {time_of_finding_str}"

        
        # Пример обработки данных
        data_beh = {
            'Возраст': int(data.get('age')),
            'Пол': str(data.get('gender')),
            'Физическое состояние': str(data.get('physical_condition')),
            'Психическое состояние': str(data.get('mental_condition')),
            'Опыт нахождения в дикой природе': str(data.get('experience')),
            'Знание местности': str(data.get('local_knowledge')),
            'Наличие телефона': str(data.get('phone')),
            'Время суток': "unknown",
            'Моральные обязательства': "unknown",
            'Внешние сигналы': "unknown",
            'Дата': data.get('date_of_loss')
        }

        behavior, _ = predict_behavior(data_beh)

        # Преобразование строк в объекты datetime
        date_time_of_loss = datetime.strptime(date_time_of_loss_str, '%d.%m.%Y %H:%M')
        date_time_of_finding = datetime.strptime(date_time_of_finding_str, '%d.%m.%Y %H:%M')

        # Разница во времени
        date_difference = date_time_of_finding - date_time_of_loss
        hours_difference = date_difference.total_seconds() // 3600  # Разница в часах
        print(hours_difference)

        # # Пример обработки данных
        # data_beh = {
        #     'Возраст': int(data.get('age')),
        #     'Пол': str(data.get('gender')),
        #     'Физическое состояние': str(data.get('physical_condition')),
        #     'Психическое состояние': str(data.get('mental_condition')),
        #     'Опыт нахождения в дикой природе': str(data.get('experience')),
        #     'Знание местности': str(data.get('local_knowledge')),
        #     'Наличие телефона': str(data.get('phone')),
        #     'Время суток': "unknown",
        #     'Моральные обязательства': "unknown",
        #     'Внешние сигналы': "unknown",
        #     'Дата': data.get('date_of_loss')
        # }

        # behavior, _ = predict_behavior(data_beh)

        # Вызов функции для получения радиуса
        radius, extra_info, prev_radius = get_radius(data, int(data.get('age')), int(hours_difference))

        return jsonify({
            'status': 'success',
            'radius': radius,
            'coords_psr': coords_psr,
            'coords_finding': coords_finding,
            'behavior': behavior,
            'extra_info': extra_info,
            'prev_radius': prev_radius
        })
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Не удалось обработать запрос'}), 500

#крот2

def get_weather_data(date:str):
    day,month,year = date.split('.')
    url = f"https://www.gismeteo.ru/diary/4079/{year}/{month}/"
    # url = f"https://arhivpogodi.ru/arhiv/sankt-peterburg/2024/02
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
                    snow_icon = any("snow" in img["src"] for img in row.find_all("img"))
                    if rain_icon or snow_icon:
                        return 'bad'
                    else:
                        return 'good'
                    break
        else:
            print(f"Информация о погоде на {day} число не найдена.")
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
    weather = get_weather_data(data.get('Дата'))
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
    return probabilities, weather

def predict_behavior(data):
    probabilities, weather = calculate_probability(data)
    # Формируем строку с процентами вероятности
    probabilities_str = "\n".join([f"{behavior}: {prob * 100:.2f}%" for behavior, prob in probabilities.items()])
    return probabilities_str, weather


if __name__ == '__main__':
    app.run(debug=True)