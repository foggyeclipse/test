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
        result_text = result_text.split('%')[1].split('\n')[1]
    print(result_text)
    if result_text == "остаться на месте:":
        behavior_coef = 0
    elif result_text == "искать укрытие:":
        behavior_coef = 0.2
    return behavior_coef, result_text.capitalize() + ' ' + str(max_percentage) + '%'

def get_behavior_data(data, current_time, current_date):
    if 6 <= current_time < 12:
        time = "morning"
    elif 12 <= current_time < 18:
        time = "day"
    elif 18 <= current_time < 24:
        time = "evening"
    else:
        time = "night"
    if 0 <= current_time < 10:
        current_time = f'0{current_time}'

    print(current_time)
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
            'Дата': current_date,
            'Время': f'{current_time}:00'
        }
    return data_beh, time

def round_to_nearest_multiple_of_6(n):
    if n < 0 or n > 24:
        raise ValueError("Число должно быть в диапазоне от 0 до 24")

    lower = (n // 6) * 6  # ближайшее кратное 6 вниз
    upper = lower + 6      # ближайшее кратное 6 вверх

    # Если верхняя граница превышает 24, оставляем только нижнюю
    if upper > 24:
        return lower

    return lower
    # Возвращаем ближайшее из двух кратных
    # if (n - lower) < (upper - n):
    #     return lower
    # else:
    #     return upper
def calc_last_day(data,time_passed, hours, normal_speed, speed_index, total_radius, list_of_radius, day):
    current_date = data.get('date_of_finding')
    data_beh, time = get_behavior_data(data, time_passed, current_date)

    behavior_data, weather = predict_behavior(data_beh)

    behavior_coef, beh_main = get_behavior_coef(behavior_data)

    interval_radius = (
        hours
        * normal_speed
        * speed_index
        * behavior_coef
    )
    total_radius += interval_radius
    print(hours,"????????#^####")

    if time_passed == 0:
        list_of_radius += f'День {day}: '

    list_of_radius +=' '.join([str(interval_radius), str(beh_main), str(weather), str(time)]) + '  '
    
    return total_radius, list_of_radius

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

    time_of_loss = int(data.get('time_of_loss').split(':')[0])
    time_of_finding = int(data.get('time_of_finding').split(':')[0])
    time_passed = round_to_nearest_multiple_of_6(time_of_loss)
    res = time_of_loss - time_passed 

    day = 1
    prev_radius = []
    first_day=True
    last_day = False

    print(time_passed)

    for i in range(0, hours_elapsed, 6):
        # if time_passed!=0:
        if(time_passed%24==0 and time_passed!=0):
            current_date = (datetime.strptime(current_date, '%d.%m.%Y') + timedelta(days=1)).strftime('%d.%m.%Y')
            time_passed = 0
            day +=1
            prev_radius.append(total_radius)
        data_beh, time = get_behavior_data(data, time_passed, current_date)

        behavior_data, weather = predict_behavior(data_beh)

        behavior_coef, beh_main = get_behavior_coef(behavior_data)

        # Рассчитываем радиус для каждых 6 часов
        interval_hours = min(6, hours_elapsed - i) 
        if interval_hours != 6:
            continue
        #     interval_hours = interval_hours + res
        # elif interval_hours !=6 and interval_hours + res > 6:
        #     a = interval_hours + res-6
        #     interval_hours = 6
        if first_day:
            interval_hours = 6 - res # Учитываем оставшиеся часы в последнем интервале
        print(interval_hours, res, "????????")
        interval_radius = (
            interval_hours
            * normal_speed
            * speed_index
            * behavior_coef
        )
        total_radius += interval_radius
        print(interval_radius)

        if(time_passed==0 or first_day):
            list_of_radius += f'День {day}: '
            first_day=False
        list_of_radius +=' '.join([str(interval_radius), str(beh_main), str(weather), str(time)]) + '  '
        time_passed += 6
        # if last_day:
    if time_passed == 24:
        time_passed = 0
        day += 1
    total_radius, list_of_radius = calc_last_day(data,time_passed, -time_passed + time_of_finding,normal_speed, speed_index, total_radius, list_of_radius, day)
    print(total_radius)
    return total_radius, list_of_radius, prev_radius

# def round_to_nearest_multiple_of_6(time_of_loss, time_of_finding):
#     # Функция для определения интервала 6 часов
#     def get_time_interval(hour, minute):
#         if (hour == 0 and minute == 0) or (hour < 6):
#             return '00:00 - 06:00'
#         elif 6 <= hour < 12:
#             return '06:00 - 12:00'
#         elif 12 <= hour < 18:
#             return '12:00 - 18:00'
#         else:
#             return '18:00 - 24:00'

#     # Преобразуем время в объект datetime, чтобы выделить часы и минуты
#     time_loss = datetime.strptime(time_of_loss, '%H:%M')
#     time_finding = datetime.strptime(time_of_finding, '%H:%M')

#     # Получаем час и минуты для каждого времени
#     loss_hour, loss_minute = time_loss.hour, time_loss.minute
#     finding_hour, finding_minute = time_finding.hour, time_finding.minute

#     # Определяем интервал для каждого времени
#     loss_interval = get_time_interval(loss_hour, loss_minute)
#     finding_interval = get_time_interval(finding_hour, finding_minute)

#     return loss_interval, finding_interval, loss_hour, finding_hour
# def get_radius(data, age, hours_elapsed, terrain_passability=None, path_curvature=None, slope_degree=None, fatigue_level=None, time_of_day=None, weather_conditions=None, group_factor=None):
#     normal_speed = 5  # Средняя скорость движения по асфальту в км/ч

#     if age < 18:
#         normal_speed = 4
#     elif age >= 60:
#         normal_speed = 3

#     # Коэффициенты понижения
#     terrain_passability_coefficient = terrain_passability if terrain_passability is not None else 1.0
#     path_curvature_coefficient = path_curvature if path_curvature is not None else 1.0
#     slope_degree_coefficient = slope_degree if slope_degree is not None else 1.0
#     fatigue_level_coefficient = fatigue_level if fatigue_level is not None else 1.0
#     time_of_day_coefficient = time_of_day if time_of_day is not None else 1.0
#     weather_conditions_coefficient = weather_conditions if weather_conditions is not None else 1.0
#     group_factor_coefficient = group_factor if group_factor is not None else 1.0

#     # Индекс скорости движения
#     speed_index = (
#         terrain_passability_coefficient
#         * path_curvature_coefficient
#         * slope_degree_coefficient
#         * fatigue_level_coefficient
#         * time_of_day_coefficient
#         * weather_conditions_coefficient
#         * group_factor_coefficient
#     )

#     # Получаем время пропажи и нахождения
#     time_of_loss = data.get('time_of_loss')
#     time_of_finding = data.get('time_of_finding')

#     # Получаем интервалы и часы для времени пропажи и нахождения
#     loss_interval, finding_interval, loss_hour, finding_hour = round_to_nearest_multiple_of_6(time_of_loss, time_of_finding)

#     # Логика для расчета радиуса для первого и последнего интервала
#     time_loss_minutes_left = (6 - (loss_hour % 6)) * 60  # Сколько минут осталось до конца интервала
#     time_finding_minutes = (finding_hour % 6) * 60       # Сколько минут прошло с начала интервала

#     # Рассчитываем радиус для времени пропажи
#     time_loss_hours_left = time_loss_minutes_left / 60
#     loss_radius = time_loss_hours_left * normal_speed * speed_index

#     # Рассчитываем радиус для времени нахождения
#     time_finding_hours = time_finding_minutes / 60
#     finding_radius = time_finding_hours * normal_speed * speed_index

#     # Логика для промежуточных интервалов по 6 часов
#     list_of_radius = ''
#     total_radius = loss_radius  # Начинаем с радиуса первого дня
#     current_date = data.get('date_of_loss')

#     # Пропуск начальных часов (время пропажи)
#     time_passed = loss_hour
#     day = 1
#     prev_radius = []
#     first_day = True

#     for i in range(0, hours_elapsed, 6):
#         # Переход на следующий день
#         if time_passed % 24 == 0 and time_passed != 0:
#             current_date = (datetime.strptime(current_date, '%d.%m.%Y') + timedelta(days=1)).strftime('%d.%m.%Y')
#             time_passed = 0
#             day += 1
#             prev_radius.append(total_radius)

#         # Получаем поведенческие данные и данные о погоде
#         data_beh, time = get_behavior_data(data, time_passed, current_date)
#         behavior_data, weather = predict_behavior(data_beh)
#         behavior_coef, beh_main = get_behavior_coef(behavior_data)

#         # Рассчитываем радиус для каждых 6 часов
#         interval_hours = min(6, hours_elapsed - i)  # Учитываем оставшиеся часы в последнем интервале

#         interval_radius = (
#             interval_hours
#             * normal_speed
#             * speed_index
#             * behavior_coef
#         )
#         total_radius += interval_radius

#         # Логирование радиусов для каждого интервала
#         if time_passed == 0 or first_day:
#             list_of_radius += f'День {day}: '
#             first_day = False

#         list_of_radius += ' '.join([str(interval_radius), str(beh_main), str(weather), str(time)]) + '  '
#         time_passed += 6

#     # Учитываем последний неполный интервал (время нахождения)
#     total_radius += finding_radius
#     list_of_radius += f' Последний интервал: {finding_radius:.2f} км'

#     print(total_radius)
#     return total_radius, list_of_radius, prev_radius

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

        # psr_lat = float('.'.join(coords_psr.get('latitude').split(',')))
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
            'Дата': data.get('date_of_loss'),
            'Время': time_of_loss_str
        }

        behavior, _ = predict_behavior(data_beh)

        # Преобразование строк в объекты datetime
        date_time_of_loss = datetime.strptime(date_time_of_loss_str, '%d.%m.%Y %H:%M')
        date_time_of_finding = datetime.strptime(date_time_of_finding_str, '%d.%m.%Y %H:%M')

        # Разница во времени
        date_difference = date_time_of_finding - date_time_of_loss
        hours_difference = date_difference.total_seconds() // 3600  # Разница в часах
        print(hours_difference)


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

def get_weather_data(date, time):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    day,month,year = date.split('.')
    hour = time.split(':')[0]
    print(day,month,year,hour)

    url = f"https://arhivpogodi.ru/arhiv/sankt-peterburg/{year}/{month}/"

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        
        date_blocks = soup.find_all('div', class_='font-size-unset d-inline-block position-sticky px-3 pb-2')
        holydate_blocks = soup.find_all('div', class_='font-size-unset d-inline-block position-sticky px-3 pb-2 text-danger')
        day_pattern = rf',\s*{day}\b'
    # Параметры поиска
        target_day_found = False
        target_temperature = None
        
        for date_block in date_blocks:
            date_text = date_block.get_text(strip=True)
            # print(date_text)
            # print(date_block)
            if re.search(day_pattern, date_text):   # Проверяем, начинается ли текст с числа дня
                # print('yes')
                target_day_found = True
                # Найти все блоки с температурой для этого дня
                hourly_blocks = date_block.find_parent('div').find_next_sibling().find_all('div', class_='d-inline-block')
                # print(hourly_blocks)
                for hour_block in hourly_blocks:
                    hour_text = hour_block.find('div', class_='text-center font-size-unset px-1 border-bottom').get_text(strip=True)
                    if hour_text == hour:
                        # print('yes??')
                        # Найти температуру в этом блоке
                        rain_block = hour_block.find('div', class_='text-center font-size-unset px-1').find('img')
                        # print(rain_block['src'])
                        temp_block = hour_block.find('div', class_='border-bottom border-top').find('span', class_='text-danger fw-bold')
                        if rain_block:
                            if rain_block['src'] == '/images/09n.png' or rain_block['src'] == '/images/09d.png' or rain_block['src'] == '/images/10n.png' or rain_block['src'] == '/images/10d.png' or rain_block['src'] == '/images/50n.png':
                                # print(1)
                                return 'bad'
                            else:
                                # print(2)
                                return 'good'
                                # print("Oi Oi Huighy 'Omlender done kill me wife and tok me bloody son Womp Womp")
                            # target_temperature = temp_block.get_text(strip=True)
                            # print(target_temperature)
                        break

        
        for date_block in holydate_blocks:
            date_text = date_block.get_text(strip=True)
            # print(date_text)
            # print(date_block)
            if re.search(day_pattern, date_text):   # Проверяем, начинается ли текст с числа дня
                # print('yes')
                target_day_found = True
                # Найти все блоки с температурой для этого дня
                hourly_blocks = date_block.find_parent('div').find_next_sibling().find_all('div', class_='d-inline-block')
                # print(hourly_blocks)
                for hour_block in hourly_blocks:
                    hour_text = hour_block.find('div', class_='text-center font-size-unset px-1 border-bottom').get_text(strip=True)
                    if hour_text == hour:
                        # print('yes??')
                        # Найти температуру в этом блоке
                        rain_block = hour_block.find('div', class_='text-center font-size-unset px-1').find('img')
                        # print(rain_block['src'])
                        temp_block = hour_block.find('div', class_='border-bottom border-top').find('span', class_='text-danger fw-bold')
                        if rain_block:
                            if rain_block['src'] == '/images/09n.png' or rain_block['src'] == '/images/09d.png' or rain_block['src'] == '/images/10n.png' or rain_block['src'] == '/images/10d.png' or rain_block['src'] == '/images/50n.png':
                                # print(11)
                                return 'bad'
                            else:
                                # print(22)
                                return 'good'
                                # print("Oi Oi Huighy 'Omlender done kill me wife and tok me bloody son Womp Womp")
                            # target_temperature = temp_block.get_text(strip=True)
                            # print(target_temperature)
                        break

                if target_temperature:
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
    weather = get_weather_data(data.get('Дата'), data.get('Время'))
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