import re
import ast
import math
import requests
from bs4 import BeautifulSoup
from ai.post_edit import post_edit
from ai.predict import predict_place
from datetime import datetime, timedelta
from clean_4 import  make_txt_mask_of_radius
from ai.make_map_screen import save_map_image
from make_mask_of_radius import make_radius_mask
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

def round_to_nearest_interval(hours):
    if not (0 <= hours <= 24):
        raise ValueError("Число должно быть в диапазоне от 0 до 24")
    
    return (hours // 6) * 6

ZOOM_LEVELS = {
    (0, 1): 3.76,
    (1, 3): 1.88, 
    (3, 6): 0.94, 
    (6, 13): 0.47,
    (13, 26): 0.235,
    (26, 53): 0.1175,
    (53, 106): 0.05875,
    (106, float('inf')): 0.029375,
}

def get_zoom_factor(radius):
    for bounds, factor in ZOOM_LEVELS.items():
        if bounds[0] <= radius < bounds[1]:
            return factor
    return 0.029375

def calculate_pixels_per_centimeter(resolution_x, resolution_y, screen_diagonal_inch):
    # Вычисляем разрешение по диагонали
    diagonal_resolution = (resolution_x ** 2 + resolution_y ** 2) ** 0.5
    
    # Вычисляем PPI (pixels per inch)
    ppi = diagonal_resolution / screen_diagonal_inch
    
    # Конвертируем PPI в пиксели на сантиметр
    pixels_per_centimeter = ppi / 2.54
    
    return pixels_per_centimeter


def make_real_radius(coords_psr, prev_radius, radius):
    coords_psr = (coords_psr['latitude'], coords_psr['longitude'])
    prev_radius.append(radius) 
    frames = []
    real_radius = []

    for i, r in enumerate(prev_radius):
        r_ceil = math.ceil(r)
        image_filename = f'map_image{i}.png'
        save_map_image(r_ceil, coords_psr, image_filename)
        frames.append(image_filename)

    predict_place(frames)

    try:
        for i, frame in enumerate(frames):
            masked_frame = f'masked_{frame}'
            post_edit(masked_frame)

            zoom_factor = get_zoom_factor(prev_radius[i])

            pixels_per_cm = calculate_pixels_per_centimeter(1920, 1080, 14)

            radius_in_pixel = int(prev_radius[i] * zoom_factor * pixels_per_cm)

            result, center = make_radius_mask(masked_frame, radius_in_pixel)
            make_txt_mask_of_radius(masked_frame, coords_psr, prev_radius[i], result, center)

    except Exception as e:
        print(f"Произошла ошибка: {e}")
    
    for f in frames:
        f = f.split('.')[0]
        with open(f'templates/masked_{f}.txt', 'r') as data:
                radius_string = data.read()
                real_radius.append(ast.literal_eval(radius_string))
    
    return real_radius

def get_weather_data(date, time):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    day, month, year = date.split('.')
    hour = time.split(':')[0]

    url = f"https://arhivpogodi.ru/arhiv/sankt-peterburg/{year}/{month}/"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Ошибка при выполнении запроса. Код состояния:", response.status_code)
        return

    soup = BeautifulSoup(response.text, "html.parser")
    day_pattern = rf',\s*({int(day)})\b'

    date_blocks = soup.find_all('div', class_='font-size-unset d-inline-block position-sticky px-3 pb-2')
    holydate_blocks = soup.find_all('div', class_='font-size-unset d-inline-block position-sticky px-3 pb-2 text-danger')

    weather = parse_weather(date_blocks, day_pattern, hour) or parse_weather(holydate_blocks, day_pattern, hour)

    if weather:
        return weather
    else:
        print(f"Информация о погоде на {day} число не найдена.")

def parse_weather(date_blocks, day_pattern, hour):
    for date_block in date_blocks:
        date_text = date_block.get_text(strip=True)
        if re.search(day_pattern, date_text):
            hourly_blocks = date_block.find_parent('div').find_next_sibling().find_all('div', class_='d-inline-block')
            for hour_block in hourly_blocks:
                hour_text = hour_block.find('div', class_='text-center font-size-unset px-1 border-bottom').get_text(strip=True)
                if hour_text == hour:
                    rain_block = hour_block.find('div', class_='text-center font-size-unset px-1').find('img')
                    if rain_block:
                        if rain_block['src'] in ['/images/09n.png', '/images/09d.png', '/images/10n.png', '/images/10d.png', '/images/50n.png']:
                            return 'bad'
                        return 'good'
    return None

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

    if age < 12:
        probabilities['остаться на месте'] += 0.5
        probabilities['искать укрытие'] += 0.2
    elif 12 <= age < 18:
        probabilities['двигаться без ориентирования'] += 0.4
    elif age >= 60:
        probabilities['остаться на месте'] += 0.3
        probabilities['искать укрытие'] += 0.3

    if gender == 'female':
        probabilities['искать укрытие'] += 0.2
        probabilities['остаться на месте'] += 0.2
    elif gender == 'male':
        probabilities['двигаться с ориентированием'] += 0.2
        probabilities['двигаться без ориентирования'] += 0.2

    total_probability = sum(probabilities.values())

    for key in probabilities:
        probabilities[key] /= total_probability

    return probabilities, weather

def predict_behavior(data):
    probabilities, weather = calculate_probability(data)
    probabilities_str = "\n".join([f"{behavior}: {prob * 100:.2f}%" for behavior, prob in probabilities.items()])
    return probabilities_str, weather

def get_behavior_data(data, current_time, current_date, bad_mentality=0):
    if 6 <= current_time < 12:
        times_of_day = "morning"
    elif 12 <= current_time < 18:
        times_of_day = "day"
    elif 18 <= current_time < 24:
        times_of_day = "evening"
    else:
        times_of_day = "night"

    if 0 <= current_time < 10:
        current_time = f"0{current_time}"

    mentality = str(data.get("mental_condition")) 
    if bad_mentality == 1 and mentality == "stable": 
        mentality = "unstable"

    data_behavior = {
        "Возраст": int(data.get("age")),
        "Пол": str(data.get("gender")),
        "Физическое состояние": str(data.get("physical_condition")),
        "Психическое состояние": mentality,
        "Опыт нахождения в дикой природе": str(data.get("experience")),
        "Знание местности": str(data.get("local_knowledge")),
        "Наличие телефона": str(data.get("phone")),
        "Время суток": times_of_day,
        "Моральные обязательства": str(data.get('moral_obligations')),
        "Внешние сигналы": str(data.get('external_signals')),
        "Дата": current_date,
        "Время": f"{current_time}:00"
    }

    return data_behavior, times_of_day

def get_behavior_coefficient(behavior_data):
    behavior_coefficient = 1.0

    percentages = re.findall(r"(\d+\.\d+)%", behavior_data)

    percentages = [float(p) for p in percentages]
    max_percentage = max(percentages)

    pattern = rf"([^\d]+)\s*{max_percentage:.2f}%"
    match = re.search(pattern, behavior_data)

    result_text = match.group(1).strip()

    if result_text != "остаться на месте:":
        result_text = result_text.split('%')[1].split('\n')[1]

    if result_text == "остаться на месте:":
        behavior_coefficient = 0.0
    elif result_text == "искать укрытие:":
        behavior_coefficient = 0.2

    return behavior_coefficient, result_text.capitalize() + ' ' + str(max_percentage) + '%'

def calculate_last_day(data,time_passed, hours, normal_speed, speed_index, total_radius, list_of_radius, day):
    current_date = data.get('date_of_finding')
    behavior_context, time = get_behavior_data(data, time_passed, current_date)

    behavior_data, weather = predict_behavior(behavior_context)
    behavior_coefficient, behavior_main = get_behavior_coefficient(behavior_data)

    interval_radius = (
        hours
        * normal_speed
        * speed_index
        * behavior_coefficient
    )
    total_radius += interval_radius

    if time_passed == 0:
        list_of_radius += f"День {day}: "

    list_of_radius += " ".join([str(round(interval_radius, 2)), str(behavior_main), str(weather), str(time)]) + " "
    
    return total_radius, list_of_radius

def get_radius(data, age, hours_elapsed, terrain_passability=None, path_curvature=None, slope_degree=None, fatigue_level=None, time_of_day=None, weather_conditions=None, group_factor=None):
    normal_speed = 5

    if age < 18:
        normal_speed = 4
    elif age >= 60:
        normal_speed = 3

    terrain_passability_coefficient = terrain_passability if terrain_passability is not None else 1.0
    path_curvature_coefficient = path_curvature if path_curvature is not None else 1.0
    slope_degree_coefficient = slope_degree if slope_degree is not None else 1.0
    fatigue_level_coefficient = fatigue_level if fatigue_level is not None else 1.0
    time_of_day_coefficient = time_of_day if time_of_day is not None else 1.0
    weather_conditions_coefficient = weather_conditions if weather_conditions is not None else 1.0
    group_factor_coefficient = group_factor if group_factor is not None else 1.0

    speed_index = (
        terrain_passability_coefficient
        * path_curvature_coefficient
        * slope_degree_coefficient
        * fatigue_level_coefficient
        * time_of_day_coefficient
        * weather_conditions_coefficient
        * group_factor_coefficient
    )

    list_of_radius = ""
    total_radius = 0
    current_date = data.get("date_of_loss")

    hours_of_loss, minutes_of_loss = map(int, data.get("time_of_loss").split(":"))
    hour_of_finding, minutes_of_finding = map(int, data.get("time_of_finding").split(":"))

    time_of_loss_total = hours_of_loss + minutes_of_loss / 60.0
    time_of_finding_total = hour_of_finding + minutes_of_finding / 60.0

    time_passed = round_to_nearest_interval(hours_of_loss)
    result = time_of_loss_total - time_passed

    day = 1
    previous_radius = []
    first_day=True 

    for i in range(0, hours_elapsed, 6):
        if(time_passed % 24 == 0 and time_passed != 0):
            current_date = (datetime.strptime(current_date, "%d.%m.%Y") + timedelta(days=1)).strftime("%d.%m.%Y")
            time_passed = 0
            day += 1
            previous_radius.append(total_radius)
        
        if day == 3: 
            behavior_context, time = get_behavior_data(data, time_passed, current_date, 1) 
        else: 
            behavior_context, time = get_behavior_data(data, time_passed, current_date)

        behavior_data, weather = predict_behavior(behavior_context)
        behavior_coefficient, behavior_main = get_behavior_coefficient(behavior_data)

        interval_hours = min(6, hours_elapsed - i) 
        if interval_hours != 6:
            continue

        if first_day:
            interval_hours = 6 - result
            behavior_coefficient, behavior_main = 1, "Двигаться c ориентированием: 100.0%"

        interval_radius = (
            interval_hours
            * normal_speed
            * speed_index
            * behavior_coefficient
        )
        total_radius += interval_radius

        if(time_passed==0 or first_day):
            list_of_radius += f'День {day}: '
            first_day=False

        list_of_radius += " ".join([str(round(interval_radius, 2)), str(behavior_main), str(weather), str(time)]) + " "
        time_passed += 6

    if time_passed == 24:
        time_passed = 0
        day += 1

    total_radius, list_of_radius = calculate_last_day(data, time_passed, -time_passed + time_of_finding_total, normal_speed, speed_index, total_radius, list_of_radius, day)
    
    return total_radius, list_of_radius, previous_radius

@app.route("/")
def index():
    return render_template('base.html')

@app.route('/radius', methods=['POST'])
def radius():
    data = request.get_json()
    try:
        coordinates_psr = data.get('coordinates_psr')
        coordinates_finding = data.get('coordinates_finding')

        date_of_loss_str = data.get('date_of_loss')
        time_of_loss_str = data.get('time_of_loss', '00:00')

        date_of_finding_str = data.get('date_of_finding')
        time_of_finding_str = data.get('time_of_finding', '00:00')

        date_time_of_loss_str = f"{date_of_loss_str} {time_of_loss_str}"
        date_time_of_finding_str = f"{date_of_finding_str} {time_of_finding_str}"
        
        date_time_of_loss = datetime.strptime(date_time_of_loss_str, '%d.%m.%Y %H:%M')
        date_time_of_finding = datetime.strptime(date_time_of_finding_str, '%d.%m.%Y %H:%M')

        hours_difference = (date_time_of_finding - date_time_of_loss).total_seconds() // 3600
    
        radius, extra_info, previous_radius = get_radius(data, int(data.get('age')), int(hours_difference))

        behavior_context = {
            'Возраст': int(data.get('age')),
            'Пол': str(data.get('gender')),
            'Физическое состояние': str(data.get('physical_condition')),
            'Психическое состояние': str(data.get('mental_condition')),
            'Опыт нахождения в дикой природе': str(data.get('experience')),
            'Знание местности': str(data.get('local_knowledge')),
            'Наличие телефона': str(data.get('phone')),
            'Время суток': extra_info.split()[-1],
            'Моральные обязательства': str(data.get('moral_obligations')),
            'Внешние сигналы': str(data.get('external_signals')),
            'Дата': data.get('date_of_finding'),
            'Время': time_of_finding_str
        }

        behavior, _ = predict_behavior(behavior_context)
        real_radius = make_real_radius(coordinates_psr, previous_radius, radius)

        return jsonify({
            'status': 'success',
            'radius': radius,
            'coordinates_psr': coordinates_psr,
            'coordinates_finding': coordinates_finding,
            'behavior': behavior,
            'extra_info': extra_info,
            'previous_radius': previous_radius,
            'real_radius': real_radius
        })
    
    except ValueError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Не удалось обработать запрос'}), 500

if __name__ == '__main__':
    app.run(debug=True)