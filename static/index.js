function formatDate(date) {
    const day = String(date.getDate()).padStart(2, '0');
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const year = date.getFullYear();
    return `${day}.${month}.${year}`;
}

const currentDate = new Date();
const formattedDate = formatDate(currentDate);
document.getElementById('date_of_finding').value = formattedDate;

function parseCoordinate(coord) {
    return parseFloat(coord.replace(',', '.'));
}

const hours = String(currentDate.getHours()).padStart(2, '0');
const minutes = String(currentDate.getMinutes()).padStart(2, '0');
const currentTime = `${hours}:${minutes}`;
document.getElementById('time_of_finding').value = currentTime;

$(document).ready(function () {
    let mapCreated = false;
    let map;

    $('#toggleAdditionalFields').click(function () {
        $('#additional_fields').toggleClass('hidden-input');
        if (!$('#additional_fields').hasClass('hidden-input')) {
            $('#myTab .nav-link').first().tab('show');
        }
    });

    $('#tab2 input[type="number"]').on('input', function () {
        var value = parseFloat($(this).val());
        if (value < 0 || value > 1) {
            alert('Коэффициент должен быть в пределах от 0 до 1');
            $(this).val('');
        }
    });

    $('#dataForm').on('submit', function (e) {
        e.preventDefault();
        
        $('#overlay').show();
        $('#loader').show();

        var data = {
            date_of_loss: $('#date_of_loss').val(),
            time_of_loss: $('#time_of_loss').val(),
            date_of_finding: $('#date_of_finding').val(),
            time_of_finding: $('#time_of_finding').val(),
            age: $('#age').val(),
            gender: $('#gender').val(),
            physical_condition: $('#physical_condition').val(),
            mental_condition: $('#mental_condition').val(),
            experience: $('#experience').val() || null,
            local_knowledge: $('#local_knowledge').val() || null,
            phone: $('#phone').val() || null,
            terrain_passability: $('#terrain_passability').val() || null,
            path_curvature: $('#path_curvature').val() || null,
            slope_angle: $('#slope_angle').val() || null,
            coordinates_psr: {
                latitude: parseCoordinate($('#psr_lat').val()),
                longitude: parseCoordinate($('#psr_lon').val())
            },
            coordinates_finding: {
                latitude: parseCoordinate($('#finding_lat').val()),
                longitude: parseCoordinate($('#finding_lon').val())
            }
        };

        $.ajax({
            url: '/radius',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(data),
            success: function (response) {
                if (response && response.coordinates_psr) {
                    if (!mapCreated) {
                        map = L.map('map').setView([response.coordinates_psr.latitude, response.coordinates_psr.longitude], 13);
                        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                            maxZoom: 18,
                            attribution: 'Map data © <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                        }).addTo(map);
                        mapCreated = true;
                    }

                    map.eachLayer(function (layer) {
                        if (layer instanceof L.Marker || layer instanceof L.Circle) {
                            map.removeLayer(layer);
                        }
                    });
                    
                    L.marker([response.coordinates_psr.latitude, response.coordinates_psr.longitude]).addTo(map)
                        .bindTooltip('Центр ПСР')
                        .openTooltip();
                    
                    var circles = [];
                    var radiusInMeters = response.radius * 1000;
                    var circle = L.circle([response.coordinates_psr.latitude, response.coordinates_psr.longitude], {
                        color: 'blue',
                        fillColor: '#add8e6',
                        fillOpacity: 0.5,
                        radius: radiusInMeters
                    }).addTo(map);

                    circle.bindTooltip("Общий радиус: " + response.radius + " км", {
                        permanent: false,
                        direction: 'top',
                        className: 'circle-tooltip'
                    });
                    circles.push(circle);


                    for (var i = response.previous_radius.length - 1; i >= 0; i--) {
                        var radius = response.previous_radius[i];
                        var radiusInMeters = radius * 1000;

                        var circle = L.circle([response.coordinates_psr.latitude, response.coordinates_psr.longitude], {
                            color: 'blue',
                            fillColor: '#add8e6',
                            fillOpacity: 0,
                            radius: radiusInMeters
                        }).addTo(map);

                        circle.bindTooltip("День " + parseInt(i+1, 10) + " радиус: " + radius + " км", {
                            permanent: false,
                            direction: 'top',
                            className: 'circle-tooltip'
                        });
                        circles.push(circle);
                    }

                    if (response.coordinates_finding) {
                        L.marker([response.coordinates_finding.latitude, response.coordinates_finding.longitude]).addTo(map)
                            .bindTooltip('Место нахождения')
                            .openTooltip();
                    } else {
                        alert('Не удалось получить координаты нахождения.');
                    }
                    document.getElementById('result').classList.add('alert', 'alert-info');
                    document.getElementById('result').innerHTML = "Вероятности поведения потерявшегося на текущий момент:<br>";
                    document.getElementById('result').innerHTML += response.behavior.replace(/%/g, '%  <br>');


                    $('#loader').hide();
                    $('#overlay').hide();
                    function generateReportFromText(text) {
                        const container = document.getElementById('report');
                        container.innerHTML = ''; 
                        
                        const days = text.split(/(?=День \d+:)/g);
                        
                        const timeOfLossInput = document.getElementById('time_of_loss');
                        const timeOfLossInitial = timeOfLossInput.value;

                        const timeOfFindingInput = document.getElementById('time_of_finding');
                        const timeOfFindingInitial = timeOfFindingInput.value;
                        
                        days.forEach((day, dayIndex) => {
                        
                        const dayMatch = day.match(/День (\d+):/);
                        if (dayMatch) {
                            const dayNumber = dayMatch[1];
                            const dayDiv = document.createElement('div');
                            dayDiv.className = 'day';
                            dayDiv.innerHTML = `<strong>День ${dayNumber}:</strong>`;

                            const activities = day.replace(/День \d+: /, '').trim().match(/\d+\.\d+ [^:]+: \d+\.\d+% [^ ]+ (morning|day|evening|night)/g);

                            if (activities) {
                                activities.forEach((activity, activityIndex) => {
                                    if (activity.trim() !== '') {
                                        const statusMatch = activity.match(/(\bgood\b|\bbad\b)/);
                                        let status = statusMatch ? statusMatch[0] : 'neutral';

                                        if (status === 'good') {
                                            weather = 'Погодные условия: хорошие';
                                        } else if (status === 'bad') {
                                            weather = 'Погодные условия: плохие';
                                        } else {
                                            weather = 'Погодные условия: нейтральные';
                                        }

                                        const timeOfDayMatch = activity.match(/\b(morning|day|evening|night)\b/);
                                        let timeOfDay = timeOfDayMatch ? timeOfDayMatch[0] : 'day';

                                        let timeRange;
                                        if (dayIndex === 0 && activityIndex === 0) {
                                            switch (timeOfDay) {
                                                case 'morning':
                                                    timeRange = `с ${timeOfLossInitial} до 11:59`;
                                                    break;
                                                case 'day':
                                                    timeRange = `с ${timeOfLossInitial} до 17:59`;
                                                    break;
                                                case 'evening':
                                                    timeRange = `с ${timeOfLossInitial} до 23:59`;
                                                    break;
                                                case 'night':
                                                    timeRange = `с ${timeOfLossInitial} до 05:59`;
                                                    break;
                                            }
                                        } else if (dayIndex === days.length - 1 && activityIndex === activities.length - 1) {
                                            switch (timeOfDay) {
                                                case 'morning':
                                                    timeRange = `с 06:00 до ${timeOfFindingInitial}`;
                                                    break;
                                                case 'day':
                                                    timeRange = `с 12:00 до ${timeOfFindingInitial}`;
                                                    break;
                                                case 'evening':
                                                    timeRange = `с 18:00 до ${timeOfFindingInitial}`;
                                                    break;
                                                case 'night':
                                                    timeRange = `с 00:00 до ${timeOfFindingInitial}`;
                                                    break;
                                            }
                                        } else {
                                            switch (timeOfDay) {
                                                case 'morning':
                                                    timeRange = 'с 06:00 до 11:59';
                                                    break;
                                                case 'day':
                                                    timeRange = 'с 12:00 до 17:59';
                                                    break;
                                                case 'evening':
                                                    timeRange = 'с 18:00 до 23:59';
                                                    break;
                                                case 'night':
                                                    timeRange = 'с 00:00 до 05:59';
                                                    break;
                                            }
                                        }

                                        if (timeOfDay === 'morning') {
                                            timeOfDay = 'Утро';
                                        } else if (timeOfDay === 'day') {
                                            timeOfDay = 'День';
                                        } else if (timeOfDay === 'evening') {
                                            timeOfDay = 'Вечер';
                                        } else {
                                            timeOfDay = 'Ночь';
                                        }
                                        
                                        activity = activity.split(' ')[0] + "км, " + activity.split(' ').slice(1, -2).join(' ');
                                        const activityHtml = `<div class="activity ${status}">${timeOfDay} ${timeRange}: ${activity.trim()}, ${weather}</div>`;
                                        dayDiv.innerHTML += activityHtml;
                                    }
                                });
                            }

                            container.appendChild(dayDiv);
                        }
                    });

                    }
                    generateReportFromText(response.extra_info.replace(/  /g, ' <br>'));
                    
                } else {
                    $('#overlay').hide();
                    $('#loader').hide();
                    alert('Не удалось получить координаты.');
                }
            },
            error: function (xhr, status, error) {
                console.error('Error:', status, error);
                $('#overlay').hide();
                $('#loader').hide();
                alert('Ошибка при отправке данных.');
            }
        });
    });
});
