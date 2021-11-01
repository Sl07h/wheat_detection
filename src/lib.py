import folium
import json
import os
import numpy as np
import pandas as pd
from branca.element import MacroElement
from exif import Image
from jinja2 import Template
from math import sin, cos, tan, radians, sqrt
from numba import njit
from shapely.geometry import Point, Polygon


class BindColormap(MacroElement):
    """Binds a colormap to a given layer.

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    """

    def __init__(self, layer, colormap):
        super(BindColormap, self).__init__()
        self.layer = layer
        self.colormap = colormap
        self._template = Template(u"""
        {% macro script(this, kwargs) %}
            {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
            {{this._parent.get_name()}}.on('overlayadd', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'block';
                }});
            {{this._parent.get_name()}}.on('overlayremove', function (eventLayer) {
                if (eventLayer.layer == {{this.layer.get_name()}}) {
                    {{this.colormap.get_name()}}.svg[0][0].style.display = 'none';
                }});
        {% endmacro %}
        """)


# обёртка, чтобы сократить код
def try_to_make_dir(path):
    try:
        os.mkdir(path)
    except:
        print('Папка {} уже существует'.format(path))


# создаём папки maps, mod, log, tmp, veg
def make_dirs(path_folder):
    try_to_make_dir('maps')
    try_to_make_dir('weights')
    # сжатые и повёрнутые изображения для карты, чтобы разметить делянки вручную
    try_to_make_dir(path_folder + 'mod')
    # готовые метаданные всех изображений в папке и распознанные
    try_to_make_dir(path_folder + 'log')
    # временные csv файлы
    try_to_make_dir(path_folder + 'tmp')
    # индексы вегетации
    try_to_make_dir(path_folder + 'veg')


# преобразуем градусы, минуты и секунды в вешественную переменную координаты
@njit
def convert_to_decimal(degree, min, sec):
    return float(degree) + \
        float(min) / 60.0 + \
        float(sec) / 3600.0


# вращаем точку вокруг другой на заданный угол
@njit
def rotate(p, sin_a, cos_a):
    y, x = p
    X = x*cos_a - y*sin_a
    Y = x*sin_a + y*cos_a
    return X, Y


# проверяем соблюдение протокола съёмки
def check_protocol_correctness(filename, yaw, pitch, roll, height):
    popup_str = 'Изображение: {}<br>рысканье: {}º<br>тангаж: {}º<br>крен: {}<br>высота: {}'.format(
        filename, yaw, pitch, roll, height)
    color_polyline = '#007800'
    is_OK = True

    wrong_parameters = []
    if abs(-90.0 - pitch) >= 2.0:
        wrong_parameters.append('тангаж')
    if abs(0.0 - roll) >= 2.0:
        wrong_parameters.append('крен')
    if height > 3.5:
        wrong_parameters.append('высота')

    # если протокол нарушен, то меняем цвет на красный и выводим ошибки
    if len(wrong_parameters) > 0:
        is_OK = False
        color_polyline = '#780000'
        popup_str_new = '<b>Ошибки:</b>'
        for wrong_parameter in wrong_parameters:
            popup_str_new += '<br>' + wrong_parameter
        popup_str_new += '<br><br>' + popup_str
    else:
        popup_str_new = popup_str

    return is_OK, color_polyline, popup_str_new


# рассчитываем охват земли по ширине и высоте, используя
# относительную высоту полёта и угол обзора
@njit
def calc_image_size(height, fov):
    diag = 2.0 * height * tan(radians(fov / 2.0))
    W = diag * 16.0 / sqrt(337.0)
    H = diag * 9.0 / sqrt(337.0)
    return W, H, diag


# рассчёт границы изображения
# возвращает центр и граничные точки (5 штук, т.к. линия замкнутая)
def calc_image_border(
    latitude,
    longtitude,
    height,
    fov,
    yaw
):
    W, H, diag = calc_image_size(height, fov)
    ratio = 31.0 * cos(radians(latitude))

    center = np.array([latitude, longtitude], dtype=float)
    points = np.array([
        [-H/2.0, -W/2.0],
        [-H/2.0, +W/2.0],
        [+H/2.0, +W/2.0],
        [+H/2.0, -W/2.0],
        [-H/2.0, -W/2.0],
    ], dtype=float)

    new_points = []
    a = radians(-yaw)
    sin_a = sin(a)
    cos_a = cos(a)
    for i, point in enumerate(points):
        X, Y = rotate(point, sin_a, cos_a)
        dx = convert_to_decimal(0.0, 0.0, X / ratio)
        dy = convert_to_decimal(0.0, 0.0, Y / 31.0)
        p = [dy + latitude, dx+longtitude]
        new_points.append(p)
    return W, H, center, new_points


def find_intersections(image_centers, image_borders):
    adjacent_list = []
    for i1, border1 in enumerate(image_borders):
        l = []
        for i2, border2 in enumerate(image_borders):
            has_intersection = False
            for point in border1:
                point = Point(point)
                polygon = Polygon(border2)
                if polygon.contains(point):
                    has_intersection = True
            for point in border2:
                point = Point(point)
                polygon = Polygon(border1)
                if polygon.contains(point):
                    has_intersection = True

            if i1 != i2 and has_intersection == True:
                l.append(i2)
        adjacent_list.append(l)
    return adjacent_list


# отображения корректности сьёмки изображений
def draw_protocol(df_metadata, map):
    feature_group_protocol = folium.FeatureGroup(name='протокол', show=False)    
    for i in range(df_metadata.shape[0]):
        data = df_metadata.loc[i]
        filename    = data['name']
        height      = data['height']
        yaw         = data['yaw']
        pitch       = data['pitch']
        roll        = data['roll']
        border      = data['border']

        is_OK, color_polyline, popup_str = check_protocol_correctness(filename, yaw, pitch, roll, height)
        iframe = folium.IFrame(html=popup_str, width=250, height=180)
        folium.PolyLine(border, color=color_polyline) \
            .add_child(folium.Popup(iframe)) \
            .add_to(feature_group_protocol)
    feature_group_protocol.add_to(map)


# 1. парсим данные через exiftool и библиотеку exif
# 2. проверяем корректность протокола
# 3. рассчитываем координаты области под дроном
# 4. сохраняем всё в 1 файл и возвращаем его
def handle_metadata(filenames, path_field_day):
    # формат полученного датафрейма
    #                is_ok   W    H     lat     long    height   yaw   pitch    roll     border
    # Image_23.jpg   True    5    3    12.34    12.34    2.7     120    -90     0.0   [[], [], [], []]
    # Image_24.jpg   False   12   7    45.67    45.67    6.0     130    -60     0.0   [[], [], [], []]
    # ...
    src_file_count = len(os.listdir(path_field_day + 'src'))
    tmp_file_count = len(os.listdir(path_field_day + 'tmp'))
    print(src_file_count, tmp_file_count)
    if src_file_count != tmp_file_count:
        try:
            for filename in filenames:
                path_img = path_field_day + 'src/' + filename
                path_csv = path_field_day + 'tmp/' + filename[:-4] + '.csv'
                command = 'exiftool-12.34/exiftool -csv {} > {}'.format(path_img, path_csv)
                os.system(command)
                df = pd.read_csv(path_csv, header=None).T
                df.to_csv(path_csv, header=False, index=False)
        except:
            print('Ошибка при выделении данных через exiftool')

    try:
        list_name = []
        list_is_OK = []
        list_W = []
        list_H = []
        list_lat = []
        list_long = []
        list_height = []
        list_yaw = []
        list_pitch = []
        list_roll = []
        list_border = []
    
        for filename in filenames:
            path_img = path_field_day + 'src/' + filename
            path_csv = path_field_day + 'tmp/' + filename[:-4] + '.csv'
            with open(path_img, 'rb') as image_file:
                my_image = Image(image_file)
                latitude = convert_to_decimal(*my_image.gps_latitude)
                longtitude = convert_to_decimal(*my_image.gps_longitude)
            df = pd.read_csv(path_csv, index_col=0).T
            print(filename)
            pitch = float(df['FlightPitchDegree'][0])       # Костыль
            yaw   = float(df['FlightYawDegree'][0])         # Костыль
            roll  = float(df['FlightRollDegree'][0])        # Костыль
            #w = float(df['ImageWidth'][0])
            #h = float(df['ImageHeight'][0])
            height = float(df['RelativeAltitude'][0])
            fov = float(df['FOV'][0].split()[0])
            print(pitch, yaw, roll, height, fov)

            is_OK, _, _ = check_protocol_correctness(filename, yaw, pitch, roll, height)
            W, H, _, border = calc_image_border(latitude, longtitude, height, fov, yaw)
            list_name.append(filename)
            list_is_OK.append(str(is_OK))
            list_W.append(W)
            list_H.append(H)
            list_lat.append(latitude)
            list_long.append(longtitude)
            list_height.append(height)
            list_yaw.append(yaw)
            list_pitch.append(pitch)
            list_roll.append(roll)
            list_border.append(border)

        df_metadata = pd.DataFrame(
            list(zip(
                list_name,
                list_is_OK,
                list_W,
                list_H,
                list_lat,
                list_long,
                list_height,
                list_yaw,
                list_pitch,
                list_roll,
                list_border
            )), columns=[
                'name',
                'is_OK',
                'W',
                'H',
                'lat',
                'long',
                'height',
                'yaw',
                'pitch',
                'roll',
                'border'
            ])
        df_metadata.to_csv(path_field_day + 'log/metadata.csv', index=False)

    except:
        print('Ошибка при формировании датафрейма')

