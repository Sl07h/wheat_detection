import folium
import json
import cv2
import os
import sys
import numpy as np
import pandas as pd
from branca.element import MacroElement
from exif import Image
from jinja2 import Template
from math import sin, cos, tan, radians, sqrt
from numba import njit
from shapely.geometry import Point, Polygon


class WheatDetectionSystem():
    # public methods
    def __init__(
        self,
        field: str,                     # 'Field_2021'
        attempt: str,                   # '1'
        cnn_model: str,                 # 'frcnn'
        kernel_size: str,               # '512'
        do_show_uncorrect: bool = True,  # True or False
        activation_treshold: int = 0.7,  # 0.7
    ):
        self.path_field_day = f'data/{field}/{attempt}'
        self.path_log_metadata = f'data/{field}/{attempt}/log/metadata.csv'
        self.path_log_bboxes = f'data/{field}/{attempt}/log/{field}.{attempt}.{cnn_model}.{kernel_size}.csv'
        self.path_log_plots = f'data/{field}/{attempt}/log/wheat_plots_result.{field}.{attempt}.{cnn_model}.{kernel_size}.csv'
        self.path_to_geojson = f'data/{field}/wheat_plots.geojson'
        self.do_show_uncorrect = do_show_uncorrect
        self.activation_treshold = activation_treshold
        self.filenames = os.listdir(f'{self.path_field_day}/src')
        self.wheat_ears = []
        self.layers = []
        self.colormaps = []
        self._read_metadata()
        self._create_map()

    def draw_wheat_plots(self):
        ''' отрисовка числа колосьев на каждой делянке '''
        feature_group_choropleth = folium.FeatureGroup(
            name='фоновая картограмма', show=True)
        df = pd.DataFrame(columns=['сорт', 'количество колосьев'])
        with open(self.path_to_geojson) as f:
            data = json.load(f)
            num_of_polygons = len(data['features'])

            max_p = np.amax(self.ears_in_polygons)
            for i in range(num_of_polygons):
                # max_p:  val = 1  ->  (0,255,0)
                # min_p:  val = 0  ->  (221,221,221)
                val = self.ears_in_polygons[i] / max_p
                hex_val = hex(int((1 - val) * 221))[2:]
                if len(hex_val) == 1:
                    hex_val = '0'+hex_val
                color = f'#{hex_val}dd{hex_val}'

                t = np.array(data['features'][i]['geometry']['coordinates'][0])
                t[:, [0, 1]] = t[:, [1, 0]]

                str_wheat_type = ''
                try:
                    str_wheat_type = data['features'][i]['properties']['name'] + '\n'
                except:
                    pass

                folium.Polygon(t,
                               color='#303030',
                               opacity=0.05,
                               fill=True,
                               fill_color=color,
                               fill_opacity=1.0
                               )\
                    .add_child(folium.Popup(str_wheat_type + str(self.ears_in_polygons[i]))) \
                    .add_to(feature_group_choropleth)
                df.append(
                    {'сорт': str_wheat_type, 'количество колосьев': self.ears_in_polygons[i]}, ignore_index=True)

        colormap = folium.LinearColormap(
            ['#dddddd', '#00ff00'], vmin=0, vmax=max_p).to_step(5)
        colormap.caption = 'количество колосьев на делянках, шт'

        self.layers.append(feature_group_choropleth)
        self.colormaps.append(colormap)
        df.to_csv(self.path_log_plots)

    def draw_protocol(self):
        ''' отображаем корректность протокола сьёмки изображений '''
        feature_group_protocol = folium.FeatureGroup(
            name='протокол', show=False)
        for i in range(self.df_metadata.shape[0]):
            line = self.df_metadata.loc[i]
            filename = line['name']
            height = line['height']
            yaw = line['yaw']
            pitch = line['pitch']
            roll = line['roll']
            border = line['border']
            _, color_polyline, popup_str = check_protocol_correctness(
                filename, yaw, pitch, roll, height)
            iframe = folium.IFrame(html=popup_str, width=250, height=180)
            folium.PolyLine(border, color=color_polyline) \
                  .add_child(folium.Popup(iframe)) \
                  .add_to(feature_group_protocol)
        feature_group_protocol.add_to(self.m)

    def draw_grids(self, size_list):
        for size in size_list:
            self._draw_grid(size)

    def save_map(self):
        ''' соединяем слои, карты цветов и сохраняем карту '''
        for layer in self.layers:
            self.m.add_child(layer)
        for colormap in self.colormaps:
            self.m.add_child(colormap)

        for layer, colormap in zip(self.layers, self.colormaps):
            self.m.add_child(BindColormap(layer, colormap))

        # self.m.add_child(folium.map.MeasureControl(collapsed=False))
        self.m.add_child(folium.map.LayerControl(collapsed=False))

        field_name = self.path_field_day.split('/')[1]
        self.m.save(f'maps/{field_name}.html')

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    def _read_metadata(self):
        ''' считываем метаданные и сохраняем промежуточные значения '''
        make_dirs(self.path_field_day)
        if not os.path.exists(self.path_log_metadata):
            handle_metadata(self.filenames, self.path_field_day)
        # считываем метаданные и координаты колосков
        self.df_bboxes = pd.read_csv(self.path_log_bboxes)
        self.df_metadata = pd.read_csv(self.path_log_metadata)
        self.df_metadata['border'] = self.df_metadata['border'].apply(
            lambda x: json.loads(x))
        self.latitude = float(self.df_metadata['lat'][0])
        self.longtitude = float(self.df_metadata['long'][0])
        # https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
        self.image_centers = list(self.df_metadata.apply(
            lambda x: [x.lat, x.long], axis=1))
        self.image_borders = list(self.df_metadata['border'])
        lat = np.array(self.image_borders).flatten()[::2]
        long = np.array(self.image_borders).flatten()[1::2]
        self.lat_min = lat.min()
        self.lat_max = lat.max()
        self.long_min = long.min()
        self.long_max = long.max()
        self.adjacent_frames = find_intersections(
            self.image_centers, self.image_borders)

        for i in range(len(self.filenames)):
            self.wheat_ears += self._calc_wheat_intersections(i)

        self.ears_in_polygons = self._calc_wheat_head_count_in_geojsons()

    def _create_map(self):
        ''' создаём карту и сохраняем объект карты как об приватное поле  '''
        self.m = folium.Map(
            [self.latitude, self.longtitude],
            tiles=None,
            prefer_canvas=True,
            control_scale=True,
            zoom_start=21
        )
        base_map = folium.FeatureGroup(
            name='Basemap', overlay=True, control=False)
        folium.TileLayer(tiles='OpenStreetMap', max_zoom=23).add_to(base_map)
        base_map.add_to(self.m)

    def _draw_grid(self, grid_size_m):
        ''' отрисовываем квадратную сетку grid_size_m * grid_size_m '''
        feature_group_grid = folium.map.FeatureGroup(
            name=f'сетка {grid_size_m:.1f}x{grid_size_m:.1f}м²',
            # overlay=False,
            show=False
        )

        ratio = 31.0 * cos(radians(self.latitude))
        d_lat = convert_to_decimal(0.0, 0.0, grid_size_m / 31.0)
        d_long = convert_to_decimal(0.0, 0.0, grid_size_m / ratio)
        delta_lat = self.lat_max - self.lat_min
        delta_long = self.long_max - self.long_min
        n_lat = int(delta_lat / d_lat)
        n_long = int(delta_long / d_long)

        wheat_counts = np.zeros((n_lat+1, n_long+1))
        coordinates = np.ndarray((n_lat+1, n_long+1, 4, 2))
        for wheat_ear in self.wheat_ears:
            _lat, _long, p = wheat_ear
            i = int((_lat - self.lat_min) / d_lat)
            j = int((_long - self.long_min) / d_long)
            wheat_counts[i][j] += p

        max_p = np.amax(wheat_counts)
        print(max_p)

        max_p /= grid_size_m*grid_size_m
        lat_cut = grid_size_m * ((delta_lat - n_lat*d_lat) / d_lat)
        long_cut = grid_size_m * ((delta_long - n_long*d_long) / d_long)
        for i in range(n_lat):
            j = n_long
            wheat_counts[i][j] /= (grid_size_m*lat_cut)

        for j in range(n_long):
            i = n_lat
            wheat_counts[i][j] /= (grid_size_m*long_cut)

        wheat_counts[n_lat][n_long] /= (lat_cut*long_cut)

        for i in range(n_lat):
            for j in range(n_long):
                if wheat_counts[i][j] > 0:
                    wheat_counts[i][j] /= (grid_size_m**2)

        # divide count of objects by region area
        for i in range(n_lat):
            for j in range(n_long):
                if wheat_counts[i][j] > 0:
                    lat_b = self.lat_min + i*d_lat
                    lat_e = lat_b + d_lat
                    long_b = self.long_min + j*d_long
                    long_e = long_b + d_long
                    coordinates[i][j] = np.array([
                        [lat_b, long_b],
                        [lat_b, long_e],
                        [lat_e, long_e],
                        [lat_e, long_b]
                    ])

                    # max_p:  val = 1  ->  (0,255,0)
                    # min_p:  val = 0  ->  (221,221,221)
                    val = wheat_counts[i][j] / max_p
                    hex_val = hex(int((1 - val) * 221))[2:]
                    if len(hex_val) == 1:
                        hex_val = '0'+hex_val
                    color = f'#{hex_val}dd{hex_val}'

                    folium.Rectangle(coordinates[i][j],
                                     color='#303030',
                                     opacity=0.05,
                                     fill=True,
                                     fill_color=color,
                                     fill_opacity=1.0
                                     )\
                        .add_child(folium.Popup(str(wheat_counts[i][j]))) \
                        .add_to(feature_group_grid)

        colormap = folium.LinearColormap(
            ['#dddddd', '#00ff00'], vmin=0, vmax=max_p).to_step(5)
        colormap.caption = 'плотность колосьев, шт/м²'
        self.layers.append(feature_group_grid)
        self.colormaps.append(colormap)

    def _read_logs(self, df, i):
        l = df.iloc[i].values[1]
        l = l.split(' ')
        k = len(l) // 5
        coords = []
        for i in range(k):
            j = i*5
            p = float(l[j])
            if p > self.activation_treshold:
                lat = int(float(l[j+1])) + int(float(l[j+3])) // 2
                lon = int(float(l[j+2])) + int(float(l[j+4])) // 2
                coords.append([lat, lon, p])
        return coords

    def _calc_wheat_intersections(self, df_i):
        ''' поворачиваем колоски по азимуту и нормируем, поподающие в несколько изображений '''
        data = self.df_metadata.loc[df_i]
        is_OK = data['is_OK']
        W = data['W']
        H = data['H']
        latitude = data['lat']
        longtitude = data['long']
        yaw = data['yaw']

        ratio = 31.0 * cos(radians(latitude))
        a = radians(-yaw)
        sin_a = sin(a)
        cos_a = cos(a)
        W = W / 2
        H = H / 2
        tmp_im = cv2.imread(f'{self.path_field_day}/src/{self.filenames[0]}')
        H_pixels, W_pixels, _ = tmp_im.shape
        H_pixels /= 2
        W_pixels /= 2

        if is_OK or self.do_show_uncorrect:
            bboxes = self._read_logs(self.df_bboxes, df_i)
            for i in range(len(bboxes)):
                x, y, p = bboxes[i]
                x = (x - W_pixels) / W_pixels * W
                y = (1.0 - (y - H_pixels)) / H_pixels * H
                point = np.array([y, x], dtype=float)
                X, Y = rotate(point, sin_a, cos_a)
                dx = convert_to_decimal(0.0, 0.0, X / ratio)
                dy = convert_to_decimal(0.0, 0.0, Y / 31.0)
                y = dy + latitude
                x = dx + longtitude
                point = Point(y, x)
                w = 1.0
                for j in self.adjacent_frames[df_i]:
                    polygon = Polygon(np.array(self.image_borders[j]))
                    if polygon.contains(point):
                        w += 1.0
                bboxes[i][0] = y
                bboxes[i][1] = x
                bboxes[i][2] = 1.0 / w
            return bboxes
        else:
            return []

    def _calc_wheat_head_count_in_geojsons(self):
        ''' считаем сколько колосков в каждом полигоне, размеченном агрономом 
        ВНИМАНИЕ в geojson координаты в другом порядке
        '''
        with open(self.path_to_geojson) as f:
            data = json.load(f)
            num_of_polygons = len(data['features'])
            ears_in_polygons = np.ones(num_of_polygons)
            for i in range(num_of_polygons):
                t = np.array(data['features'][i]['geometry']
                             ['coordinates'][0][:4])
                # docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
                t[:, [0, 1]] = t[:, [1, 0]]
                wheat_plot_polygon = Polygon(t)

                tmp = t.T
                lat_min = tmp[0].min()
                lat_max = tmp[0].max()
                long_min = tmp[1].min()
                long_max = tmp[1].max()

                wheat_ears_candidates = list(filter(lambda x: (
                    lat_min < x[0] and x[0] < lat_max and
                    long_min < x[1] and x[1] < long_max), self.wheat_ears))

                for wheat_ear in wheat_ears_candidates:
                    point = Point(np.array(wheat_ear[:2]))
                    if wheat_plot_polygon.contains(point):
                        ears_in_polygons[i] += wheat_ear[2]
        return ears_in_polygons


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


# создаём папки maps, weights, mod, log, tmp, veg
def make_dirs(path_folder):
    try_to_make_dir('maps')
    try_to_make_dir('weights')
    # сжатые и повёрнутые изображения для карты, чтобы разметить делянки вручную
    try_to_make_dir(f'{path_folder}/mod')
    # готовые метаданные всех изображений в папке и распознанные
    try_to_make_dir(f'{path_folder}/log')
    # временные csv файлы
    try_to_make_dir(f'{path_folder}/tmp')
    # индексы вегетации
    try_to_make_dir(f'{path_folder}/veg')


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


# 1. парсим данные через exiftool и библиотеку exif
# 2. проверяем корректность протокола
# 3. рассчитываем координаты области под дроном
# 4. сохраняем всё в 1 файл и возвращаем его
def handle_metadata(filenames, path_field_day):
    # формат полученного датафрейма
    # name           is_ok   W    H     lat     long    height   yaw   pitch    roll     border
    # Image_23.jpg   True    5    3    12.34    12.34    2.7     120    -90     0.0   [[], [], [], []]
    # Image_24.jpg   False   12   7    45.67    45.67    6.0     130    -60     0.0   [[], [], [], []]
    # ...
    src_file_count = len(os.listdir(f'{path_field_day}/src'))
    tmp_file_count = len(os.listdir(f'{path_field_day}/tmp'))
    print(src_file_count, tmp_file_count)

    exiftool_script_name = 'exiftool'
    if sys.platform == 'win32':
        exiftool_script_name = 'windows_exiftool'

    if src_file_count != tmp_file_count:
        try:
            for filename in filenames:
                path_img = f'{path_field_day}/src/{filename}'
                path_csv = f'{path_field_day}/tmp/{filename[:-4]}.csv'
                command = 'exiftool-12.34/{} -csv {} > {}'.format(
                    exiftool_script_name, path_img, path_csv)
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
            yaw = float(df['FlightYawDegree'][0])         # Костыль
            roll = float(df['FlightRollDegree'][0])        # Костыль
            #w = float(df['ImageWidth'][0])
            #h = float(df['ImageHeight'][0])
            height = float(df['RelativeAltitude'][0])
            fov = float(df['FOV'][0].split()[0])
            print(pitch, yaw, roll, height, fov)

            is_OK, _, _ = check_protocol_correctness(
                filename, yaw, pitch, roll, height)
            W, H, _, border = calc_image_border(
                latitude, longtitude, height, fov, yaw)
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
