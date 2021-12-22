import folium
import json
import cv2
import gc
import os
import sys
import numpy as np
import pandas as pd
from folium.plugins import Draw, MeasureControl
from branca.element import MacroElement
from exif import Image
from jinja2 import Template
from math import sin, cos, tan, radians, sqrt, isnan
from numba import njit
from shapely.geometry import Point, Polygon
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

#from effdet import get_efficientdet_config, EfficientDet, DetBenchEval
#from effdet.efficientdet import HeadNet

from effdet.config import get_efficientdet_config
from effdet.efficientdet import EfficientDet, HeadNet
from effdet.bench import DetBenchPredict


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
        self.path_field_day    = f'data/{field}/{attempt}'
        self.path_log_metadata = f'data/{field}/{attempt}/log/metadata.csv'
        self.path_log_bboxes   = f'data/{field}/{attempt}/log/bboxes.{field}.{attempt}.{cnn_model}.{kernel_size}.csv'
        self.path_log_plots    = f'data/{field}/{attempt}/log/result.{field}.{attempt}.{cnn_model}.{kernel_size}.csv'
        self.path_to_geojson   = f'data/{field}/wheat_plots.geojson'
        self.cnn_model = cnn_model
        self.kernel_size = int(kernel_size)
        self.do_show_uncorrect = do_show_uncorrect
        self.activation_treshold = activation_treshold
        self.filenames = os.listdir(f'{self.path_field_day}/src')
        self.wheat_ears = []
        self.layers = []
        self.colormaps = []
        # сокрытые от пользователя операции
        make_dirs(self.path_field_day)
        if not os.path.exists(self.path_log_bboxes):
            self._detect_wheat_heads()
        else:
            print(f'Найден файл: {self.path_log_bboxes}')
        if not os.path.exists(self.path_log_metadata):
            print(f'Нет файла: {self.path_log_metadata}. Собираю:')
            handle_metadata(self.filenames, self.path_field_day)
        else:
            print(f'Найден файл: {self.path_log_metadata}')
        self._read_metadata()
        
        # считаем колосья в прямоугольниках карты плотности и делянках
        self.adjacent_frames = find_intersections(self.image_borders)
        print('Нашёл пересечение кадров')
        n = len(self.filenames)
        for i in range(n):
            print(f'{i} / {n}')
            self.wheat_ears += self._calc_wheat_intersections(i)

        self.ears_in_polygons = self._calc_wheat_head_count_in_geojsons()
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
                    str_wheat_type = data['features'][i]['properties']['name']
                except:
                    pass

                folium.Polygon(t,
                               color='#303030',
                               opacity=0.05,
                               fill=True,
                               fill_color=color,
                               fill_opacity=1.0
                               )\
                    .add_child(folium.Popup(f'{str_wheat_type}\n{self.ears_in_polygons[i]}')) \
                    .add_to(feature_group_choropleth)
                df = df.append(
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
            yaw = line['gimbal_yaw']
            pitch = line['gimbal_pitch']
            roll = line['gimbal_roll']
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

        Draw(export=True).add_to(self.m)
        MeasureControl(collapsed=False).add_to(self.m)
        folium.map.LayerControl(collapsed=False).add_to(self.m)

        field_name = self.path_field_day.split('/')[1]
        self.m.save(f'maps/{field_name}.{self.kernel_size}.html')

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------

    def _read_metadata(self):
        ''' считываем метаданные и сохраняем промежуточные значения '''
        self.df_bboxes = pd.read_csv(self.path_log_bboxes)
        self.df_metadata = pd.read_csv(self.path_log_metadata)
        self.df_metadata['border'] = self.df_metadata['border'].apply(
            lambda x: json.loads(x))
        self.latitude = float(self.df_metadata['lat'][0])
        self.longtitude = float(self.df_metadata['long'][0])
        # https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
        self.image_borders = list(self.df_metadata['border'])
        lat = np.array(self.image_borders).flatten()[::2]
        long = np.array(self.image_borders).flatten()[1::2]
        self.lat_min = lat.min()
        self.lat_max = lat.max()
        self.long_min = long.min()
        self.long_max = long.max()
        print(f'Считал файл: {self.path_log_bboxes}')
        print(f'Считал файл: {self.path_log_metadata}')

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
        coords = []
        l = df.iloc[i].values[1]
        try:
            if isnan(l):
                return coords
        except:
            pass
        l = l.split(' ')
        k = len(l) // 5
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
        yaw = data['gimbal_yaw']

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

    def _detect_wheat_heads(self):

        results = []

        detection_threshold, nms_treshold = 0.6, 0.8
        kernel_size = self.kernel_size
        stride_size = kernel_size // 2
        batch_size = 1
        num_classes = 2
        path = f'{self.path_field_day}/src/{self.filenames[0]}'
        H, W, _ = cv2.imread(path).shape

        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        val_transforms = transforms.Compose([
           transforms.ToTensor(),
        ])

        if self.cnn_model == 'frcnn':
            path_to_weight = 'weights/fasterrcnn_resnet50_fpn_best.pth'
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            model.load_state_dict(torch.load(path_to_weight, map_location=device))
        elif self.cnn_model == 'effdet':
            path_to_weight = 'weights/fold0-best-all-states.bin'
            # source: https://www.kaggle.com/shonenkov/inference-efficientdet
            model = load_net(path_to_weight)

        model.eval()
        model.to(device)

        test_data_loader = DataLoader(
            WheatTestDataset(f'{self.path_field_day}/src', stride_size, val_transforms),
            batch_size=1,
            shuffle=False,
            drop_last=False
        )

        image_i = 0
        n = len(self.filenames)
        for batch_images, batch_image_names in test_data_loader:
            image = batch_images[0]
            image_name = batch_image_names[0]
            _, H, W = image.shape
            print(f'\n{image_i} / {n}  Изображение {image_name}: {W}x{H}')
            image_i += 1
            x = batch_images
            kc, kh, kw = 3, kernel_size, kernel_size
            dc, dh, dw = 3, stride_size, stride_size
            # https://discuss.pytorch.org/t/patch-making-does-pytorch-have-anything-to-offer/33850/11
            patches_unfold = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
            patches = patches_unfold.contiguous().view(-1, kc, kh, kw)
            print(patches.shape)
            pos_list = []
            boxes_list = []
            scores_list = []

            for k in range(patches.shape[0]//batch_size):
                batch = patches[k*batch_size:(k+1)*batch_size].to(device)
                if self.cnn_model == 'effdet':
                    batch = transforms.functional.resize(batch, 512)
                outputs = model(batch)
                for i, image in enumerate(batch):
                    # преобразуем в numpy
                    if self.cnn_model == 'frcnn':
                        boxes = outputs[i]['boxes'].data.cpu().numpy()
                        scores = outputs[i]['scores'].data.cpu().numpy()
                    elif self.cnn_model == 'effdet':
                        boxes = outputs[i].detach().cpu().numpy()[:,:4]    
                        scores = outputs[i].detach().cpu().numpy()[:,4]
                    # фильтруем совсем слабые результаты
                    boxes = boxes[scores >= detection_threshold].astype(np.int32)
                    scores = scores[scores >= detection_threshold]
                    # (x0,y0), (x1,y1) -> X,Y,W,H
                    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
                    # сохраняем относительные bbox-ы и их вероятность
                    boxes_list.append(boxes)
                    scores_list.append(scores)
                    pos_list.append(k*batch_size+i)

            boxes_list, scores_list = fix_coordinates(pos_list, boxes_list, scores_list, W // stride_size - 1, stride_size)
            result = {
                'image_id': batch_image_names[0],
                'PredictionString': format_prediction_string(boxes_list, scores_list, nms_treshold)
            }
            results.append(result)

        test_df = pd.DataFrame(
            results, columns=['image_id', 'PredictionString'])
        test_df.to_csv(self.path_log_bboxes, index=False)
        print(f'Saved file {self.path_log_bboxes}.')


class BindColormap(MacroElement):
    '''Binds a colormap to a given layer.

    Parameters
    ----------
    colormap : branca.colormap.ColorMap
        The colormap to bind.
    '''

    def __init__(self, layer, colormap):
        super(BindColormap, self).__init__()
        self.layer = layer
        self.colormap = colormap
        self._template = Template(u'''
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
        ''')


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
    if abs(-90.0 - pitch) >= 5.0:
        wrong_parameters.append('тангаж')
    if abs(0.0 - roll) >= 3.0:
        wrong_parameters.append('крен')
    if abs(3.0 - height) > 0.2:
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


def find_intersections(image_borders):
    adjacent_list = []
    for i1, border1 in enumerate(image_borders):
        l = []
        for i2, border2 in enumerate(image_borders):
            has_intersection = False
            polygon1 = Polygon(border1)
            polygon2 = Polygon(border2)
            has_intersection = polygon1.intersects(polygon2)
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
                command = f'exiftool-12.34/{exiftool_script_name} -csv {path_img} > {path_csv}'
                os.system(command)
                df = pd.read_csv(path_csv, header=None).T
                df.to_csv(path_csv, header=False, index=False)
        except:
            print('Ошибка. Число файлов в src и tmp не совпало')

    try:
        df_metadata = pd.DataFrame(columns=[
            'name',
            'is_OK',
            'W', 'H',
            'lat', 'long',
            'height',
            'gimbal_yaw', 'gimbal_pitch', 'gimbal_roll',
            'flight_yaw', 'flight_pitch', 'flight_roll',
            'border',
        ])
        for filename in filenames:
            path_img = f'{path_field_day}/src/{filename}'
            path_csv = f'{path_field_day}/tmp/{filename[:-4]}.csv'
            with open(path_img, 'rb') as image_file:
                my_image = Image(image_file)
                latitude = convert_to_decimal(*my_image.gps_latitude)
                longtitude = convert_to_decimal(*my_image.gps_longitude)
            df = pd.read_csv(path_csv, index_col=0).T
            gimbal_pitch = float(df['GimbalPitchDegree'][0])
            gimbal_yaw   = float(df['GimbalYawDegree'][0])
            gimbal_roll  = float(df['GimbalRollDegree'][0])
            flight_pitch = float(df['FlightPitchDegree'][0])
            flight_yaw   = float(df['FlightYawDegree'][0])
            flight_roll  = float(df['FlightRollDegree'][0])
            height       = float(df['RelativeAltitude'][0])
            fov          = float(df['FOV'][0].split()[0])

            is_OK, _, _ = check_protocol_correctness(filename, gimbal_yaw, gimbal_pitch, gimbal_roll, height)
            W, H, _, border = calc_image_border(latitude, longtitude, height, fov, gimbal_yaw)
            df_metadata = df_metadata.append({
                'name':          filename,
                'is_OK':         str(is_OK),
                'W':             W,
                'H':             H,
                'lat':           latitude,
                'long':          longtitude,
                'height':        height,
                'gimbal_yaw':    gimbal_yaw,
                'gimbal_pitch':  gimbal_pitch,
                'gimbal_roll':   gimbal_roll,
                'flight_yaw':    flight_yaw,
                'flight_pitch':  flight_pitch,
                'flight_roll':   flight_roll,
                'border': border
            }, ignore_index=True)
       
        df_metadata = df_metadata.sort_values(by=['name'])
        df_metadata.to_csv(f'{path_field_day}/log/metadata.csv', index=False)

    except:
        print('Ошибка при формировании датафрейма')


# https://discuss.pytorch.org/t/how-to-load-images-from-different-folders-in-the-same-batch/18942
class WheatTestDataset(Dataset):

    def __init__(self, dir, stride_size, augs=None):
        self.dir = dir
        self.image_names = os.listdir(self.dir)
        self.augs = augs
        path = os.path.join(self.dir, self.image_names[0])
        H, W, _ = cv2.imread(path).shape
        pad_w = stride_size - W % stride_size
        pad_h = stride_size - H % stride_size
        if H % stride_size == 0:
            pad_h = 0
        if W % stride_size == 0:
            pad_w = 0
        if pad_w != 0 or pad_h != 0:
            self.do_padding = True 
            self.pad_h = pad_h
            self.pad_w = pad_w
        else: 
            self.do_padding = False

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        path = os.path.join(self.dir, image_name)
        image = cv2.imread(path)
        #image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
        if self.do_padding:
            image = cv2.copyMakeBorder(
                image, 0, self.pad_h, 0, self.pad_w, cv2.BORDER_CONSTANT, 0)
        if self.augs:
            img_tensor = self.augs(image)
        print(img_tensor.size(), img_tensor.dtype)
        return img_tensor, image_name


def format_prediction_string(boxes, scores, nms_treshold):
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    t = torchvision.ops.nms(boxes, scores, nms_treshold)
    boxes = boxes[t]
    scores = scores[t]

    pred_strings = []
    for score, bbox in zip(scores, boxes):
        pred_strings.append('{0:.4f} {1} {2} {3} {4}'.format(
            score, bbox[0], bbox[1], bbox[2], bbox[3]))

    return ' '.join(pred_strings)


def fix_coordinates(list_i, list_bbox, list_prob, W_boxes, stride_size):
    for i, bbox, prob in zip(list_i, list_bbox, list_prob):
        x = (i % W_boxes) * stride_size
        y = (i // W_boxes) * stride_size
        for i in range(bbox.shape[0]):
            bbox[i][0] += x
            bbox[i][1] += y
    return np.vstack(list_bbox), np.hstack(list_prob)


def load_net(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    config.num_classes = 1
    config.image_size = [512, 512]
    config.norm_kwargs = dict(eps=.001, momentum=.01)
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    gc.collect()
    model = DetBenchPredict(net)
    return model
