import folium
import glob
import json
import cv2
import gc
import os
import numpy as np
import pandas as pd
from folium.plugins import Draw, MeasureControl
from branca.element import MacroElement
from exif import Image
from jinja2 import Template
from math import sin, cos, tan, radians, sqrt, isnan
from numba import njit
from shapely.geometry import Point, Polygon
from collections import defaultdict
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

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
        self.prefix_string     = f'data/{field}/{attempt}/log/{field}.{attempt}'
        self.path_log_metadata = f'{self.prefix_string}.metadata.csv'
        self.path_log_bboxes   = f'{self.prefix_string}.{cnn_model}.{kernel_size}.bboxes.csv'
        self.path_log_plots    = f'{self.prefix_string}.{cnn_model}.{kernel_size}.result.csv'
        self.path_to_geojson   = f'data/{field}/{field}.geojson'
        self.path_to_map       = f'maps/{field}.{attempt}.html'
        self.cnn_model = cnn_model
        self.kernel_size = int(kernel_size)
        self.do_show_uncorrect = do_show_uncorrect
        self.activation_treshold = activation_treshold
        self.max_image_size = 150
        self.max_mask_size = 5472
        self.filenames = []
        for datatype in ['[jJ][pP][gG]', '[jJ][pP][eE][gG]', '[pP][nN][gG]']:
            files = glob.glob(f'{self.path_field_day}/src/*.{datatype}')
            self.filenames += sorted(map(os.path.basename, files))
        self.wheat_ears = []
        self.layers = []
        self.colormaps = []
        # сокрытые от пользователя операции
        make_dirs(self.path_field_day)        

    def read_metadata(self):
        if not os.path.exists(self.path_log_metadata):
            handle_metadata(self.filenames, self.path_field_day, self.path_log_metadata)

        self.df_metadata = pd.read_csv(self.path_log_metadata)
        self.df_metadata['border'] = self.df_metadata['border'].apply(lambda x: json.loads(x))
        print(f'[+] считал файл: {self.path_log_metadata}')
        self.latitude = float(self.df_metadata['latitude'][0])
        self.longtitude = float(self.df_metadata['longtitude'][0])
        # https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
        self.image_borders = list(self.df_metadata['border'])
        lat = np.array(self.image_borders).flatten()[::2]
        long = np.array(self.image_borders).flatten()[1::2]
        self.lat_min = lat.min()
        self.lat_max = lat.max()
        self.long_min = long.min()
        self.long_max = long.max()

    def read_bboxes(self):
        '''
        1. производим детекцию колосьев (если надо)
        2. считываем файл с диска '''
        if not os.path.exists(self.path_log_bboxes):
            self._detect_wheat_heads()

        self.df_bboxes = pd.read_csv(self.path_log_bboxes)
        print(f'[+] считал файл: {self.path_log_bboxes}')

    def perform_calculations(self):
        ''' считаем колосья в прямоугольниках карты плотности и делянках '''
        self.adjacent_frames = calc_adjacency_list(self.image_borders)
        print('Нашёл пересечение кадров')
        n = len(self.filenames)
        for i in range(n):
            print(f'{i} / {n}')
            self.wheat_ears += self._calc_wheat_intersections(i)
        self.ears_in_polygons = self._calc_wheat_head_count_in_geojsons()

    def draw_wheat_plots(self):
        ''' отрисовка числа колосьев на каждой делянке '''
        feature_group_choropleth = folium.FeatureGroup(
            name='wheat plots', show=True)
        df = pd.DataFrame(columns=['сорт', 'количество колосьев'])
        with open(self.path_to_geojson) as f:
            data = json.load(f)
            num_of_polygons = len(data['features'])

            max_p = np.amax(self.ears_in_polygons)
            for i in range(num_of_polygons):
                val = self.ears_in_polygons[i] / max_p
                hex_color = calc_hex_color(val)

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
                               fill_color=hex_color,
                               fill_opacity=1.0
                               )\
                    .add_child(folium.Popup(f'{str_wheat_type}\n{self.ears_in_polygons[i]}')) \
                    .add_to(feature_group_choropleth)
                df = df.append(
                    {'сорт': str_wheat_type, 'количество колосьев': self.ears_in_polygons[i]}, ignore_index=True)

        colormap = folium.LinearColormap(
            ['#dddddd', '#00ff00'], vmin=0, vmax=max_p).to_step(5)
        colormap.caption = 'count of spikes in wheat plots, pcs'
        self.layers.append(feature_group_choropleth)
        self.colormaps.append(colormap)
        df.to_csv(self.path_log_plots)

    def draw_protocol(self):
        ''' отображаем корректность протокола сьёмки изображений '''
        feature_group_protocol = folium.FeatureGroup(
            name='protocol', show=True)
        for i in range(self.df_metadata.shape[0]):
            line = self.df_metadata.loc[i]
            filename = line['name']
            flight_altitude_m = line['flight_altitude_m']
            gimbal_yaw_deg   = line['gimbal_yaw_deg']
            gimbal_pitch_deg = line['gimbal_pitch_deg']
            gimbal_roll_deg  = line['gimbal_roll_deg']
            flight_yaw_deg   = line['flight_yaw_deg']
            flight_pitch_deg = line['flight_pitch_deg']
            flight_roll_deg  = line['flight_roll_deg']
            border = line['border']
            wrong_parameters = check_protocol_correctness(gimbal_pitch_deg, gimbal_roll_deg, flight_altitude_m)


            popup_str = f'<b>filename: {filename}</b><br>' + \
                        f'gimbal_yaw_deg: {gimbal_yaw_deg}º<br>' + \
                        f'gimbal_pitch_deg: {gimbal_pitch_deg}º<br>' + \
                        f'gimbal_roll_deg: {gimbal_roll_deg}º<br>' + \
                        f'flight_yaw_deg: {flight_yaw_deg}º<br>' + \
                        f'flight_pitch_deg: {flight_pitch_deg}º<br>' + \
                        f'flight_roll_deg: {flight_roll_deg}º<br>' + \
                        f'flight_altitude_m: {flight_altitude_m}'

            # если протокол нарушен, то меняем цвет на красный и выводим ошибки
            color_polyline = '#007800'
            if len(wrong_parameters) > 0:
                color_polyline = '#780000'
                popup_str += '<br><br><b>Ошибки:</b>'
                for wrong_parameter in wrong_parameters:
                    popup_str += '<br>' + wrong_parameter

            popup_str = f'''  <table style="width: 100%; vertical-align: bottom;">
            <colgroup>
            <col span="1" style="width: 30%; vertical-align: top;">
            <col span="1" style="width: 70%;">
            </colgroup>

            <tr>
                <td>{popup_str}</td>
                <td><img src="http://127.0.0.1:9999/{self.path_field_day}/src/{filename}" width="547" height="308"></td>
            </tr>
            </table>'''


            iframe = folium.IFrame(html=popup_str, width=800, height=350)
            folium.PolyLine(border, color=color_polyline) \
                  .add_child(folium.Popup(iframe)) \
                  .add_to(feature_group_protocol)
        feature_group_protocol.add_to(self.m)

    def draw_grids(self, size_list):
        for size in size_list:
            self._draw_grid(size)

    def draw_images_on_map(self, do_rewrite=True):
        feature_group_images = folium.FeatureGroup(name='images', show=False)
        feature_group_masks = folium.FeatureGroup(name='masks', show=False)
        for i in range(len(self.filenames)):
            data = self.df_metadata.loc[i]
            filename = data['name']
            is_OK = data['is_OK']
            yaw = data['gimbal_yaw_deg']
            border = data['border']

            path_img_src = f'{self.path_field_day}/src/{filename}'
            path_img_mod = f'{self.path_field_day}/mod/{filename[:-4]}.webp'
            path_mask_src = f'{self.path_field_day}/masks/model_2_mask_{filename[:-4]}.png'
            path_mask_mod = f'{self.path_field_day}/mod/mask_{filename[:-4]}.webp'
            
            self._rotate_and_save(path_img_src, path_img_mod, yaw, self.max_image_size, do_rewrite, is_OK)
            self._rotate_and_save(path_mask_src, path_mask_mod, yaw, self.max_mask_size, do_rewrite, is_OK)

            ar = np.array(border).T
            bounds = [[np.min(ar[0]), np.min(ar[1])], [np.max(ar[0]), np.max(ar[1])]]
            if is_OK or self.do_show_uncorrect:
                folium.raster_layers.ImageOverlay(
                    name="Инструмент для разметки делянок",
                    image=path_img_mod,
                    bounds=bounds,
                    opacity=1.0,
                    control=False,
                    zindex=1,
                ).add_to(feature_group_images)
                folium.raster_layers.ImageOverlay(
                    name="Инструмент для разметки делянок",
                    image=path_mask_mod,
                    bounds=bounds,
                    opacity=1.0,
                    control=False,
                    zindex=1,
                ).add_to(feature_group_masks)
        feature_group_images.add_to(self.m)
        feature_group_masks.add_to(self.m)
    
    def create_mask(self, index_type = 'tgi'):
        if index_type == 'tgi':
            self._calc_tgi()

    def _calc_tgi(self):
        if os.path.exists(f'{self.path_field_day}/tgi'):
            print(f'[+] {self.path_field_day}/tgi уже существует')
            return None
        try_to_make_dir(f'{self.path_field_day}/tgi')
        filenames = glob.glob(f'{self.path_field_day}/src/*')
        filenames = sorted(map(os.path.basename, filenames))
        for filename in filenames:
            path_img = f'{self.path_field_day}/src/{filename}'
            img = cv2.imread(path_img)
            B, G, R = cv2.split(img)
            b,r,p = -0.14114779800465022, -0.825274736150845, 16.039116721891254
            tgi = b*B + G + r*R
            _, mask_tgi = cv2.threshold(tgi, p, 255, cv2.THRESH_BINARY)
            kernel = np.ones((6,6), np.uint8)
            opening = cv2.morphologyEx(mask_tgi, cv2.MORPH_OPEN, kernel)
            kernel_dilation = np.ones((70,70), np.uint8)
            opening_dilation = cv2.dilate(opening, kernel_dilation, iterations = 1)
            result = cv2.bitwise_and(opening_dilation, mask_tgi)
            filename=os.path.splitext(filename)[0]
            cv2.imwrite(f'{self.path_field_day}/tgi/{filename}.png', result)

    def create_map(self):
        ''' создаём карту и сохраняем объект карты как об приватное поле  '''
        self.m = folium.Map(
            [self.latitude, self.longtitude],
            tiles=None,
            prefer_canvas=True,
            control_scale=True,
            zoom_start=21
        )
        base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
        folium.TileLayer(tiles='OpenStreetMap', max_zoom=25).add_to(base_map)
        base_map.add_to(self.m)

    def save_map(self):
        ''' соединяем слои, карты цветов и сохраняем карту '''
        for layer in self.layers:
            self.m.add_child(layer)
        for colormap in self.colormaps:
            self.m.add_child(colormap)

        for layer, colormap in zip(self.layers, self.colormaps):
            self.m.add_child(BindColormap(layer, colormap))

        export_filename = f'{self.path_field_day.split("/")[1]}.geojson'
        Draw(position='topright', export=True, filename=export_filename).add_to(self.m)
        MeasureControl(position='bottomleft', collapsed=False).add_to(self.m)
        folium.map.LayerControl(position='topleft', collapsed=False).add_to(self.m)

        self.m.save(self.path_to_map)

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------       
    def _draw_grid(self, grid_size_m):
        ''' отрисовываем квадратную сетку grid_size_m * grid_size_m '''
        feature_group_grid = folium.map.FeatureGroup(
            name=f'grid {grid_size_m:.2f}x{grid_size_m:.2f}м²',
            # overlay=False,
            show=False
        )

        d_lat = convert_meters_to_lat(grid_size_m)
        d_long = convert_meters_to_long(grid_size_m, self.latitude)

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
                    value = wheat_counts[i][j] / max_p
                    hex_color = calc_hex_color(value)
                    folium.Rectangle(coordinates[i][j],
                                     color='#303030',
                                     opacity=0.05,
                                     fill=True,
                                     fill_color=hex_color,
                                     fill_opacity=1.0
                                     )\
                        .add_child(folium.Popup(f'{wheat_counts[i][j]:.2f}')) \
                        .add_to(feature_group_grid)

        colormap = folium.LinearColormap(
            ['#dddddd', '#00ff00'], vmin=0, vmax=max_p).to_step(5)
        colormap.caption = 'spike density, pcs/m²'
        self.layers.append(feature_group_grid)
        self.colormaps.append(colormap)

    def _handle_image(
        self, dir,
        d_images_grid, d_images_overlay,
        image_index, d_lat, d_long, grid_size_m, grid_size_px,
        is_OK, yaw, do_calc_overlay_dict=False
    ):
        ''' Этапы:
        1. считываем оригинальное изображение
        2. поворачиваем
        3. наращиваем до размеров сетки
        4. делаем маску изображения
        5. сохраняем в таблице кусочки в заданном разрешении '''

        # если протокол нарушен, то пропускаем изображение
        if not (is_OK or self.do_show_uncorrect):
            return d_images_grid, d_images_overlay

        tmp = np.array(self.image_borders[image_index]).T
        img_lat_min = tmp[0].min()
        img_lat_max = tmp[0].max()
        img_long_min = tmp[1].min()
        img_long_max = tmp[1].max()
        img_lat_m = convert_lat_to_meters(img_lat_max - img_lat_min)
        img_long_m = convert_long_to_meters(img_long_max - img_long_min, self.latitude)

        i_min = int((img_lat_min  - self.lat_min) / d_lat)
        j_min = int((img_long_min - self.long_min) / d_long)
        i_max = int((img_lat_max  - self.lat_min) / d_lat) + 1
        j_max = int((img_long_max - self.long_min) / d_long) + 1

        filenames = glob.glob(f'{self.path_field_day}/{dir}/*')
        filenames = sorted(map(os.path.basename, filenames))
        path_src = f'{self.path_field_day}/{dir}/{filenames[image_index]}'
        print(path_src)
        image_src = cv2.imread(path_src, 0)
        # image_src[:,:100] = 255
        # image_src[:,-100:] = 255
        # image_src[:100,:] = 255
        # image_src[-100:,:] = 255

        img_rotated = rotate_image(image_src, -yaw)
        img_rotated = img_rotated[::-1] # чтобы координатные оси совпадали
        h, w = img_rotated.shape[:2]

        # рассчитываем padding в градусах и пикселях
        pad_b_ratio = ((img_lat_min - self.grid_lat_min) % d_lat) / d_lat
        pad_t_ratio = ((self.grid_lat_max - img_lat_max) % d_lat) / d_lat
        pad_l_ratio = ((img_long_min - self.grid_long_min) % d_long) / d_long
        pad_r_ratio = ((self.grid_long_max - img_long_max) % d_long) / d_long
        pad_b_px =  int(h * pad_b_ratio * grid_size_m / img_lat_m)
        pad_t_px =  int(h * pad_t_ratio * grid_size_m / img_lat_m)
        pad_l_px =  int(w * pad_l_ratio * grid_size_m / img_long_m)
        pad_r_px =  int(w * pad_r_ratio * grid_size_m / img_long_m)
        pad_t_px, pad_b_px = pad_b_px, pad_t_px # чтобы координатные оси совпадали

        # print(f'{pad_b_ratio:.1f} {pad_t_ratio:.1f}    {pad_l_ratio:.1f} {pad_r_ratio:.1f}')
        # print(f'{pad_b_px:.1f} {pad_t_px:.1f}    {pad_l_px:.1f} {pad_r_px:.1f}')

        new_image = cv2.copyMakeBorder(
            img_rotated,
            pad_t_px, pad_b_px,
            pad_l_px, pad_r_px,
            cv2.BORDER_CONSTANT, (0,0,0))
        h, w = new_image.shape[:2]

        # модифицируем таблицу
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                # h0 = h * 
                h0 = int(float(h) * (i - i_min) / (i_max - i_min))
                h1 = int(float(h) * (i - i_min + 1) / (i_max - i_min))
                w0 = int(float(w) * (j - j_min) / (j_max - j_min))
                w1 = int(float(w) * (j - j_min + 1) / (j_max - j_min))
                img_crop = new_image[h0:h1, w0:w1]
                img_crop = cv2.resize(img_crop, (grid_size_px, grid_size_px))
                d_images_grid[i,j] += img_crop

        if do_calc_overlay_dict == False:
            return d_images_grid, d_images_overlay

        # рассчитываем границу
        mask = np.ones(image_src.shape, np.uint8)
        img_overlay = rotate_image(mask, -yaw)
        img_overlay = img_overlay[::-1] # чтобы координатные оси совпадали
        img_overlay = cv2.copyMakeBorder(
            img_overlay,
            pad_t_px, pad_b_px,
            pad_l_px, pad_r_px,
            cv2.BORDER_CONSTANT, (0,0,0))
       
        # модифицируем таблицу
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                h0 = int(float(h) * (i - i_min) / (i_max - i_min))
                h1 = int(float(h) * (i - i_min + 1) / (i_max - i_min))
                w0 = int(float(w) * (j - j_min) / (j_max - j_min))
                w1 = int(float(w) * (j - j_min + 1) / (j_max - j_min))
                img_crop = img_overlay[h0:h1, w0:w1]
                img_crop = cv2.resize(img_crop, (grid_size_px, grid_size_px))
                d_images_overlay[i,j] += img_crop

        return d_images_grid, d_images_overlay

    def _create_grid_from_images(self, dir, grid_size_m, grid_size_px = 200):
        ''' @brief строим сетку изображений из диретории dir
        @param dir строка папки [src/ndvi/etc.]
        @param grid_size_m размер кусочка плитки в метрах
        @param grid_size_px размер кусочка плитки в пикселях '''
        subdir = f'{self.path_field_day}/grid_{dir}_{grid_size_m:.2f}'
        if os.path.exists(subdir):
            print(f'[+] {subdir} уже существует')
            return None
        d_lat = convert_meters_to_lat(grid_size_m)
        d_long = convert_meters_to_long(grid_size_m, self.latitude)
        # запоминаем какие изображения попадают в каждый элемент сетки
        d = defaultdict(list)
        d_images_grid = defaultdict(lambda: np.zeros((grid_size_px, grid_size_px), np.int16))
        d_images_overlay = defaultdict(lambda: np.zeros((grid_size_px, grid_size_px), np.uint8))
        for index in range(len(self.filenames)):
            tmp = np.array(self.image_borders[index]).T
            img_lat_min = tmp[0].min()
            img_lat_max = tmp[0].max()
            img_long_min = tmp[1].min()
            img_long_max = tmp[1].max()
            i_min = int((img_lat_min  - self.lat_min) / d_lat)
            j_min = int((img_long_min - self.long_min) / d_long)
            i_max = int((img_lat_max  - self.lat_min) / d_lat) + 1
            j_max = int((img_long_max - self.long_min) / d_long) + 1
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    d[i,j].append(index)
                    d_images_grid[i,j] = np.zeros((grid_size_px, grid_size_px), np.int16)
                    d_images_overlay[i,j] = np.zeros((grid_size_px, grid_size_px), np.uint8)

        tmp = np.array(list(d.keys())).T
        i_max = tmp[0].max() + 1
        j_max = tmp[1].max() + 1
        self.grid_lat_min = self.lat_min
        self.grid_long_min = self.long_min
        self.grid_lat_max = self.grid_lat_min + i_max * d_lat
        self.grid_long_max = self.grid_long_min + j_max * d_long

        for index in range(len(self.filenames)):
            print(' '*80+f'\r{index} / {len(self.filenames)}', end='\r')
            data = self.df_metadata.loc[index]
            yaw = data['gimbal_yaw_deg']
            d_images_grid, d_images_overlay = self._handle_image(
                dir, d_images_grid, d_images_overlay,
                index, d_lat, d_long, grid_size_m, grid_size_px,
                True, yaw, True)

        try_to_make_dir(subdir)
        for i,j in d.keys():
            a = np.array(d_images_grid[i,j], np.float32)
            b = np.array(d_images_overlay[i,j], np.float32)
            # нормируем, пропуская пиксели в которых нет изображений и считаем индекс
            division = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            cv2.imwrite(f'{subdir}/{i}_{j}.png', division[::-1])
        print(' '*80 + f'\r[+] {dir} сетка {grid_size_m:.2f}x{grid_size_m:.2f} м^2')

    def draw_vegetation(self, grid_size_px = 200):
        ''' отрисовываем долю зелёных пикселей на квадратную сетку grid_size_m * grid_size_m '''
        grid_size_m = 0.2
        dir = 'tgi'
        self._create_grid_from_images(dir, grid_size_m, grid_size_px)
        print(f'[+] плитка {dir} создана')
        filenames = os.listdir(f'{self.path_field_day}/grid_{dir}_{grid_size_m:.2f}')

        grid = np.ndarray((len(filenames), 3))
        for index, filename in enumerate(filenames):
            i, j = filename[:-4].split('_')
            i = int(i)
            j = int(j)
            image = cv2.imread(f'{self.path_field_day}/grid_{dir}_{grid_size_m:.2f}/{i}_{j}.png')
            grid[index] = np.array([i, j, 100.0*np.mean(image / 255)])
        max_value = np.max(grid.T[2])
        grid.T[2] /=max_value


        d_lat = convert_meters_to_lat(grid_size_m)
        d_long = convert_meters_to_long(grid_size_m, self.latitude)
        feature_group_grid = folium.map.FeatureGroup(
            name=f'{dir} {grid_size_m:.2f}x{grid_size_m:.2f}м²',
            show=True
        )
        # divide count of objects by region area
        for i, j, value in grid:
            if value > 0:
                lat_b = self.lat_min + i*d_lat
                lat_e = lat_b + d_lat
                long_b = self.long_min + j*d_long
                long_e = long_b + d_long
                coordinates = np.array([
                    [lat_b, long_b],
                    [lat_e, long_e]
                ])
                hex_color = calc_hex_color(value)
                folium.Rectangle(coordinates,
                                    color='#303030',
                                    opacity=0.05,
                                    fill=True,
                                    fill_color=hex_color,
                                    fill_opacity=1.0
                                    )\
                    .add_child(folium.Popup(f'{i} {j}    {100*value:.2f}')) \
                    .add_to(feature_group_grid)
        colormap = folium.LinearColormap(['#dddddd', '#00ff00'], vmin=0, vmax=max_value).to_step(5)
        colormap.caption = 'share of green land, %'
        self.layers.append(feature_group_grid)
        self.colormaps.append(colormap)

        self._save_to_klm(grid, dir, grid_size_m)
    
    def _save_to_klm(self, grid, data_type, grid_size_m):
        ''' @brief сохраняем сетку к .klm файл
        @param grid [i, j, value]
        @param grid_size_m размер сетки '''
        f = open(f'{self.prefix_string}.{data_type}_{grid_size_m:.2f}.klm', 'w')
        f.write('''<?xml version="1.0" encoding="utf-8" ?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document id="root_doc">
<Schema name="canopy_example" id="canopy_example">
    <SimpleField name="percent" type="float"></SimpleField>
</Schema>
<Folder><name>canopy_example</name>
''')
        d_lat = convert_meters_to_lat(grid_size_m)
        d_long = convert_meters_to_long(grid_size_m, self.latitude)
        for i, j, value in grid:
            if value > 0:
                lat_b = self.lat_min + i*d_lat
                lat_e = lat_b + d_lat
                long_b = self.long_min + j*d_long
                long_e = long_b + d_long
                coordinates = np.array([
                    [lat_b, long_b],
                    [lat_b, long_e],
                    [lat_e, long_e],
                    [lat_e, long_b]
                ])
                hex_color = calc_hex_color(value)
                self._save_grid_element_to_klm(f, coordinates, value, hex_color)
        f.write('\n</Folder>\n</Document></kml>')
        f.close()

    def draw_masks(self):
        ''' отрисовываем маски на сетке grid_size_m * grid_size_m '''
        dirs = glob.glob(f'{self.path_field_day}/grid*')
        dirs = sorted(map(os.path.basename, dirs))
        for dir in dirs:
            _, type, grid_size_m = dir.split('_')
            grid_size_m = float(grid_size_m)
            feature_group_grid = folium.map.FeatureGroup(
                name=f'сетка {type} {grid_size_m:.2f}x{grid_size_m:.2f}м²',
                show=False
            )

            d_lat = convert_meters_to_lat(grid_size_m)
            d_long = convert_meters_to_long(grid_size_m, self.latitude)

            filenames = glob.glob(f'{self.path_field_day}/{dir}/*.png')
            filenames = sorted(map(os.path.basename, filenames))
            for filename in filenames:
                i, j = filename[:-4].split('_') # i_j.png
                i = int(i)
                j = int(j)
                bounds = [
                    [self.lat_min + i*d_lat, self.long_min + j*d_long],
                    [self.lat_min + (i+1)*d_lat, self.long_min + (j+1)*d_long],
                ]
                folium.raster_layers.ImageOverlay(
                    name="Инструмент для разметки делянок",
                    image=f'{self.path_field_day}/{dir}/{filename}',
                    bounds=bounds,
                    opacity=1.0,
                    control=False,
                    zindex=1,
                ).add_to(feature_group_grid)

            self.layers.append(feature_group_grid)

    def _create_tiles(self, tile_type):
        ''' @brief собирает плитку 200x200 в большие изображения и сжимает до 1000x1000 '''
        try_to_make_dir(f'{self.path_field_day}/tiles_{tile_type}')
        grid_size_m = 0.2
        dir = f'grid_{tile_type}_{grid_size_m:.2f}'
        self._create_grid_from_images(tile_type, grid_size_m)
        self.unite_every_k_images = 25
        filenames = glob.glob(f'{self.path_field_day}/{dir}/*')
        filenames = sorted(map(os.path.basename, filenames))
        path = f'{self.path_field_day}/{dir}/{filenames[0]}'
        grid_size_px, _ = cv2.imread(f'{self.path_field_day}/{dir}/{filenames[0]}', 0).shape

        # определяем какие плитки останутся
        d_images_grid = set()
        for filename in filenames:
            i, j = filename[:-4].split('_') # i_j.png
            i_tile = int(i) // self.unite_every_k_images
            j_tile = int(j) // self.unite_every_k_images
            d_images_grid |= {f'{i_tile}_{j_tile}'}
        # собираем эти плитки
        d_images_grid = sorted(d_images_grid)
        tile_size = grid_size_px * self.unite_every_k_images
        for i_j_str in d_images_grid:
            i_tile, j_tile = i_j_str.split('_')
            i_tile = int(i_tile)
            j_tile = int(j_tile)
            tile = np.zeros((tile_size, tile_size), np.uint8)
            for filename in filenames:
                i, j = filename[:-4].split('_') # i_j.png
                i = int(i)
                j = int(j)
                i_src, j_src = i, j
                i %= self.unite_every_k_images
                j %= self.unite_every_k_images
                if i_tile <= i_src / self.unite_every_k_images < i_tile + 1 and j_tile <= j_src / self.unite_every_k_images < j_tile + 1:
                    im = cv2.imread(f'{self.path_field_day}/{dir}/{filename}', 0)
                    tile[tile_size - (i+1)*grid_size_px:tile_size - i*grid_size_px, j*grid_size_px:(j+1)*grid_size_px] = im#[::-1]
                    # cv2.putText(tile, f'{i}_{j}', (i*grid_size_px,j*grid_size_px + 200), cv2.FONT_HERSHEY_COMPLEX, 2, (64, 64, 64),1,2)
                    # cv2.putText(tile, f'{i_src}_{j_src}', (i*grid_size_px,j*grid_size_px + 100), cv2.FONT_HERSHEY_COMPLEX, 2, (64, 64, 64),1,2)
            
            # for i in range(unite_every_k_images):
            #     tile[i*grid_size_px,:] = 128
            #     tile[:,i*grid_size_px] = 128
                #cv2.putText(tile, f'{i}_{i}', (i*grid_size_px,i*grid_size_px + 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255),1,2)
            tile = cv2.resize(tile, (500, 500))
            cv2.imwrite(f'{self.path_field_day}/tiles_{tile_type}/{i_tile}_{j_tile}.webp', tile)

    def draw_tiles(self, tile_type):
        ''' отрисовываем маски на сетке grid_size_m * grid_size_m '''
        self._create_tiles(tile_type)
        grid_size_m = 0.2
        unite_every_k_images = 25
        tile_size = grid_size_m * unite_every_k_images
        feature_group_grid = folium.map.FeatureGroup(
            name=f'tiles {tile_type} {tile_size:.2f}x{tile_size:.2f}м²',
            show=False
        )

        d_lat = convert_meters_to_lat(tile_size)
        d_long = convert_meters_to_long(tile_size, self.latitude)

        filenames = glob.glob(f'{self.path_field_day}/tiles_{tile_type}/*')
        filenames = sorted(map(os.path.basename, filenames))
        for filename in filenames:
            i, j = filename.split('.')[0].split('_') # i_j.png
            i = int(i)
            j = int(j)
            bounds = [
                [self.lat_min + i*d_lat, self.long_min + j*d_long],
                [self.lat_min + (i+1)*d_lat, self.long_min + (j+1)*d_long],
            ]
            folium.raster_layers.ImageOverlay(
                name="Инструмент для разметки делянок",
                image=f'http://127.0.0.1:9999/{self.path_field_day}/tiles_{tile_type}/{filename}',
                bounds=bounds,
                opacity=1.0,
                control=False,
                zindex=1,
            ).add_to(feature_group_grid)

        self.layers.append(feature_group_grid)


    def _rotate_and_save(self, path_src, path_mod, yaw, max_image_size, do_rewrite, is_OK):
        if not os.path.exists(path_mod) or do_rewrite and (is_OK or self.do_show_uncorrect):
            image = cv2.imread(path_src)
            h, w, _ = image.shape
            image_src = cv2.resize(image, (max_image_size, int(float(max_image_size*h)/w)))
            image = rotate_image(image_src, -yaw)
            trans_mask = image[:,:,2] == 0
            new_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
            new_image[trans_mask] = [0,0,0,0]
            cv2.imwrite(path_mod, new_image)

    def _read_coords_and_p(self, df, i):
        ''' @brief читаем координаты и вероятности колосьев \n return coords_and_p[lat, long, p] '''
        coords_and_p = []
        l = df.iloc[i].values[1]
        try: # если ничего не обнаружили
            if isnan(l):
                return coords_and_p
        except:
            pass
        l = l.split(' ')
        k = len(l) // 5
        for i in range(k):
            j = i*5
            p = float(l[j])
            if p > self.activation_treshold:
                lat = int(float(l[j+1])) + int(float(l[j+3])) // 2
                long = int(float(l[j+2])) + int(float(l[j+4])) // 2
                coords_and_p.append([lat, long, p])
        return coords_and_p

    def _calc_wheat_intersections(self, df_i):
        ''' поворачиваем колоски по азимуту и нормируем, поподающие в несколько изображений '''
        data = self.df_metadata.loc[df_i]
        is_OK       = data['is_OK']
        width_m     = data['width_m']
        height_m    = data['height_m']
        latitude    = data['latitude']
        longtitude  = data['longtitude']
        yaw_deg     = data['gimbal_yaw_deg']

        yaw_rad = radians(-yaw_deg)
        sin_a = sin(yaw_rad)
        cos_a = cos(yaw_rad)
        width_m /= 2
        height_m /= 2
        tmp_im = cv2.imread(f'{self.path_field_day}/src/{self.filenames[0]}')
        height_px, width_px, _ = tmp_im.shape
        height_px /= 2
        width_px /= 2

        if is_OK or self.do_show_uncorrect:
            coords_and_p = self._read_coords_and_p(self.df_bboxes, df_i)
            for i in range(len(coords_and_p)):
                x, y, p = coords_and_p[i]
                x = width_m *  (x - width_px) / width_px
                y = height_m * (1.0 - (y - height_px)) / height_px
                point = np.array([y, x], dtype=float)
                Y, X = rotate(point, sin_a, cos_a)
                dy = convert_meters_to_lat(Y)
                dx = convert_meters_to_long(X, latitude)
                point = Point([dy + latitude, dx + longtitude])
                w = 1.0
                for j in self.adjacent_frames[df_i]:
                    polygon = Polygon(np.array(self.image_borders[j]))
                    if polygon.contains(point):
                        w += 1.0
                coords_and_p[i] = [y, x, 1.0 / w]
            return coords_and_p
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

    def _save_grid_element_to_klm(self, f, coordinates, percent, color):
        coordinates_str = ''
        for lat, long in coordinates:
            coordinates_str += f'{long},{lat} '
        lat, long = coordinates[0]
        coordinates_str += f'{long},{lat}'
        f.write(f'''
    <Placemark>
        <Style><LineStyle><color>ff{color[1:]}</color></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>
        <ExtendedData><SchemaData schemaUrl="#canopy_example">
            <SimpleData name="percent">{percent}</SimpleData>
        </SchemaData></ExtendedData>
            <MultiGeometry><Polygon><outerBoundaryIs><LinearRing><coordinates>{coordinates_str}</coordinates></LinearRing></outerBoundaryIs></Polygon></MultiGeometry>
    </Placemark>''')

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
        print(f'[+] {path} уже существует')


# создаём папки maps, weights, log, tmp
def make_dirs(path_folder):
    try_to_make_dir('maps')
    try_to_make_dir('weights')
    # готовые метаданные всех изображений в папке и распознанные
    try_to_make_dir(f'{path_folder}/log')
    # временные csv файлы
    try_to_make_dir(f'{path_folder}/tmp')

''' 
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
'''

def rotate_image(image, angle_deg):
    h, w = image.shape[:2]
    image_center = (w / 2.0, h / 2)

    rot = cv2.getRotationMatrix2D(image_center, angle_deg, 1)

    angle_rad = radians(angle_deg)
    sin_a = sin(angle_rad)
    cos_a = cos(angle_rad)
    b_w = int((h * abs(sin_a)) + (w * abs(cos_a)))
    b_h = int((h * abs(cos_a)) + (w * abs(sin_a)))

    rot[0, 2] += ((b_w / 2) - image_center[0])
    rot[1, 2] += ((b_h / 2) - image_center[1])

    result = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return result


# преобразуем градусы, минуты и секунды в вешественную переменную координаты
@njit
def convert_to_decimal(degree, min, sec):
    return float(degree) + \
        float(min) / 60.0 + \
        float(sec) / 3600.0


@njit
def rotate(point, sin_a, cos_a):
    ''' @brief вращаем точку вокруг начала координат на заданный угол.
    @param point - точка [lat, long] (y, x)
    @param sin_a - синус угла
    @param cos_a - косинус угла
    @return Y, X '''
    y, x = point
    X = x*cos_a - y*sin_a
    Y = x*sin_a + y*cos_a
    return Y, X


def check_protocol_correctness(
    gimbal_pitch_deg: float,
    gimbal_roll_deg: float,
    flight_altitude_m: float,
):
    ''' @brief проверяем соблюдение протокола съёмки
    @param gimbal_pitch_deg - тангаж подвеса
    @param gimbal_roll_deg - крен подвеса
    @param flight_altitude_m - высота полёта
    @return wrong_parameters['тангаж', 'крен', 'высота'] '''
    wrong_parameters = []
    if abs(-90.0 - gimbal_pitch_deg) >= 3.0:
        wrong_parameters.append('тангаж')
    if abs(0.0 - gimbal_roll_deg) >= 3.0:
        wrong_parameters.append('крен')
    if abs(3.0 - flight_altitude_m) > 0.2:
        wrong_parameters.append('высота')
    return wrong_parameters


@njit
def calc_image_size(flight_altitude_m, fov_deg, width_px, height_px):
    ''' рассчитываем размер охваченной области под изображением.
    @param flight_altitude_m - высота полёта
    @param fov_deg - угол обзора камеры
    @param width_px - ширина изображения
    @param height_px - высота изображения
    @return width_m, height_m '''
    diag_m = 2.0 * flight_altitude_m * tan(radians(fov_deg / 2.0))
    diag_px = sqrt(width_px**2 + height_px**2)
    width_m  = (width_px  * diag_m) / diag_px
    height_m = (height_px * diag_m) / diag_px
    return width_m, height_m


# вместо WGS пока сфера, растянутая центробежной силой
# https://en.wikipedia.org/wiki/Circumference_of_the_Earth
EARTH_CIRCUMFERENSE_ALONG_MERIDIAN_KM = 40007.863
EARTH_CIRCUMFERENSE_ALONG_EQUATOR_KM  = 40075.017
LENGTH_OF_1_LAT_DEGREE_M = 1000 * EARTH_CIRCUMFERENSE_ALONG_MERIDIAN_KM / 360.0
LENGTH_OF_1_LONG_DEGREE_M = 1000 * EARTH_CIRCUMFERENSE_ALONG_EQUATOR_KM / 360.0

@njit
def convert_lat_to_meters(latitude_deg):
    ''' @brief переводит градусы широты в метры 
    @param latitude_deg - градус широты [-90, 90]
    @return lat_length_m'''
    lat_length_m = latitude_deg * LENGTH_OF_1_LAT_DEGREE_M
    if -90 <= latitude_deg <= 90:
        return lat_length_m
    else:
        raise ValueError
    

@njit
def convert_meters_to_lat(latitude_m):
    ''' @brief переводит метры в градусы широты
    @param latitude_m - метры вдоль меридиана
    @return latitude_deg'''
    latitude_deg = latitude_m / LENGTH_OF_1_LAT_DEGREE_M
    if -90 <= latitude_deg <= 90:
        return latitude_deg
    else:
        raise ValueError


@njit
def convert_long_to_meters(longitude_deg, latitude_deg):
    ''' @brief переводит градусы долготы в метры
    @param longitude_deg - градус долготы [-180, 180]
    @return longitude_m'''
    longitude_m = longitude_deg * LENGTH_OF_1_LONG_DEGREE_M * cos(radians(latitude_deg))
    if -180 <= longitude_deg <= 180:
        return longitude_m
    else:
        raise ValueError


@njit
def convert_meters_to_long(longitude_m, latitude_deg):
    ''' @brief переводит метры в градусы долготы
    @param longitude_m - метры вдоль Экватора
    @return longitude_deg'''
    longitude_deg = longitude_m / (LENGTH_OF_1_LONG_DEGREE_M * cos(radians(latitude_deg)))
    if -180 <= longitude_deg <= 180:
        return longitude_deg
    else:
        raise ValueError


def calc_image_border(latitude_deg, longtitude_deg, width_m, height_m, yaw_deg):
    ''' @brief рассчёт границы изображения
    @param latitude_deg - широта
    @param longtitude_deg - долгота
    @param width_m - ширина изображения
    @param height_m - высота изображения
    @param yaw_deg - рысканье
    @return border[lat, long] '''
    points = np.array([
        [-height_m/2.0, -width_m/2.0],
        [-height_m/2.0, +width_m/2.0],
        [+height_m/2.0, +width_m/2.0],
        [+height_m/2.0, -width_m/2.0],
        [-height_m/2.0, -width_m/2.0],
    ], dtype=float)

    border = []
    yaw_rad = radians(-yaw_deg)
    sin_a = sin(yaw_rad)
    cos_a = cos(yaw_rad)
    for point in points:
        y_m, x_m = rotate(point, sin_a, cos_a)
        dy_deg = convert_meters_to_lat(y_m)
        dx_deg = convert_meters_to_long(x_m, latitude_deg)
        border.append([dy_deg + latitude_deg, dx_deg+longtitude_deg])
    return border


def calc_adjacency_list(image_borders):
    ''' @brief строит список смежных полигонов
    @return adjacent_list[adj_image_1, adj_image_2,...] '''
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
def handle_metadata(filenames, path_field_day, path_log_metadata):
    # формат полученного датафрейма
    # name           is_ok   W    H     lat     long    height   yaw   pitch    roll     border
    # Image_23.jpg   True    5    3    12.34    12.34    2.7     120    -90     0.0   [[], [], [], []]
    # Image_24.jpg   False   12   7    45.67    45.67    6.0     130    -60     0.0   [[], [], [], []]
    # ...
    src_files = []
    for datatype in ['[jJ][pP][gG]', '[jJ][pP][eE][gG]', '[pP][nN][gG]']:
        files = glob.glob(f'{path_field_day}/src/*.{datatype}')
        src_files += sorted(map(os.path.basename, files))
    src_file_count = len(src_files) 
    tmp_file_count = len(os.listdir(f'{path_field_day}/tmp'))
    print(src_file_count, tmp_file_count)

    if src_file_count != tmp_file_count:
        try:
            for filename in filenames:
                path_img = f'{path_field_day}/src/{filename}'
                path_csv = f'{path_field_day}/tmp/{filename[:-4]}.csv'
                command = os.path.join('exiftool-12.34', f'exiftool -csv {path_img} > {path_csv}')
                os.system(command)
                df = pd.read_csv(path_csv, header=None).T
                df.to_csv(path_csv, header=False, index=False)
        except:
            print('[-] число файлов в src и tmp не совпало')
    else:
        print('[+] число файлов совпало')
            
    try:
        df_metadata = pd.DataFrame(columns=[
            'name',
            'is_OK',
            'width_m', 'height_m',
            'latitude', 'longtitude', 'flight_altitude_m',
            'gimbal_yaw_deg', 'gimbal_pitch_deg', 'gimbal_roll_deg',
            'flight_yaw_deg', 'flight_pitch_deg', 'flight_roll_deg',
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
            width_px            = float(df['ExifImageWidth'][0])
            height_px           = float(df['ExifImageHeight'][0])
            gimbal_yaw_deg      = float(df['GimbalYawDegree'][0])
            gimbal_pitch_deg    = float(df['GimbalPitchDegree'][0])
            gimbal_roll_deg     = float(df['GimbalRollDegree'][0])
            flight_yaw_deg      = float(df['FlightYawDegree'][0])
            flight_pitch_deg    = float(df['FlightPitchDegree'][0])
            flight_roll_deg     = float(df['FlightRollDegree'][0])
            flight_altitude_m   = float(df['RelativeAltitude'][0])
            fov_deg             = float(df['FOV'][0].split()[0])
            
            wrong_parameters = check_protocol_correctness(gimbal_pitch_deg, gimbal_roll_deg, flight_altitude_m)
            is_OK = (len(wrong_parameters) == 0)
            width_m, height_m = calc_image_size(flight_altitude_m, fov_deg, width_px, height_px)
            border = calc_image_border(latitude, longtitude, width_m, height_m, gimbal_yaw_deg)
            df_metadata = df_metadata.append({
                'name':                 filename,
                'is_OK':                str(is_OK),
                'width_m':              width_m,
                'height_m':             height_m,
                'latitude':             latitude,
                'longtitude':           longtitude,
                'flight_altitude_m':    flight_altitude_m,
                'gimbal_yaw_deg':       gimbal_yaw_deg,
                'gimbal_pitch_deg':     gimbal_pitch_deg,
                'gimbal_roll_deg':      gimbal_roll_deg,
                'flight_yaw_deg':       flight_yaw_deg,
                'flight_pitch_deg':     flight_pitch_deg,
                'flight_roll_deg':      flight_roll_deg,
                'border':               border
            }, ignore_index=True)
       
        df_metadata = df_metadata.sort_values(by=['name'])
        df_metadata.to_csv(path_log_metadata, index=False)

    except:
        print('[-] датафрейм не сформирован')

def calc_hex_color(value):
    ''' @brief отображает значение [0,1] в цвет [серый, зелёный]
    max_p:  val = 1  ->  (0,255,0)
    min_p:  val = 0  ->  (221,221,221)
    @param value [0, 1]
    @return hex_color '''
    hex_val = hex(int((1 - value) * 221))[2:]
    if len(hex_val) == 1:
        hex_val = '0'+hex_val
    hex_color = f'#{hex_val}dd{hex_val}'
    return hex_color

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
