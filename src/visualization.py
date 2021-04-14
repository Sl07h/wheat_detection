from lib import *


path_field_day      = 'data/Field2_3_2019/07_25/'
path_log_bboxes     = path_field_day + 'log/Field_2_3.frcnn.512.csv'
path_log_metadata   = path_field_day + 'log/metadata.csv'
path_to_geojson     = 'data/Field2_3_2019/wheat_plots.geojson'
latitude            = 54.87890
longtitude          = 82.99877
do_show_uncorrect   = False
grid_sizes          = [0.5, 2.0]
activation_treshold = 0.95

layers = []
colormaps = []


# поворачиваем колоски по азимуту и нормируем, поподающие в несколько изображений
def calc_wheat_intersections(df_i, df_bboxes, df_metadata, adjacent_frames, image_borders):
    data = df_metadata.loc[df_i]
    is_OK       = data['is_OK']
    W           = data['W']
    H           = data['H']
    latitude    = data['lat']
    longtitude  = data['long']
    yaw         = data['yaw']

    ratio = 31.0 * cos(radians(latitude))
    a = radians(-yaw)
    sin_a = sin(a)
    cos_a = cos(a)
    W = W / 2
    H = H / 2

    if is_OK or do_show_uncorrect:
        bboxes = read_logs(df_bboxes, df_i)
        for i in range(len(bboxes)):
            x, y, p = bboxes[i]
            x = (x - 2736) / 2736 * W
            y = (1.0 - (y - 1539)) / 1539 * H
            point = np.array([y, x], dtype=float)
            X, Y = rotate(point, sin_a, cos_a)
            dx = convert_to_decimal(0.0, 0.0, X / ratio)
            dy = convert_to_decimal(0.0, 0.0, Y / 31.0)
            y = dy + latitude
            x = dx + longtitude
            point = Point(y, x)
            w = 1.0
            for j in adjacent_frames[df_i]:
                polygon = Polygon(np.array(image_borders[j]))
                if polygon.contains(point):
                    w += 1.0
            bboxes[i][0] = y
            bboxes[i][1] = x
            bboxes[i][2] = 1.0 / w
        return bboxes
    else:
        return []


def read_logs(df, i):
    l = df.iloc[i].values[1]
    l = l.split(' ')
    k = len(l) // 5
    coords = []
    for i in range(k):
        j = i*5
        p = float(l[j])
        if p > activation_treshold:
            lat = int(l[j+1]) + int(l[j+3]) // 2
            lon = int(l[j+2]) + int(l[j+4]) // 2
            coords.append([lat, lon, p])
    return coords


# отрисовываем квадратную сетку grid_size_m * grid_size_m
def draw_grid(
        wheat_ears,
        lat_min,
        lat_max,
        long_min,
        long_max,
        grid_size_m):

    global layers
    global colormaps

    feature_group_grid = folium.map.FeatureGroup(
        name='сетка {:.1f}x{:.1f}м²'.format(grid_size_m, grid_size_m),
        # overlay=False,
        show=False
    )
    
    ratio  = 31.0 * cos(radians(latitude))
    d_lat  = convert_to_decimal(0.0, 0.0, grid_size_m / 31.0)
    d_long = convert_to_decimal(0.0, 0.0, grid_size_m / ratio)
    delta_lat = lat_max - lat_min
    delta_long = long_max - long_min
    n_lat = int(delta_lat / d_lat)
    n_long = int(delta_long / d_long)

    wheat_counts = np.zeros((n_lat+1, n_long+1))
    coordinates = np.ndarray((n_lat+1, n_long+1, 4, 2))
    for wheat_ear in wheat_ears:
        _lat, _long, p = wheat_ear
        i = int((_lat - lat_min) / d_lat)
        j = int((_long - long_min) / d_long)
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
                lat_b = lat_min + i*d_lat
                lat_e = lat_b + d_lat
                long_b = long_min + j*d_long
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
                color = '#{}dd{}'.format(hex_val, hex_val)

                folium.Rectangle(coordinates[i][j],
                                 color='#303030',
                                 opacity=0.05,
                                 fill=True,
                                 fill_color=color,
                                 fill_opacity=1.0
                                 )\
                    .add_child(folium.Popup(str(wheat_counts[i][j]))) \
                    .add_to(feature_group_grid)

    colormap = folium.LinearColormap(['#dddddd', '#00ff00'], vmin=0, vmax=max_p).to_step(5)
    colormap.caption = 'плотность колосьев, шт/м²'
    layers.append(feature_group_grid)
    colormaps.append(colormap)


# считаем сколько колосков в каждом полигоне, размеченном агрономом
# ВНИМАНИЕ в geojson координаты в другом порядке
def draw_with_geojson(path_to_geojson, wheat_ears, m):
    feature_group_choropleth = folium.FeatureGroup(name='фоновая картограмма', show=True)
    with open(path_to_geojson) as f:
        data = json.load(f)
        num_of_polygons = len(data['features'])
        ears_in_polygons = np.ones(num_of_polygons)
        for i in range(num_of_polygons):
            t = np.array(data['features'][i]['geometry']['coordinates'][0][:4])
            # docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
            t[:,[0, 1]] = t[:,[1, 0]]
            wheat_plot_polygon = Polygon(t)
            for wheat_ear in wheat_ears:
                point = Point(np.array(wheat_ear[:2]))
                if wheat_plot_polygon.contains(point):
                    ears_in_polygons[i] += wheat_ear[2]

        max_p = np.amax(ears_in_polygons)
        for i in range(num_of_polygons):
            # max_p:  val = 1  ->  (0,255,0)
            # min_p:  val = 0  ->  (221,221,221)
            val = ears_in_polygons[i] / max_p
            hex_val = hex(int((1 - val) * 221))[2:]
            if len(hex_val) == 1:
                hex_val = '0'+hex_val
            color = '#{}dd{}'.format(hex_val, hex_val)

            t = np.array(data['features'][i]['geometry']['coordinates'][0])
            t[:,[0, 1]] = t[:,[1, 0]]
            folium.Polygon(t,
                                color='#303030',
                                opacity=0.05,
                                fill=True,
                                fill_color=color,
                                fill_opacity=1.0
                                )\
                .add_child(folium.Popup(str(ears_in_polygons[i]))) \
                .add_to(feature_group_choropleth)
    
    colormap = folium.LinearColormap(['#dddddd', '#00ff00'], vmin=0, vmax=max_p).to_step(5)
    colormap.caption = 'количество колосьев на делянках, шт'

    m.add_child(feature_group_choropleth)
    m.add_child(colormap)
    m.add_child(BindColormap(feature_group_choropleth, colormap))




# считаем сколько колосков в каждом полигоне, размеченном агрономом
# ВНИМАНИЕ в geojson координаты в другом порядке
def calc_wheat_head_count_in_geojsons(path_to_geojson, wheat_ears):
    with open(path_to_geojson) as f:
        data = json.load(f)
        num_of_polygons = len(data['features'])
        ears_in_polygons = np.ones(num_of_polygons)
        for i in range(num_of_polygons):
            t = np.array(data['features'][i]['geometry']['coordinates'][0][:4])
            # docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
            t[:,[0, 1]] = t[:,[1, 0]]
            wheat_plot_polygon = Polygon(t)
            for wheat_ear in wheat_ears:
                point = Point(np.array(wheat_ear[:2]))
                if wheat_plot_polygon.contains(point):
                    ears_in_polygons[i] += wheat_ear[2]
    return ears_in_polygons


if __name__ == "__main__":
    m = folium.Map([latitude, longtitude], tiles=None,
                   prefer_canvas=True, control_scale=True, zoom_start=21)
    base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
    folium.TileLayer(tiles='OpenStreetMap', max_zoom=23).add_to(base_map)
    base_map.add_to(m)

    filenames = os.listdir(path_field_day + 'src')
    if not os.path.exists(path_log_metadata):
        make_dirs(path_field_day)
        handle_metadata(filenames, path_field_day)

    # считываем метаданные и координаты колосков
    df_bboxes = pd.read_csv(path_log_bboxes)
    df_metadata = pd.read_csv(path_log_metadata)
    df_metadata['border'] = df_metadata['border'].apply(lambda x: json.loads(x))
    # https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe
    image_centers = list(df_metadata.apply(lambda x: [x.lat, x.long], axis=1))
    image_borders = list(df_metadata['border'])


    adjacent_frames = find_intersections(image_centers, image_borders)
    wheat_ears = []
    for i in range(len(filenames)):
        wheat_ears += calc_wheat_intersections(i, df_bboxes, df_metadata, adjacent_frames, image_borders)

    lat = np.array(image_borders).flatten()[::2]
    long = np.array(image_borders).flatten()[1::2]
    lat_min = lat.min()
    lat_max = lat.max()
    long_min = long.min()
    long_max = long.max()

    
    # ears_in_polygons = calc_wheat_head_count_in_geojsons(path_to_geojson, wheat_ears)
    
    draw_with_geojson(path_to_geojson, wheat_ears, m)
    # feature_group_choropleth = folium.FeatureGroup(name='фоновая картограмма', show=True)
    # folium.Choropleth(
    #     path_to_geojson,
    #     pd.Series(ears_in_polygons)
    # ).add_to(feature_group_choropleth)
    # feature_group_choropleth.add_to(m)


    for grid_size in grid_sizes:
        draw_grid(wheat_ears, lat_min, lat_max, long_min, long_max, grid_size)

    draw_protocol(df_metadata, m)

    for layer in layers:
        m.add_child(layer)
    for colormap in colormaps:
        m.add_child(colormap)


    for layer, colormap in zip(layers, colormaps):
        m.add_child(BindColormap(layer, colormap))

    m.add_child(folium.map.LayerControl(collapsed=False))

    field_name = path_field_day.split('/')[1]
    m.save('maps/{}.html'.format(field_name))