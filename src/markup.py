import cv2
from folium.plugins import Draw
from scipy.ndimage import rotate as rotate_sp
from lib import *


path_field_day      = 'data/Field2_3_2019/07_25/'
path_log_bboxes     = path_field_day + 'log/Field_2_3.frcnn.512.csv'
path_log_metadata   = path_field_day + 'log/metadata.csv'
latitude            = 54.878876
longtitude          = 82.9987
do_show_uncorrect   = False
do_rotate_force     = True
opacity             = 0.6
decrease_by         = 10


# отрисовываем повёрнутое и сжатое изображение на карте для разметки делянок
def draw_image_on_map(i, df_metadata, do_rewrite):
    data = df_metadata.loc[i]
    filename = data['name']
    is_OK = data['is_OK']
    height = data['height']
    yaw = data['yaw']
    border = data['border']

    path_img_src = r'{}src/{}'.format(path_field_day, filename)
    path_img_mod = r'{}mod/{}.png'.format(path_field_day, filename[:-4])
    path_img_fix = r'{}mod/90_{}.png'.format(path_field_day, filename[:-4])

    if do_rewrite:
        image = cv2.imread(path_img_src)
        h, w, _ = image.shape
        image_src = cv2.resize(image, (w // decrease_by, h // decrease_by))
        image = rotate_sp(image_src, -yaw, reshape=True)
        trans_mask = image[:,:,2] == 0
        new_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        new_image[trans_mask] = [0,0,0,0]
        cv2.imwrite(path_img_mod, new_image)
        new_image = rotate_sp(image_src, 90.0, reshape=True)
        cv2.imwrite(path_img_fix, new_image)

    ar = np.array(border).T
    bounds = [[np.min(ar[0]), np.min(ar[1])], [np.max(ar[0]), np.max(ar[1])]]
    if is_OK or do_show_uncorrect:
        img = folium.raster_layers.ImageOverlay(
            name="Инструмент для разметки делянок",
            image=path_img_mod,
            bounds=bounds,
            opacity=opacity,    
            control=False,
            zindex=1,
        ).add_to(feature_group_yaw)

        img = folium.raster_layers.ImageOverlay(
            name="Инструмент для разметки делянок",
            image=path_img_fix,
            bounds=bounds,
            opacity=opacity,
            control=False,
            zindex=1,
        ).add_to(feature_group_fix)


if __name__ == "__main__":
    m = folium.Map([latitude, longtitude], tiles=None,
                   prefer_canvas=True, control_scale=True, zoom_start=21)
    base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
    folium.TileLayer(tiles='OpenStreetMap', max_zoom=24).add_to(base_map)
    base_map.add_to(m)

    filenames = os.listdir(path_field_day + 'src')
    if not os.path.exists(path_log_metadata):
        make_dirs(path_field_day)
        handle_metadata(filenames, path_field_day)

    df_metadata = pd.read_csv(path_log_metadata)
    df_metadata['border'] = df_metadata['border'].apply(
        lambda x: json.loads(x))

    feature_group_yaw = folium.FeatureGroup(
        name='rotate_by_yaw', overlay=False)
    feature_group_fix = folium.FeatureGroup(
        name='rotate_by_90º', overlay=False)

    do_rewrite = False
    num_of_images = df_metadata.shape[0]
    if len(os.listdir(path_field_day + 'mod')) < num_of_images or do_rotate_force:
        do_rewrite = True
    for i in range(len(filenames)):
        draw_image_on_map(i, df_metadata, do_rewrite)

    feature_group_yaw.add_to(m)
    feature_group_fix.add_to(m)

    draw_protocol(df_metadata, m)
    d = Draw(export=True).add_to(m)
    folium.LayerControl().add_to(m)
    m.save('maps/markup.html')