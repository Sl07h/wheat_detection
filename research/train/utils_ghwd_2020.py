import cv2
import os
import json
import numpy as np
import pandas as pd

dataset_dir = 'data/gwhd_2020'

train_dirs = [
    'arvalis_1',    # Франция
    'arvalis_2',    # Франция
    'arvalis_3',    # Франция
    'ethz_1',       # Швейцария
    'inrae_1',      # Франция
    'rres_1',       # Англия
    'usask_1',      # Канада
]

test_dirs = [
    'uq_1',         # Австралия
    'utokyo_1',     # Япония
    'utokyo_2',     # Япония
    'nau_1',        # Китай
]


def convert_json_to_csv():
    for dir in train_dirs:
        df = pd.DataFrame(columns=['image_path', 'bbox_list'])
        path_by_id = {}
        bbox_by_id = {}
        path_to_json = f'{dataset_dir}/{dir}.json'
        f = open(path_to_json) 
        data = json.load(f)
        
        annotations_list = data['images']
        for image in annotations_list:
            path_by_id[image['id']] = image['path']
            bbox_by_id[image['id']] = []
        
        annotations_list = data['annotations']
        for bbox_str in annotations_list:
            image_id, bbox = bbox_str['image_id'], bbox_str['bbox']
            bbox_by_id[image_id] += [bbox]

        for image_id in path_by_id.keys():
            image_path = path_by_id[image_id]
            bbox_list  = bbox_by_id[image_id]
            df = df.append({'image_path': image_path, 'bbox_list': bbox_list}, ignore_index=True)
        df.to_csv(f'{dataset_dir}/{dir}.csv', index=False)


def check_markup():
    for dir in train_dirs:
        dir_path = f'research/result/visualisation/{dir}'
        try:
            os.mkdir(dir_path)
        except:
            print(f'[+] папка {dir} уже существует')
        df = pd.read_csv(f'{dataset_dir}/{dir}.csv')
        df['bbox_list'] = df['bbox_list'].apply(lambda x: json.loads(x))
        for i in range(df.shape[0] // 50):
            line = df.iloc[i]
            image_path = line['image_path']
            bbox_list = np.array(line['bbox_list'])
            im = cv2.imread(f'{dataset_dir}/{dir}/{image_path}')
            for bbox in bbox_list:
                im = cv2.rectangle(
                    im,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    (220, 0, 0), 3)
            cv2.imwrite(f'{dir_path}/{image_path}', im)


convert_json_to_csv()
# check_markup()
