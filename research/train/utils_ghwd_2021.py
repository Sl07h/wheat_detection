import os
import shutil
import pandas as pd


def split_into_dirs(root_dir):
    df_train = pd.read_csv(f'{root_dir}/gwhd_2021_train.csv')
    df_val   = pd.read_csv(f'{root_dir}/gwhd_2021_val.csv')
    df_test  = pd.read_csv(f'{root_dir}/gwhd_2021_test.csv')
    df_all   = pd.concat([df_train, df_val, df_test])
    dirs     = df_all['domain'].unique()
    print(dirs, len(dirs))
    for dir in dirs:
        os.mkdir(f'{root_dir}/{dir}')
    for index, row in df_all.iterrows():
        name = row['image_name']
        dir = row['domain']
        path_src = f'{root_dir}/images/{name}'
        path_dst = f'{root_dir}/{dir}/{name}'
        shutil.copy(path_src, path_dst)


split_into_dirs('data/gwhd_2021_copy')
