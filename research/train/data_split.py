# выделим выборки из соревнований 2020 и 2021 года на основе таблиц
# https://spj.sciencemag.org/journals/plantphenomics/2021/9846158/tab1/
# https://spj.sciencemag.org/journals/plantphenomics/2021/9846158/tab3/

# [+] 2020 год
gwhd_2020_train = [
    'Arvalis_1', # 66
    'Arvalis_2', # 401
    'Arvalis_3', # 588
    'Arvalis_4', # 204
    'arvalis_5', # 448
    'arvalis_6', # 160
    'Ethz_1',    # 747
    'Inrae_1',   # 176
    'Rres_1',    # 432
    'Usask_1',   # 200
]                # 3422 изображения

gwhd_2020_test = [
    'Utokyo_1',  # 538
    'Utokyo_2',  # 456
    'Utokyo_3',  # 120
    'NAU_1',     # 20
    'UQ_1',      # 22
    'UQ_2',      # 16
    'UQ_3',      # 14
    'UQ_4',      # 30
    'UQ_5',      # 30
    'UQ_6',      # 30
]                # 1276 изображений


# [+] 2021 год
# Ethz_1, Rres_1, Inrae_1, Arvalis (all), NMBU (all), ULiège-GxABT (all)
gwhd_2021_train = [
    'Ethz_1',         # 747
    'Rres_1',         # 432
    'Inrae_1',        # 176
    'Arvalis_1',      # 66
    'Arvalis_2',      # 401
    'Arvalis_3',      # 588
    'Arvalis_4',      # 204
    'arvalis_5',      # 448
    'arvalis_6',      # 160
    'arvalis_7',      # 24
    'arvalis_8',      # 20
    'arvalis_9',      # 32
    'arvalis_10',     # 60
    'arvalis_11',     # 60
    'arvalis_12',     # 29
    'NMBU_1',         # 82
    'NMBU_2',         # 98
    'ULiège-GxABT_1', # 30
]                     # 3657 изображений
# UQ_1 to UQ_6, Utokyo (all), NAU_1, Usask_1
gwhd_2021_val = [
    'UQ_1',      # 22
    'UQ_2',      # 16
    'UQ_3',      # 14
    'UQ_4',      # 30
    'UQ_5',      # 30
    'UQ_6',      # 30
    'Utokyo_1',  # 538
    'Utokyo_2',  # 456
    'Utokyo_3',  # 120
    'NAU_1',     # 20
    'Usask_1',   # 200
]                # 1476 изображений
# UQ_7 to UQ_11, Ukyoto_1, NAU_2 and NAU_3, ARC_1, CIMMYT (all), KSU (all), Terraref (all)
gwhd_2021_test = [
    'UQ_7',      # 17
    'UQ_8',      # 41
    'UQ_9',      # 33
    'UQ_10',     # 53
    'UQ_11',     # 42
    'Ukyoto_1',  # 60
    'NAU_2',     # 100
    'NAU_3',     # 100
    'ARC_1',     # 30
    'CIMMYT_1',  # 69
    'CIMMYT_2',  # 77
    'CIMMYT_3',  # 60
    'KSU_1',     # 100
    'KSU_2',     # 100
    'KSU_3',     # 95
    'KSU_4',     # 60
    'Terraref_1',# 144
    'Terraref_2',# 106
]                # 1287 изображений
