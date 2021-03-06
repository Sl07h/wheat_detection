﻿# Система компьютерного зрения для анализа урожайности посевов пшеницы

Работа была выполнена в рамках практики на ФИТ НГУ и проходила в ИЦиГ СО РАН.



## Установка и запуск

Для того, чтобы получить результат по отдельным делянкам, нужно:

1. Распределить снимки по папкам полей и дней съёмки внутри директории data

2. Запустить скрипт ```markup.py``` в тот, день когда видна структура делянок.

3. Разметить делянки при помощи карты в ```maps/markup.html``` и сохранить полученный geojson-файл в директории поля.

4. Запустить скрипт ```visualisation.py``` в этой папке

Все зависимости собраны в файле requirements.txt.

Pytorch ставить коммандой ```pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html```



[Веса сети faster rcnn с kaggle](
https://www.kaggle.com/dataset/7d5f1ed9454c848ecb909c109c6fa8e573ea4de299e249c79edc6f47660bf4c5?select=fasterrcnn_resnet50_fpn_best.pth
)





## Структура папок и назначение файлов

```
📦data
 ┗ 📂field_name_year[_season]
 ┃ ┣ 📂mm_dd
 ┃ ┃ ┣ 📂src        исходные файлы
 ┃ ┃ ┣ 📂log        результаты распознания и метаданные в одной таблице
 ┃ ┃ ┣ 📂mod        повернутые и сжатые для разметки делянок
 ┃ ┃ ┗ 📂tmp        временные файлы
 ┃ ┗ 📜wheat_plots.geojson
📦maps
 ┣ 📜visualization.html
 ┗ 📜markup.html
📦settings
📦src
 ┣ 📜clean.py
 ┣ 📜frcnn.py
 ┣ 📜lib.py
 ┣ 📜markup.py
 ┗ 📜visualization.py
📦weights
📜.gitignore
📜README.md
```

field_name_year может содержать в конце номер сезона, если в году несколько урожаев. 

wheat_plots.geojson размечается вручную 1 раз за сезон.



