﻿# Система компьютерного зрения для анализа урожайности посевов пшеницы

Работа была выполнена в рамках практики на ФИТ НГУ и проходила в ИЦиГ СО РАН.




## Структура папок и назначение файлов

```
📦data
 ┗ 📂test_set1             поле
 ┃ ┗ 📂attempt
 ┃ ┃ ┣ 📂src               исходные файлы
 ┃ ┃ ┣ 📂log
 ┃ ┃ ┃ ┣ 📜test_set1.07_25.frcnn.400.bboxes.csv       найденные колоски
 ┃ ┃ ┃ ┣ 📜test_set1.07_25.frcnn.400.metadata.csv     обработанные метаданные
 ┃ ┃ ┃ ┗ 📜test_set1.07_25.frcnn.400.result.csv       файл результатов (делянка - число колосьев)
 ┃ ┃ ┣ 📂mod               повернутые и сжатые для разметки делянок
 ┃ ┃ ┣ 📂tmp               временные файлы при работе exiftool
 ┃ ┗ 📜test_set1.geojson
📦maps
 ┣ 📜visualization.html
 ┗ 📜markup.html
📦src
 ┣ 📜clean.py
 ┣ 📜install.sh
 ┣ 📜main.py
 ┣ 📜plot.py
 ┗ 📜wds.py
📦weights
📜.gitignore
📜README.md
```

<!-- field_name_year может содержать в конце номер сезона, если в году несколько урожаев.  -->

Файл *.geojson размечается вручную 1 раз за сезон.




## Установка и запуск

### Установка

```
git clone https://github.com/Sl07h/wheat_detection/
cd wheat_detection
. src/install.sh
```

### Запуск

Для того, чтобы получить результат по отдельным делянкам, нужно:

1. Распределить снимки по папкам полей и дней съёмки внутри директории data

2. Выполнить команду ```python src/plot.py``` в тот, день когда видна структура делянок.

3. Разметить делянки при помощи карты в директории ```maps``` и сохранить полученный geojson-файл в директории поля.

4. Отредактировать geojson файл, подписав сорта и условия эксперимента. Можно использовать сервис https://geojson.io

5. Выполнить команду ```python src/main.py```. Получится карта и csv-файл с результатами


Подробная инструкция:

