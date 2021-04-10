# Система компьютерного зрения для анализа урожайности посевов пшеницы

Работа была выполнена в рамках практики на ФИТ НГУ и проходила в ИЦиГ СО РАН.





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
 ┣ 📜lib.py
 ┣ 📜markup.py
 ┗ 📜visualization.py
📜.gitignore
📜README.md
```

field_name_year может содержать в конце номер сезона, если в году несколько урожаев. 

wheat_plots.geojson размечается вручную 1 раз за сезон.



## Установка и запуск

```
folium==0.12.1
pandas==1.2.3
exif
shapely==1.7.1
numba==0.53.1
opencv-python==4.5.1
scipy==1.6.2
```