DownloadFromGoogleDrive () {
    # $1 = file ID
    URL="https://docs.google.com/uc?export=download&id="${1}
    token=$(wget --quiet \
                 --save-cookies /tmp/cookies.txt \
                 --keep-session-cookies \
                 --no-check-certificate $URL \
                 -O- | sed -rn 's/.*confirm=(\w\w\w\w).*/\1/p')
    DownloadURL=$URL"&confirm="$token
    echo $DownloadURL
    wget --load-cookies /tmp/cookies.txt $DownloadURL -O arch.zip 
    7z x arch.zip;
    rm qwe.zip
}

# 0. подключим обновим адреса, откуда качать пакеты
sudo apt-get update

# 1.python и pip
sudo apt install python3-pip
sudo apt install python3-venv

# 2. 7zip
sudo apt install p7zip-full

# 3. Сама система
wget "https://github.com/Sl07h/wheat_detection/archive/refs/heads/master.zip"
7z x master.zip
mv wheat_detection-master wheat_detection
rm master.zip
cd wheat_detection

# 4.exiftool
wget https://github.com/exiftool/exiftool/archive/refs/tags/12.34.zip
7z x 12.34.zip
rm 12.34.zip

# 5. создаём виртуальную переменную
python3 -m venv wds_venv

# 6. активируем её
. wds_venv/bin/activate

# 7. ставим библиотеки питона
pip install -r requirements.txt

# 8. качаем веса нейросетей
mkdir weights
cd weights
# 8.1 frcnn
DownloadFromGoogleDrive "1nCK-yJ-Y8jTg-hENr31HXNMGTb2OuYgV"
# 8.2 effdet
DownloadFromGoogleDrive "1--0eoJ-SEr0bohgo0E7LOj0gJr9vcNnA"
cd ../

# 9. Загрузка данных
wget "https://drive.google.com/uc?export=download&id=18KD97J_GbJ2xBMs6jAIfJS-xBk2tyr7N" -O "data.zip"
7z x data.zip
rm data.zip
