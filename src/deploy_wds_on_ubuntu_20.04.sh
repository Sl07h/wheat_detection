sudo apt-get update
sudo apt install git
sudo apt install p7zip-full
sudo apt install python3-pip
sudo apt install python3-venv

git clone https://github.com/Sl07h/wheat_detection/
cd wheat_detection

# 1. install exiftool
wget https://github.com/exiftool/exiftool/archive/refs/tags/12.34.zip
7z x 12.34.zip
rm 12.34.zip

# 2. create virtual enviroment for wheat detection system libraries
python3 -m venv wds_venv

# 3. activate virtual enviroment
. wds_venv/bin/activate

# 4. install python libraries
pip install -r requirements.txt

# 5. download model weights
mkdir weights
cd weights
# 5.1 frcnn
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nCK-yJ-Y8jTg-hENr31HXNMGTb2OuYgV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nCK-yJ-Y8jTg-hENr31HXNMGTb2OuYgV" -O qwe.zip && rm -rf /tmp/cookies.txt
7z x qwe.zip

# 5.2 effdet
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16q8F-jEOSvI0VVSDEVW4ZEA9di1Dsp0u' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16q8F-jEOSvI0VVSDEVW4ZEA9di1Dsp0u" -O qwe.zip && rm -rf /tmp/cookies.txt
7z x qwe.zip;

rm qwe.zip
cd ../