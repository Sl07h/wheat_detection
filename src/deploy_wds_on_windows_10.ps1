Function DownloadFromGoogleDrive ([String]$URL)
{
    Invoke-WebRequest -Uri $URL -OutFile qwe.txt -SessionVariable googleDriveSession
    $text = Get-Content .\qwe.txt -Raw 
    $pattern="confirm=(\w\w\w\w)"
    $text -match $pattern
    $token = $matches[0]
    $DownloadURL = -join($URL, '&', $token)
    $DownloadURL
    $res = Invoke-WebRequest -Uri $DownloadURL -OutFile "arch.zip" -WebSession $googleDriveSession
    $res
    7z x arch.zip
    rm arch.zip
    rm qwe.txt
}

# 0. подключим протокол передачи
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# 1.python и pip
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe" -OutFile "python-3.8.10-amd64.exe"
# https://stackoverflow.com/questions/1741490/how-to-tell-powershell-to-wait-for-each-command-to-end-before-starting-the-next
./python-3.8.10-amd64.exe /quiet `
                        PrependPath=1 `
                        InstallAllUsers=0 `
                        Include_launcher=0 `
                        Include_doc=0 `
                        Include_tcltk=0 `
                        Include_test=0 | Out-Null
rm python-3.8.10-amd64.exe
$env:Path += ";" + $env:LOCALAPPDATA + "\Programs\Python\Python38"

# 2. 7zip
Invoke-WebRequest -Uri "https://www.7-zip.org/a/7z2107-x64.exe" -OutFile "7z2107-x64.exe"
Start-Process -FilePath 7z2107-x64.exe -Args "/S" -Verb RunAs -Wait
rm 7z2107-x64.exe
$env:Path += ";C:\Program Files\7-Zip"

# 3. Сама система
Invoke-WebRequest -Uri "https://github.com/Sl07h/wheat_detection/archive/refs/heads/master.zip" -OutFile "master.zip"
7z x master.zip
mv wheat_detection-master wheat_detection
rm master.zip
cd wheat_detection

# 4.exiftool
Invoke-WebRequest -Uri "https://exiftool.org/exiftool-12.34.zip" -OutFile "exiftool-12.34.zip"
7z x exiftool-12.34.zip
mkdir exiftool-12.34
mv "exiftool(-k).exe" "exiftool-12.34/exiftool.exe"
rm exiftool-12.34.zip

# 5. создаём виртуальную переменную
python -m venv wds_venv

# 6. активируем её
wds_venv/Scripts/Activate.ps1

# 7. ставим библиотеки питона
Invoke-WebRequest -Uri "https://drive.google.com/uc?export=download&id=1K9e_K0wlHYYqY5cy3Zxx4_hMB2lR6qlZ" -OutFile "pycocotools-2.0.4-cp38-cp38-win_amd64.zip"
$env:Path += ";C:\Program Files\7-Zip"
7z x pycocotools-2.0.4-cp38-cp38-win_amd64.zip
pip install pycocotools-2.0.4-cp38-cp38-win_amd64.whl
rm pycocotools-2.0.4-cp38-cp38-win_amd64.zip
rm pycocotools-2.0.4-cp38-cp38-win_amd64.whl
pip install -r requirements.txt


# 8. качаем веса нейросетей
mkdir weights
cd weights
# 8.1 frcnn
DownloadFromGoogleDrive("https://drive.google.com/uc?id=1nCK-yJ-Y8jTg-hENr31HXNMGTb2OuYgV&export=download")
# 8.2 effdet
DownloadFromGoogleDrive("https://drive.google.com/uc?id=1--0eoJ-SEr0bohgo0E7LOj0gJr9vcNnA&export=download")
cd ../

# 9. Загрузка данных
Invoke-WebRequest -Uri "https://drive.google.com/uc?export=download&id=18KD97J_GbJ2xBMs6jAIfJS-xBk2tyr7N" -OutFile "data.zip"
7z x data.zip
rm data.zip
