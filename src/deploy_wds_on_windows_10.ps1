Function DownloadFromGoogleDrive ([String]$URL)
{
    (Invoke-WebRequest -Uri $URL).Links.Href > qwe.txt
    (Invoke-WebRequest -Uri $URL).Links.Href > qwe.txt
    $text = Get-Content .\qwe.txt -Raw 
    $pattern="confirm=(\w\w\w\w)"
    $text -match $pattern
    $token = $matches[0]
    $DownloadURL = -join($URL, '&', $token)
    $DownloadURL
    Invoke-WebRequest -Uri $DownloadURL -OutFile "arch.zip"
    7z x arch.zip
    rm arch.zip
    rm qwe.txt
}

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# 1. 7zip
Invoke-WebRequest -Uri "https://www.7-zip.org/a/7z2107-x64.exe" -OutFile "7z2107-x64.exe"
Start-Process -FilePath 7z2107-x64.exe -Args "/S" -Verb RunAs -Wait
rm 7z2107-x64.exe

# 2. Сама система
Invoke-WebRequest -Uri "https://github.com/Sl07h/wheat_detection/archive/refs/heads/master.zip" -OutFile "master.zip"
7z x master.zip
mv wheat_detection-master wheat_detection
rm master.zip
cd wheat_detection

# 3.python и pip
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe" -OutFile "python-3.8.10-amd64.exe"
# https://stackoverflow.com/questions/1741490/how-to-tell-powershell-to-wait-for-each-command-to-end-before-starting-the-next
./python-3.8.10-amd64.exe /quiet PrependPath=1 InstallAllUsers=0 InstallLauncherAllUsers=0 Include_launcher=0 Include_doc=0 Include_tcltk=0 Include_test=0  | Out-Null
rm python-3.8.10-amd64.exe


# 4.exiftool
Invoke-WebRequest -Uri "https://exiftool.org/exiftool-12.34.zip" -OutFile "exiftool-12.34.zip"
7z x exiftool-12.34.zip
mkdir exiftool-12.34
mv "exiftool(-k).exe" "exiftool-12.34/exiftool.exe"
rm "exiftool-12.34.zip"

# 5. create virtual enviroment for wheat detection system libraries
python -m venv wds_venv

# 6. activate virtual enviroment
wds_venv/Scripts/Activate.ps1

7. install python libraries
pip install -r requirements.txt


# 8. download model weights
mkdir weights
cd weights
# 8.1 frcnn
DownloadFromGoogleDrive("https://drive.google.com/uc?id=1nCK-yJ-Y8jTg-hENr31HXNMGTb2OuYgV&export=download")

# 8.2 effdet
DownloadFromGoogleDrive("https://drive.google.com/uc?id=1--0eoJ-SEr0bohgo0E7LOj0gJr9vcNnA&export=download")

cd ../

