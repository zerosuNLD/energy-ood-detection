import os
import urllib.request
import tarfile
import zipfile

def download_and_extract(url, out_file, extract_to):
    if not os.path.exists(out_file):
        print(f"Downloading {out_file}...")
        urllib.request.urlretrieve(url, out_file)
        print("Download complete.")
    if not os.path.exists(extract_to):
        print(f"Extracting {out_file}...")
        if out_file.endswith('.tar.gz'):
            with tarfile.open(out_file, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(extract_to))
        elif out_file.endswith('.zip'):
            with zipfile.ZipFile(out_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(extract_to))
        print("Extraction complete.")

base_dir = os.path.join(os.getcwd(), 'data')
lsun_tar = os.path.join(base_dir, 'LSUN_resize.tar.gz')
lsun_dir = os.path.join(base_dir, 'LSUN_resize')

download_and_extract(
    "https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz?dl=1",
    lsun_tar,
    lsun_dir
)
