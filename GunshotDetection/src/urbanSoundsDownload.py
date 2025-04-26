import os
import tarfile
import zipfile
import gdown
import requests
from tqdm import tqdm

# Config
URBAN_ID = "1sgymYB-HCNmhdGhfXsdnhqXM6Wxd6SoV"  
URBAN_ARCHIVE = "UrbanSound8K.tar.gz"

GUNSHOT_URL = "https://zenodo.org/records/7004819/files/edge-collected-gunshot-audio.zip?download=1"
GUNSHOT_ARCHIVE = "edge-collected-gunshot-audio.zip"

def download_from_google_drive(file_id, output):
    print("Downloading UrbanSound8K...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)
    print("UrbanSound8K download complete.")

def download_from_url(url, output):
    print(f"Downloading {output}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(output, 'wb') as f, tqdm(
            desc=output,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"{output} download complete.")

def extract_tar_gz(filepath, extract_path="."):
    print(f"Extracting {filepath}...")
    with tarfile.open(filepath, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting .tar.gz"):
            tar.extract(member, path=extract_path)
    print(f"{filepath} extraction complete.")

def extract_zip(filepath, extract_path="."):
    print(f"Extracting {filepath}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting .zip"):
            zip_ref.extract(file, extract_path)
    print(f"{filepath} extraction complete.")

def delete_file(filepath):
    print(f"Deleting {filepath}...")
    if os.path.exists(filepath):
        os.remove(filepath)
        print("Deleted.")
    else:
        print("File not found.")

def main():
    print("Starting full dataset setup...")

    download_from_google_drive(URBAN_ID, URBAN_ARCHIVE)
    extract_tar_gz(URBAN_ARCHIVE)
    delete_file(URBAN_ARCHIVE)

    download_from_url(GUNSHOT_URL, GUNSHOT_ARCHIVE)
    extract_zip(GUNSHOT_ARCHIVE)
    delete_file(GUNSHOT_ARCHIVE)

    print("All datasets are ready.")

if __name__ == "__main__":
    main()
