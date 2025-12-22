import os
import time
import zipfile
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from azure.storage.blob import BlobServiceClient
load_dotenv()

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")
AZURE_BLOB_FOLDER = os.getenv("AZURE_BLOB_FOLDER")

CSV_FOLDER_PATH = os.getenv("CSV_FOLDER_PATH")


# Initialize Azure Blob Service Client
AZURE_CONN_STR = (
    f"DefaultEndpointsProtocol=https;"
    f"AccountName={AZURE_STORAGE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_STORAGE_ACCOUNT_KEY};"
    f"EndpointSuffix=core.windows.net"
)
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
container_client = blob_service_client.get_container_client(AZURE_BLOB_CONTAINER)


# Download folder for Selenium
download_dir = os.path.abspath("downloaded_attachments")
os.makedirs(download_dir, exist_ok=True)


# Configure Selenium Chrome options to automatically download PDFs and ZIPs instead of opening them
chrome_options = Options()
chrome_options.page_load_strategy = 'eager'  # Return control after DOMContentLoaded but before full load
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_dir,
    "plugins.always_open_pdf_externally": True,  # Force download PDFs and ZIPs
    "download.prompt_for_download": False,
    "download.directory_upgrade": True,
})
driver = webdriver.Chrome(options=chrome_options)
driver.set_page_load_timeout(120)  # Set page load timeout to 2 minutes


def blob_exists(blob_name: str) -> bool:
    blob_path = f"{AZURE_BLOB_FOLDER}/{blob_name}"
    blob_client = container_client.get_blob_client(blob_path)
    return blob_client.exists()


def upload_file_to_blob(file_path: str, blob_name: str):
    blob_path = f"{AZURE_BLOB_FOLDER}/{blob_name}"
    blob_client = container_client.get_blob_client(blob_path)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=True)
    print(f"[+] Uploaded {blob_name} to Azure Blob Storage under {blob_path}")


def wait_for_download(filename, timeout=120):
    filepath = os.path.join(download_dir, filename)
    waited = 0
    while waited < timeout:
        # File exists and not a partial .crdownload
        if os.path.exists(filepath) and not os.path.exists(filepath + ".crdownload"):
            return filepath
        time.sleep(1)
        waited += 1
    raise TimeoutError(f"Download timed out for file {filename}")


def safe_navigate(url, retries=2):
    for attempt in range(retries):
        try:
            driver.get(url)
            return
        except TimeoutException:
            print(f"[!] Timeout loading {url}, retry {attempt + 1} of {retries}")
    raise TimeoutException(f"Failed to load {url} after {retries} retries")


def extract_pdfs_from_zip(zip_path, extract_to):
    print(f"[•] Extracting PDFs from: {zip_path}")
    extracted_files = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            pdf_files = [f for f in zip_ref.namelist() if f.lower().endswith('.pdf')]
            for pdf_file in pdf_files:
                print(f"[•] Extracting {pdf_file}")
                extracted_path = zip_ref.extract(pdf_file, extract_to)
                extracted_files.append(extracted_path)
        print(f"[+] Extracted {len(pdf_files)} PDFs.")
    except Exception as e:
        print(f"[!] Failed to extract PDFs from ZIP {zip_path}: {e}")
    return extracted_files


def process_attachments(csv_path: str):
    df = pd.read_csv(csv_path, usecols=['LINK'])


    for link in df['LINK'].dropna().unique():
        link = link.strip()
        if not link:
            continue


        file_name = urlparse(link).path.split('/')[-1]


        if file_name.lower().endswith('.zip'):
            if blob_exists(file_name):
                print(f"[=] ZIP file already in Azure Blob, skipping: {file_name}")
                continue
            try:
                print(f"[↓] Navigating to ZIP URL: {link}")
                safe_navigate(link)
                zip_path = wait_for_download(file_name)
                print(f"[+] ZIP download complete: {zip_path}")


                # Extract PDFs and upload them
                pdf_files = extract_pdfs_from_zip(zip_path, download_dir)
                for pdf_path in pdf_files:
                    pdf_name = os.path.basename(pdf_path)
                    if not blob_exists(pdf_name):
                        upload_file_to_blob(pdf_path, pdf_name)
                    else:
                        print(f"[=] PDF {pdf_name} already exists in Azure Blob, skipping upload.")
            except Exception as e:
                print(f"[!] Error processing ZIP {file_name}: {e}")


        elif file_name.lower().endswith('.pdf'):
            if blob_exists(file_name):
                print(f"[=] PDF file already in Azure Blob, skipping: {file_name}")
                continue
            try:
                print(f"[↓] Navigating to PDF URL: {link}")
                safe_navigate(link)
                pdf_path = wait_for_download(file_name)
                print(f"[+] PDF download complete: {pdf_path}")
                upload_file_to_blob(pdf_path, file_name)
            except Exception as e:
                print(f"[!] Error processing PDF {file_name}: {e}")
        else:
            print(f"[!] Unsupported file extension for link, skipping: {file_name}")


if __name__ == "__main__":
    try:
        # Loop through all CSV files in the folder
        for csv_file in os.listdir(CSV_FOLDER_PATH):
            if csv_file.lower().endswith(".csv"):
                csv_path = os.path.join(CSV_FOLDER_PATH, csv_file)
                print(f"\n=== Processing CSV: {csv_file} ===")
                try:
                    process_attachments(csv_path)
                except Exception as e:
                    print(f"[!] Error processing {csv_file}: {e}")
    finally:
        driver.quit()
