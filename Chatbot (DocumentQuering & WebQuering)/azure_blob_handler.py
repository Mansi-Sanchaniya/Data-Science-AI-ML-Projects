from azure.storage.blob import BlobServiceClient
import os
import json
from dotenv import load_dotenv
load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

if not AZURE_STORAGE_CONNECTION_STRING:
    raise RuntimeError(" AZURE_STORAGE_CONNECTION_STRING not set in .env")

os.environ["AZURE_STORAGE_CONNECTION_STRING"] = AZURE_STORAGE_CONNECTION_STRING

# Patch requests if needed
import urllib3
urllib3.disable_warnings()

def download_pdfs_from_azure(connection_string, container_name, base_dir):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string, connection_verify=False)
    container_client = blob_service_client.get_container_client(container_name)

    for blob in container_client.list_blobs():
        if not blob.name.lower().endswith(".pdf"):
            continue

        # Save with folder structure like Compliance/abc.pdf
        relative_path = os.path.normpath(blob.name)  
        local_path = os.path.join(base_dir, relative_path)

        if os.path.exists(local_path):
            print(f" Skipping already downloaded: {relative_path}")
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            print(f" Downloading: {relative_path}")
            with open(local_path, "wb") as f:
                f.write(container_client.download_blob(blob.name).readall())
        except Exception as e:
            print(f"Failed to download {blob.name}: {e}")
