from pydantic import SecretStr
import os
import shutil
import asyncio
from time import sleep
from browser_use import Agent, Browser, ChatAzureOpenAI
from dotenv import load_dotenv
load_dotenv()


default_download_folder = os.getenv("DEFAULT_DOWNLOAD_FOLDER")
target_folder = os.getenv("TARGET_FOLDER")

os.makedirs(target_folder, exist_ok=True)

browser = Browser(
    executable_path=os.getenv("CHROME_EXECUTABLE_PATH"),
    user_data_dir=os.getenv("CHROME_USER_DATA_DIR"),
    profile_directory=os.getenv("CHROME_PROFILE_DIRECTORY"),
)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = SecretStr(os.getenv("AZURE_OPENAI_API_KEY"))

# Initialize the model
llm = ChatAzureOpenAI(
    model="gpt-4o-mini",
    api_version="2025-01-01-preview",
    api_key=AZURE_OPENAI_API_KEY.get_secret_value(),
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    temperature=0.0  # Very deterministic, no creativity
)

task_description = f"""
You must follow these exact steps ONLY, no additional actions or improvisation. While scrolling make sure to scroll slowly and scroll little to identify files quickly:
1. Go to https://www.nseindia.com/resources/exchange-communication-circulars#
2. Find on the page 1D button and click it
2. Parse the page and identify a link or a button or a csv file where Download (.csv) is written and click Download (.csv). (It will be clearly visible on the right half of the page above the tabular structure find properly)
3. Download the CSV file by clicking on the link or button Download (.csv) (will save to default downloads folder).
Return a list of all downloaded CSV files with their name, URL, and local path.
"""

agent = Agent(
    task=task_description,
    llm=llm,
    browser=browser,
)

def move_csv_files(src_folder, dest_folder):
    for filename in os.listdir(src_folder):
        if filename.lower().endswith(".csv"):
            src_path = os.path.join(src_folder, filename)
            dest_path = os.path.join(dest_folder, filename)
            if os.path.exists(dest_path):
                print(f"Skipping {filename} as it already exists in the target folder.")
                continue
            print(f"Moving {src_path} to {dest_path}")
            shutil.move(src_path, dest_path)

async def main():
    try:
        history = await agent.run()
        print("Agent run history:")
        print(history)  # print full history object
        
        # Attempt to print downloads if present as a key or attribute
        downloads = getattr(history, "downloads", None)
        if callable(downloads):
            downloads = downloads()
        if downloads:
            print("Downloaded Files:")
            for d in downloads:
                print(f"- Filename: {d.get('file_name', 'N/A')}, URL: {d.get('url', 'N/A')}, Local Path: {d.get('path', 'N/A')}")
        else:
            print("No downloads found in history object.")

        print("\nAll actions taken by the agent:")
        print(getattr(history, "actions", lambda: [])())
        
        # Wait for downloads then move files as before
        print("Waiting for downloads to complete...")
        await asyncio.sleep(10)
        move_csv_files(default_download_folder, target_folder)
        print("File move complete.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())

    