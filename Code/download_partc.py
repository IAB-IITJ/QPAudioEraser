import os
import time
import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry
from urllib3.exceptions import ProtocolError, IncompleteRead

# URL of the dataset
url = "https://cn01.mmai.io/download/voxceleb?key=7085032242d2c994b8f08a55ca2ff9af83017e82dcf409ed35a9ec1ae33fa28926d8a2235f003b7ad0bdc1d4352d517bff6c94ec3ffc3b5338d19a507a08a36774a4826e3de2457941009f638880887ad675d7bd1917bd3d51a0ab66c5d89b6735f532c2271b709890f35e3670ebc2f61f807a1d2d0404822e7cd7fe532a4e55&file=vox1_dev_wav_partac"

# Local filename
filename = "vox1_dev_wav_partac"

# Set chunk size
chunk_size = 4096  # Reduce chunk size to handle network issues

# Maximum retries
max_retries = 5
retry_wait = 10  # Seconds to wait before retrying

# Create a session with retry mechanism
session = requests.Session()
retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_file_size():
    """Check the existing file size to resume downloading."""
    return os.path.getsize(filename) if os.path.exists(filename) else 0

def download_file():
    """Downloads the file with error handling and auto-resume."""
    headers = {}
    resume_byte_pos = get_file_size()

    # If resuming, add a 'Range' header
    if resume_byte_pos > 0:
        headers["Range"] = f"bytes={resume_byte_pos}-"

    for attempt in range(max_retries):
        try:
            response = session.get(url, stream=True, verify=False, headers=headers, timeout=60)
            response.raise_for_status()

            # Get total size including resumed part
            total_size = int(response.headers.get("content-length", 0)) + resume_byte_pos

            with open(filename, "ab") as file, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=filename, initial=resume_byte_pos
            ) as progress_bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

            print(f"\nDownload complete: {filename}")
            return  # Exit function if download succeeds

        except (requests.exceptions.ChunkedEncodingError, IncompleteRead, ProtocolError) as e:
            print(f"Network error: {e}. Retrying in {retry_wait} seconds...")
            time.sleep(retry_wait)

    print("Download failed after multiple attempts.")

download_file()
