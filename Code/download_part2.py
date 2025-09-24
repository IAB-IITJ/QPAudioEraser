import requests
from tqdm import tqdm

# URL of the dataset
url = "https://cn01.mmai.io/download/voxceleb?key=7085032242d2c994b8f08a55ca2ff9af83017e82dcf409ed35a9ec1ae33fa28926d8a2235f003b7ad0bdc1d4352d517bff6c94ec3ffc3b5338d19a507a08a36774a4826e3de2457941009f638880887ad675d7bd1917bd3d51a0ab66c5d89b6735f532c2271b709890f35e3670ebc2f61f807a1d2d0404822e7cd7fe532a4e55&file=vox1_dev_wav_partab"

# Local filename
filename = "vox1_dev_wav_partab"

# Disable SSL verification (temporary workaround)
response = requests.get(url, stream=True, verify=False)
response.raise_for_status()

total_size = int(response.headers.get('content-length', 0))
chunk_size = 8192

with open(filename, "wb") as file, tqdm(
    total=total_size, unit="B", unit_scale=True, desc=filename
) as progress_bar:
    for chunk in response.iter_content(chunk_size=chunk_size):
        if chunk:
            file.write(chunk)
            progress_bar.update(len(chunk))

print(f"\nDownload complete: {filename}")
