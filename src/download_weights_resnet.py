
import os
import urllib.request
import ssl
import certifi

url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"

save_path = os.path.expanduser(
    "~/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
)

os.makedirs(os.path.dirname(save_path), exist_ok=True)

context = ssl.create_default_context(cafile=certifi.where())

print("Starting ResNet50 weights download...", flush=True)

try:
    with urllib.request.urlopen(url, context=context, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        with open(save_path, "wb") as file:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                file.write(chunk)
                downloaded += len(chunk)

                if total_size:
                    percent = downloaded / total_size * 100
                    print(f"Downloaded: {percent:.1f}%", flush=True)

    print("Download successful!", flush=True)
    print("Saved to:", save_path, flush=True)

except Exception as error:
    print("Download failed:", repr(error), flush=True)
