import os

folders = [
    "data/videos",
    "data/frames",
    "data/labels",
    "scripts"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")
