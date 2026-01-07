import urllib.request
import sys

urls = [
    "https://storage.googleapis.com/mediapipe-models/pose/pose_landmarker/float16/1/pose_landmarker_lite.task",
    "https://storage.googleapis.com/mediapipe-models/pose/pose_landmarker/float16/latest/pose_landmarker_lite.task",
    "https://storage.googleapis.com/mediapipe-models/pose/pose_landmarker/full/1/pose_landmarker_full.task",
    "https://storage.googleapis.com/mediapipe-models/pose/pose_landmarker/float16/1/pose_landmarker_heavy.task"
]

for url in urls:
    print(f"Trying {url}...", flush=True)
    try:
        urllib.request.urlretrieve(url, "pose_landmarker.task")
        print("Success!", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"Failed: {e}", flush=True)

print("All failed.", flush=True)
sys.exit(1)
