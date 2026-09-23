import os
import sys
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_NAME = "yolov8n.pt"
MODEL_URL = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{DEFAULT_MODEL_NAME}"

def ensure_model(model_name: str = DEFAULT_MODEL_NAME, target_dir: Path = MODEL_DIR) -> str:
    """
    Ensure the YOLOv8 model weights exist in target_dir.
    If not, download from the official GitHub releases.
    Returns the absolute path to the model file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_name

    if model_path.exists() and model_path.stat().st_size > 1000:
        return str(model_path)

    print(f"[ModelDownloader] '{model_name}' not found in {target_dir}. Downloading from {MODEL_URL}...")
    try:
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.2f}MB / {total_size / (1024*1024):.2f}MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, str(model_path), reporthook=progress)
        print(f"\n[ModelDownloader] Model saved successfully to {model_path}")
    except Exception as e:
        print(f"\n[ModelDownloader] Direct download failed ({e}). Falling back to ultralytics download...")
        try:
            from ultralytics import YOLO
            temp_model = YOLO(model_name)
            # Move or save to target dir if needed
            temp_path = Path(model_name)
            if temp_path.exists() and temp_path.resolve() != model_path.resolve():
                temp_path.replace(model_path)
            print(f"[ModelDownloader] Ultralytics downloaded model to {model_path}")
        except Exception as e2:
            raise RuntimeError(f"Failed to download YOLOv8 model: {e2}")

    return str(model_path)

if __name__ == "__main__":
    path = ensure_model()
    print(f"Verified model at: {path}")
