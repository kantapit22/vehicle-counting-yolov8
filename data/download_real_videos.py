"""
download_real_videos.py - Download real traffic videos from Pexels
Downloads two real-world traffic videos:
1. Cloudy traffic - overcast/cloudy weather driving
2. Heavy traffic - congested/busy road traffic

Uses Pexels API (free, requires API key from https://www.pexels.com/api/)
"""

import os
import sys
import io
import requests
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent

# Pexels free API - get your key at https://www.pexels.com/api/
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

SEARCH_URL = "https://api.pexels.com/videos/search"


def search_and_download(query: str, output_filename: str, min_duration: int = 8, max_duration: int = 60):
    """
    Search Pexels for a video matching the query, then download it.
    """
    if not PEXELS_API_KEY:
        print("=" * 60)
        print("ERROR: Pexels API Key is required!")
        print()
        print("How to get a free API key (takes 30 seconds):")
        print("1. Go to https://www.pexels.com/api/")
        print("2. Click 'Get Started' and create a free account")
        print("3. Your API key will be shown on the dashboard")
        print()
        print("Then run this script with:")
        print(f'  set PEXELS_API_KEY=your_key_here')
        print(f'  python {__file__}')
        print("=" * 60)
        return False

    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": 15,
        "orientation": "landscape",
        "size": "medium",
    }

    print(f"\n[SEARCH] Searching Pexels for: '{query}'...")
    resp = requests.get(SEARCH_URL, headers=headers, params=params)

    if resp.status_code == 401:
        print("[ERROR] Invalid API key! Please check your PEXELS_API_KEY.")
        return False
    if resp.status_code != 200:
        print(f"[ERROR] API error: {resp.status_code} - {resp.text}")
        return False

    data = resp.json()
    videos = data.get("videos", [])

    if not videos:
        print(f"[ERROR] No videos found for query: '{query}'")
        return False

    # Filter by duration
    suitable = [v for v in videos if min_duration <= v.get("duration", 0) <= max_duration]
    if not suitable:
        suitable = videos

    video = suitable[0]
    video_id = video["id"]
    duration = video.get("duration", "?")
    print(f"[FOUND] Video: ID={video_id}, Duration={duration}s")
    print(f"   URL: https://www.pexels.com/video/{video_id}/")

    video_files = video.get("video_files", [])
    if not video_files:
        print("[ERROR] No downloadable files found!")
        return False

    # Sort by width, prefer around 960px wide (HD but not too large)
    target_width = 960
    video_files_sorted = sorted(video_files, key=lambda f: abs(f.get("width", 0) - target_width))
    best_file = video_files_sorted[0]

    download_url = best_file["link"]
    width = best_file.get("width", "?")
    height = best_file.get("height", "?")
    file_type = best_file.get("file_type", "video/mp4")
    print(f"   Resolution: {width}x{height} ({file_type})")

    output_path = DATA_DIR / output_filename
    print(f"[DOWNLOAD] Downloading to: {output_path}")

    resp = requests.get(download_url, stream=True)
    total_size = int(resp.headers.get("content-length", 0))

    downloaded = 0
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r   Progress: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

    print(f"\n[OK] Downloaded: {output_path} ({downloaded / (1024*1024):.1f} MB)")
    return True


def download_without_api():
    """
    Alternative: Download directly using known Pexels video IDs (no API key needed).
    These are popular, well-known free traffic videos on Pexels.
    """
    videos = {
        "cloudy_traffic_real.mp4": {
            "urls": [
                "https://www.pexels.com/video/1721294/download/",
                "https://www.pexels.com/video/3173312/download/",
                "https://www.pexels.com/video/2053100/download/",
            ],
            "description": "Cars on highway - overcast weather",
        },
        "heavy_traffic_real.mp4": {
            "urls": [
                "https://www.pexels.com/video/3048225/download/",
                "https://www.pexels.com/video/2053855/download/",
                "https://www.pexels.com/video/856116/download/",
            ],
            "description": "Heavy traffic congestion on road",
        },
    }

    for filename, info in videos.items():
        output_path = DATA_DIR / filename
        print(f"\n[VIDEO] {info['description']}")
        print(f"[DOWNLOAD] Downloading to: {output_path}")

        success = False
        for url in info["urls"]:
            try:
                print(f"   Trying: {url}")
                resp = requests.get(url, stream=True, allow_redirects=True, timeout=30)
                if resp.status_code == 200:
                    content_type = resp.headers.get("content-type", "")
                    total_size = int(resp.headers.get("content-length", 0))

                    if "video" in content_type or "octet-stream" in content_type or total_size > 100000:
                        downloaded = 0
                        with open(output_path, "wb") as f:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total_size > 0:
                                        pct = downloaded / total_size * 100
                                        mb = downloaded / (1024 * 1024)
                                        total_mb = total_size / (1024 * 1024)
                                        print(f"\r   Progress: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)

                        if downloaded > 100000:  # At least 100KB
                            print(f"\n[OK] Downloaded: {output_path} ({downloaded / (1024*1024):.1f} MB)")
                            success = True
                            break
                        else:
                            print(f"\n   [WARN] File too small ({downloaded} bytes), trying next URL...")
                            output_path.unlink(missing_ok=True)
                    else:
                        print(f"   [WARN] Not a video response (content-type: {content_type})")
                else:
                    print(f"   [WARN] HTTP {resp.status_code}, trying next URL...")
            except Exception as e:
                print(f"   [WARN] Error: {e}, trying next URL...")

        if not success:
            print(f"[FAIL] Could not download {filename}.")
            print(f"       Please download manually from https://www.pexels.com/search/videos/traffic/")


def main():
    print("=" * 60)
    print("  Real Traffic Video Downloader")
    print("=" * 60)

    if PEXELS_API_KEY:
        print(f"[OK] Pexels API key found")
        print()

        search_and_download(
            query="cars driving highway overcast cloudy weather",
            output_filename="cloudy_traffic_real.mp4",
            min_duration=8,
            max_duration=60,
        )

        search_and_download(
            query="heavy traffic congestion busy road cars",
            output_filename="heavy_traffic_real.mp4",
            min_duration=8,
            max_duration=60,
        )
    else:
        print("[INFO] No Pexels API key found. Using direct download links...")
        download_without_api()

    print()
    print("=" * 60)
    print("  Done! You can now run the project with real videos:")
    print()
    print("  Cloudy scenario:")
    print("    python src/main.py --source data/cloudy_traffic_real.mp4 --scenario cloudy --show")
    print()
    print("  Heavy traffic scenario:")
    print("    python src/main.py --source data/heavy_traffic_real.mp4 --scenario heavy_traffic --show")
    print("=" * 60)


if __name__ == "__main__":
    main()
