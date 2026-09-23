"""
download_highway_video.py - Download highway traffic videos for vehicle counting testing.
Downloads free, open-source traffic surveillance videos suitable for YOLO vehicle detection.
"""

import sys
import io
import os
import urllib.request
import ssl

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Known working direct-download video URLs for traffic surveillance testing
# These are commonly used in vehicle detection/counting research
VIDEOS = [
    {
        "name": "highway_surveillance_1.mp4",
        "description": "Highway traffic - overhead surveillance camera (Supervisely sample)",
        "urls": [
            # Supervisely open-source highway tracking sample
            "https://github.com/supervisely-ecosystem/demo-video-retail/releases/download/v0.0.1/videos.zip",
        ],
        "fallback_search": "highway traffic overhead camera",
    },
    {
        "name": "highway_traffic_test.mp4",
        "description": "Highway traffic from elevated/overhead static camera",
        "urls": [
            # Common test video used in vehicle counting repos
            "https://github.com/nicholaskajoh/Vehicle-Counting/raw/master/data/videos/sample_traffic.mp4",
            "https://github.com/ahmetozlu/vehicle_counting_tensorflow/raw/master/input_video.mp4",
        ],
        "fallback_search": "highway traffic surveillance",
    },
]


def download_file(url, output_path, description=""):
    """Download a file from URL with progress reporting."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")

    try:
        # Create SSL context that doesn't verify (for GitHub redirects)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            content_type = response.headers.get("Content-Type", "")

            print(f"  Content-Type: {content_type}")
            if total_size > 0:
                print(f"  File Size: {total_size / (1024*1024):.1f} MB")

            downloaded = 0
            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        print(f"\r  Progress: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r  Downloaded: {mb:.1f} MB", end="", flush=True)

            file_size = os.path.getsize(output_path)
            if file_size < 50000:  # Less than 50KB is likely an error page
                print(f"\n  [WARN] File too small ({file_size} bytes), probably not a video")
                os.remove(output_path)
                return False

            print(f"\n  [OK] Downloaded successfully! ({file_size / (1024*1024):.1f} MB)")
            return True

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def try_pexels_api(output_path, query="highway traffic overhead"):
    """Try downloading from Pexels using API if key is available."""
    api_key = os.environ.get("PEXELS_API_KEY", "")
    if not api_key:
        return False

    try:
        import json
        url = f"https://api.pexels.com/videos/search?query={query}&per_page=5&orientation=landscape"
        req = urllib.request.Request(url, headers={"Authorization": api_key})

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            data = json.loads(response.read())
            videos = data.get("videos", [])
            if videos:
                # Find best resolution around 960px wide
                video = videos[0]
                files = video.get("video_files", [])
                best = sorted(files, key=lambda f: abs(f.get("width", 0) - 960))[0]
                return download_file(best["link"], output_path, f"Pexels: {query}")
    except Exception as e:
        print(f"  [INFO] Pexels API failed: {e}")

    return False


def download_with_yt_dlp(search_query, output_path):
    """Try to download a video using yt-dlp search."""
    try:
        import subprocess
        # Check if yt-dlp is available
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False

        print(f"\n  [INFO] Using yt-dlp to search: '{search_query}'")
        result = subprocess.run(
            [
                "yt-dlp",
                f"ytsearch1:{search_query}",
                "-f", "best[height<=720][ext=mp4]",
                "--max-downloads", "1",
                "--no-playlist",
                "-o", output_path,
            ],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0 and os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 100000:
                print(f"  [OK] Downloaded via yt-dlp ({file_size / (1024*1024):.1f} MB)")
                return True
        return False
    except Exception:
        return False


def main():
    print("=" * 60)
    print("  Highway Traffic Video Downloader")
    print("  For Vehicle Counting & Tracking System - YOLOv8")
    print("=" * 60)

    success_count = 0

    for video_info in VIDEOS:
        output_path = os.path.join(DATA_DIR, video_info["name"])

        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 100000:
                print(f"\n[SKIP] {video_info['name']} already exists ({file_size / (1024*1024):.1f} MB)")
                success_count += 1
                continue

        print(f"\n[DOWNLOAD] {video_info['description']}")

        # Method 1: Try Pexels API
        if try_pexels_api(output_path, video_info.get("fallback_search", "highway traffic")):
            success_count += 1
            continue

        # Method 2: Try direct URLs
        downloaded = False
        for url in video_info["urls"]:
            if download_file(url, output_path, video_info["description"]):
                downloaded = True
                break

        if downloaded:
            success_count += 1
            continue

        # Method 3: Try yt-dlp
        if download_with_yt_dlp(video_info["fallback_search"], output_path):
            success_count += 1
            continue

        print(f"  [FAIL] Could not download {video_info['name']}")

    print("\n" + "=" * 60)
    print(f"  Results: {success_count}/{len(VIDEOS)} videos downloaded")

    if success_count > 0:
        print("\n  You can now test with:")
        for video_info in VIDEOS:
            path = os.path.join(DATA_DIR, video_info["name"])
            if os.path.exists(path):
                print(f"    python src/main.py --source data/{video_info['name']} --scenario sunny --show")
    else:
        print("\n  [INFO] Automatic download failed. Please try manually:")
        print("  1. Go to https://www.pexels.com/search/videos/highway+traffic/")
        print("  2. Download a video with overhead/elevated camera angle")
        print(f"  3. Save it to: {DATA_DIR}")

    print("=" * 60)


if __name__ == "__main__":
    main()
