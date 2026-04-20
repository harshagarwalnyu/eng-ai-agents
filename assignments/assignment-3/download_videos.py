import argparse
import sys
from pathlib import Path

import yt_dlp

VIDEOS = {
    "drone_video_1.mp4": "https://www.youtube.com/watch?v=DhmZ6W1UAv4",
    "drone_video_2.mp4": "https://www.youtube.com/watch?v=YrydHPwRelI",
}


def download_videos(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filename, url in VIDEOS.items():
        out_path = output_dir / filename
        if out_path.exists():
            print(f"Skipping {filename}, already exists at {out_path}")
            continue

        print(f"Downloading {filename} from {url}...")
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(out_path),
            "merge_output_format": "mp4",
            "quiet": False,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}", file=sys.stderr)
            return 1

    print("All videos downloaded successfully.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download input videos for assignment 3.")
    parser.add_argument("--output-dir", type=Path, default=Path("videos"), help="Directory to save the videos")
    args = parser.parse_args()
    sys.exit(download_videos(args.output_dir))
