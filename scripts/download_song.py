"""Download één liedje als audio vanaf een YouTube-URL."""

import argparse
import logging
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_audio import OUTPUT_DIR, _detect_tools, download_audio, download_transcript, get_video_id

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download audio van een YouTube-video.")
    parser.add_argument("url", help="YouTube-URL (bijv. https://www.youtube.com/watch?v=...)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Pad naar het audiobestand (standaard: ../data/songs/<video_id>/audio.webm)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    _detect_tools()

    try:
        video_id = get_video_id(args.url)
    except ValueError as e:
        logger.error("Ongeldige URL: %s", e)
        raise SystemExit(1) from e

    DEFAULT_SONGS_DIR = SCRIPTS_DIR.parent / "data" / "songs"
    output_file = args.output or (DEFAULT_SONGS_DIR / video_id / "audio.webm")
    transcript_path = DEFAULT_SONGS_DIR / video_id / "transcript.txt"

    try:
        download_audio(args.url, output_file, video_id)
        download_transcript(video_id, transcript_path)
    except Exception as e:
        logger.error("Download mislukt: %s", e)
        raise SystemExit(1) from e

    print(f"Audio opgeslagen: {output_file.resolve()}")


if __name__ == "__main__":
    main()
