from pathlib import Path
import subprocess
import sys

EPISODES_AUDIO_DIR = Path(__file__).resolve().parents[1] / "data" / "episodes_audio"


def convert_webm_to_wav(webm_path: Path) -> Path:
    wav_path = webm_path.with_suffix(".wav")
    if wav_path.exists():
        print(f"Skipping (already exists): {wav_path}")
        return wav_path
    cmd = ["ffmpeg", "-y", "-i", str(webm_path), "-ac", "1", "-ar", "16000", str(wav_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"Converted: {webm_path} -> {wav_path}")
    return wav_path


def main() -> None:
    webm_files = sorted(EPISODES_AUDIO_DIR.rglob("*.webm"))
    if not webm_files:
        print("No .webm files found.")
        sys.exit(0)
    for webm in webm_files:
        convert_webm_to_wav(webm)


if __name__ == "__main__":
    main()
