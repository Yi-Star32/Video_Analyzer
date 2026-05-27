"""Batch-download audio en transcripten van YouTube-URLs uit data/urls.txt."""

import logging
import shutil
import subprocess
import sys
import time
from concurrent import futures
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
URLS_FILE = DATA_DIR / "urls.txt"
OUTPUT_DIR = DATA_DIR / "episodes_audio"
MAX_WORKERS = 4
MAX_RETRIES = 3
BACKOFF_BASE = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

YTDLP_BASE_CMD = [
    "yt-dlp",
    "--remote-components",
    "ejs:github",
    "--no-warnings",
    "--cookies-from-browser",
    "firefox",
    "--geo-bypass",
    "--user-agent",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

ARIA2C_AVAILABLE = False
JS_RUNTIME_ARGS: list[str] = []


def _detect_tools() -> None:
    global ARIA2C_AVAILABLE, JS_RUNTIME_ARGS

    ARIA2C_AVAILABLE = bool(shutil.which("aria2c"))

    if shutil.which("node"):
        JS_RUNTIME_ARGS = ["--js-runtimes", "node"]
    elif shutil.which("deno"):
        JS_RUNTIME_ARGS = ["--js-runtimes", "deno"]
    else:
        JS_RUNTIME_ARGS = []


def _ytdlp_cmd() -> list[str]:
    if shutil.which("yt-dlp"):
        base = list(YTDLP_BASE_CMD)
    else:
        venv_ytdlp = PROJECT_ROOT.parent / "env" / "Scripts" / "yt-dlp.exe"
        if venv_ytdlp.exists():
            base = [str(venv_ytdlp), *YTDLP_BASE_CMD[1:]]
        else:
            base = list(YTDLP_BASE_CMD)

    return [*base, *JS_RUNTIME_ARGS]


def _progress_args(video_id: str) -> list[str]:
    return [
        "--newline",
        "--progress",
        "--progress-template",
        f"[{video_id}] %(progress._percent_str)s | "
        "%(progress._speed_str)s | ETA %(progress._eta_str)s",
    ]


def _run_logged(
    cmd: list[str],
    label: str,
    *,
    video_id: str | None = None,
    show_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    prefix = f"[{video_id}] " if video_id else ""
    logger.info("%s%s…", prefix, label)

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=not show_output,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"{label} mislukt na {elapsed:.1f}s"
            + (f": {detail[-2000:]}" if detail else "")
        )

    logger.info("%s%s klaar (%.1fs)", prefix, label, elapsed)
    return result


def load_urls(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    logger.info("Geladen: %d URL(s) uit %s", len(urls), path.name)
    return urls


def get_video_id(url: str) -> str:
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    raise ValueError("Kon video ID niet vinden")


def _ytdlp_common_args(video_id: str) -> list[str]:
    args = [
        *_ytdlp_cmd(),
        *_progress_args(video_id),
        "-f",
        "bestaudio[ext=webm]/bestaudio/best",
        "--no-playlist",
        "--concurrent-fragments",
        "8",
    ]
    if ARIA2C_AVAILABLE:
        args.extend(
            [
                "--downloader",
                "aria2c",
                "--downloader-args",
                "aria2c:-x 8 -s 8 -k 1M --file-allocation=none",
            ]
        )
    return args


def download_audio(url: str, output_file: Path, video_id: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.unlink(missing_ok=True)

    cmd = [*_ytdlp_common_args(video_id), "-o", str(output_file), url]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _run_logged(cmd, "Audio downloaden", video_id=video_id)
            break
        except RuntimeError:
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_BASE * attempt
            logger.warning(
                "[%s] Opnieuw over %ds (poging %d/%d)", video_id, wait, attempt, MAX_RETRIES
            )
            time.sleep(wait)

    if not output_file.exists() or output_file.stat().st_size == 0:
        raise RuntimeError(f"yt-dlp produceerde geen bestand: {output_file}")

    size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info("[%s] audio.webm %.1f MB", video_id, size_mb)


def download_transcript(video_id: str, output_file: Path) -> None:
    transcript = YouTubeTranscriptApi().fetch(
        video_id,
        languages=("en", "en-US", "nl"),
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# language: {transcript.language_code}\n")
        for snippet in transcript:
            start = snippet.start
            end = snippet.start + snippet.duration
            f.write(f"[{start:.2f}s - {end:.2f}s] {snippet.text}\n")

    if output_file.stat().st_size == 0:
        output_file.unlink(missing_ok=True)
        raise RuntimeError(f"Leeg transcript ({transcript.language_code})")


def _process_url(idx: int, url: str, total: int) -> dict | None:
    time.sleep((idx % MAX_WORKERS) * 0.5)

    try:
        video_id = get_video_id(url)
    except ValueError as e:
        logger.error("[%d/%d] Ongeldige URL: %s — %s", idx + 1, total, url, e)
        return None

    video_dir = OUTPUT_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_path = video_dir / "audio.webm"
    transcript_path = video_dir / "transcript.txt"

    logger.info("[%d/%d] Start: %s", idx + 1, total, video_id)
    job_t0 = time.perf_counter()

    try:
        download_audio(url, audio_path, video_id)
    except Exception as e:
        logger.error("[%d/%d] Audio mislukt (%s): %s", idx + 1, total, video_id, e)
        return None

    transcript_ok = True
    try:
        logger.info("[%d/%d] %s — Transcript downloaden…", idx + 1, total, video_id)
        download_transcript(video_id, transcript_path)
    except Exception as e:
        transcript_ok = False
        logger.warning("[%d/%d] Transcript niet beschikbaar (%s): %s", idx + 1, total, video_id, e)

    logger.info(
        "[%d/%d] Klaar: %s (%.1fs totaal) → %s",
        idx + 1,
        total,
        video_id,
        time.perf_counter() - job_t0,
        video_dir,
    )
    return {
        "video_id": video_id,
        "url": url,
        "audio": str(audio_path.resolve()),
        "transcript": str(transcript_path.resolve()) if transcript_ok else None,
        "dir": str(video_dir.resolve()),
    }


def main() -> None:
    _detect_tools()

    logger.info(
        "Tools: aria2c=%s | js=%s",
        "ja" if ARIA2C_AVAILABLE else "nee",
        " ".join(JS_RUNTIME_ARGS[1:]) if JS_RUNTIME_ARGS else "standaard (deno)",
    )
    if not JS_RUNTIME_ARGS:
        logger.warning(
            "Geen node/deno op PATH — YouTube-extractie kan trager of onbetrouwbaar zijn."
        )

    if not URLS_FILE.exists():
        logger.error("Bestand niet gevonden: %s", URLS_FILE)
        raise SystemExit(1)

    urls = load_urls(URLS_FILE)
    if not urls:
        logger.error("Geen URLs in %s", URLS_FILE)
        raise SystemExit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failed = 0
    batch_t0 = time.perf_counter()

    with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_process_url, idx, url, len(urls)): idx
            for idx, url in enumerate(urls)
        }
        for future in futures.as_completed(future_map):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.error("Onverwachte fout (idx %d): %s", future_map[future], e)

    print("\n" + "=" * 60)
    print(f"BATCH KLAAR — {len(results)}/{len(urls)} geslaagd ({time.perf_counter() - batch_t0:.1f}s)")
    print(f"Outputmap: {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    for r in sorted(results, key=lambda x: x["video_id"]):
        print(f"\n  {r['video_id']}/")
        print(f"    audio.webm        → {r['audio']}")
        if r["transcript"]:
            print(f"    transcript.txt  → {r['transcript']}")

    if failed:
        print(f"\n{failed} video('s) mislukt — zie logs hierboven.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
