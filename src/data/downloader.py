"""
Download raw datasets from public GitHub sources.

Datasets used:
  1. results.csv        — ~47 000 international match results (1872–present)
                          Source: github.com/martj42/international_results
  2. shootouts.csv      — penalty shootout outcomes for drawn knockout matches
  3. goalscorers.csv    — individual goalscorer events (used for derived stats)
  4. fifa_rankings.csv  — monthly FIFA world rankings (2018–present)
                          Source: github.com/saimatkhan/fifa-world-rankings

All files are saved to data/raw/.
"""

from __future__ import annotations
from pathlib import Path
import requests
from tqdm import tqdm

from src.utils.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path(CONFIG["paths"]["raw_data"])

DATASETS = {
    "results.csv": CONFIG["data"]["results_url"],
    "shootouts.csv": CONFIG["data"]["shootouts_url"],
    "goalscorers.csv": CONFIG["data"]["goalscorers_url"],
    "fifa_rankings.csv": CONFIG["data"]["fifa_ranking_url"],
}


def _download_file(url: str, dest: Path) -> None:
    """Stream-download a single file with a progress bar."""
    logger.info(f"Downloading {dest.name} ...")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)

    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    logger.info(f"Saved → {dest}")


def download_all(force: bool = False) -> dict[str, Path]:
    """Download every dataset. Skip files that already exist unless force=True."""
    downloaded: dict[str, Path] = {}
    for filename, url in DATASETS.items():
        dest = RAW_DIR / filename
        if dest.exists() and not force:
            logger.info(f"Already exists, skipping: {dest}")
            downloaded[filename] = dest
            continue
        try:
            _download_file(url, dest)
            downloaded[filename] = dest
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
    return downloaded


if __name__ == "__main__":
    download_all()
