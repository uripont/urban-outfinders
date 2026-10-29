"""Download the product catalogue and the NLTK resources the project needs.

The catalogue is published as an asset of a GitHub release. The public
download URL is tried first; while the repository is private it falls back
to the GitHub CLI, which uses your own `gh auth login` credentials.

Usage: uv run python scripts/fetch_data.py
"""

import gzip
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import nltk

REPO = "uripont/urban-outfinders"
TAG = "data-v1"
ASSET = "fashion_products_dataset.json.gz"
SHA256 = "adbfea4369a8efd849aa20942789aaf501d1d02cb38fdde6affeab9f8d9e15fb"
NLTK_RESOURCES = ["stopwords", "punkt_tab", "wordnet"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET = DATA_DIR / "fashion_products_dataset.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(dest_dir: Path) -> Path:
    dest = dest_dir / ASSET
    url = f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except urllib.error.HTTPError:
        pass  # Not public (yet): use the GitHub CLI instead.

    if shutil.which("gh") is None:
        sys.exit("Could not download the dataset. Install the GitHub CLI and run `gh auth login`.")
    subprocess.run(
        ["gh", "release", "download", TAG, "--repo", REPO, "--pattern", ASSET, "--dir", str(dest_dir)],
        check=True,
    )
    return dest


def fetch_dataset() -> None:
    if DATASET.exists() and sha256(DATASET) == SHA256:
        print(f"Dataset already present: {DATASET}")
        return

    DATA_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive = download_asset(Path(tmp))
        with gzip.open(archive, "rb") as src, DATASET.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    if sha256(DATASET) != SHA256:
        DATASET.unlink()
        sys.exit("Downloaded dataset failed the checksum check.")
    print(f"Dataset ready: {DATASET}")


def fetch_nltk_resources() -> None:
    for resource in NLTK_RESOURCES:
        nltk.download(resource, quiet=True)
    print(f"NLTK resources ready: {', '.join(NLTK_RESOURCES)}")


if __name__ == "__main__":
    fetch_dataset()
    fetch_nltk_resources()
