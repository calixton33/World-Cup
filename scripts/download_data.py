"""Download the required public CSVs from a single pinned GitHub revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "martj42/international_results"
FILES = ("results.csv", "former_names.csv", "shootouts.csv")


def fetch(url: str) -> bytes:
    """Read a public source URL with a timeout and an identifying user agent."""
    request = Request(url, headers={"User-Agent": "international-football-portfolio"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def download(raw_dir: Path, ref: str = "master", refresh: bool = False) -> None:
    """Fetch atomically and save revision and content hashes next to ignored data."""
    if (raw_dir / "results.csv").exists() and not refresh:
        print(f"Using existing {raw_dir / 'results.csv'} (use --refresh to replace it).")
        return
    revision = json.loads(fetch(f"https://api.github.com/repos/{REPOSITORY}/commits/{ref}"))["sha"]
    payloads: dict[str, bytes] = {}
    for name in FILES:
        try:
            payloads[name] = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{name}")
        except HTTPError as error:
            if name != "results.csv" and error.code == 404:
                print(f"Optional {name} is absent at this revision.")
            else:
                raise
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        temporary = raw_dir / f".{name}.tmp"
        temporary.write_bytes(payload)
        temporary.replace(raw_dir / name)
    for name in set(FILES) - set(payloads):
        (raw_dir / name).unlink(missing_ok=True)
    manifest = {
        "repository": f"https://github.com/{REPOSITORY}", "revision": revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()},
    }
    (raw_dir / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {', '.join(payloads)} at {revision} into {raw_dir}")


def main() -> None:
    """Offer a local-file fallback when this environment cannot reach GitHub."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--ref", default="master", help="Commit or branch; resolved to a commit")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    try:
        download(args.raw_dir, args.ref, args.refresh)
    except (URLError, TimeoutError, OSError) as error:
        print(f"Download unavailable: {error}. Place results.csv (and, if available, "
              f"former_names.csv and shootouts.csv) in {args.raw_dir}, then rerun with "
              "--skip-download.", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
