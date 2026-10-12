# Data

The public [martj42 international results repository](https://github.com/martj42/international_results) supplies real men’s match results, historic team aliases, and shootout records under CC0.

Run `python scripts/download_data.py` from the repository root. It resolves one source commit and downloads `results.csv`, optional `former_names.csv`, and `shootouts.csv` into ignored `data/raw/`, with a revision and checksum manifest. If network access is unavailable, place these files there yourself and use `python scripts/run_all.py --skip-download`.

`data/processed/` is also ignored and contains reproducible pre-match features. The app uses only precomputed files in `models/` and `reports/`. Source details and cleaning decisions are recorded in `reports/data_audit.json` and `reports/metrics.json`.

Score outcomes include extra time and exclude penalties. Shootout winners are never used as labels or features. Team names follow the source’s canonical/successor conventions; aliases are only applied in their historical date windows.
