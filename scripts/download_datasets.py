from __future__ import annotations

import os
from pathlib import Path

import kagglehub


DATASETS = {
    "vehicle_classification": "mohamedmaher5/vehicle-classification",
    "vehicle_damage": "eashankaushik/car-damage-detection",
    "customer_support": "thoughtvector/customer-support-on-twitter",
    "vehicle_metadata": "nehalbirla/vehicle-dataset-from-cardekho",
}


def main() -> None:
    base = Path("data")
    base.mkdir(exist_ok=True)

    if not Path.home().joinpath(".kaggle", "kaggle.json").exists():
        raise SystemExit("Missing ~/.kaggle/kaggle.json. Add Kaggle API credentials first.")

    for folder, dataset in DATASETS.items():
        print(f"Downloading {dataset} ...")
        path = kagglehub.dataset_download(dataset)
        print(f"Downloaded to cache: {path}")
        out = base / folder
        out.mkdir(parents=True, exist_ok=True)
        print(f"Please copy / extract files from cache into: {out}")


if __name__ == "__main__":
    main()
