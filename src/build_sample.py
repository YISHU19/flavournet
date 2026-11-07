"""Build data/coffee_tea_sample.csv: 10,000 reviews of Coffee & Tea businesses
from the Yelp Open Dataset.

The dataset is free but cannot be redistributed, so it is not in this repo.
Download it from https://business.yelp.com/data/resources/open-dataset/,
extract the JSON files, then:

    python src/build_sample.py path/to/yelp_dataset

It streams the 5 GB review file once, keeps reviews of businesses whose
categories include "Coffee & Tea", and draws a seeded random sample.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CATEGORY = "Coffee & Tea"
COLUMNS = ["review_id", "text", "stars", "date", "business_name", "city", "state", "categories"]


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("yelp_dir", type=Path, help="folder holding the yelp_academic_dataset_*.json files")
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "coffee_tea_sample.csv")
    args = ap.parse_args()

    businesses = {
        b["business_id"]: (b["name"], b["city"], b["state"], b["categories"])
        for b in read_lines(args.yelp_dir / "yelp_academic_dataset_business.json")
        if b.get("categories") and CATEGORY in b["categories"]
    }
    print(f"{len(businesses):,} {CATEGORY} businesses")

    rows = []
    for i, r in enumerate(read_lines(args.yelp_dir / "yelp_academic_dataset_review.json")):
        b = businesses.get(r["business_id"])
        if b:
            rows.append((r["review_id"], r["text"], int(r["stars"]), r["date"], *b))
        if i % 1_000_000 == 0:
            print(f"  scanned {i:,} reviews, kept {len(rows):,}", end="\r")
    pool = pd.DataFrame(rows, columns=COLUMNS)
    print(f"\n{len(pool):,} {CATEGORY} reviews")

    sample = pool.sample(args.n, random_state=args.seed).reset_index(drop=True)
    args.out.parent.mkdir(exist_ok=True)
    sample.to_csv(args.out, index=False)
    print(f"saved {len(sample):,} reviews to {args.out}")


if __name__ == "__main__":
    main()
