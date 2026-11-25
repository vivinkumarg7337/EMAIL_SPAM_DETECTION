"""Split the raw dataset and build the single `text` column Atlas expects."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "emails.csv"


def main():
    df = pd.read_csv(RAW)
    df["text"] = df["subject"].fillna("") + "\n" + df["body"].fillna("")
    for split in ("train", "validation", "test"):
        part = df[df["split"] == split].drop(columns=["split"]).reset_index(drop=True)
        # shuffle so the engine's own split does not see template blocks
        part = part.sample(frac=1.0, random_state=42).reset_index(drop=True)
        out = ROOT / "data" / f"{split}.csv"
        part.to_csv(out, index=False)
        print(f"{split:10s} {len(part):5d} rows -> {out}  ({part['label'].mean():.1%} spam)")


if __name__ == "__main__":
    main()
