from __future__ import annotations

import pandas as pd

from ctrsdf.config import ProjectConfig


def assign_splits(frame: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    df = frame.copy()
    df["month_end"] = pd.to_datetime(df["month_end"])
    splits = config.raw["splits"]
    df["split"] = "unused"
    df.loc[df["month_end"] <= pd.Timestamp(splits["train_end"]), "split"] = "train"
    df.loc[
        (df["month_end"] >= pd.Timestamp(splits["validation_start"]))
        & (df["month_end"] <= pd.Timestamp(splits["validation_end"])),
        "split",
    ] = "validation"
    df.loc[
        (df["month_end"] >= pd.Timestamp(splits["holdout_start"]))
        & (df["month_end"] <= pd.Timestamp(splits["holdout_end"])),
        "split",
    ] = "holdout"
    return df
