import pandas as pd

from ctrsdf.config import ProjectConfig
from ctrsdf.models.splits import assign_splits


def test_assign_splits():
    cfg = ProjectConfig(
        raw={
            "project": {"seed": 1, "sample_start": "1996-01-01", "sample_end": "2026-03-31", "smoke_start": "2020-01-01", "smoke_end": "2020-03-31"},
            "splits": {
                "train_end": "2006-12-31",
                "validation_start": "2007-01-01",
                "validation_end": "2012-12-31",
                "holdout_start": "2013-01-01",
                "holdout_end": "2026-03-31",
            },
        },
        root=".",
    )
    frame = pd.DataFrame({"month_end": ["2006-12-31", "2009-01-31", "2015-01-31"]})
    out = assign_splits(frame, cfg)
    assert out["split"].tolist() == ["train", "validation", "holdout"]
