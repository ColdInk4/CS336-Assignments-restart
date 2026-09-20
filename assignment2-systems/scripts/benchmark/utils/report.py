# utils/report.py
from collections.abc import Sequence

import pandas as pd


def results_to_wide(
    long_df: pd.DataFrame,
    modes: Sequence[str],
    sizes: Sequence[str],
) -> pd.DataFrame:
    """pivot 成 size × mode，每个格子是 'mean ± std' 或失败状态。"""
    df = long_df.copy()
    df["time"] = df.apply(
        lambda r: (
            f"{r['mean_ms']:.2f} ± {r['std_ms']:.2f}"
            if r["status"] == "ok"
            else r["status"]
        ),
        axis=1,
    )
    wide = df.pivot(index="size", columns="mode", values="time")
    wide = wide.reindex(columns=[m for m in modes if m in wide.columns])
    wide = wide.reindex(index=[s for s in sizes if s in wide.index])
    return wide
