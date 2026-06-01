from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


def plot_cumulative_returns(returns: pd.DataFrame, outdir: Path, ret_col: str = "long_short") -> list[Path]:
    set_style()
    outdir.mkdir(parents=True, exist_ok=True)
    data = returns.copy()
    data["wealth"] = (1.0 + data[ret_col].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=160)
    ax.plot(pd.to_datetime(data["month_end"]), data["wealth"], color="#1f77b4", lw=1.8)
    ax.set_title("Conditional Tail-Risk Strategy")
    ax.set_ylabel("Cumulative gross return")
    ax.set_xlabel("")
    paths = [outdir / "cumulative_returns.png", outdir / "cumulative_returns.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_drawdown(returns: pd.DataFrame, outdir: Path, ret_col: str = "long_short") -> list[Path]:
    set_style()
    outdir.mkdir(parents=True, exist_ok=True)
    data = returns.copy()
    wealth = (1.0 + data[ret_col].fillna(0)).cumprod()
    data["drawdown"] = wealth / wealth.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=160)
    ax.fill_between(pd.to_datetime(data["month_end"]), data["drawdown"], 0, color="#d62728", alpha=0.35)
    ax.plot(pd.to_datetime(data["month_end"]), data["drawdown"], color="#8c1d18", lw=1.2)
    ax.set_title("Strategy Drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    paths = [outdir / "drawdown.png", outdir / "drawdown.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_rolling_ic(predictions: pd.DataFrame, outdir: Path, target: str = "ret_excess_fwd_1m") -> list[Path]:
    set_style()
    outdir.mkdir(parents=True, exist_ok=True)
    target = target if target in predictions.columns else "ret_fwd_1m"
    rows = []
    for date, group in predictions.dropna(subset=[target, "prediction"]).groupby("month_end"):
        rows.append({"month_end": date, "ic": group[target].corr(group["prediction"], method="spearman")})
    data = pd.DataFrame(rows).sort_values("month_end")
    data["ic_12m"] = data["ic"].rolling(12, min_periods=6).mean()
    fig, ax = plt.subplots(figsize=(7.0, 3.5), dpi=160)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.plot(pd.to_datetime(data["month_end"]), data["ic"], color="#9ecae1", lw=0.8, label="Monthly IC")
    ax.plot(pd.to_datetime(data["month_end"]), data["ic_12m"], color="#08519c", lw=1.8, label="12-month average")
    ax.set_title("Rolling Rank Information Coefficient")
    ax.set_ylabel("Spearman IC")
    ax.set_xlabel("")
    ax.legend(frameon=False)
    paths = [outdir / "rolling_ic.png", outdir / "rolling_ic.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_model_leaderboard(metrics_path: Path, outdir: Path) -> list[Path]:
    set_style()
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [
        {
            "model": name.replace("prediction_", "").replace("_", " "),
            "spearman_ic": values.get("spearman_ic"),
            "oos_r2": values.get("oos_r2"),
        }
        for name, values in metrics.get("models", {}).items()
    ]
    data = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=160)
    sns.barplot(data=data, y="model", x="spearman_ic", ax=axes[0], color="#4c78a8")
    sns.barplot(data=data, y="model", x="oos_r2", ax=axes[1], color="#59a14f")
    axes[0].set_title("Rank IC")
    axes[1].set_title("OOS R2")
    for ax in axes:
        ax.set_ylabel("")
        ax.axvline(0, color="#333333", lw=0.8)
    fig.suptitle("Model Leaderboard", y=1.02)
    paths = [outdir / "model_leaderboard.png", outdir / "model_leaderboard.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_data_coverage(panel: pd.DataFrame, outdir: Path) -> list[Path]:
    set_style()
    outdir.mkdir(parents=True, exist_ok=True)
    data = panel.copy()
    data["year"] = pd.to_datetime(data["month_end"]).dt.year
    blocks = {
        "CRSP returns": "ret",
        "Compustat chars": "book_equity",
        "Option surface": "atm_iv_30",
        "VIX state": "vix_level",
        "Rates state": "treasury_slope",
    }
    rows = []
    for label, col in blocks.items():
        if col in data:
            share = data.groupby("year")[col].apply(lambda x: x.notna().mean())
            rows.extend({"year": year, "block": label, "coverage": value} for year, value in share.items())
    cov = pd.DataFrame(rows)
    heat = cov.pivot(index="block", columns="year", values="coverage").sort_index()
    fig, ax = plt.subplots(figsize=(9.5, 2.8), dpi=160)
    sns.heatmap(heat, ax=ax, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "Nonmissing share"})
    ax.set_title("Data Coverage by Year")
    ax.set_xlabel("")
    ax.set_ylabel("")
    paths = [outdir / "data_coverage.png", outdir / "data_coverage.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_feature_missingness(panel: pd.DataFrame, outdir: Path) -> list[Path]:
    set_style()
    outdir.mkdir(parents=True, exist_ok=True)
    numeric = panel.select_dtypes("number")
    miss = numeric.isna().mean().sort_values(ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=160)
    miss.sort_values().plot.barh(ax=ax, color="#4c78a8")
    ax.set_xlabel("Missing share")
    ax.set_title("Feature Missingness")
    paths = [outdir / "feature_missingness.png", outdir / "feature_missingness.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return paths
