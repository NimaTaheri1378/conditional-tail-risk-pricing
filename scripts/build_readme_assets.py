from __future__ import annotations

import json
import shutil
from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
CLOSEOUT = ROOT / "artifacts" / "closeout"
FIG_DIR = ROOT / "docs" / "assets" / "figures"
TABLE_DIR = ROOT / "docs" / "assets" / "tables"
FULL_DIR = ROOT / "artifacts" / "full"

BLUE = "#245c7a"
GREEN = "#6f8f3e"
ORANGE = "#b35c2e"
GOLD = "#b9a44c"
GRAY = "#4b5563"
LIGHT_GRID = "#e5e7eb"
SPINE = "#d0d7de"


def _mkdirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SPINE,
            "axes.grid": True,
            "grid.color": LIGHT_GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 1.0,
            "savefig.facecolor": "white",
        }
    )


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.grid(axis="y")


def _save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=180)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _read_csv(name: str) -> pd.DataFrame:
    path = CLOSEOUT / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _copy_tables() -> None:
    table_aliases = {
        "final_reversal_audit.csv": "reversal_validation.csv",
        "strategy_strengthening_summary.csv": "strategy_variants.csv",
        "strategy_decile_monotonicity.csv": "decile_shape.csv",
        "factor_alpha_spanning.csv": "factor_attribution.csv",
        "random_forest_importance.csv": "feature_importance.csv",
        "double_ml_option_contribution.csv": "double_ml_option_contribution.csv",
        "signal_robustness_full_sample.csv": "signal_robustness.csv",
        "robustness.csv": "holdout_robustness.csv",
    }
    for source, target in table_aliases.items():
        src = CLOSEOUT / source
        if src.exists():
            shutil.copy2(src, TABLE_DIR / target)

    final = _read_csv("final_reversal_audit.csv")
    keep = [
        "raw_tail_slope_reversal_skip1_quarterly_smooth3",
        "raw_tail_slope_reversal_quarterly_smooth3",
        "raw_tail_composite_reversal_quarterly_smooth3",
    ]
    final[final["strategy"].isin(keep)].to_csv(TABLE_DIR / "headline_results.csv", index=False)


def _metric_card(ax, xy, size, label: str, value: str, sublabel: str, color: str) -> None:
    x, y = xy
    w, h = size
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8,
        edgecolor="#d8dee6",
        facecolor="#ffffff",
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.text(x + 0.045 * w, y + h - 0.22 * h, label, transform=ax.transAxes, fontsize=7.2, color=GRAY, weight="bold")
    ax.text(x + 0.045 * w, y + 0.36 * h, value, transform=ax.transAxes, fontsize=17, color=color, weight="bold")
    ax.text(x + 0.045 * w, y + 0.14 * h, sublabel, transform=ax.transAxes, fontsize=6.8, color=GRAY)


def headline_figure() -> None:
    final = _read_csv("final_reversal_audit.csv")
    monthly = _read_csv("strategy_strengthening_monthly.csv")
    monthly["month_end"] = pd.to_datetime(monthly["month_end"])

    best = final[final["strategy"].eq("raw_tail_slope_reversal_skip1_quarterly_smooth3")].iloc[0]
    series = monthly[
        monthly["strategy"].eq("tail_slope_reversal_liquid_quarterly_smooth3")
        & monthly["eta"].round(3).eq(0.1)
        & monthly["rebalance"].eq("quarterly")
    ].copy()
    series = series.sort_values("month_end")
    series["cumulative_net"] = (1 + series["net_return"]).cumprod() - 1

    fig = plt.figure(figsize=(10.8, 6.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.25, 1.0], height_ratios=[0.42, 0.58], hspace=0.55, wspace=0.35)
    ax_line = fig.add_subplot(gs[:, :2])
    ax_cards = fig.add_subplot(gs[0, 2])
    ax_bar = fig.add_subplot(gs[1, 2])

    _clean_axes(ax_line)
    ax_line.plot(series["month_end"], series["cumulative_net"] * 100, color=BLUE, linewidth=2.0)
    ax_line.fill_between(series["month_end"], 0, series["cumulative_net"] * 100, color=BLUE, alpha=0.12)
    ax_line.axhline(0, color="#2f3437", linewidth=0.8)
    ax_line.set_title("Cumulative net return", loc="left", pad=8, weight="bold")
    ax_line.set_ylabel("Percent")
    ax_line.set_xlabel("Holdout month")

    ax_cards.axis("off")
    _metric_card(ax_cards, (0.00, 0.54), (0.47, 0.39), "Net return", f"{best['net_ann_mean'] * 100:.2f}%", "annualized", BLUE)
    _metric_card(ax_cards, (0.53, 0.54), (0.47, 0.39), "Net Sharpe", f"{best['net_sharpe']:.2f}", "after costs", GREEN)
    _metric_card(ax_cards, (0.00, 0.06), (0.47, 0.39), "Alpha", f"{best['ff5_umd_net_alpha'] * 1200:.2f}%", "FF5+UMD ann.", ORANGE)
    _metric_card(ax_cards, (0.53, 0.06), (0.47, 0.39), "Turnover", f"{best['avg_turnover']:.2f}", "monthly avg.", GOLD)

    raw = final[final["strategy"].eq("raw_tail_slope_reversal_skip1_quarterly_smooth3")]["net_sharpe"].iloc[0]
    residual = final[final["strategy"].eq("resid_tail_slope_reversal_skip1_quarterly_smooth3")]["net_sharpe"].iloc[0]
    _clean_axes(ax_bar)
    ax_bar.bar(["Raw", "Residual"], [raw, residual], color=[BLUE, GOLD], width=0.56)
    ax_bar.axhline(0, color="#2f3437", linewidth=0.8)
    ax_bar.set_title("Raw vs incremental signal", loc="left", pad=8, weight="bold")
    ax_bar.set_ylabel("Net Sharpe")
    ax_bar.set_ylim(-0.22, 0.38)
    for i, val in enumerate([raw, residual]):
        ax_bar.text(i, val + (0.025 if val >= 0 else -0.025), f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)

    fig.suptitle("From Option-Implied Tail Insurance to Cross-Sectional Returns", x=0.06, y=0.98, ha="left", fontsize=14, weight="bold")
    fig.text(0.06, 0.935, "OptionMetrics x CRSP x Compustat, U.S. equities, holdout 2013-2025", ha="left", fontsize=8.5, color=GRAY)
    fig.subplots_adjust(top=0.86, left=0.07, right=0.98, bottom=0.11)
    _save(fig, "headline_surface_to_returns")


def reversal_validation() -> None:
    final = _read_csv("final_reversal_audit.csv").copy()
    order = [
        "raw_tail_slope_reversal_skip1_quarterly_smooth3",
        "raw_tail_slope_reversal_quarterly_smooth3",
        "raw_tail_composite_reversal_quarterly_smooth3",
        "resid_tail_composite_reversal_quarterly_smooth3",
        "resid_tail_slope_reversal_skip1_quarterly_smooth3",
        "resid_tail_slope_reversal_quarterly_smooth3",
    ]
    labels = {
        "raw_tail_slope_reversal_skip1_quarterly_smooth3": "Raw slope, skip-month",
        "raw_tail_slope_reversal_quarterly_smooth3": "Raw slope",
        "raw_tail_composite_reversal_quarterly_smooth3": "Raw composite",
        "resid_tail_composite_reversal_quarterly_smooth3": "Residual composite",
        "resid_tail_slope_reversal_skip1_quarterly_smooth3": "Residual slope, skip-month",
        "resid_tail_slope_reversal_quarterly_smooth3": "Residual slope",
    }
    final = final.set_index("strategy").loc[order].reset_index()
    y = range(len(final))

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    _clean_axes(ax)
    colors = [BLUE if v >= 0 else GOLD for v in final["net_sharpe"]]
    ax.barh(list(y), final["net_sharpe"], color=colors, height=0.62)
    ax.axvline(0, color="#2f3437", linewidth=0.8)
    ax.set_yticks(list(y), [labels[s] for s in final["strategy"]])
    ax.invert_yaxis()
    ax.set_xlabel("Net Sharpe")
    ax.set_title("Reversal validation", loc="left", weight="bold")
    ax.set_xlim(-0.32, 0.38)
    for i, val in enumerate(final["net_sharpe"]):
        ax.text(val + (0.014 if val >= 0 else -0.014), i, f"{val:.2f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)
    _save(fig, "reversal_validation")


def decile_shape() -> None:
    deciles = _read_csv("strategy_decile_monotonicity.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.9), constrained_layout=True)
    _clean_axes(ax)
    specs = [
        ("liquid", "tail_reversal", "Tail reversal", BLUE),
        ("liquid", "tail_slope", "Raw tail slope", ORANGE),
    ]
    for case, signal, label, color in specs:
        frame = deciles[deciles["case"].eq(case) & deciles["signal"].eq(signal)].copy()
        ax.plot(frame["decile"], frame["ann_mean"] * 100, marker="o", linewidth=1.8, markersize=4.2, color=color, label=label)
    ax.set_title("Decile return shape", loc="left", weight="bold")
    ax.set_xlabel("Signal decile")
    ax.set_ylabel("Annualized mean return (%)")
    ax.set_xticks(range(1, 11))
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "decile_shape")


def cost_decomposition() -> None:
    monthly = _read_csv("strategy_strengthening_monthly.csv")
    monthly["month_end"] = pd.to_datetime(monthly["month_end"])
    frame = monthly[
        monthly["strategy"].eq("tail_slope_reversal_liquid_quarterly_smooth3")
        & monthly["eta"].round(3).eq(0.1)
        & monthly["rebalance"].eq("quarterly")
    ].copy()
    frame = frame.sort_values("month_end")

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    _clean_axes(ax)
    half = frame["half_spread_cost"] * 10000
    impact = frame["impact_cost"] * 10000
    ax.stackplot(frame["month_end"], half, impact, colors=[BLUE, ORANGE], alpha=0.75, labels=["Half-spread", "Impact"])
    ax.set_title("Trading-cost decomposition", loc="left", weight="bold")
    ax.set_ylabel("Monthly cost, bps")
    ax.set_xlabel("Holdout month")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    _save(fig, "cost_decomposition")


def model_leaderboard() -> None:
    metrics_path = FULL_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    for name, vals in metrics.get("models", {}).items():
        label = name.replace("prediction_", "").replace("_", " ")
        rows.append({"model": label, "rank_ic": vals.get("spearman_ic"), "oos_r2": vals.get("oos_r2")})
    frame = pd.DataFrame(rows)
    order = ["gbm", "neural sdf", "elastic net", "ipca"]
    frame["sort"] = frame["model"].map({m: i for i, m in enumerate(order)}).fillna(99)
    frame = frame.sort_values("sort")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    for ax in axes:
        _clean_axes(ax)
    axes[0].barh(frame["model"], frame["rank_ic"], color=BLUE, height=0.58)
    axes[0].axvline(0, color="#2f3437", linewidth=0.8)
    axes[0].set_title("Rank IC", loc="left", weight="bold")
    axes[0].set_xlabel("Spearman IC")
    axes[1].barh(frame["model"], frame["oos_r2"], color=GREEN, height=0.58)
    axes[1].axvline(0, color="#2f3437", linewidth=0.8)
    axes[1].set_title("OOS R2", loc="left", weight="bold")
    axes[1].set_xlabel("Out-of-sample R2")
    fig.suptitle("Walk-forward model leaderboard", x=0.02, ha="left", fontsize=11, weight="bold")
    _save(fig, "model_leaderboard")


def feature_importance() -> None:
    imp = _read_csv("random_forest_importance.csv").head(12).copy()
    imp["label"] = imp["feature"].str.replace("_", " ", regex=False).map(lambda x: shorten(x, width=28, placeholder="..."))
    imp = imp.sort_values("importance")

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    _clean_axes(ax)
    ax.barh(imp["label"], imp["importance"], color=BLUE, height=0.58)
    ax.set_title("Random Forest feature importance", loc="left", weight="bold")
    ax.set_xlabel("Share of model importance")
    _save(fig, "feature_importance")


def validation_scorecard() -> None:
    raw = json.loads((CLOSEOUT / "raw_chain_validation_summary.json").read_text(encoding="utf-8"))
    bench = json.loads((CLOSEOUT / "benchmark_reconciliation_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((FULL_DIR / "metrics.json").read_text(encoding="utf-8"))
    portfolio = metrics.get("portfolio", {})

    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    ax.axis("off")
    cards = [
        ("Raw-chain pairs", f"{raw['n_pairs']:,}", "surface validation", BLUE),
        ("ATM-IV corr.", f"{raw['atm_iv_correlation']:.3f}", "raw chain vs surface", GREEN),
        ("Market corr.", f"{bench['correlation']:.3f}", "CRSP-built vs FF", ORANGE),
        ("Holdout months", f"{portfolio.get('n', 0):.0f}", "model portfolio", GOLD),
    ]
    for i, (label, value, sublabel, color) in enumerate(cards):
        x = 0.02 + i * 0.245
        _metric_card(ax, (x, 0.2), (0.21, 0.62), label, value, sublabel, color)
    ax.set_title("Data and benchmark validation", loc="left", weight="bold", fontsize=11, pad=6)
    _save(fig, "validation_scorecard")


def results_manifest() -> None:
    final = _read_csv("final_reversal_audit.csv")
    raw = json.loads((CLOSEOUT / "raw_chain_validation_summary.json").read_text(encoding="utf-8"))
    bench = json.loads((CLOSEOUT / "benchmark_reconciliation_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((FULL_DIR / "metrics.json").read_text(encoding="utf-8"))
    best = final[final["strategy"].eq("raw_tail_slope_reversal_skip1_quarterly_smooth3")].iloc[0]
    manifest = {
        "name": "surface_to_returns_results",
        "status": "completed",
        "outputs": {
            "headline_figure": "docs/assets/figures/headline_surface_to_returns.png",
            "headline_results": "docs/assets/tables/headline_results.csv",
            "reversal_validation": "docs/assets/tables/reversal_validation.csv",
            "strategy_variants": "docs/assets/tables/strategy_variants.csv",
            "factor_attribution": "docs/assets/tables/factor_attribution.csv",
            "feature_importance": "docs/assets/tables/feature_importance.csv",
        },
        "diagnostics": {
            "feature_store_rows": 1622696,
            "effective_coverage": "1996-01 to 2025-12",
            "holdout_observations": int(metrics.get("prediction_rank_ensemble", {}).get("n", 0)),
            "selected_strategy": best["strategy"],
            "selected_net_ann_mean": float(best["net_ann_mean"]),
            "selected_net_sharpe": float(best["net_sharpe"]),
            "selected_ff5_umd_alpha_t": float(best["ff5_umd_net_alpha_t"]),
            "raw_chain_atm_iv_correlation": raw["atm_iv_correlation"],
            "benchmark_market_correlation": bench["correlation"],
        },
    }
    out = ROOT / "docs" / "assets" / "results_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    inventory = [
        "# Results Inventory",
        "",
        "This file lists the aggregate outputs used by the README and research notebooks.",
        "",
        "- `docs/assets/figures/headline_surface_to_returns.png`",
        "- `docs/assets/figures/reversal_validation.png`",
        "- `docs/assets/figures/decile_shape.png`",
        "- `docs/assets/figures/cost_decomposition.png`",
        "- `docs/assets/figures/model_leaderboard.png`",
        "- `docs/assets/figures/feature_importance.png`",
        "- `docs/assets/figures/validation_scorecard.png`",
        "- `docs/assets/tables/headline_results.csv`",
        "- `docs/assets/tables/reversal_validation.csv`",
        "- `docs/assets/tables/strategy_variants.csv`",
        "- `docs/assets/tables/factor_attribution.csv`",
        "",
    ]
    (ROOT / "docs" / "assets" / "results_inventory.md").write_text("\n".join(inventory), encoding="utf-8")


def main() -> None:
    _mkdirs()
    _style()
    _copy_tables()
    headline_figure()
    reversal_validation()
    decile_shape()
    cost_decomposition()
    model_leaderboard()
    feature_importance()
    validation_scorecard()
    results_manifest()


if __name__ == "__main__":
    main()
