from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from ctrsdf.config import ProjectConfig, load_config
from ctrsdf.analysis.results import run_results_package
from ctrsdf.data.extract import full_extract, smoke_extract
from ctrsdf.data.schema import run_schema_audit
from ctrsdf.evaluation.metrics import performance_metrics, prediction_metrics
from ctrsdf.features.store import build_feature_store
from ctrsdf.models.baselines import fit_elastic_net, numeric_features
from ctrsdf.models.boosting import fit_gradient_boosting
from ctrsdf.models.ipca import SimpleIPCA
from ctrsdf.models.splits import assign_splits
from ctrsdf.plots.figures import (
    plot_cumulative_returns,
    plot_data_coverage,
    plot_drawdown,
    plot_feature_missingness,
    plot_model_leaderboard,
    plot_rolling_ic,
)
from ctrsdf.portfolio.backtest import long_short_decile_returns
from ctrsdf.sdf.neural import fit_neural_sdf, torch_available
from ctrsdf.utils.logging import configure_logging
from ctrsdf.utils.manifest import Manifest

LOGGER = logging.getLogger(__name__)


def _prepare(config: ProjectConfig, label: str) -> Path:
    configure_logging(config.path("logs"), label)
    for key in ["data_raw", "data_cache", "data_interim", "data_processed", "artifacts", "manifests", "figures", "model_cards"]:
        config.path(key).mkdir(parents=True, exist_ok=True)
    return config.path("logs") / f"{label}.log"


def cmd_schema_audit(config: ProjectConfig) -> None:
    _prepare(config, "schema_audit")
    output = run_schema_audit(config)
    LOGGER.info("Schema audit written to %s", output)


def cmd_smoke(config: ProjectConfig) -> None:
    _prepare(config, "smoke")
    if not (config.path("manifests") / "schema_audit.parquet").exists():
        run_schema_audit(config)
    smoke_extract(config)
    feature_path = build_feature_store(config, "smoke")
    _run_models(config, feature_path, "smoke", neural_epochs=3)


def cmd_full(config: ProjectConfig) -> None:
    _prepare(config, "full")
    if not (config.path("manifests") / "schema_audit.parquet").exists():
        run_schema_audit(config)
    full_extract(config)
    feature_path = build_feature_store(config, "full")
    _run_models(config, feature_path, "full", neural_epochs=120)


def cmd_figures(config: ProjectConfig) -> None:
    _prepare(config, "figures")
    for label in ("full", "smoke"):
        panel_path = config.path("data_processed") / label / "monthly_feature_store.parquet"
        returns_path = config.path("artifacts") / label / "portfolio_returns.parquet"
        predictions_path = config.path("artifacts") / label / "predictions.parquet"
        metrics_path = config.path("artifacts") / label / "metrics.json"
        if panel_path.exists():
            panel = pd.read_parquet(panel_path)
            plot_feature_missingness(panel, config.path("figures") / label)
            plot_data_coverage(panel, config.path("figures") / label)
        if returns_path.exists():
            returns = pd.read_parquet(returns_path)
            plot_cumulative_returns(returns, config.path("figures") / label)
            plot_drawdown(returns, config.path("figures") / label)
        if predictions_path.exists():
            plot_rolling_ic(pd.read_parquet(predictions_path), config.path("figures") / label)
        if metrics_path.exists():
            plot_model_leaderboard(metrics_path, config.path("figures") / label)


def cmd_results(config: ProjectConfig) -> None:
    _prepare(config, "surface_to_returns_results")
    path = run_results_package(config, "full")
    LOGGER.info("Surface-to-returns results written to %s", path)


def cmd_closeout(config: ProjectConfig) -> None:
    cmd_results(config)


def _run_models(config: ProjectConfig, feature_path: Path, label: str, neural_epochs: int) -> None:
    panel = pd.read_parquet(feature_path)
    panel = assign_splits(panel, config)
    target = "ret_excess_fwd_1m" if "ret_excess_fwd_1m" in panel.columns else "ret_fwd_1m"
    if panel[target].notna().sum() == 0 and target != "ret_fwd_1m":
        target = "ret_fwd_1m"
    if target not in panel:
        raise ValueError(f"Feature store missing target {target}.")
    leakage_cols = {
        "permno",
        "mcap",
        "vol",
        "trading_days",
        "ret_fwd_1m",
        "ret_excess_fwd_1m",
        "rf_fwd_1m",
    }
    features = numeric_features(panel, target, leakage_cols)
    features = [c for c in features if c not in {"ret"}]
    train = panel[panel["split"].isin(["train", "validation"])]
    holdout = panel[panel["split"] == "holdout"].copy()
    if len(train.dropna(subset=[target])) < 50:
        train = panel.dropna(subset=[target])
        holdout = panel.copy()
    if not features:
        raise ValueError("No numeric model features available.")
    features = [c for c in features if train[c].notna().any()]
    if not features:
        raise ValueError("No non-empty numeric model features available.")
    panel[features] = panel[features].replace([float("inf"), float("-inf")], float("nan"))
    train = panel[panel["split"].isin(["train", "validation"])]
    holdout = panel[panel["split"] == "holdout"].copy()
    if len(train.dropna(subset=[target])) < 50:
        train = panel.dropna(subset=[target])
        holdout = panel.copy()

    enet, enet_features = fit_elastic_net(train, target, features)
    holdout["prediction_elastic_net"] = enet.predict(
        holdout[enet_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([float("inf"), float("-inf")], float("nan"))
    )

    gbm = fit_gradient_boosting(train, target, features, use_gpu=bool(config.raw["models"].get("gpu", True)))
    holdout["prediction_gbm"] = gbm.predict(
        holdout[features].apply(pd.to_numeric, errors="coerce").replace([float("inf"), float("-inf")], float("nan"))
    )

    ipca = SimpleIPCA(n_factors=min(5, max(1, len(features) // 2))).fit(train, features, target)
    holdout["prediction_ipca"] = ipca.predict(holdout)

    if torch_available() and bool(config.raw["models"].get("gpu", True)):
        neural = fit_neural_sdf(train, target, features, epochs=neural_epochs)
        holdout["prediction_neural_sdf"] = neural.predict(
            holdout[features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([float("inf"), float("-inf")], float("nan"))
            .fillna(0)
            .astype("float64")
            .to_numpy()
        )

    prediction_cols = [c for c in holdout.columns if c.startswith("prediction_")]
    rank_cols = []
    for col in prediction_cols:
        rank_col = f"{col}_rank"
        holdout[rank_col] = holdout.groupby("month_end")[col].rank(pct=True)
        rank_cols.append(rank_col)
    holdout["prediction"] = holdout[rank_cols].mean(axis=1)
    returns = long_short_decile_returns(holdout, score="prediction", ret=target)
    outdir = config.path("artifacts") / label
    outdir.mkdir(parents=True, exist_ok=True)
    holdout.to_parquet(outdir / "predictions.parquet", index=False)
    returns.to_parquet(outdir / "portfolio_returns.parquet", index=False)

    model_metrics = {col: prediction_metrics(holdout, target, col) for col in prediction_cols}
    metrics = {
        "prediction_rank_ensemble": prediction_metrics(holdout, target, "prediction"),
        "models": model_metrics,
        "portfolio": performance_metrics(returns["long_short"]) if "long_short" in returns else {"n": 0},
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    plot_feature_missingness(panel, config.path("figures") / label)
    plot_data_coverage(panel, config.path("figures") / label)
    if not returns.empty:
        plot_cumulative_returns(returns, config.path("figures") / label)
        plot_drawdown(returns, config.path("figures") / label)
    plot_rolling_ic(holdout, config.path("figures") / label, target=target)
    plot_model_leaderboard(outdir / "metrics.json", config.path("figures") / label)
    Manifest(
        name=f"{label}_models",
        status="completed",
        outputs={
            "predictions": str(outdir / "predictions.parquet"),
            "portfolio_returns": str(outdir / "portfolio_returns.parquet"),
            "metrics": str(outdir / "metrics.json"),
        },
        diagnostics=metrics,
    ).write(config.path("manifests") / f"{label}_models.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["schema-audit", "smoke", "full", "figures", "results", "closeout"])
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    commands = {
        "schema-audit": cmd_schema_audit,
        "smoke": cmd_smoke,
        "full": cmd_full,
        "figures": cmd_figures,
        "results": cmd_results,
        "closeout": cmd_closeout,
    }
    commands[args.command](config)


if __name__ == "__main__":
    main()
