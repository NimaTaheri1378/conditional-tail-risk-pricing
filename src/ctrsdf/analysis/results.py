from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from ctrsdf.config import ProjectConfig
from ctrsdf.models.baselines import fama_macbeth
from ctrsdf.plots.figures import set_style
from ctrsdf.utils.io import atomic_write_parquet
from ctrsdf.utils.manifest import Manifest

LOGGER = logging.getLogger(__name__)

TARGET = "ret_excess_fwd_1m"
MODEL_EXCLUDE = {
    "permno",
    "mcap",
    "vol",
    "trading_days",
    "ret_fwd_1m",
    "ret_excess_fwd_1m",
    "rf_fwd_1m",
}
CORE_FEATURES = [
    "ret",
    "mcap",
    "quoted_spread",
    "adv_dollar",
    "book_equity",
    "gross_profitability",
    "operating_profitability",
    "asset_growth",
    "leverage",
    "cash_assets",
    "atm_iv_30",
    "atm_iv_60",
    "atm_iv_91",
    "skew_25d",
    "tail_slope_10p_25p",
    "term_slope",
    "curvature",
    "vix_level",
    "treasury_slope",
    "skew_25d_x_vix_level",
    "skew_25d_x_leverage",
    "atm_iv_30_x_cash_assets",
    "tail_slope_10p_25p_x_asset_growth",
]


@dataclass
class CloseoutPaths:
    outdir: Path
    figdir: Path
    rawdir: Path
    processed: Path


def run_results_package(config: ProjectConfig, label: str = "full") -> Path:
    paths = CloseoutPaths(
        outdir=config.path("artifacts") / "closeout",
        figdir=config.path("figures") / label,
        rawdir=config.path("data_raw") / label,
        processed=config.path("data_processed") / label,
    )
    paths.outdir.mkdir(parents=True, exist_ok=True)
    paths.figdir.mkdir(parents=True, exist_ok=True)

    _ensure_small_wrds_extracts(config, paths)
    panel = pd.read_parquet(paths.processed / "monthly_feature_store.parquet")
    predictions = pd.read_parquet(config.path("artifacts") / label / "predictions.parquet")
    ff5 = _load_ff5(paths.rawdir)
    security = _load_security_info(paths.rawdir)
    predictions = _attach_security_info(predictions, security)

    outputs: dict[str, str] = {}
    diagnostics: dict[str, object] = {}

    benchmark_path = _benchmark_reconciliation(panel, ff5, paths)
    outputs["benchmark_reconciliation"] = str(benchmark_path)
    diagnostics["benchmark_reconciliation"] = _read_json_summary(benchmark_path)
    outputs["fama_macbeth"] = str(_run_fama_macbeth(panel, paths))
    rf_pred, rf_metrics = _run_random_forest(panel, predictions, paths)
    outputs["random_forest_predictions"] = str(rf_pred)
    diagnostics["random_forest"] = rf_metrics
    outputs["double_ml"] = str(_run_double_ml(panel, paths))
    cost_path, cost_diag = _cost_turnover_backtests(predictions, paths)
    outputs["cost_turnover"] = str(cost_path)
    diagnostics["cost_turnover"] = cost_diag
    alpha_path, alpha_diag = _factor_alpha_and_spanning(paths, ff5)
    outputs["factor_alpha_spanning"] = str(alpha_path)
    diagnostics["factor_alpha_spanning"] = alpha_diag
    outputs["robustness"] = str(_robustness_tests(predictions, paths))
    signal_path = _signal_robustness_tests(panel, paths)
    outputs["signal_robustness"] = str(signal_path)
    diagnostics["signal_robustness"] = _summarize_signal_robustness(signal_path)
    strategy_path, strategy_diag = _strategy_strengthening_tests(predictions, ff5, paths)
    outputs["strategy_strengthening"] = str(strategy_path)
    diagnostics["strategy_strengthening"] = strategy_diag
    final_path, final_diag = _final_reversal_audit(predictions, ff5, paths)
    outputs["final_reversal_audit"] = str(final_path)
    diagnostics["final_reversal_audit"] = final_diag
    raw_chain_path = _raw_chain_validation(paths)
    outputs["raw_chain_validation"] = str(raw_chain_path)
    diagnostics["raw_chain_validation"] = _read_json_summary(raw_chain_path)
    outputs["interpretability"] = str(_interpretability(panel, predictions, paths))
    outputs["results_inventory"] = str(_results_inventory(config, paths, diagnostics))

    manifest = Manifest(
        name="surface_to_returns_results",
        status="completed",
        outputs=outputs,
        diagnostics=diagnostics,
    )
    return manifest.write(config.path("manifests") / "results_manifest.json")


def _ensure_small_wrds_extracts(config: ProjectConfig, paths: CloseoutPaths) -> None:
    ff5_path = paths.rawdir / "ff5_factors.parquet"
    security_path = paths.rawdir / "crsp_security_info.parquet"
    raw_chain_path = paths.rawdir / "raw_chain_validation_sample.parquet"
    if ff5_path.exists() and security_path.exists() and raw_chain_path.exists():
        return
    try:
        import wrds
    except ImportError:
        LOGGER.warning("WRDS package unavailable; skipping small closeout extracts.")
        return
    db = wrds.Connection()
    try:
        if not ff5_path.exists():
            query = """
                select date as month_end, mktrf, smb, hml, rmw, cma, rf, umd
                from ff_all.fivefactors_monthly
                where date between %(start)s and %(end)s
            """
            frame = db.raw_sql(query, params={"start": config.sample_start, "end": config.sample_end})
            atomic_write_parquet(frame, ff5_path)
        if not security_path.exists():
            query = """
                select permno, secinfostartdt, secinfoenddt, siccd, primaryexch
                from crsp_a_stock.stksecurityinfohist
                where coalesce(secinfoenddt, '2100-01-01') >= %(start)s
                  and secinfostartdt <= %(end)s
            """
            frame = db.raw_sql(query, params={"start": config.sample_start, "end": config.sample_end})
            atomic_write_parquet(frame, security_path)
        if not raw_chain_path.exists():
            frames = []
            years = [1996, 2000, 2005, 2010, 2015, 2020, 2025]
            for year in years:
                table = f"optionm_all.opprcd{year}"
                query = f"""
                    with sample_date as (
                      select max(date) as date
                      from {table}
                      where date between %(start)s and %(end)s
                    )
                    select o.secid, o.date, o.cp_flag, o.delta, o.impl_volatility,
                           o.best_bid, o.best_offer, o.volume, o.open_interest,
                           o.exdate, o.strike_price
                    from {table} o
                    join sample_date s on o.date = s.date
                    where o.best_bid > 0
                      and o.best_offer > o.best_bid
                      and o.impl_volatility between 0.01 and 3.0
                      and o.delta is not null
                      and o.exdate between o.date + interval '20 day' and o.date + interval '220 day'
                """
                try:
                    frames.append(
                        db.raw_sql(
                            query,
                            params={"start": f"{year}-12-01", "end": f"{year}-12-31"},
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Raw-chain validation skip for %s: %s", year, type(exc).__name__)
            if frames:
                atomic_write_parquet(pd.concat(frames, ignore_index=True), raw_chain_path)
    finally:
        db.close()


def _load_ff5(rawdir: Path) -> pd.DataFrame:
    path = rawdir / "ff5_factors.parquet"
    if path.exists():
        ff5 = pd.read_parquet(path)
    else:
        ff5 = pd.read_parquet(rawdir / "ff_factors.parquet")
        ff5["rmw"] = np.nan
        ff5["cma"] = np.nan
    ff5["month_end"] = pd.to_datetime(ff5["month_end"]) + pd.offsets.MonthEnd(0)
    return ff5


def _load_security_info(rawdir: Path) -> pd.DataFrame:
    path = rawdir / "crsp_security_info.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["permno", "month_end", "siccd", "primaryexch", "ff12"])
    sec = pd.read_parquet(path)
    sec["secinfostartdt"] = pd.to_datetime(sec["secinfostartdt"])
    sec["secinfoenddt"] = pd.to_datetime(sec["secinfoenddt"]).fillna(pd.Timestamp("2100-01-01"))
    sec["siccd"] = pd.to_numeric(sec["siccd"], errors="coerce")
    sec["ff12"] = sec["siccd"].map(ff12_from_sic)
    return sec


def _read_json_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "unreadable_json", "path": str(path)}


def _attach_security_info(frame: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    if sec.empty or "siccd" in frame.columns:
        return frame
    merged = frame.merge(sec, on="permno", how="left")
    keep = merged["siccd"].isna() | (
        (merged["month_end"] >= merged["secinfostartdt"])
        & (merged["month_end"] <= merged["secinfoenddt"])
    )
    merged = merged.loc[keep].sort_values(["permno", "month_end", "secinfostartdt"])
    return merged.drop_duplicates(["permno", "month_end"], keep="last")


def _numeric_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _model_features(frame: pd.DataFrame) -> list[str]:
    features = [
        c
        for c in CORE_FEATURES
        if c in frame.columns and c not in MODEL_EXCLUDE and pd.api.types.is_numeric_dtype(frame[c])
    ]
    return [c for c in features if frame[c].notna().any()]


def _benchmark_reconciliation(panel: pd.DataFrame, ff5: pd.DataFrame, paths: CloseoutPaths) -> Path:
    data = panel.dropna(subset=["ret", "mcap"]).copy()
    data["mcap_lag"] = data.groupby("permno")["mcap"].shift(1)
    vw = (
        data.dropna(subset=["mcap_lag"])
        .groupby("month_end")
        .apply(lambda x: np.average(x["ret"], weights=x["mcap_lag"].clip(lower=0)))
        .reset_index(name="crsp_vw_ret")
    )
    bench = ff5[["month_end", "mktrf", "rf"]].copy()
    bench["ff_market_ret"] = bench["mktrf"] + bench["rf"]
    out = vw.merge(bench, on="month_end", how="inner")
    out["diff"] = out["crsp_vw_ret"] - out["ff_market_ret"]
    summary = {
        "n_months": int(len(out)),
        "correlation": float(out["crsp_vw_ret"].corr(out["ff_market_ret"])),
        "mean_difference": float(out["diff"].mean()),
        "tracking_error_monthly": float(out["diff"].std()),
        "post_2024_mean_difference": float(out.loc[out["month_end"] >= "2024-01-01", "diff"].mean()),
    }
    out.to_csv(paths.outdir / "benchmark_reconciliation.csv", index=False)
    (paths.outdir / "benchmark_reconciliation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    set_style()
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=160)
    ax.plot(out["month_end"], out["diff"], color="#4c78a8", lw=1.0)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_title("CRSP-Built Market Return minus FF Market Return")
    ax.set_ylabel("Monthly difference")
    fig.savefig(paths.figdir / "benchmark_reconciliation.png", bbox_inches="tight")
    fig.savefig(paths.figdir / "benchmark_reconciliation.pdf", bbox_inches="tight")
    plt.close(fig)
    return paths.outdir / "benchmark_reconciliation_summary.json"


def _run_fama_macbeth(panel: pd.DataFrame, paths: CloseoutPaths) -> Path:
    features = [
        c
        for c in [
            "atm_iv_30",
            "skew_25d",
            "tail_slope_10p_25p",
            "term_slope",
            "curvature",
            "asset_growth",
            "leverage",
            "cash_assets",
            "vix_level",
            "skew_25d_x_vix_level",
            "skew_25d_x_leverage",
            "atm_iv_30_x_cash_assets",
        ]
        if c in panel.columns
    ]
    result = fama_macbeth(panel, TARGET, features)
    path = paths.outdir / "fama_macbeth.csv"
    result.to_csv(path, index=False)
    return path


def _run_random_forest(panel: pd.DataFrame, predictions: pd.DataFrame, paths: CloseoutPaths) -> tuple[Path, dict[str, float]]:
    path = paths.outdir / "random_forest_predictions.parquet"
    if path.exists():
        out = pd.read_parquet(path)
        return path, _prediction_metrics(out[TARGET], out["prediction_random_forest"])
    features = _model_features(panel)
    train = panel[(panel["month_end"] < "2013-01-01") & panel[TARGET].notna()].copy()
    holdout = predictions[predictions[TARGET].notna()].copy()
    if len(train) > 300_000:
        train = train.sample(300_000, random_state=20260527)
    x_train = _numeric_matrix(train, features)
    med = x_train.median().fillna(0.0)
    x_train = x_train.fillna(med).fillna(0.0)
    y_train = pd.to_numeric(train[TARGET], errors="coerce").astype(float)
    model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=100,
        max_features="sqrt",
        n_jobs=1,
        random_state=20260527,
    )
    model.fit(x_train, y_train)
    x_holdout = _numeric_matrix(holdout, features).fillna(med).fillna(0.0)
    pred = model.predict(x_holdout)
    out = holdout[["permno", "month_end", TARGET]].copy()
    out["prediction_random_forest"] = pred
    atomic_write_parquet(out, path)
    metrics = _prediction_metrics(out[TARGET], out["prediction_random_forest"])
    importances = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    importances.to_csv(paths.outdir / "random_forest_importance.csv", index=False)
    set_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=160)
    sns.barplot(data=importances.head(20), y="feature", x="importance", ax=ax, color="#4c78a8")
    ax.set_title("Random Forest Feature Importance")
    ax.set_ylabel("")
    fig.savefig(paths.figdir / "random_forest_importance.png", bbox_inches="tight")
    fig.savefig(paths.figdir / "random_forest_importance.pdf", bbox_inches="tight")
    plt.close(fig)
    return path, metrics


def _prediction_metrics(y: pd.Series, pred: pd.Series | np.ndarray) -> dict[str, float]:
    y_arr = pd.to_numeric(y, errors="coerce").astype(float).to_numpy()
    p_arr = np.asarray(pred, dtype=float)
    mask = np.isfinite(y_arr) & np.isfinite(p_arr)
    y_arr, p_arr = y_arr[mask], p_arr[mask]
    denom = np.square(y_arr - y_arr.mean()).sum()
    return {
        "n": float(mask.sum()),
        "mae": float(mean_absolute_error(y_arr, p_arr)),
        "mse": float(mean_squared_error(y_arr, p_arr)),
        "oos_r2": float(1 - np.square(p_arr - y_arr).sum() / denom) if denom else np.nan,
        "spearman_ic": float(stats.spearmanr(y_arr, p_arr, nan_policy="omit").statistic),
    }


def _run_double_ml(panel: pd.DataFrame, paths: CloseoutPaths) -> Path:
    path = paths.outdir / "double_ml_option_contribution.csv"
    if path.exists():
        return path
    data = panel.dropna(subset=[TARGET, "skew_25d"]).copy()
    controls = [c for c in _model_features(data) if c not in {"skew_25d", "tail_slope_10p_25p", "atm_iv_30", "atm_iv_60", "atm_iv_91"}]
    if len(data) > 250_000:
        data = data.sample(250_000, random_state=20260527)
    months = np.array(sorted(data["month_end"].dropna().unique()))
    splitter = TimeSeriesSplit(n_splits=5)
    residual_rows = []
    for train_month_idx, test_month_idx in splitter.split(months):
        train_months = set(months[train_month_idx])
        test_months = set(months[test_month_idx])
        train = data[data["month_end"].isin(train_months)]
        test = data[data["month_end"].isin(test_months)]
        x_train = _numeric_matrix(train, controls)
        med = x_train.median().fillna(0.0)
        x_train = x_train.fillna(med).fillna(0.0)
        x_test = _numeric_matrix(test, controls).fillna(med).fillna(0.0)
        y_model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=200, n_jobs=1, random_state=20260527)
        d_model = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=200, n_jobs=1, random_state=20260528)
        y_model.fit(x_train, train[TARGET].astype(float))
        d_model.fit(x_train, train["skew_25d"].astype(float))
        residual_rows.append(
            pd.DataFrame(
                {
                    "y_resid": test[TARGET].astype(float).to_numpy() - y_model.predict(x_test),
                    "d_resid": test["skew_25d"].astype(float).to_numpy() - d_model.predict(x_test),
                }
            )
        )
    resid = pd.concat(residual_rows, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    x = sm.add_constant(resid["d_resid"], has_constant="add")
    fit = sm.OLS(resid["y_resid"], x).fit(cov_type="HC1")
    result = pd.DataFrame(
        [
            {
                "treatment": "skew_25d",
                "coefficient": fit.params["d_resid"],
                "tstat": fit.tvalues["d_resid"],
                "pvalue": fit.pvalues["d_resid"],
                "n": fit.nobs,
            }
        ]
    )
    result.to_csv(path, index=False)
    return path


def _cost_turnover_backtests(predictions: pd.DataFrame, paths: CloseoutPaths) -> tuple[Path, dict[str, float]]:
    path = paths.outdir / "cost_turnover_backtests.csv"
    if path.exists():
        out = pd.read_csv(path)
        diag = {
            "rows": int(len(out)),
            "baseline_net_sharpe_eta_010": float(
                _annualized_sharpe(out[(out["scheme"] == "sector_value") & (out["eta"] == 0.10)]["net_return"])
            ),
        }
        return path, diag
    data = predictions.dropna(subset=["prediction", TARGET]).copy()
    data = data[(data["prc"].abs() >= 5) & (data["adv_dollar"].fillna(0) >= 1_000_000)]
    variants = []
    for scheme in ["equal", "value", "sector_value"]:
        weights = _make_weights(data, scheme)
        variants.append(_portfolio_from_weights(data, weights, scheme))
    out = pd.concat(variants, ignore_index=True)
    out.to_csv(path, index=False)
    _plot_cost_turnover(out, paths.figdir)
    diag = {
        "rows": int(len(out)),
        "baseline_net_sharpe_eta_010": float(
            _annualized_sharpe(out[(out["scheme"] == "sector_value") & (out["eta"] == 0.10)]["net_return"])
        ),
    }
    return path, diag


def _make_weights(data: pd.DataFrame, scheme: str) -> pd.DataFrame:
    rows = []
    for date, group in data.groupby("month_end"):
        sub = group.dropna(subset=["prediction"]).copy()
        if sub["prediction"].nunique() < 10:
            continue
        sub["bucket"] = pd.qcut(sub["prediction"], 10, labels=False, duplicates="drop")
        low = sub[sub["bucket"] == sub["bucket"].min()].copy()
        high = sub[sub["bucket"] == sub["bucket"].max()].copy()
        high["side"], low["side"] = 1.0, -1.0
        active = pd.concat([high, low], ignore_index=True)
        if scheme == "equal":
            active["base"] = 1.0
        else:
            active["base"] = active["mcap"].clip(lower=0).fillna(0)
            active.loc[active["base"] <= 0, "base"] = 1.0
        if scheme == "sector_value" and "ff12" in active.columns:
            active["ff12"] = active["ff12"].fillna("Other")
            parts = []
            for side, side_group in active.groupby("side"):
                industries = sorted(side_group["ff12"].unique())
                industry_budget = 0.5 / max(len(industries), 1)
                for _, industry_group in side_group.groupby("ff12"):
                    w = industry_group["base"] / industry_group["base"].sum()
                    tmp = industry_group.copy()
                    tmp["weight"] = side * industry_budget * w
                    parts.append(tmp)
            active = pd.concat(parts, ignore_index=True)
        else:
            parts = []
            for side, side_group in active.groupby("side"):
                w = side_group["base"] / side_group["base"].sum()
                tmp = side_group.copy()
                tmp["weight"] = side * 0.5 * w
                parts.append(tmp)
            active = pd.concat(parts, ignore_index=True)
        active["weight"] = _cap_and_renormalize(active["weight"], active["side"], cap=0.02)
        rows.append(active[["month_end", "permno", "weight"]])
    if not rows:
        return pd.DataFrame(columns=["month_end", "permno", "weight"])
    return pd.concat(rows, ignore_index=True)


def _make_rebalanced_weights(data: pd.DataFrame, scheme: str, rebalance_every: int = 3) -> pd.DataFrame:
    months = sorted(pd.to_datetime(data["month_end"].dropna().unique()))
    if not months:
        return pd.DataFrame(columns=["month_end", "permno", "weight"])
    rebalance_months = months[::rebalance_every]
    target = _make_weights(data[data["month_end"].isin(rebalance_months)].copy(), scheme)
    if target.empty:
        return target
    expanded = []
    month_positions = {month: idx for idx, month in enumerate(months)}
    for rebalance_month in rebalance_months:
        start = month_positions[rebalance_month]
        hold_months = months[start : min(start + rebalance_every, len(months))]
        weights = target[target["month_end"].eq(rebalance_month)].copy()
        if weights.empty:
            continue
        for hold_month in hold_months:
            tmp = weights.copy()
            tmp["month_end"] = hold_month
            expanded.append(tmp)
    if not expanded:
        return pd.DataFrame(columns=["month_end", "permno", "weight"])
    return pd.concat(expanded, ignore_index=True)


def _delay_weights(weights: pd.DataFrame, delay_months: int) -> pd.DataFrame:
    if delay_months <= 0 or weights.empty:
        return weights
    out = weights.copy()
    out["month_end"] = pd.to_datetime(out["month_end"]) + pd.offsets.MonthEnd(delay_months)
    return out


def _cap_and_renormalize(weights: pd.Series, side: pd.Series, cap: float) -> pd.Series:
    out = weights.clip(lower=-cap, upper=cap)
    for sign in [1.0, -1.0]:
        mask = side == sign
        total = out.loc[mask].abs().sum()
        if total > 0:
            out.loc[mask] = out.loc[mask] * (0.5 / total)
    return out


def _portfolio_from_weights(data: pd.DataFrame, weights: pd.DataFrame, scheme: str) -> pd.DataFrame:
    merged = weights.merge(
        data[["month_end", "permno", TARGET, "quoted_spread", "adv_dollar", "ret"]],
        on=["month_end", "permno"],
        how="left",
    ).sort_values(["month_end", "permno"])
    merged = merged.dropna(subset=[TARGET]).copy()
    prev = merged[["month_end", "permno", "weight"]].copy()
    prev["month_end"] = prev["month_end"] + pd.offsets.MonthEnd(1)
    prev = prev.rename(columns={"weight": "prev_weight"})
    merged = merged.merge(prev, on=["month_end", "permno"], how="left")
    merged["trade"] = merged["weight"] - merged["prev_weight"].fillna(0.0)
    merged["gross_return"] = merged["weight"] * merged[TARGET].astype(float)
    merged["half_spread_cost"] = merged["trade"].abs() * (merged["quoted_spread"].fillna(0).clip(lower=0, upper=0.25) / 2.0)
    merged["participation"] = (merged["trade"].abs() * 100_000_000 / (20 * merged["adv_dollar"].replace(0, np.nan))).clip(lower=0, upper=1)
    merged["vol_proxy"] = merged["ret"].abs().fillna(merged["ret"].abs().median()).clip(upper=1)
    rows = []
    for eta in [0.05, 0.10, 0.20]:
        tmp = merged.copy()
        tmp["impact_cost"] = tmp["trade"].abs() * eta * tmp["vol_proxy"] * np.sqrt(tmp["participation"].fillna(0))
        monthly = tmp.groupby("month_end").agg(
            gross_return=("gross_return", "sum"),
            turnover=("trade", lambda x: x.abs().sum()),
            half_spread_cost=("half_spread_cost", "sum"),
            impact_cost=("impact_cost", "sum"),
            p95_participation=("participation", lambda x: x.quantile(0.95)),
        ).reset_index()
        monthly["scheme"] = scheme
        monthly["eta"] = eta
        monthly["net_return"] = monthly["gross_return"] - monthly["half_spread_cost"] - monthly["impact_cost"]
        rows.append(monthly)
    return pd.concat(rows, ignore_index=True)


def _plot_cost_turnover(out: pd.DataFrame, figdir: Path) -> None:
    set_style()
    base = out[(out["scheme"] == "sector_value") & (out["eta"] == 0.10)].copy()
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=160)
    ax.stackplot(
        base["month_end"],
        base["half_spread_cost"],
        base["impact_cost"],
        labels=["Half-spread", "Impact"],
        colors=["#4c78a8", "#f58518"],
        alpha=0.8,
    )
    ax.set_title("Turnover-Cost Decomposition")
    ax.set_ylabel("Monthly cost")
    ax.legend(frameon=False)
    fig.savefig(figdir / "turnover_cost_decomposition.png", bbox_inches="tight")
    fig.savefig(figdir / "turnover_cost_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)


def _factor_alpha_and_spanning(paths: CloseoutPaths, ff5: pd.DataFrame) -> tuple[Path, dict[str, float]]:
    path = paths.outdir / "factor_alpha_spanning.csv"
    if path.exists():
        alpha = pd.read_csv(path)
        spanning = paths.outdir / "ff25_spanning_alphas.csv"
        mean_abs = pd.read_csv(spanning)["alpha"].abs().mean() if spanning.exists() else np.nan
        diagnostics = {
            "net_ff5_umd_alpha_monthly": float(
                alpha[(alpha["return"] == "net_return") & (alpha["model"] == "FF5_UMD")]["alpha"].iloc[0]
            )
            if ((alpha["return"] == "net_return") & (alpha["model"] == "FF5_UMD")).any()
            else np.nan,
            "ff25_mean_abs_alpha": float(mean_abs),
        }
        return path, diagnostics
    costs = pd.read_csv(paths.outdir / "cost_turnover_backtests.csv", parse_dates=["month_end"])
    strategy = costs[(costs["scheme"] == "sector_value") & (costs["eta"] == 0.10)][["month_end", "gross_return", "net_return"]]
    factors = ff5.copy()
    joined = strategy.merge(factors, on="month_end", how="inner")
    specs = {
        "CAPM": ["mktrf"],
        "FF3": ["mktrf", "smb", "hml"],
        "FF3_UMD": ["mktrf", "smb", "hml", "umd"],
        "FF5": ["mktrf", "smb", "hml", "rmw", "cma"],
        "FF5_UMD": ["mktrf", "smb", "hml", "rmw", "cma", "umd"],
    }
    rows = []
    for ret_col in ["gross_return", "net_return"]:
        for name, cols in specs.items():
            available = [c for c in cols if c in joined and joined[c].notna().any()]
            fit = _hac_alpha(joined.dropna(subset=[ret_col] + available), ret_col, available)
            rows.append({"return": ret_col, "model": name, **fit})
    alpha = pd.DataFrame(rows)
    alpha.to_csv(path, index=False)
    _plot_alpha(alpha, paths.figdir)
    spanning = _test_asset_spanning(paths, ff5)
    diagnostics = {
        "net_ff5_umd_alpha_monthly": float(
            alpha[(alpha["return"] == "net_return") & (alpha["model"] == "FF5_UMD")]["alpha"].iloc[0]
        )
        if ((alpha["return"] == "net_return") & (alpha["model"] == "FF5_UMD")).any()
        else np.nan,
        "ff25_mean_abs_alpha": spanning.get("mean_abs_alpha", np.nan),
    }
    return path, diagnostics


def _hac_alpha(data: pd.DataFrame, ret_col: str, factors: list[str]) -> dict[str, float]:
    cols = [ret_col] + factors
    data = data[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < len(factors) + 12:
        return {"alpha": np.nan, "alpha_t": np.nan, "n": float(len(data))}
    x = sm.add_constant(data[factors].astype("float64"), has_constant="add")
    fit = sm.OLS(data[ret_col].astype("float64"), x).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return {"alpha": float(fit.params["const"]), "alpha_t": float(fit.tvalues["const"]), "n": float(fit.nobs)}


def _plot_alpha(alpha: pd.DataFrame, figdir: Path) -> None:
    set_style()
    data = alpha[alpha["return"] == "net_return"].copy()
    fig, ax = plt.subplots(figsize=(6.8, 3.5), dpi=160)
    sns.barplot(data=data, x="model", y="alpha", ax=ax, color="#59a14f")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_title("Net Strategy Alpha by Benchmark")
    ax.set_ylabel("Monthly alpha")
    ax.set_xlabel("")
    fig.savefig(figdir / "alpha_bar.png", bbox_inches="tight")
    fig.savefig(figdir / "alpha_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def _test_asset_spanning(paths: CloseoutPaths, ff5: pd.DataFrame) -> dict[str, float]:
    pt_path = paths.rawdir / "ff_test_portfolios.parquet"
    if not pt_path.exists():
        return {}
    pt = pd.read_parquet(pt_path)
    pt["month_end"] = pd.to_datetime(pt["date"]) + pd.offsets.MonthEnd(0)
    data = pt.merge(ff5, on="month_end", how="inner")
    factors = [c for c in ["mktrf", "smb", "hml", "rmw", "cma"] if c in data and data[c].notna().any()]
    rows = []
    for col in [c for c in pt.columns if c.endswith("_vwret")]:
        tmp = data.dropna(subset=[col] + factors + ["rf"]).copy()
        tmp["excess"] = tmp[col].astype(float) - tmp["rf"].astype(float)
        fit = _hac_alpha(tmp, "excess", factors)
        rows.append({"portfolio": col, **fit})
    out = pd.DataFrame(rows)
    out.to_csv(paths.outdir / "ff25_spanning_alphas.csv", index=False)
    return {"mean_abs_alpha": float(out["alpha"].abs().mean())}


def _robustness_tests(predictions: pd.DataFrame, paths: CloseoutPaths) -> Path:
    data = predictions.dropna(subset=["prediction", TARGET]).copy()
    data["year"] = data["month_end"].dt.year
    data["mcap_rank"] = data.groupby("month_end")["mcap"].rank(pct=True)
    cases = {
        "all": pd.Series(True, index=data.index),
        "large_cap": data["mcap_rank"] >= 0.5,
        "ex_microcap": data["mcap_rank"] >= 0.2,
        "ex_2020": data["year"] != 2020,
        "pre_gfc": data["month_end"] < "2007-07-01",
        "post_gfc": data["month_end"] >= "2009-07-01",
        "post_2020": data["month_end"] >= "2021-01-01",
    }
    if "siccd" in data:
        cases["ex_financials"] = ~data["siccd"].between(6000, 6999)
    rows = []
    for name, mask in cases.items():
        subset = data[mask].copy()
        ret = _quick_decile_returns(subset)
        perf = _return_summary(ret["long_short"]) if not ret.empty else {}
        rows.append({"case": name, **perf})
    out = pd.DataFrame(rows)
    path = paths.outdir / "robustness.csv"
    out.to_csv(path, index=False)
    _plot_robustness(out, paths.figdir)
    return path


def _signal_robustness_tests(panel: pd.DataFrame, paths: CloseoutPaths) -> Path:
    data = panel.dropna(subset=["skew_25d", TARGET]).copy()
    data["year"] = data["month_end"].dt.year
    data["mcap_rank"] = data.groupby("month_end")["mcap"].rank(pct=True)
    cases = {
        "all": pd.Series(True, index=data.index),
        "large_cap": data["mcap_rank"] >= 0.5,
        "ex_microcap": data["mcap_rank"] >= 0.2,
        "ex_2020": data["year"] != 2020,
        "pre_gfc": data["month_end"] < "2007-07-01",
        "post_gfc": data["month_end"] >= "2009-07-01",
        "post_2020": data["month_end"] >= "2021-01-01",
    }
    rows = []
    for signal in ["skew_25d", "tail_slope_10p_25p", "atm_iv_30"]:
        if signal not in data:
            continue
        for name, mask in cases.items():
            subset = data[mask].rename(columns={signal: "prediction"}).copy()
            ret = _quick_decile_returns(subset.dropna(subset=["prediction"]))
            perf = _return_summary(ret["long_short"]) if not ret.empty else {}
            rows.append({"signal": signal, "case": name, **perf})
    out = pd.DataFrame(rows)
    path = paths.outdir / "signal_robustness_full_sample.csv"
    out.to_csv(path, index=False)
    set_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    plot_data = out[out["signal"] == "skew_25d"].sort_values("ann_mean")
    sns.pointplot(data=plot_data, y="case", x="ann_mean", ax=ax, color="#4c78a8")
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_title("Full-Sample Signal Robustness: Skew")
    ax.set_xlabel("Annualized mean")
    ax.set_ylabel("")
    fig.savefig(paths.figdir / "signal_robustness_forest.png", bbox_inches="tight")
    fig.savefig(paths.figdir / "signal_robustness_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _summarize_signal_robustness(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    data = pd.read_csv(path)
    if data.empty:
        return {"status": "empty", "path": str(path)}
    key_cases = data[data["case"].isin(["all", "pre_gfc", "post_gfc", "post_2020"])]
    best = data.sort_values("sharpe", ascending=False).iloc[0]
    worst = data.sort_values("sharpe", ascending=True).iloc[0]
    return {
        "rows": int(len(data)),
        "signals": sorted(data["signal"].dropna().unique().tolist()),
        "cases": sorted(data["case"].dropna().unique().tolist()),
        "best_sharpe": {
            "signal": str(best["signal"]),
            "case": str(best["case"]),
            "sharpe": float(best["sharpe"]),
            "ann_mean": float(best["ann_mean"]),
        },
        "worst_sharpe": {
            "signal": str(worst["signal"]),
            "case": str(worst["case"]),
            "sharpe": float(worst["sharpe"]),
            "ann_mean": float(worst["ann_mean"]),
        },
        "key_case_mean_ann_return": float(key_cases["ann_mean"].mean()),
    }


def _strategy_strengthening_tests(
    predictions: pd.DataFrame,
    ff5: pd.DataFrame,
    paths: CloseoutPaths,
) -> tuple[Path, dict[str, object]]:
    summary_path = paths.outdir / "strategy_strengthening_summary.csv"
    monthly_path = paths.outdir / "strategy_strengthening_monthly.csv"
    alpha_path = paths.outdir / "strategy_strengthening_alphas.csv"
    monotonic_path = paths.outdir / "strategy_decile_monotonicity.csv"
    expected_strategy = "tail_reversal_liquid_quarterly_smooth3"
    if summary_path.exists() and monthly_path.exists() and alpha_path.exists() and monotonic_path.exists():
        summary = pd.read_csv(summary_path)
        if expected_strategy in set(summary.get("strategy", pd.Series(dtype=str)).dropna()):
            _plot_strategy_strengthening(summary, paths.figdir)
            _plot_strategy_monotonicity(pd.read_csv(monotonic_path), paths.figdir)
            return summary_path, _strategy_strengthening_diagnostics(summary)

    data = predictions.copy()
    data["month_end"] = pd.to_datetime(data["month_end"]) + pd.offsets.MonthEnd(0)
    required = [TARGET, "prediction", "mcap", "prc", "adv_dollar", "quoted_spread", "ret"]
    missing = [col for col in required if col not in data.columns]
    if missing:
        empty = pd.DataFrame({"status": ["missing_columns"], "missing_columns": [",".join(missing)]})
        empty.to_csv(summary_path, index=False)
        return summary_path, {"status": "missing_columns", "missing_columns": missing}

    data = data.dropna(subset=[TARGET, "mcap", "prc", "adv_dollar"]).copy()
    data = data[(data["prc"].abs() >= 5) & (data["adv_dollar"].fillna(0) >= 5_000_000)]
    data["mcap_rank"] = data.groupby("month_end")["mcap"].rank(pct=True)
    data["leverage_rank"] = _rank_by_month(data, "leverage")
    data["asset_growth_rank"] = _rank_by_month(data, "asset_growth")
    data = _attach_state_regimes(data)
    data = _add_tail_strategy_scores(data)
    data = _add_smoothed_scores(
        data,
        ["prediction", "score_tail_composite", "score_tail_slope", "score_tail_reversal", "score_tail_slope_reversal"],
        window=3,
    )

    strategies = [
        ("ml_score_liquid", "prediction", data["mcap_rank"] >= 0.20, "monthly"),
        ("ml_score_liquid_quarterly_smooth3", "prediction_smooth3", data["mcap_rank"] >= 0.20, "quarterly"),
        ("tail_slope_liquid", "score_tail_slope", data["mcap_rank"] >= 0.20, "monthly"),
        ("tail_slope_liquid_quarterly_smooth3", "score_tail_slope_smooth3", data["mcap_rank"] >= 0.20, "quarterly"),
        ("tail_slope_reversal_liquid", "score_tail_slope_reversal", data["mcap_rank"] >= 0.20, "monthly"),
        (
            "tail_slope_reversal_liquid_quarterly_smooth3",
            "score_tail_slope_reversal_smooth3",
            data["mcap_rank"] >= 0.20,
            "quarterly",
        ),
        ("tail_composite_liquid", "score_tail_composite", data["mcap_rank"] >= 0.20, "monthly"),
        (
            "tail_composite_liquid_quarterly_smooth3",
            "score_tail_composite_smooth3",
            data["mcap_rank"] >= 0.20,
            "quarterly",
        ),
        ("tail_reversal_liquid", "score_tail_reversal", data["mcap_rank"] >= 0.20, "monthly"),
        (
            "tail_reversal_liquid_quarterly_smooth3",
            "score_tail_reversal_smooth3",
            data["mcap_rank"] >= 0.20,
            "quarterly",
        ),
        (
            "tail_reversal_large_liquid_quarterly_smooth3",
            "score_tail_reversal_smooth3",
            data["mcap_rank"] >= 0.50,
            "quarterly",
        ),
        ("tail_reversal_high_vix", "score_tail_reversal", (data["mcap_rank"] >= 0.20) & data["high_vix_state"], "monthly"),
        ("tail_composite_large_liquid", "score_tail_composite", data["mcap_rank"] >= 0.50, "monthly"),
        (
            "tail_composite_large_liquid_quarterly_smooth3",
            "score_tail_composite_smooth3",
            data["mcap_rank"] >= 0.50,
            "quarterly",
        ),
        ("tail_composite_high_vix", "score_tail_composite", (data["mcap_rank"] >= 0.20) & data["high_vix_state"], "monthly"),
        ("tail_composite_low_vix", "score_tail_composite", (data["mcap_rank"] >= 0.20) & ~data["high_vix_state"], "monthly"),
        (
            "tail_composite_high_drawdown",
            "score_tail_composite",
            (data["mcap_rank"] >= 0.20) & data["market_drawdown_state"],
            "monthly",
        ),
        (
            "tail_composite_constrained",
            "score_tail_composite",
            (data["mcap_rank"] >= 0.20) & (data["leverage_rank"] >= 0.50),
            "monthly",
        ),
        (
            "tail_composite_aggressive_investment",
            "score_tail_composite",
            (data["mcap_rank"] >= 0.20) & (data["asset_growth_rank"] >= 0.50),
            "monthly",
        ),
    ]

    monthly_frames = []
    constituent_rows = []
    for name, score, mask, rebalance in strategies:
        if score not in data.columns:
            continue
        subset = data.loc[mask].dropna(subset=[score]).copy()
        if subset.empty:
            continue
        subset["prediction"] = subset[score]
        if subset.groupby("month_end")["prediction"].nunique().max() < 10:
            continue
        if rebalance == "quarterly":
            weights = _make_rebalanced_weights(subset, "sector_value", rebalance_every=3)
        else:
            weights = _make_weights(subset, "sector_value")
        if weights.empty:
            continue
        monthly = _portfolio_from_weights(subset, weights, "sector_value")
        monthly["strategy"] = name
        monthly["score"] = score
        monthly["rebalance"] = rebalance
        monthly_frames.append(monthly)
        counts = weights.groupby("month_end")["permno"].nunique().reset_index(name="n_names")
        counts["strategy"] = name
        constituent_rows.append(counts)

    if monthly_frames:
        monthly_all = pd.concat(monthly_frames, ignore_index=True)
    else:
        monthly_all = pd.DataFrame()
    monthly_all.to_csv(monthly_path, index=False)

    counts_all = pd.concat(constituent_rows, ignore_index=True) if constituent_rows else pd.DataFrame()
    summary = _summarize_strategy_monthly(monthly_all, counts_all)
    summary.to_csv(summary_path, index=False)
    alphas = _strategy_strengthening_alphas(monthly_all, ff5)
    alphas.to_csv(alpha_path, index=False)
    monotonic = _strategy_decile_monotonicity(data)
    monotonic.to_csv(monotonic_path, index=False)
    _plot_strategy_strengthening(summary, paths.figdir)
    _plot_strategy_monotonicity(monotonic, paths.figdir)
    return summary_path, _strategy_strengthening_diagnostics(summary)


def _rank_by_month(data: pd.DataFrame, col: str) -> pd.Series:
    if col not in data:
        return pd.Series(np.nan, index=data.index)
    return data.groupby("month_end")[col].rank(pct=True)


def _attach_state_regimes(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    state_cols = ["month_end"] + [col for col in ["vix_level", "mktrf"] if col in out.columns]
    if "vix_level" in out:
        state = (
            out[state_cols]
            .drop_duplicates("month_end")
            .sort_values("month_end")
            .copy()
        )
        state["vix_median_lagged"] = state["vix_level"].shift(1).expanding(min_periods=60).median()
        state["high_vix_state"] = state["vix_level"] >= state["vix_median_lagged"]
    else:
        state = out[state_cols].drop_duplicates("month_end").sort_values("month_end").copy()
        state["high_vix_state"] = False
    if "mktrf" in state:
        state["market_index"] = (1.0 + state["mktrf"].fillna(0).astype(float)).cumprod()
        state["market_peak"] = state["market_index"].cummax()
        state["market_drawdown"] = state["market_index"] / state["market_peak"] - 1.0
        state["market_drawdown_state"] = state["market_drawdown"] <= -0.10
    else:
        state["market_drawdown"] = np.nan
        state["market_drawdown_state"] = False
    out = out.merge(
        state[["month_end", "high_vix_state", "market_drawdown", "market_drawdown_state"]],
        on="month_end",
        how="left",
    )
    out["high_vix_state"] = out["high_vix_state"].fillna(False).astype(bool)
    out["market_drawdown_state"] = out["market_drawdown_state"].fillna(False).astype(bool)
    return out


def _add_tail_strategy_scores(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    if "tail_slope_10p_25p" in out:
        out["score_tail_slope"] = out["tail_slope_10p_25p"]
    else:
        out["score_tail_slope"] = np.nan
    if "skew_25d_x_vix_level" not in out and {"skew_25d", "vix_level"}.issubset(out.columns):
        out["skew_25d_x_vix_level"] = out["skew_25d"].astype(float) * out["vix_level"].astype(float)
    if "tail_slope_10p_25p_x_asset_growth" not in out and {
        "tail_slope_10p_25p",
        "asset_growth",
    }.issubset(out.columns):
        out["tail_slope_10p_25p_x_asset_growth"] = out["tail_slope_10p_25p"].astype(float) * out[
            "asset_growth"
        ].astype(float)
    score_parts = []
    for col in ["tail_slope_10p_25p", "skew_25d_x_vix_level", "tail_slope_10p_25p_x_asset_growth"]:
        if col in out:
            rank = out.groupby("month_end")[col].rank(pct=True)
            score_parts.append(rank)
    if score_parts:
        out["score_tail_composite"] = pd.concat(score_parts, axis=1).mean(axis=1)
    else:
        out["score_tail_composite"] = np.nan
    out["score_tail_reversal"] = -out["score_tail_composite"]
    out["score_tail_slope_reversal"] = -out["score_tail_slope"]
    return out


def _add_smoothed_scores(data: pd.DataFrame, columns: list[str], window: int = 3) -> pd.DataFrame:
    out = data.sort_values(["permno", "month_end"]).copy()
    for col in columns:
        if col not in out:
            continue
        out[f"{col}_smooth{window}"] = (
            out.groupby("permno")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            .astype(float)
        )
    return out


def _add_residualized_tail_scores(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["log_mcap"] = np.log(out["mcap"].where(out["mcap"] > 0))
    out["log_adv_dollar"] = np.log(out["adv_dollar"].where(out["adv_dollar"] > 0))
    controls = [
        "log_mcap",
        "log_adv_dollar",
        "quoted_spread",
        "ret",
        "atm_iv_30",
        "leverage",
        "operating_profitability",
        "gross_profitability",
        "asset_growth",
        "prediction",
    ]
    controls = [col for col in controls if col in out and out[col].notna().any()]
    for score in ["score_tail_slope_reversal", "score_tail_reversal"]:
        if score not in out:
            continue
        resid_col = f"{score}_resid"
        out[resid_col] = np.nan
        for _, group in out.groupby("month_end"):
            idx = group.index[group[score].notna()]
            if len(idx) <= len(controls) + 25:
                continue
            x = group.loc[idx, controls].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            x = x.fillna(x.median()).fillna(0.0)
            y = pd.to_numeric(group.loc[idx, score], errors="coerce")
            valid = y.notna() & np.isfinite(y.astype(float))
            if valid.sum() <= len(controls) + 25:
                continue
            x = x.loc[valid]
            y = y.loc[valid]
            scale = x.std(ddof=0).replace(0, np.nan)
            keep_cols = scale[scale.notna() & (scale > 1e-12)].index.tolist()
            if not keep_cols:
                out.loc[x.index, resid_col] = y - y.mean()
                continue
            x = (x[keep_cols] - x[keep_cols].mean()) / scale[keep_cols]
            x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            design = np.column_stack([np.ones(len(x)), x.to_numpy(dtype=float)])
            y_arr = y.to_numpy(dtype=float)
            xtx = design.T @ design
            ridge = np.eye(xtx.shape[0]) * max(len(x), 1) * 1e-6
            ridge[0, 0] = 0.0
            coef = np.linalg.solve(xtx + ridge, design.T @ y_arr)
            out.loc[x.index, resid_col] = y_arr - design @ coef
    return out


def _summarize_strategy_monthly(monthly: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    rows = []
    counts_lookup = {}
    if not counts.empty:
        counts_lookup = counts.groupby("strategy")["n_names"].mean().to_dict()
    for (strategy, eta), group in monthly.groupby(["strategy", "eta"]):
        gross = _return_summary(group["gross_return"])
        net = _return_summary(group["net_return"])
        rows.append(
            {
                "strategy": strategy,
                "eta": float(eta),
                "n_months": int(group["month_end"].nunique()),
                "avg_n_names": float(counts_lookup.get(strategy, np.nan)),
                "gross_ann_mean": gross["ann_mean"],
                "gross_sharpe": gross["sharpe"],
                "net_ann_mean": net["ann_mean"],
                "net_sharpe": net["sharpe"],
                "avg_turnover": float(group["turnover"].mean()),
                "avg_cost": float((group["half_spread_cost"] + group["impact_cost"]).mean()),
                "p95_participation": float(group["p95_participation"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def _strategy_strengthening_alphas(monthly: pd.DataFrame, ff5: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    factors = ff5.copy()
    specs = {
        "CAPM": ["mktrf"],
        "FF3": ["mktrf", "smb", "hml"],
        "FF5": ["mktrf", "smb", "hml", "rmw", "cma"],
        "FF5_UMD": ["mktrf", "smb", "hml", "rmw", "cma", "umd"],
    }
    rows = []
    base = monthly[monthly["eta"].eq(0.10)].merge(factors, on="month_end", how="inner")
    for strategy, group in base.groupby("strategy"):
        for ret_col in ["gross_return", "net_return"]:
            for model, cols in specs.items():
                available = [c for c in cols if c in group and group[c].notna().any()]
                fit = _hac_alpha(group.dropna(subset=[ret_col] + available), ret_col, available)
                rows.append({"strategy": strategy, "return": ret_col, "model": model, **fit})
    return pd.DataFrame(rows)


def _strategy_decile_monotonicity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cases = {
        "liquid": data["mcap_rank"] >= 0.20,
        "large_liquid": data["mcap_rank"] >= 0.50,
        "high_vix": (data["mcap_rank"] >= 0.20) & data["high_vix_state"],
        "low_vix": (data["mcap_rank"] >= 0.20) & ~data["high_vix_state"],
    }
    signals = {
        "ml_score": "prediction",
        "tail_slope": "score_tail_slope",
        "tail_composite": "score_tail_composite",
        "tail_reversal": "score_tail_reversal",
    }
    for case, mask in cases.items():
        for signal, col in signals.items():
            if col not in data:
                continue
            decile_rows = []
            subset = data.loc[mask].dropna(subset=[col, TARGET]).copy()
            for date, group in subset.groupby("month_end"):
                if group[col].nunique() < 10:
                    continue
                group = group.copy()
                group["decile"] = pd.qcut(group[col], 10, labels=False, duplicates="drop") + 1
                means = {}
                for decile, decile_group in group.groupby("decile"):
                    weights = decile_group["mcap"].clip(lower=0).fillna(0)
                    if weights.sum() > 0:
                        ret = np.average(decile_group[TARGET], weights=weights)
                    else:
                        ret = decile_group[TARGET].mean()
                    means[int(decile)] = float(ret)
                for decile, ret in means.items():
                    decile_rows.append({"month_end": date, "decile": int(decile), "ret": float(ret)})
            deciles = pd.DataFrame(decile_rows)
            if deciles.empty:
                continue
            annual = deciles.groupby("decile")["ret"].mean() * 12
            for decile, ret in annual.items():
                rows.append(
                    {
                        "case": case,
                        "signal": signal,
                        "decile": int(decile),
                        "ann_mean": float(ret),
                        "spread_d10_d1": float(annual.get(10, np.nan) - annual.get(1, np.nan)),
                        "decile_spearman": float(pd.Series(annual.index).corr(pd.Series(annual.values), method="spearman")),
                    }
                )
    return pd.DataFrame(rows)


def _final_reversal_audit(
    predictions: pd.DataFrame,
    ff5: pd.DataFrame,
    paths: CloseoutPaths,
) -> tuple[Path, dict[str, object]]:
    summary_path = paths.outdir / "final_reversal_audit.csv"
    monthly_path = paths.outdir / "final_reversal_monthly.csv"
    alpha_path = paths.outdir / "final_reversal_alphas.csv"
    monotonic_path = paths.outdir / "final_reversal_monotonicity.csv"
    required_strategy = "resid_tail_slope_reversal_skip1_quarterly_smooth3"
    if summary_path.exists() and alpha_path.exists() and monotonic_path.exists():
        summary = pd.read_csv(summary_path)
        if required_strategy in set(summary.get("strategy", pd.Series(dtype=str)).dropna()):
            _plot_final_reversal_audit(summary, paths.figdir)
            return summary_path, _final_reversal_diagnostics(summary)

    data = predictions.copy()
    data["month_end"] = pd.to_datetime(data["month_end"]) + pd.offsets.MonthEnd(0)
    data = data.dropna(subset=[TARGET, "prediction", "mcap", "prc", "adv_dollar"]).copy()
    data = data[(data["prc"].abs() >= 5) & (data["adv_dollar"].fillna(0) >= 5_000_000)]
    data["mcap_rank"] = data.groupby("month_end")["mcap"].rank(pct=True)
    data = _attach_state_regimes(data)
    data = _add_tail_strategy_scores(data)
    data = _add_residualized_tail_scores(data)
    data = _add_smoothed_scores(
        data,
        [
            "score_tail_slope_reversal",
            "score_tail_reversal",
            "score_tail_slope_reversal_resid",
            "score_tail_reversal_resid",
        ],
        window=3,
    )
    liquid = data["mcap_rank"] >= 0.20
    variants = [
        (
            "raw_tail_slope_reversal_quarterly_smooth3",
            "score_tail_slope_reversal_smooth3",
            liquid,
            3,
            0,
        ),
        (
            "raw_tail_slope_reversal_skip1_quarterly_smooth3",
            "score_tail_slope_reversal_smooth3",
            liquid,
            3,
            1,
        ),
        (
            "resid_tail_slope_reversal_quarterly_smooth3",
            "score_tail_slope_reversal_resid_smooth3",
            liquid,
            3,
            0,
        ),
        (
            "resid_tail_slope_reversal_skip1_quarterly_smooth3",
            "score_tail_slope_reversal_resid_smooth3",
            liquid,
            3,
            1,
        ),
        (
            "raw_tail_composite_reversal_quarterly_smooth3",
            "score_tail_reversal_smooth3",
            liquid,
            3,
            0,
        ),
        (
            "resid_tail_composite_reversal_quarterly_smooth3",
            "score_tail_reversal_resid_smooth3",
            liquid,
            3,
            0,
        ),
    ]

    monthly_frames = []
    count_rows = []
    for name, score, mask, rebalance_every, delay_months in variants:
        if score not in data:
            continue
        subset = data.loc[mask].dropna(subset=[score]).copy()
        if subset.empty or subset.groupby("month_end")[score].nunique().max() < 10:
            continue
        subset["prediction"] = subset[score]
        weights = _make_rebalanced_weights(subset, "sector_value", rebalance_every=rebalance_every)
        weights = _delay_weights(weights, delay_months)
        if weights.empty:
            continue
        monthly = _portfolio_from_weights(subset, weights, "sector_value")
        monthly["strategy"] = name
        monthly["score"] = score
        monthly["delay_months"] = delay_months
        monthly["rebalance_every"] = rebalance_every
        monthly_frames.append(monthly)
        counts = weights.groupby("month_end")["permno"].nunique().reset_index(name="n_names")
        counts["strategy"] = name
        count_rows.append(counts)

    monthly_all = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    monthly_all.to_csv(monthly_path, index=False)
    counts_all = pd.concat(count_rows, ignore_index=True) if count_rows else pd.DataFrame()
    summary = _summarize_strategy_monthly(monthly_all, counts_all)
    alphas = _strategy_strengthening_alphas(monthly_all, ff5)
    monotonic = _final_reversal_monotonicity(data)
    summary = _merge_final_alpha_and_monotonicity(summary, alphas, monotonic)
    summary.to_csv(summary_path, index=False)
    alphas.to_csv(alpha_path, index=False)
    monotonic.to_csv(monotonic_path, index=False)
    _plot_final_reversal_audit(summary, paths.figdir)
    return summary_path, _final_reversal_diagnostics(summary)


def _final_reversal_monotonicity(data: pd.DataFrame) -> pd.DataFrame:
    signals = {
        "raw_tail_slope_reversal_quarterly_smooth3": "score_tail_slope_reversal_smooth3",
        "resid_tail_slope_reversal_quarterly_smooth3": "score_tail_slope_reversal_resid_smooth3",
        "raw_tail_composite_reversal_quarterly_smooth3": "score_tail_reversal_smooth3",
        "resid_tail_composite_reversal_quarterly_smooth3": "score_tail_reversal_resid_smooth3",
    }
    rows = []
    subset_base = data[data["mcap_rank"] >= 0.20].copy()
    for strategy, col in signals.items():
        if col not in subset_base:
            continue
        decile_rows = []
        subset = subset_base.dropna(subset=[col, TARGET]).copy()
        for date, group in subset.groupby("month_end"):
            if group[col].nunique() < 10:
                continue
            group = group.copy()
            group["decile"] = pd.qcut(group[col], 10, labels=False, duplicates="drop") + 1
            for decile, decile_group in group.groupby("decile"):
                weights = decile_group["mcap"].clip(lower=0).fillna(0)
                ret = (
                    np.average(decile_group[TARGET], weights=weights)
                    if weights.sum() > 0
                    else decile_group[TARGET].mean()
                )
                decile_rows.append({"month_end": date, "decile": int(decile), "ret": float(ret)})
        deciles = pd.DataFrame(decile_rows)
        if deciles.empty:
            continue
        annual = deciles.groupby("decile")["ret"].mean() * 12
        rows.append(
            {
                "strategy": strategy,
                "spread_d10_d1": float(annual.get(10, np.nan) - annual.get(1, np.nan)),
                "decile_spearman": float(
                    pd.Series(annual.index).corr(pd.Series(annual.values), method="spearman")
                ),
            }
        )
    return pd.DataFrame(rows)


def _merge_final_alpha_and_monotonicity(
    summary: pd.DataFrame,
    alphas: pd.DataFrame,
    monotonic: pd.DataFrame,
) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary[summary["eta"].eq(0.10)].copy()
    if not alphas.empty:
        alpha = alphas[(alphas["return"] == "net_return") & (alphas["model"] == "FF5_UMD")][
            ["strategy", "alpha", "alpha_t", "n"]
        ].rename(columns={"alpha": "ff5_umd_net_alpha", "alpha_t": "ff5_umd_net_alpha_t", "n": "alpha_n"})
        out = out.merge(alpha, on="strategy", how="left")
    if not monotonic.empty:
        out = out.merge(monotonic, on="strategy", how="left")
    out["holdout_start"] = out["n_months"].map(lambda _: "2013-01")
    out["holdout_end"] = out["n_months"].map(lambda _: "2025-12")
    return out.sort_values("net_sharpe", ascending=False)


def _final_reversal_diagnostics(summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        return {"status": "empty"}
    best = summary.sort_values("net_sharpe", ascending=False).iloc[0]
    diag: dict[str, object] = {
        "rows": int(len(summary)),
        "best": {
            "strategy": str(best["strategy"]),
            "net_ann_mean": float(best["net_ann_mean"]),
            "net_sharpe": float(best["net_sharpe"]),
            "avg_turnover": float(best["avg_turnover"]),
            "ff5_umd_net_alpha": float(best.get("ff5_umd_net_alpha", np.nan)),
            "ff5_umd_net_alpha_t": float(best.get("ff5_umd_net_alpha_t", np.nan)),
        },
    }
    for strategy in [
        "raw_tail_slope_reversal_quarterly_smooth3",
        "raw_tail_slope_reversal_skip1_quarterly_smooth3",
        "resid_tail_slope_reversal_quarterly_smooth3",
        "resid_tail_slope_reversal_skip1_quarterly_smooth3",
    ]:
        row = summary[summary["strategy"].eq(strategy)]
        if not row.empty:
            item = row.iloc[0]
            diag[strategy] = {
                "net_ann_mean": float(item["net_ann_mean"]),
                "net_sharpe": float(item["net_sharpe"]),
                "avg_turnover": float(item["avg_turnover"]),
                "ff5_umd_net_alpha": float(item.get("ff5_umd_net_alpha", np.nan)),
                "ff5_umd_net_alpha_t": float(item.get("ff5_umd_net_alpha_t", np.nan)),
                "spread_d10_d1": float(item.get("spread_d10_d1", np.nan)),
            }
    return diag


def _plot_final_reversal_audit(summary: pd.DataFrame, figdir: Path) -> None:
    if summary.empty:
        return
    set_style()
    data = summary.sort_values("net_sharpe").copy()
    data["label"] = data["strategy"].map(_final_reversal_label)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
    sns.barplot(data=data, y="label", x="net_sharpe", ax=ax, color="#59a14f")
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_title("Final Reversal Audit")
    ax.set_xlabel("Net Sharpe, eta=0.10")
    ax.set_ylabel("")
    fig.savefig(figdir / "final_reversal_audit.png", bbox_inches="tight")
    fig.savefig(figdir / "final_reversal_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def _final_reversal_label(strategy: str) -> str:
    labels = {
        "raw_tail_slope_reversal_quarterly_smooth3": "Raw tail slope, slow",
        "raw_tail_slope_reversal_skip1_quarterly_smooth3": "Raw tail slope, skip",
        "resid_tail_slope_reversal_quarterly_smooth3": "Residual tail slope, slow",
        "resid_tail_slope_reversal_skip1_quarterly_smooth3": "Residual tail slope, skip",
        "raw_tail_composite_reversal_quarterly_smooth3": "Raw composite, slow",
        "resid_tail_composite_reversal_quarterly_smooth3": "Residual composite, slow",
    }
    return labels.get(strategy, strategy.replace("_", " "))


def _strategy_strengthening_diagnostics(summary: pd.DataFrame) -> dict[str, object]:
    if summary.empty:
        return {"status": "empty"}
    eta = summary[summary["eta"].eq(0.10)].copy()
    if eta.empty:
        eta = summary.copy()
    best = eta.sort_values("net_sharpe", ascending=False).iloc[0]
    diag: dict[str, object] = {
        "rows": int(len(summary)),
        "strategies": sorted(summary["strategy"].dropna().unique().tolist()),
        "best_eta010": {
            "strategy": str(best["strategy"]),
            "net_ann_mean": float(best["net_ann_mean"]),
            "net_sharpe": float(best["net_sharpe"]),
            "avg_turnover": float(best["avg_turnover"]),
        },
    }
    for strategy in [
        "ml_score_liquid",
        "ml_score_liquid_quarterly_smooth3",
        "tail_composite_liquid",
        "tail_composite_high_vix",
        "tail_reversal_liquid",
        "tail_reversal_liquid_quarterly_smooth3",
    ]:
        row = eta[eta["strategy"].eq(strategy)]
        if not row.empty:
            item = row.iloc[0]
            diag[strategy] = {
                "net_ann_mean": float(item["net_ann_mean"]),
                "net_sharpe": float(item["net_sharpe"]),
                "avg_turnover": float(item["avg_turnover"]),
            }
    return diag


def _plot_strategy_strengthening(summary: pd.DataFrame, figdir: Path) -> None:
    if summary.empty:
        return
    set_style()
    data = summary[summary["eta"].eq(0.10)].sort_values("net_sharpe").copy()
    if data.empty:
        return
    data["label"] = data["strategy"].map(_strategy_plot_label)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    sns.barplot(data=data, y="label", x="net_sharpe", ax=ax, color="#4c78a8")
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_title("Cost-Aware Strategy Strengthening")
    ax.set_xlabel("Net Sharpe, eta=0.10")
    ax.set_ylabel("")
    fig.savefig(figdir / "strategy_strengthening.png", bbox_inches="tight")
    fig.savefig(figdir / "strategy_strengthening.pdf", bbox_inches="tight")
    plt.close(fig)


def _strategy_plot_label(strategy: str) -> str:
    labels = {
        "ml_score_liquid": "ML score, monthly",
        "ml_score_liquid_quarterly_smooth3": "ML score, slow",
        "tail_slope_liquid": "Tail slope, monthly",
        "tail_slope_liquid_quarterly_smooth3": "Tail slope, slow",
        "tail_slope_reversal_liquid": "Tail slope reversal, monthly",
        "tail_slope_reversal_liquid_quarterly_smooth3": "Tail slope reversal, slow",
        "tail_composite_liquid": "Tail composite, monthly",
        "tail_composite_liquid_quarterly_smooth3": "Tail composite, slow",
        "tail_composite_large_liquid": "Tail composite, large",
        "tail_composite_large_liquid_quarterly_smooth3": "Tail composite, large slow",
        "tail_composite_high_vix": "Tail composite, high VIX",
        "tail_composite_low_vix": "Tail composite, low VIX",
        "tail_composite_high_drawdown": "Tail composite, drawdown",
        "tail_composite_constrained": "Tail composite, constrained",
        "tail_composite_aggressive_investment": "Tail composite, aggressive inv.",
        "tail_reversal_liquid": "Tail reversal, monthly",
        "tail_reversal_liquid_quarterly_smooth3": "Tail reversal, slow",
        "tail_reversal_large_liquid_quarterly_smooth3": "Tail reversal, large slow",
        "tail_reversal_high_vix": "Tail reversal, high VIX",
    }
    return labels.get(strategy, strategy.replace("_", " "))


def _plot_strategy_monotonicity(monotonic: pd.DataFrame, figdir: Path) -> None:
    if monotonic.empty:
        return
    set_style()
    data = monotonic[
        (monotonic["signal"] == "tail_composite") & (monotonic["case"].isin(["liquid", "high_vix", "low_vix"]))
    ].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    sns.lineplot(data=data, x="decile", y="ann_mean", hue="case", marker="o", ax=ax)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_title("Tail Composite Decile Monotonicity")
    ax.set_xlabel("Signal decile")
    ax.set_ylabel("Annualized next-month excess return")
    fig.savefig(figdir / "strategy_decile_monotonicity.png", bbox_inches="tight")
    fig.savefig(figdir / "strategy_decile_monotonicity.pdf", bbox_inches="tight")
    plt.close(fig)


def _quick_decile_returns(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date, group in data.groupby("month_end"):
        if group["prediction"].nunique() < 10:
            continue
        group = group.copy()
        group["bucket"] = pd.qcut(group["prediction"], 10, labels=False, duplicates="drop")
        high = group[group["bucket"] == group["bucket"].max()]
        low = group[group["bucket"] == group["bucket"].min()]
        rows.append({"month_end": date, "long_short": high[TARGET].mean() - low[TARGET].mean()})
    return pd.DataFrame(rows)


def _return_summary(ret: pd.Series) -> dict[str, float]:
    ret = ret.dropna()
    if ret.empty:
        return {"n": 0.0, "ann_mean": np.nan, "sharpe": np.nan}
    ann_mean = ret.mean() * 12
    ann_vol = ret.std() * np.sqrt(12)
    return {"n": float(len(ret)), "ann_mean": float(ann_mean), "sharpe": float(ann_mean / ann_vol) if ann_vol else np.nan}


def _plot_robustness(out: pd.DataFrame, figdir: Path) -> None:
    set_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=160)
    sns.pointplot(data=out.sort_values("ann_mean"), y="case", x="ann_mean", ax=ax, color="#4c78a8")
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_title("Robustness: Annualized Long-Short Mean")
    ax.set_xlabel("Annualized mean")
    ax.set_ylabel("")
    fig.savefig(figdir / "robustness_forest.png", bbox_inches="tight")
    fig.savefig(figdir / "robustness_forest.pdf", bbox_inches="tight")
    plt.close(fig)


def _raw_chain_validation(paths: CloseoutPaths) -> Path:
    summary_path = paths.outdir / "raw_chain_validation_summary.json"
    if summary_path.exists():
        return summary_path
    raw_path = paths.rawdir / "raw_chain_validation_sample.parquet"
    if not raw_path.exists():
        return summary_path
    raw = pd.read_parquet(raw_path)
    if raw.empty:
        return summary_path
    raw["month_end"] = pd.to_datetime(raw["date"]) + pd.offsets.MonthEnd(0)
    raw["delta_scaled"] = raw["delta"].astype(float)
    raw.loc[raw["delta_scaled"].abs() <= 1.5, "delta_scaled"] *= 100
    raw_features = []
    for (secid, month), group in raw.groupby(["secid", "month_end"]):
        row = {"secid": secid, "month_end": month}
        for name, cp, target_delta in [
            ("raw_put25_iv", "P", -25),
            ("raw_put10_iv", "P", -10),
            ("raw_call25_iv", "C", 25),
            ("raw_atm_call_iv", "C", 50),
        ]:
            sub = group[group["cp_flag"].eq(cp)].copy()
            if sub.empty:
                row[name] = np.nan
            else:
                idx = (sub["delta_scaled"] - target_delta).abs().idxmin()
                row[name] = sub.loc[idx, "impl_volatility"]
        raw_features.append(row)
    rawf = pd.DataFrame(raw_features)
    rawf["raw_skew_25d"] = rawf["raw_put25_iv"] - rawf["raw_call25_iv"]
    rawf["raw_tail_slope"] = rawf["raw_put10_iv"] - rawf["raw_put25_iv"]
    surf_frames = []
    for year in sorted(rawf["month_end"].dt.year.unique()):
        path = paths.rawdir / "option_surface" / f"year={year}.parquet"
        if path.exists():
            surf_frames.append(pd.read_parquet(path))
    if not surf_frames:
        return summary_path
    surf = pd.concat(surf_frames, ignore_index=True)
    from ctrsdf.features.options import build_option_features

    surff = build_option_features(surf)
    comp = rawf.merge(surff, on=["secid", "month_end"], how="inner")
    summary = {
        "n_pairs": int(len(comp)),
        "skew_correlation": float(comp["raw_skew_25d"].corr(comp["skew_25d"])),
        "atm_iv_correlation": float(comp["raw_atm_call_iv"].corr(comp["atm_iv_30"])),
    }
    comp.to_csv(paths.outdir / "raw_chain_validation.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    set_style()
    fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=160)
    sample = comp.dropna(subset=["raw_skew_25d", "skew_25d"]).sample(min(len(comp), 5000), random_state=20260527)
    ax.scatter(sample["raw_skew_25d"], sample["skew_25d"], s=4, alpha=0.25, color="#4c78a8")
    ax.set_title("Raw Chain vs Surface Skew")
    ax.set_xlabel("Raw chain 25-delta skew")
    ax.set_ylabel("Surface 25-delta skew")
    fig.savefig(paths.figdir / "raw_chain_validation.png", bbox_inches="tight")
    fig.savefig(paths.figdir / "raw_chain_validation.pdf", bbox_inches="tight")
    plt.close(fig)
    return summary_path


def _interpretability(panel: pd.DataFrame, predictions: pd.DataFrame, paths: CloseoutPaths) -> Path:
    if (paths.figdir / "shap_beeswarm.png").exists() and (paths.figdir / "ale_skew_vix.png").exists():
        return paths.figdir / "ale_skew_vix.png"
    features = _model_features(panel)
    train = panel[(panel["month_end"] < "2013-01-01") & panel[TARGET].notna()].copy()
    if len(train) > 160_000:
        train = train.sample(160_000, random_state=20260527)
    x = _numeric_matrix(train, features)
    med = x.median().fillna(0.0)
    x = x.fillna(med).fillna(0.0)
    y = train[TARGET].astype(float)
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="reg:squarederror",
            random_state=20260527,
            n_jobs=8,
        )
        model.fit(x, y)
        _plot_shap(model, x, paths.figdir)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("SHAP/XGBoost interpretability skipped: %s", type(exc).__name__)
    _plot_ale_like(predictions, paths.figdir)
    return paths.figdir / "ale_skew_vix.png"


def _plot_shap(model, x: pd.DataFrame, figdir: Path) -> None:
    try:
        import shap
    except ImportError:
        return
    sample = x.sample(min(len(x), 2500), random_state=20260527)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    plt.figure(figsize=(7.2, 5.0), dpi=160)
    shap.summary_plot(values, sample, show=False, max_display=20, plot_size=None)
    plt.title("SHAP Summary: Boosted Tree")
    plt.savefig(figdir / "shap_beeswarm.png", bbox_inches="tight")
    plt.savefig(figdir / "shap_beeswarm.pdf", bbox_inches="tight")
    plt.close()


def _plot_ale_like(predictions: pd.DataFrame, figdir: Path) -> None:
    set_style()
    for feature in ["skew_25d", "atm_iv_30", "leverage"]:
        if feature not in predictions:
            continue
        data = predictions.dropna(subset=[feature, "prediction"]).copy()
        if data.empty:
            continue
        data["bin"] = pd.qcut(data[feature], 20, duplicates="drop")
        shape = data.groupby("bin", observed=True).agg(x=(feature, "median"), y=("prediction", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=160)
        ax.plot(shape["x"], shape["y"], color="#4c78a8", lw=1.8)
        ax.set_title(f"Shape Diagnostic: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Mean rank signal")
        fig.savefig(figdir / f"ale_{feature}.png", bbox_inches="tight")
        fig.savefig(figdir / f"ale_{feature}.pdf", bbox_inches="tight")
        plt.close(fig)
    for left, right, name in [
        ("skew_25d", "vix_level", "ale_skew_vix"),
        ("skew_25d", "leverage", "ale_skew_leverage"),
    ]:
        if left not in predictions or right not in predictions:
            continue
        data = predictions.dropna(subset=[left, right, "prediction"]).copy()
        if data.empty:
            continue
        data["left_bin"] = pd.qcut(data[left], 10, labels=False, duplicates="drop")
        data["right_bin"] = pd.qcut(data[right], 10, labels=False, duplicates="drop")
        heat = data.pivot_table(index="right_bin", columns="left_bin", values="prediction", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(5.4, 4.2), dpi=160)
        sns.heatmap(heat, ax=ax, cmap="vlag", center=data["prediction"].mean(), cbar_kws={"label": "Mean rank signal"})
        ax.set_title(f"Interaction Shape: {left} x {right}")
        ax.set_xlabel(left)
        ax.set_ylabel(right)
        fig.savefig(figdir / f"{name}.png", bbox_inches="tight")
        fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)


def _results_inventory(config: ProjectConfig, paths: CloseoutPaths, diagnostics: dict[str, object]) -> Path:
    items = [
        ("Data schema discovery", "artifacts/manifests/schema_audit.json"),
        ("Restartable extraction manifests", "artifacts/manifests/full_extract.json"),
        ("Monthly feature-store manifest", "artifacts/manifests/full_feature_store.json"),
        ("Walk-forward model metrics", "artifacts/manifests/full_models.json"),
        ("Fama-MacBeth tests", "artifacts/closeout/fama_macbeth.csv"),
        ("Random Forest feature importance", "artifacts/closeout/random_forest_importance.csv"),
        ("Double ML option contribution", "artifacts/closeout/double_ml_option_contribution.csv"),
        ("Raw-chain option validation", "artifacts/closeout/raw_chain_validation_summary.json"),
        ("Benchmark reconciliation", "artifacts/closeout/benchmark_reconciliation_summary.json"),
        ("Transaction-cost grid", "artifacts/closeout/cost_turnover_backtests.csv"),
        ("Factor attribution", "artifacts/closeout/factor_alpha_spanning.csv"),
        ("Holdout robustness", "artifacts/closeout/robustness.csv"),
        ("Signal robustness", "artifacts/closeout/signal_robustness_full_sample.csv"),
        ("Strategy variants", "artifacts/closeout/strategy_strengthening_summary.csv"),
        ("Reversal validation", "artifacts/closeout/final_reversal_audit.csv"),
        ("Interpretability figures", "artifacts/figures/full/shap_beeswarm.png"),
        ("Reference SQL", "sql/wrds/"),
        ("Review notebooks", "notebooks/"),
    ]
    text = ["# Results Inventory", ""]
    text.append(f"Configured sample window: `{config.sample_start}` to `{config.sample_end}`.")
    text.append("")
    text.append("This inventory records the reproducible outputs generated by the surface-to-returns pipeline.")
    text.append("")
    for item, evidence in items:
        text.append(f"- **{item}**: `{evidence}`")
    text.append("")
    text.append("## Diagnostic Snapshot")
    text.append("")
    text.append("```json")
    text.append(json.dumps(diagnostics, indent=2, default=str))
    text.append("```")
    path = paths.outdir / "results_inventory.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def _annualized_sharpe(ret: pd.Series) -> float:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if ret.empty or ret.std() == 0:
        return np.nan
    return float(ret.mean() * 12 / (ret.std() * np.sqrt(12)))


def ff12_from_sic(sic: float | int | None) -> str:
    if pd.isna(sic):
        return "Other"
    sic = int(sic)
    if 100 <= sic <= 999 or 2000 <= sic <= 2399 or 2700 <= sic <= 2749 or 2770 <= sic <= 2799 or 3100 <= sic <= 3199 or 3940 <= sic <= 3989:
        return "Consumer NonDurables"
    if 2500 <= sic <= 2519 or 2590 <= sic <= 2599 or 3630 <= sic <= 3659 or 3710 <= sic <= 3711 or 3714 <= sic <= 3714 or 3716 <= sic <= 3716 or 3750 <= sic <= 3751 or 3792 <= sic <= 3792 or 3900 <= sic <= 3939 or 3990 <= sic <= 3999:
        return "Consumer Durables"
    if 2520 <= sic <= 2589 or 2600 <= sic <= 2699 or 2750 <= sic <= 2769 or 3000 <= sic <= 3099 or 3200 <= sic <= 3569 or 3580 <= sic <= 3629 or 3700 <= sic <= 3709 or 3712 <= sic <= 3713 or 3715 <= sic <= 3715 or 3717 <= sic <= 3749 or 3752 <= sic <= 3791 or 3793 <= sic <= 3799 or 3830 <= sic <= 3839 or 3860 <= sic <= 3899:
        return "Manufacturing"
    if 1200 <= sic <= 1399 or 2900 <= sic <= 2999:
        return "Energy"
    if 2800 <= sic <= 2829 or 2840 <= sic <= 2899:
        return "Chemicals"
    if 3570 <= sic <= 3579 or 3660 <= sic <= 3692 or 3694 <= sic <= 3699 or 3810 <= sic <= 3829 or 7370 <= sic <= 7379:
        return "Business Equipment"
    if 4800 <= sic <= 4899:
        return "Telecom"
    if 4900 <= sic <= 4949:
        return "Utilities"
    if 5000 <= sic <= 5999 or 7200 <= sic <= 7299 or 7600 <= sic <= 7699:
        return "Shops"
    if 2830 <= sic <= 2839 or 3693 <= sic <= 3693 or 3840 <= sic <= 3859 or 8000 <= sic <= 8099:
        return "Healthcare"
    if 6000 <= sic <= 6999:
        return "Finance"
    return "Other"
