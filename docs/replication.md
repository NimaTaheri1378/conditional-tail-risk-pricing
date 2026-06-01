# Reproducibility

1. Confirm WRDS access through `~/.pgpass` without printing credentials.
2. Run `ctrsdf schema-audit` to resolve physical WRDS table names.
3. Run `ctrsdf smoke` on the configured smoke window.
4. Inspect manifests and logs.
5. Run `ctrsdf full` only after smoke outputs pass validation.
6. Run `ctrsdf results` to refresh benchmark reconciliation, Fama-MacBeth, Random Forest, Double ML, cost/turnover, factor attribution, robustness, raw-chain validation, reversal validation, and interpretability artifacts from the cached full run.
7. Run `ctrsdf figures` to refresh the README/result figures.
8. Run `ctrsdf secret-audit` and confirm licensed row-level panels, protected caches, local credentials, and run logs are excluded from version control.
