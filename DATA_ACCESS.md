# Data Access

This repository contains code, configurations, synthetic fixtures, aggregate result tables, and rendered research figures.

Raw WRDS, CRSP, Compustat, OptionMetrics, Cboe, Fama-French-on-WRDS, Treasury, and other vendor data are not included. Users must run the extract pipeline under their own authorized data licenses.

Credentials and API keys must be supplied through secure local mechanisms such as `~/.pgpass`, environment variables, or cluster secret stores. They must not be written into tracked files, logs, notebooks, manifests, or GitHub releases.

Before publishing or rerunning the project in a new environment, run:

```bash
python -m ctrsdf.utils.secret_audit --root .
python -m pytest
```
