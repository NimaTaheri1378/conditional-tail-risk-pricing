from __future__ import annotations

import wrds


TABLES = [
    ("crsp_a_stock", "stkdlysecuritydata"),
    ("crsp_a_stock", "stkmthsecuritydata"),
    ("crsp_a_stock", "stksecurityinfohist"),
    ("crsp_a_stock", "stkdelists"),
    ("crsp_a_ccm", "ccmxpf_linktable"),
    ("comp_na_daily_all", "funda"),
    ("comp_na_daily_all", "fundq"),
    ("optionm_all", "vsurfd1996"),
    ("optionm_all", "opprcd1996"),
    ("optionm_all", "secprd1996"),
    ("optionm_all", "secnmd"),
    ("wrdsapps_link_crsp_optionm", "opcrsphist"),
    ("cboe_all", "cboe"),
    ("ff_all", "factors_monthly"),
    ("ff_all", "fivefactors_monthly"),
    ("ff_all", "portfolios25"),
    ("frb_all", "rates_daily"),
    ("frb_all", "rates_monthly"),
]


def main() -> None:
    db = wrds.Connection()
    try:
        for library, table in TABLES:
            print(f"\n## {library}.{table}")
            try:
                desc = db.describe_table(library=library, table=table)
            except Exception as exc:  # noqa: BLE001
                print(f"ERR {type(exc).__name__}: {str(exc)[:160]}")
                continue
            print(desc[["name", "type"]].to_string(index=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
