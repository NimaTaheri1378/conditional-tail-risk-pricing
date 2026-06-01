from __future__ import annotations

import wrds


KEYWORDS = {
    "optionm_all": ["surf", "vol", "vsurf", "std", "impl", "sec", "opprc", "zero"],
    "wrdsapps_link_crsp_optionm": ["option", "crsp", "link", "secid"],
    "cboe_all": ["vix", "vol", "skew"],
    "ff_all": ["25", "port", "factor", "monthly", "size", "book"],
    "frb_all": ["rate", "yield", "treas", "dgs", "fed"],
}


def main() -> None:
    db = wrds.Connection()
    try:
        for lib, keys in KEYWORDS.items():
            print(f"\n## {lib}")
            try:
                tables = db.list_tables(library=lib)
            except Exception as exc:  # noqa: BLE001
                print(f"ERR {type(exc).__name__}: {str(exc)[:120]}")
                continue
            if lib == "cboe_all":
                print("ALL:", ", ".join(tables))
            for table in tables:
                lower = table.lower()
                if any(key in lower for key in keys):
                    print(table)
    finally:
        db.close()


if __name__ == "__main__":
    main()
