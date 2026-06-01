from __future__ import annotations

import wrds


def main() -> None:
    db = wrds.Connection()
    try:
        query = """
            select days, delta, cp_flag, count(*) as n
            from optionm_all.vsurfd2020
            where date between '2020-01-02' and '2020-01-10'
            group by days, delta, cp_flag
            order by days, delta, cp_flag
        """
        print(db.raw_sql(query).to_string(index=False))
    finally:
        db.close()


if __name__ == "__main__":
    main()
