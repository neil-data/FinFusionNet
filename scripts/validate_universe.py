"""FinFusionNet CLI: Validate Point-in-Time Universe and Rebalances."""

import pandas as pd
from finfusionnet.data.trading_calendar import get_trading_calendar
from finfusionnet.data.universe import PointInTimeUniverse, REBALANCES


def validate_universe():
    print("==========================================================")
    print("FinFusionNet: Point-in-Time NIFTY 50 Universe Validation")
    print("==========================================================")

    u = PointInTimeUniverse()
    print(f"Total unique tickers across 2020-2026: {len(u.all_tickers)}")
    print(f"Number of scheduled index rebalances: {len(REBALANCES)}")

    # Check that for every trading date from 2021-01-01 to 2026-09-30,
    # the constituent count is EXACTLY 50!
    cal = get_trading_calendar("2021-01-01", "2026-09-30")
    print(f"Auditing constituent count across all {len(cal)} trading sessions...")

    violations = []
    for d in cal:
        members = u.get_constituents_on_date(d)
        if len(members) != 50:
            violations.append((d.strftime("%Y-%m-%d"), len(members)))

    if violations:
        print(f"FAIL: Found {len(violations)} trading dates with non-50 constituent count!")
        for dt, count in violations[:5]:
            print(f"  {dt}: {count} members")
        raise AssertionError("Constituent count violation")
    else:
        print(f"SUCCESS: Exactly 50 constituents confirmed on all {len(cal)} trading sessions.")

    # Print rebalance schedule
    print("\nRebalance Schedule:")
    for r in REBALANCES:
        print(f"  {r['effective_date']}: +{r['in']} | -{r['out']}")

    print("\nUniverse metadata validation passed.")


if __name__ == "__main__":
    validate_universe()
