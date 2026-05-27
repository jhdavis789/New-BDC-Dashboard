#!/usr/bin/env python3
"""Run a V2 adapter for a single (ticker, period) and write output.

Usage:
    python run_adapter.py ARCC 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from adapters import get_adapter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("period")
    ap.add_argument("--html", default=None, help="Override HTML path")
    ap.add_argument("--accession", default=None)
    ap.add_argument(
        "--out",
        default=None,
        help="Override output path (default: PARSER_V2/out/v2_raw/<TICKER>/<TICKER>_<period>.json)",
    )
    args = ap.parse_args()

    Adapter = get_adapter(args.ticker)
    html = args.html or os.path.join(
        HERE, "ground_truth", args.ticker, args.period, f"{args.ticker.lower()}-{args.period.replace('-', '')}.htm"
    )
    if not os.path.exists(html):
        # Fall back to the default cache location
        html = os.path.join(
            os.path.dirname(HERE), "BDC", "soi_cache", args.ticker,
            f"{args.ticker}_{args.period}_10-K.html"
        )
    if not os.path.exists(html):
        print(f"ERROR: could not find HTML at {html}", file=sys.stderr)
        return 2

    parser = Adapter(html_path=html, period_end=args.period, accession=args.accession)
    parser.load()
    invs = parser.extract()

    out_path = args.out or os.path.join(
        HERE, "out", "v2_raw", args.ticker, f"{args.ticker}_{args.period}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Schedule of Unfunded Commitments — separate top-level array.
    unfunded = getattr(parser, "unfunded_commitments", None) or []
    unfunded_total = getattr(parser, "unfunded_disclosed_total", None)
    result = {
        "ticker": args.ticker,
        "period_end": args.period,
        "accession": args.accession,
        "unit_multiplier": parser.unit_multiplier,
        "n_investments": len(invs),
        "investments": [asdict(i) for i in invs],
        "section_totals": [asdict(t) for t in parser.section_totals],
        "footnote_legend": parser.footnote_legend,
        "unfunded_commitments": [asdict(u) for u in unfunded],
        "unfunded_disclosed_total": unfunded_total,
    }
    # Some adapters (e.g., ADS) separate cash-equivalent positions out of the
    # investment-portfolio list because XBRL's grand-total fact excludes them.
    # If the adapter exposes a `cash_equivalents` attribute, serialize it too
    # so reconciliation tests and audit harnesses can still see the rows.
    cash_equivs = getattr(parser, "cash_equivalents", None)
    if cash_equivs:
        result["cash_equivalents"] = [asdict(i) for i in cash_equivs]
    # OSCF and (in the future) other BDCs split derivative-schedule
    # tables (FX forwards, interest-rate swaps) into a structured
    # `derivatives` dict keyed by instrument type.  Mirrors the
    # `cash_equivalents` opt-in path above.
    derivatives = getattr(parser, "derivatives", None)
    if derivatives:
        result["derivatives"] = derivatives
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    fv = sum((i.fair_value or 0) for i in invs)
    cost = sum((i.amortized_cost or 0) for i in invs)
    print(
        f"{args.ticker} {args.period}: "
        f"{len(invs)} rows, FV=${fv:,.0f}, Cost=${cost:,.0f}, "
        f"footnotes in legend={len(parser.footnote_legend)}, "
        f"section_totals={len(parser.section_totals)}"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
