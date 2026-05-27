# Data schema — BDC SOI parser & dashboard

This documents the two JSON shapes that sit underneath the BDC dashboard:

1. **Parser output** — one file per (BDC, period), produced by `run_adapter.py`.
2. **Dashboard data file** — the single aggregated JSON the dashboard reads.

---

## 1. Parser output (`<TICKER>_<period>.json`)

One file per BDC filing. Top-level object:

| Key | Type | Meaning |
|---|---|---|
| `ticker` | str | BDC ticker, e.g. `"ARCC"`. |
| `period_end` | str | Fiscal period end, ISO date, e.g. `"2025-12-31"`. |
| `accession` | str\|null | SEC accession number of the source 10-K/10-Q. |
| `unit_multiplier` | float | Scale factor applied to reported dollar columns (filings report in thousands or millions). All dollar fields below are already scaled to **whole dollars**. |
| `n_investments` | int | Count of rows in `investments`. |
| `investments` | array | Schedule-of-Investments rows (see below). |
| `section_totals` | array | Disclosed subtotal/total rows captured verbatim for reconciliation (`label`, `amortized_cost`, `fair_value`, `pct_net_assets`). |
| `footnote_legend` | object | Footnote marker → disclosed footnote text. |
| `unfunded_commitments` | array | Schedule of unfunded commitments (separate disclosure table). |
| `unfunded_disclosed_total` | float\|null | Filer-disclosed total unfunded commitment, for reconciliation. |
| `cash_equivalents` | array | Money-market / cash-equivalent rows (excluded from portfolio totals by convention). |
| `derivatives` | object | Forward / swap positions where disclosed. |

### `investments[]` row

Every row is one investment tranche. Dollar fields are whole dollars; rates are percent (e.g. `7.25` = 7.25%); spreads are basis points.

| Field | Type | Notes |
|---|---|---|
| `row_id` | str | Stable hash-based id for the row. |
| `ticker`, `period_end`, `accession` | str | Provenance, copied from the file header. |
| `parent_entity`, `parent_row_id`, `link_id` | str\|null | SPV / funded↔unfunded sibling linkage. |
| `company_name_raw` | str | Name exactly as printed. |
| `company_name` | str | Footnote- and address-stripped. |
| `canonical_borrower_name` | str\|null | Cross-filing canonical issuer name. |
| `industry` | str\|null | SOI-text industry/sector label. |
| `category` | str\|null | Affiliation bucket (non-controlled/non-affiliated, etc.). |
| `business_description` | str\|null | Where the filer prints one. |
| `investment_type_raw` | str\|null | Lien/type text as printed. |
| `investment_type` | str\|null | Normalized: `1L`, `2L`, `Mezz`, `Unsec`, `Equity`, `Preferred`, `Warrant`, `JV`, `Revolver`, `DDT`. |
| `investment_category` | str\|null | `Debt`, `Equity`, `Unfunded Commitment`, `Derivative`. |
| `reference_rate` | str\|null | `SOFR`, `EURIBOR`, `Prime`, `Fixed`. |
| `reference_rate_text_raw` | str\|null | Reference-rate text as printed. |
| `spread_bps` | float\|null | Credit spread over the reference rate, in bps. |
| `spread_text_raw` | str\|null | Spread text as printed. |
| `cash_rate_pct` | float\|null | All-in cash coupon. |
| `pik_rate_pct`, `pik_rate_max_pct`, `pik_type` | float/str\|null | PIK coupon, cap, and `toggle`/`mandatory`/`accrued`. |
| `floor_pct` | float\|null | Rate floor. |
| `maturity_date`, `acquisition_date` | str\|null | ISO dates. |
| `par_principal` | float\|null | Par / principal (or shares-equivalent for equity). |
| `amortized_cost` | float\|null | Amortized cost. |
| `fair_value` | float\|null | Fair value. |
| `shares` | float\|null | Share/unit count for equity. |
| `pct_net_assets` | float\|null | Disclosed % of net assets. |
| `unfunded_amount` | float\|null | Non-null on unfunded sibling rows. |
| `currency`, `country`, `geographic_region` | str | `USD`/ISO currency; ISO country; `Americas`/`EMEA`/`APAC`/`Other`. |
| `footnote_markers` | array | Footnote markers attached to the row. |
| `footnote_meanings` | object | Marker → resolved meaning. |
| `footnote_flags` | array | Controlled-vocabulary flags derived from footnotes. |
| `is_non_accrual` | bool | Non-accrual status. |
| `is_non_qualifying_asset` / `is_qualifying_asset` | bool | 1940-Act 30% basket classification. |
| `is_unfunded_commitment` | bool | Row represents an unfunded commitment. |
| `is_controlled_affiliate` / `is_non_controlled_affiliate` | bool | Affiliation. |
| `is_pledged`, `pledged_facilities` | bool/array | Pledged to a named facility. |
| `is_cash_equivalent` | bool | Cash-equivalent row. |
| `is_level_3` | bool | Fair-value hierarchy Level 3. |
| `_source_table_index`, `_source_row_index` | int\|null | Pointer back to the source HTML table/row (audit trail). |

---

## 2. Dashboard data file

A single aggregated JSON built from the per-BDC parser outputs plus market data. Top-level keys:

| Key | Type | Meaning |
|---|---|---|
| `tickers` | array | BDCs included. |
| `sectors`, `seniority_types` | array | Axis label vocabularies. |
| `portfolios` | object | Per-BDC rollups (see below). |
| `positions` | object | Per-BDC position-level rows used in drill-downs. |
| `companies` | object | Per-borrower aggregation across BDCs (overlap analysis). |
| `capital_structure` | object | Per-BDC debt stack / leverage. |
| `liquidity` | object | Per-BDC liquidity (cash, undrawn revolver, unfunded). |
| `cash_flow_data` | object | Per-BDC cash-flow inputs. |
| `pik_summary`, `non_accrual_summary` | object | Credit-quality rollups. |
| `aggregate` | object | Universe-wide totals. |
| `historical` | object | Per-BDC time series. |
| `rates_curve`, `bond_market` | object | Reference-rate proxies and traded bond marks. |
| `audit` | object | Reconciliation metadata. |

### `portfolios[ticker]`

| Field | Meaning |
|---|---|
| `total_fv`, `total_cost`, `total_par` | Portfolio totals (whole dollars). |
| `mark_pct`, `debt_mark_pct` | FV / cost. |
| `by_sector`, `by_seniority` | 1-D composition maps. |
| `sector_x_seniority` | 2-D composition matrix; `_na` / `_pik` variants carry non-accrual and PIK overlays. |
| `top_companies` | Largest positions. |
| `maturity_buckets` | FV by maturity year-bucket. |
| `position_count`, `unique_companies` | Counts. |
| `wa_spread_bps`, `pct_floating`, `wa_maturity_years`, `wa_spread_dur`, `wa_dts` | Weighted-average risk stats. |
| `spread_dist`, `spread_coverage` | Spread distribution and coverage. |

---

*By JAOD & (mainly) Claude.*
