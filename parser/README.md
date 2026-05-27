# BDC SOI parser — reference adapters

Structural parsers that turn a BDC's **Schedule of Investments** (the giant
holdings table in a 10-K / 10-Q) into clean, reconciled JSON. This folder is a
**runnable sample** of the engine behind the BDC dashboards — it ships two
reference adapters (ARCC, OBDC) plus the shared runtime.

## What's here

```
parser/
  README.md            this file
  SCHEMA.md            full data schema (parser output + dashboard data file)
  requirements.txt     beautifulsoup4 + lxml
  run_adapter.py       CLI: parse one filing
  adapters/
    base.py            shared structural parsing engine (V2SOIParser)
    _shared.py         tokenizers, footnote + rate/spread classifiers
    _unfunded.py       unfunded-commitment table extraction
    arcc.py            Ares Capital (ARCC) adapter
    obdc.py            Blue Owl Capital (OBDC) adapter
    registry.py        ticker -> adapter map
```

## Design principle

Parsers use **structural cues** — table headers, column semantics, footnote
legends, XBRL tags — never positional column indices or known-value
whitelists. A row is assumed wrong until the source HTML proves it right. Every
parsed value carries a pointer back to its source table/row for audit.

## Run it

```bash
pip install -r requirements.txt

# Point at a downloaded 10-K HTML for a supported ticker:
python run_adapter.py ARCC 2025-12-31 --html /path/to/arcc-10k.htm --out arcc.json
python run_adapter.py OBDC 2025-12-31 --html /path/to/obdc-10k.htm --out obdc.json
```

Output conforms to **SCHEMA.md → section 1**. The two adapters here reconcile
to the filer's XBRL `Total Investments` to within rounding (ARCC −0.001%,
OBDC +0.000%).

## Scope of this sample

- **Provided:** the two adapters above + the full shared runtime, runnable
  end-to-end, and the complete data schema.
- **Not provided here:** the other ~40 BDC adapters, the reconciliation /
  audit harness, the dashboard build pipeline, and the orchestration docs.
  Architecture is identical across all adapters; the rest is available on
  request.

---

*By JAOD & (mainly) Claude.*
