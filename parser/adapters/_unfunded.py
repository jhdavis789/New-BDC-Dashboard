"""Shared unfunded-commitment / revolver / DDTL extractor.

BDC 10-Ks/10-Qs disclose unfunded debt and equity commitments in three
structural places:

  1. **Schedule of Unfunded Commitments table** — usually appears
     IMMEDIATELY AFTER the main SOI run. Per-facility detail with
     columns like:
       (a) BXSL/OBDC variant:
           Investments | Commitment Type | Commitment Expiration Date |
           [Funded Commitment] | (Unfunded) Commitment | Fair Value
       (b) FSK variant:
           Category / Company | Commitment Amount
           ... category banners interleave the data rows.
  2. **Notes to financial statements** ("Commitments and Contingencies")
     — narrative aggregate (e.g. GBDC: "$927.9 million ... including
     $252.6 million of unfunded commitments on revolvers").
  3. **iXBRL tags** — `us-gaap:UnfundedLoanCommitmentLiability`,
     `us-gaap:OtherCommitmentDueWithinOneYear`,
     `us-gaap:InvestmentCompanyCommitmentToFundInvestments` (BDC custom).

This module emits a list of `UnfundedRecord` instances representing
per-facility detail when (1) is available, with `source` = `'soi_table'`,
and falls back to a single aggregate record (`source = 'note'`) when
only the narrative aggregate is disclosable.

Hard rule (per `research/SKILLS/parsing/flexible-structural/SKILL.md`):
detection is purely STRUCTURAL — we key off table-header tokens and
narrative captions, never hardcoded dollar values or known issuer
names. The only ticker-specific allowance here is the choice between
the two table variants (a) vs (b), which itself is detected
structurally from header columns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from bs4 import Tag

from ._shared import (
    clean_text,
    is_total_like_text,
    normalize_label,
    parse_number,
    strip_footnotes,
)

if TYPE_CHECKING:
    from .base import V2SOIParser


# ────────────────────────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────────────────────────


@dataclass
class UnfundedRecord:
    """One entry in the schedule of unfunded commitments.

    For per-row records (`source='soi_table'`), all dollar fields are in
    raw USD (already scaled by the parent parser's unit_multiplier).
    For aggregate fallback records (`source='note'` or `'xbrl'`), only
    `undrawn_amount` is populated and `company_name` is None.
    """

    company_name: Optional[str]
    company_name_raw: Optional[str]
    investment_description: Optional[str]   # commitment type (Revolver / DDT / etc.)
    commitment_amount: Optional[float]      # total commitment, if disclosed
    drawn_amount: Optional[float]           # funded portion, if disclosed
    undrawn_amount: Optional[float]         # the unfunded $ figure
    fair_value_of_undrawn: Optional[float]  # fair value of unfunded portion
    expiration_date: Optional[str]          # ISO YYYY-MM-DD where parseable
    source: str                             # 'soi_table' | 'note' | 'xbrl'
    # Lineage
    _source_table_index: int = -1
    _source_row_index: int = -1
    # Linkage to a parent SOI investment (matched by canonical name).
    link_id: Optional[str] = None
    parent_row_id: Optional[str] = None
    # Aggregate-only context tag: 'revolvers', 'delayed_draw', 'equity',
    # 'debt', 'total' — used for GBDC-style aggregate disclosures.
    aggregate_kind: Optional[str] = None


# ────────────────────────────────────────────────────────────────────
# Header / caption detection
# ────────────────────────────────────────────────────────────────────


# Variant (a) BXSL/OBDC: at least one column header contains
# "unfunded commitment" or "commitment expiration date" as a structural
# signature.
_VARIANT_A_HEADER_TOKENS = {
    # Required: a Company-name-like column.
    "company_col": ("investments", "company", "portfolio company", "issuer"),
}

# Variant (b) FSK: header has exactly the two columns
# "Category / Company" and "Commitment Amount".
_VARIANT_B_HEADER_TOKENS = {
    "category_company": ("category / company", "category/company",
                         "company / category", "company/category"),
    "amount_col": ("commitment amount", "unfunded commitment",
                   "amount of commitment"),
}


def _row_normalized_labels(parser: "V2SOIParser", row: Tag) -> list[str]:
    return [normalize_label(t) for _, _, t in parser._row_spans(row) if t]


def _is_unfunded_header_variant_a(parser: "V2SOIParser", row: Tag) -> bool:
    """Header signature for the BXSL/OBDC/ARCC table variant.

    Required:
      • a column whose label matches one of `_VARIANT_A_HEADER_TOKENS["company_col"]`
      • at least one column whose label contains an "unfunded ... commitment(s)"
        phrase (the BDC-universal marker for the schedule of unfunded
        commitments). We allow modifiers between the two words to capture
        ARCC-style "Total unfunded equity commitments" as well as the
        bare "Unfunded Commitment" used by BXSL/OBDC. We also accept
        "Commitment Expiration Date" as a strong-enough single-column
        signature even without "unfunded" in the same row.
    """
    labels = _row_normalized_labels(parser, row)
    if not labels:
        return False
    has_company = any(
        any(c == lbl or c in lbl for c in _VARIANT_A_HEADER_TOKENS["company_col"])
        for lbl in labels
    )
    if not has_company:
        return False
    if any("commitment expiration date" in lbl for lbl in labels):
        return True
    # "unfunded ... commitment" pattern (allow up to ~3 modifying words
    # in between, matching ARCC "total unfunded equity commitments",
    # "total unfunded delayed draw loan commitments", etc.).
    unf_re = re.compile(r"\bunfunded\b(?:\s+\w+){0,4}\s+commitments?\b")
    return any(unf_re.search(lbl) for lbl in labels)


def _is_unfunded_header_variant_b(parser: "V2SOIParser", row: Tag) -> bool:
    """Header signature for the FSK 'Category / Company' table variant."""
    labels = _row_normalized_labels(parser, row)
    if not labels:
        return False
    has_cat_co = any(
        any(c in lbl for c in _VARIANT_B_HEADER_TOKENS["category_company"])
        for lbl in labels
    )
    has_amt = any(
        any(a in lbl for a in _VARIANT_B_HEADER_TOKENS["amount_col"])
        for lbl in labels
    )
    return has_cat_co and has_amt


# ────────────────────────────────────────────────────────────────────
# Period guard — reuse parser's prior-period detector
# ────────────────────────────────────────────────────────────────────


def _table_is_current_period(parser: "V2SOIParser", table: Tag) -> bool:
    """Return True iff the narrative caption preceding this table refers
    to the current period (i.e., not a prior-period comparative)."""
    try:
        return not parser._table_preceding_caption_mentions_prior_period(table)
    except Exception:
        # If the parser hasn't initialised the narrative cache (e.g. very
        # old filing without proper as-of captions), be permissive.
        return True


# ────────────────────────────────────────────────────────────────────
# Variant A: BXSL / OBDC per-row table
# ────────────────────────────────────────────────────────────────────


_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_MONTH_YEAR_RE = re.compile(r"^\s*(\d{1,2})/(\d{2,4})\s*$")


def _parse_expiration_date(text: str) -> Optional[str]:
    """Parse a date cell into ISO YYYY-MM-DD where possible.

    Accepts 'M/D/YYYY', 'MM/DD/YY', 'M/YYYY' (month-only → first of month).
    Returns None for non-date strings (e.g. 'N/A').
    """
    if not text:
        return None
    t = text.strip()
    if not t or t.upper() == "N/A":
        return None
    m = _DATE_RE.search(t)
    if m:
        mo, d, yr = m.group(1), m.group(2), m.group(3)
        if len(yr) == 2:
            yr = "20" + yr if int(yr) < 80 else "19" + yr
        try:
            return f"{int(yr):04d}-{int(mo):02d}-{int(d):02d}"
        except ValueError:
            return None
    m2 = _MONTH_YEAR_RE.match(t)
    if m2:
        mo, yr = m2.group(1), m2.group(2)
        if len(yr) == 2:
            yr = "20" + yr if int(yr) < 80 else "19" + yr
        try:
            return f"{int(yr):04d}-{int(mo):02d}-01"
        except ValueError:
            return None
    return None


def _build_variant_a_schema(parser: "V2SOIParser", header_row: Tag) -> dict[int, str]:
    """Return a map: cell_logical_start_index -> field_name.

    Field names: 'company', 'commitment_type', 'expiration', 'commitment',
    'funded', 'unfunded', 'fair_value'.
    """
    field_map: dict[int, str] = {}
    running = 0
    for cell in header_row.find_all(["td", "th"]):
        cs = int(cell.get("colspan", 1) or 1)
        label = normalize_label(clean_text(cell.get_text()))
        # Strip leading "(in thousands)" / "(in millions)" caption prefix.
        label = re.sub(
            r"^\(in\s+(millions|thousands|billions)\)\s*",
            "",
            label,
            flags=re.I,
        ).strip()
        # OBDC's variant has split header text "unfunded\ncompany" — the
        # actual company column carries the literal label "company" or
        # "portfolio company" after the <br/> normalisation.
        f = None
        if not label:
            pass
        elif "investments" in label and "fair value" not in label:
            # BXSL header literally uses "Investments" as the company col.
            f = "company"
        elif label in ("company", "portfolio company", "issuer"):
            f = "company"
        elif "commitment type" in label:
            f = "commitment_type"
        elif "commitment expiration date" in label:
            f = "expiration"
        elif (
            "funded commitment" in label and "unfunded" not in label
        ) or (
            label.startswith("less:") and "funded" in label and "unfunded" not in label
        ):
            f = "funded"
        elif (
            "unfunded commitment" in label
            or label == "commitment"
            or "amount of commitment" in label
            or re.search(r"\bunfunded\b(?:\s+\w+){0,4}\s+commitments?\b", label)
        ):
            # OBDC's "Commitment" col adjacent to "Funded Commitment" is the
            # UNFUNDED amount (header text "Unfunded\nCommitment" splits
            # across lines — only "Commitment" survives in the cell text).
            # ARCC uses "Total Unfunded [Equity] Commitments" / "Total
            # Unfunded [Revolving and Delayed Draw] Commitments".
            f = "unfunded"
        elif "fair value" in label:
            f = "fair_value"
        elif (
            "total commitment" in label
            or re.search(r"^total\s+\w+\s+commitments?$", label)
            or re.search(r"^total\s+revolving\s+and\s+delayed[-\s]?draw", label)
            or re.search(r"^total\s+equity\s+commitments?$", label)
        ):
            # Aggregate total-commitment column, distinct from the
            # unfunded-commitment column.
            f = "commitment"
        if f is not None:
            field_map[running] = f
        running += cs
    return field_map


def _resolve_field(field_map: dict[int, str], col_start: int) -> Optional[str]:
    """Look up the field for a logical column start index. The field_map
    keys are header-cell logical starts; data cells with non-trivial
    colspans need a containment match (start ≤ col < next_start)."""
    keys = sorted(field_map.keys())
    if not keys:
        return None
    last = None
    for k in keys:
        if k <= col_start:
            last = k
        else:
            break
    return field_map.get(last) if last is not None else None


def _extract_variant_a_table(
    parser: "V2SOIParser",
    table: Tag,
    table_doc_idx: int,
) -> tuple[list[UnfundedRecord], Optional[float]]:
    """Extract per-row records from a BXSL/OBDC-style unfunded table.

    Returns (records, total_from_filing). The total is captured if the
    table has a 'Total Unfunded Commitments' row — useful for
    reconciliation against the row sum.
    """
    rows = table.find_all("tr")
    # Find the header row (first row matching variant-A signature).
    header_row = None
    header_idx = -1
    for ri, row in enumerate(rows[:8]):
        if _is_unfunded_header_variant_a(parser, row):
            header_row = row
            header_idx = ri
            break
    if header_row is None:
        return [], None
    field_map = _build_variant_a_schema(parser, header_row)
    if "company" not in field_map.values() or "unfunded" not in field_map.values():
        return [], None

    out: list[UnfundedRecord] = []
    total_from_filing: Optional[float] = None
    mult = parser.unit_multiplier
    for ri, row in enumerate(rows):
        if ri <= header_idx:
            continue
        spans = parser._row_spans(row)
        non_empty = [s for s in spans if s[2]]
        if not non_empty:
            continue
        first_text = non_empty[0][2]

        # "Total Unfunded Commitments $X $Y" reconciliation row.
        if is_total_like_text(first_text):
            # Capture the unfunded total for reconciliation.
            unf_total: Optional[float] = None
            for start, _, txt in spans:
                if not txt:
                    continue
                f = _resolve_field(field_map, start)
                if f == "unfunded":
                    val = parse_number(txt)
                    if val is not None:
                        unf_total = val * mult
                        break
            if unf_total is not None:
                total_from_filing = unf_total
            continue

        # Section banner ("Non-controlled/non-affiliated - debt commitments")
        # — detected as a single-cell row whose first text doesn't look
        # like a company name AND has no numeric content.
        has_numeric = any(
            parse_number(s[2]) is not None for s in spans
            if s[2] and s[2] not in ("$", "—", "-", "(", ")")
        )
        if not has_numeric and len(non_empty) <= 2:
            continue

        # Pull fields by schema.
        rec_fields: dict[str, Optional[str | float]] = {
            "company": None,
            "commitment_type": None,
            "expiration": None,
            "funded": None,
            "unfunded": None,
            "commitment": None,
            "fair_value": None,
        }
        for start, _, txt in spans:
            if not txt:
                continue
            f = _resolve_field(field_map, start)
            if f is None:
                continue
            if f in ("company", "commitment_type", "expiration"):
                if rec_fields[f] is None:
                    rec_fields[f] = txt
            elif f in ("funded", "unfunded", "commitment", "fair_value"):
                # Skip currency-symbol-only cells.
                if txt.strip() in ("$", "—", "-"):
                    continue
                val = parse_number(txt)
                if val is None:
                    continue
                if rec_fields[f] is None:
                    rec_fields[f] = val * mult

        company = rec_fields["company"]
        company_str = company.strip() if isinstance(company, str) else ""
        # Total/aggregate rows often have an empty company column or just
        # a `$` (or `—`) in it. Treat them as totals: capture the unfunded
        # figure into `total_from_filing` and skip emitting a record.
        if not company_str or not any(ch.isalpha() for ch in company_str):
            row_unf = rec_fields.get("unfunded")
            if isinstance(row_unf, float):
                # Sum across multiple total rows (one per cluster table)
                # — ARCC repeats no totals, BXSL repeats the SAME total
                # in every continuation table. Track the LARGEST seen as
                # the disclosed-total candidate; cluster aggregation is
                # handled by the caller.
                if total_from_filing is None or row_unf > total_from_filing:
                    total_from_filing = row_unf
            continue
        # Drop pure-banner rows (no numeric fields present).
        if (
            rec_fields["unfunded"] is None
            and rec_fields["commitment"] is None
            and rec_fields["funded"] is None
        ):
            continue

        out.append(UnfundedRecord(
            company_name=strip_footnotes(company_str),
            company_name_raw=company_str,
            investment_description=(
                rec_fields["commitment_type"]  # type: ignore[arg-type]
                if isinstance(rec_fields["commitment_type"], str) else None
            ),
            commitment_amount=rec_fields["commitment"],  # type: ignore[arg-type]
            drawn_amount=rec_fields["funded"],            # type: ignore[arg-type]
            undrawn_amount=rec_fields["unfunded"],        # type: ignore[arg-type]
            fair_value_of_undrawn=rec_fields["fair_value"],  # type: ignore[arg-type]
            expiration_date=_parse_expiration_date(
                rec_fields["expiration"] if isinstance(rec_fields["expiration"], str) else ""
            ),
            source="soi_table",
            _source_table_index=table_doc_idx,
            _source_row_index=ri,
        ))
    return out, total_from_filing


# ────────────────────────────────────────────────────────────────────
# Variant B: FSK "Category / Company" table
# ────────────────────────────────────────────────────────────────────


def _extract_variant_b_run(
    parser: "V2SOIParser",
    tables: list[tuple[int, Tag]],
) -> tuple[list[UnfundedRecord], Optional[float]]:
    """Variant B (FSK): a contiguous run of small tables sharing the
    "Category / Company | Commitment Amount" header. Section banner rows
    (e.g. 'Senior Secured Loans—First Lien') interleave the data rows
    and apply to subsequent rows. The final table's last data row is
    typically 'Total $X' followed by 'Unfunded Asset Based Finance/Other
    commitments $Y' — both are aggregates, not per-facility.
    """
    out: list[UnfundedRecord] = []
    total_from_filing: Optional[float] = None
    current_category: Optional[str] = None
    mult = parser.unit_multiplier

    for table_doc_idx, table in tables:
        rows = table.find_all("tr")
        # Find header
        header_idx = -1
        for ri, row in enumerate(rows[:6]):
            if _is_unfunded_header_variant_b(parser, row):
                header_idx = ri
                break
        if header_idx < 0:
            # Continuation table without its own header — assume header is
            # row 0 and skip if no 'commitment amount' tokens appear.
            header_idx = 0

        for ri, row in enumerate(rows):
            if ri <= header_idx:
                continue
            spans = parser._row_spans(row)
            non_empty = [s for s in spans if s[2]]
            if not non_empty:
                continue
            first_text = non_empty[0][2]
            first_norm = normalize_label(first_text)

            # Skip header repetitions.
            if "category / company" in first_norm or "category/company" in first_norm:
                continue

            # 'Total $X' — reconciliation aggregate (sum of debt commitments).
            if first_norm == "total":
                for _, _, txt in spans[1:]:
                    if not txt or txt.strip() in ("$", "—", "-"):
                        continue
                    val = parse_number(txt)
                    if val is not None:
                        total_from_filing = val * mult
                        break
                continue

            # 'Unfunded Asset Based Finance/Other commitments $Y' — aggregate.
            if (
                "unfunded" in first_norm
                and "commitment" in first_norm
                and not any(
                    parse_number(s[2]) is not None
                    for s in spans
                    if s[2] and s is not non_empty[0]
                )
            ):
                # No numeric → just a banner; treat as category.
                current_category = first_text
                continue
            if "unfunded" in first_norm and "commitment" in first_norm:
                # Aggregate row with a $ value — emit as aggregate record.
                for _, _, txt in spans[1:]:
                    if not txt or txt.strip() in ("$", "—", "-"):
                        continue
                    val = parse_number(txt)
                    if val is not None:
                        out.append(UnfundedRecord(
                            company_name=None,
                            company_name_raw=None,
                            investment_description=first_text,
                            commitment_amount=None,
                            drawn_amount=None,
                            undrawn_amount=val * mult,
                            fair_value_of_undrawn=None,
                            expiration_date=None,
                            source="note",
                            _source_table_index=table_doc_idx,
                            _source_row_index=ri,
                            aggregate_kind="other",
                        ))
                        break
                continue

            # Determine if this row has a numeric amount.
            amount: Optional[float] = None
            for s in spans:
                txt = s[2]
                if not txt or txt.strip() in ("$", "—", "-"):
                    continue
                if s is non_empty[0]:
                    continue
                val = parse_number(txt)
                if val is not None:
                    amount = val * mult
                    break

            if amount is None:
                # Treat as a section banner / category header.
                # Heuristic: short text with no numeric content.
                if len(first_text) <= 120:
                    current_category = first_text
                continue

            # Per-company commitment row.
            out.append(UnfundedRecord(
                company_name=strip_footnotes(first_text),
                company_name_raw=first_text,
                investment_description=current_category,
                commitment_amount=None,
                drawn_amount=None,
                undrawn_amount=amount,
                fair_value_of_undrawn=None,
                expiration_date=None,
                source="soi_table",
                _source_table_index=table_doc_idx,
                _source_row_index=ri,
            ))
    return out, total_from_filing


# ────────────────────────────────────────────────────────────────────
# Aggregate-only fallback (GBDC-style note disclosure)
# ────────────────────────────────────────────────────────────────────


_GBDC_AGGREGATE_RES: list[tuple[str, re.Pattern]] = [
    # "outstanding commitments to fund investments totaling $ 927,887"
    ("total", re.compile(
        r"outstanding\s+commitments\s+to\s+fund\s+investments\s+"
        r"totaling\s+\$\s*([\d,\.]+)",
        re.I,
    )),
    # "including $ 252,574 of commitments on undrawn revolvers"
    ("revolvers", re.compile(
        r"including\s+\$\s*([\d,\.]+)\s+of\s+commitments\s+on\s+"
        r"undrawn\s+revolvers",
        re.I,
    )),
    # OSCF: "off-balance sheet arrangements consisted of $ 910.9 million
    # of unfunded commitments to provide debt financing ..."
    ("total", re.compile(
        r"off[-\s]balance\s+sheet\s+arrangements\s+consisted\s+of\s+"
        r"\$\s*([\d,\.]+)\s*million\s+of\s+unfunded\s+commitments",
        re.I,
    )),
    # Common variant: "...consisted of $X million in unfunded commitments"
    ("total", re.compile(
        r"consisted\s+of\s+\$\s*([\d,\.]+)\s*million\s+(?:in|of)\s+"
        r"unfunded\s+commitments",
        re.I,
    )),
    # ADS / Apollo-style precise subtotal table grand total:
    # "Total Unfunded Commitments (4) $ 4,522,110"
    # The trailing parenthesized footnote number (e.g. "(4)") is required
    # to discriminate the rollup row from the column header.
    ("total", re.compile(
        r"Total\s+Unfunded\s+Commitments?\s*\(\d+\)\s*\$\s*([\d,]+)",
        re.I,
    )),
    # ADS narrative aggregate: "we had unfunded commitments... with an
    # aggregate principal amount of $4.5 billion". Falls back to the
    # narrative billions phrasing if the table grand total isn't captured.
    ("total", re.compile(
        r"unfunded\s+commitments[^\.]{0,200}?aggregate\s+principal\s+"
        r"amount\s+of\s+\$\s*([\d,\.]+)\s*billion",
        re.I,
    )),
]


def _extract_aggregate_from_notes(parser: "V2SOIParser") -> list[UnfundedRecord]:
    """Walk the document text for aggregate-only commitment disclosures.

    Returns a list of aggregate records with `source='note'` and
    `aggregate_kind` set to 'total' or 'revolvers'. Used for GBDC where no
    per-facility table exists in the filing.

    Detection is structural: we look for paragraphs near a 'Commitments'
    or 'Commitments and Contingencies' note heading. To avoid double-
    counting we require the matched sentence to mention the current
    fiscal-year-end month/year in a 'as of <Month> <day>, <year>' phrase.
    """
    if parser.soup is None:
        return []
    period = parser.period_end or ""
    if not period:
        return []
    yr = period[:4]
    # Map ISO month → English month name.
    months = {"01": "January", "02": "February", "03": "March", "04": "April",
              "05": "May", "06": "June", "07": "July", "08": "August",
              "09": "September", "10": "October", "11": "November",
              "12": "December"}
    cur_month = months.get(period[5:7], "")
    if not cur_month:
        return []
    # Build a single text blob from the body to scan.
    text = parser.soup.get_text(" ", strip=True)
    # Normalise non-breaking spaces.
    text = text.replace("\xa0", " ")
    out: list[UnfundedRecord] = []
    seen_kinds: set[str] = set()
    # Sentence-level guard: each match must sit inside a sentence that
    # also names the current period.
    period_re = re.compile(
        rf"as\s+of\s+{cur_month}\s+\d{{1,2}},?\s+{yr}", re.I
    )
    for kind, rx in _GBDC_AGGREGATE_RES:
        if kind in seen_kinds:
            continue
        for m in rx.finditer(text):
            # Look back for a current-period anchor. Default 250 chars
            # works for narrative aggregates ("As of <date>, the Company's
            # off-balance sheet arrangements consisted of $X million...")
            # which sit immediately after the date phrase. Widen to 800
            # chars for table grand-totals which sit at the END of a
            # multi-row table whose introduction ("had the following
            # unfunded commitments...") is a few rows further up.
            window = 800 if "Total Unfunded Commitments" in m.group(0) else 250
            ctx_start = max(0, m.start() - window)
            ctx = text[ctx_start:m.end() + 50]
            if not period_re.search(ctx):
                continue
            num_str = m.group(1).replace(",", "")
            try:
                val = float(num_str)
            except ValueError:
                continue
            # GBDC's note is in raw $; the SOI is in thousands. The note
            # phrases the number identically to the underlying ledger
            # ($ 927,887 = 927,887 thousand = $927.887M). Apply scaling
            # only when the document uses the same unit caption around
            # the note. Heuristic: if `val` is < 100,000 we treat it as
            # the raw $M number (e.g. "$927.9 million"); otherwise treat
            # it as thousands matching the SOI.
            # ADS variant: "aggregate principal amount of $4.5 billion"
            # — explicitly billions-denominated, scale by 1B. Check both
            # the matched substring (in case "billion" is inside the regex
            # match) and the tail context (in case it follows the captured
            # number).
            matched = m.group(0).lower()
            tail_ctx = text[m.end():m.end() + 30].lower()
            if ("billion" in matched or "billion" in tail_ctx) and val < 10_000:
                undrawn = val * 1_000_000_000
            elif val < 100_000 and "." in num_str:
                # Likely millions-form ("$927.9 million").
                undrawn = val * 1_000_000
            else:
                undrawn = val * parser.unit_multiplier
            out.append(UnfundedRecord(
                company_name=None,
                company_name_raw=None,
                investment_description=None,
                commitment_amount=None,
                drawn_amount=None,
                undrawn_amount=undrawn,
                fair_value_of_undrawn=None,
                expiration_date=None,
                source="note",
                _source_table_index=-1,
                _source_row_index=-1,
                aggregate_kind=kind,
            ))
            seen_kinds.add(kind)
            break
    return out


# ────────────────────────────────────────────────────────────────────
# Top-level extractor
# ────────────────────────────────────────────────────────────────────


def extract_unfunded_commitments(
    parser: "V2SOIParser",
) -> tuple[list[UnfundedRecord], Optional[float]]:
    """Discover and parse the unfunded-commitments schedule(s) for the
    parser's filing. Returns (records, disclosed_total).

    Behaviour:
      • Walks every <table> in the document AFTER the last main SOI
        table.
      • For each candidate, classifies the header into variant A or B.
      • For variant A: extracts per-row records from each contiguous
        cluster, summing the disclosed totals.
      • For variant B (FSK): processes the contiguous run as a single
        unit because category banners interleave across multiple
        narrow tables.
      • If no per-row table is found, falls back to the narrative
        aggregate (`_extract_aggregate_from_notes`).
      • Period-guards against prior-period comparative tables (uses the
        parser's existing `_table_preceding_caption_mentions_prior_period`
        helper).
    """
    if parser.soup is None:
        return [], None
    soi_tables = getattr(parser, "_main_soi_tables", None) or parser.find_soi_tables()
    if not soi_tables:
        return _extract_aggregate_from_notes(parser), None
    last_soi_idx = max(parser.tables.index(t) for t in soi_tables)

    out_records: list[UnfundedRecord] = []
    disclosed_total: Optional[float] = None

    # Group variant-A tables into clusters that share a column-schema
    # signature (same field map). Each cluster contributes its grand
    # total = the LARGEST per-table total seen within the cluster (BXSL
    # repeats one total in every continuation table; ARCC has a single
    # grand-total row in the LAST table of the run; OBDC has multi-level
    # subtotals plus a final grand total).
    #
    # The filing's disclosed total is the SUM of distinct cluster totals
    # (debt schedule + equity schedule for ARCC; one schedule for the
    # others). Cluster boundary = (a) different header schema, OR
    # (b) > 2 non-A tables in between, OR (c) a current-period guard
    # flips to prior-period.
    variant_b_run: list[tuple[int, Tag]] = []
    cluster_totals: list[float] = []
    cluster_largest: Optional[float] = None
    cluster_signature: Optional[tuple] = None
    last_a_idx = -10

    def _flush_cluster() -> None:
        nonlocal cluster_largest, cluster_signature
        if cluster_largest is not None:
            cluster_totals.append(cluster_largest)
        cluster_largest = None
        cluster_signature = None

    for i in range(last_soi_idx + 1, len(parser.tables)):
        t = parser.tables[i]
        rows = t.find_all("tr")
        v_a = False
        v_b = False
        a_header_row = None
        for row in rows[:8]:
            if _is_unfunded_header_variant_a(parser, row):
                v_a = True
                a_header_row = row
                break
            if _is_unfunded_header_variant_b(parser, row):
                v_b = True
                break
        if v_a:
            if not _table_is_current_period(parser, t):
                _flush_cluster()
                last_a_idx = -10
                continue
            # Cluster signature = sorted tuple of (start, field) pairs.
            field_map = _build_variant_a_schema(parser, a_header_row)
            sig = tuple(sorted(field_map.items()))
            if (
                cluster_signature is not None
                and (sig != cluster_signature or i - last_a_idx > 2)
            ):
                _flush_cluster()
            cluster_signature = sig
            recs, tot = _extract_variant_a_table(parser, t, i)
            out_records.extend(recs)
            if tot is not None:
                if cluster_largest is None or tot > cluster_largest:
                    cluster_largest = tot
            last_a_idx = i
        elif v_b:
            _flush_cluster()
            last_a_idx = -10
            if not _table_is_current_period(parser, t):
                continue
            variant_b_run.append((i, t))
        else:
            if i - last_a_idx > 4:
                _flush_cluster()
                last_a_idx = -10
    _flush_cluster()

    if cluster_totals:
        disclosed_total = sum(cluster_totals)

    if variant_b_run:
        recs, tot = _extract_variant_b_run(parser, variant_b_run)
        out_records.extend(recs)
        if tot is not None and disclosed_total is None:
            disclosed_total = tot

    if out_records:
        # Link to parent SOI rows by canonical company name.
        _link_to_parents(parser, out_records)
        return out_records, disclosed_total

    # No per-row schedule found — fall back to aggregate from notes.
    agg = _extract_aggregate_from_notes(parser)
    if agg:
        # Take the 'total' aggregate as the disclosed total.
        for r in agg:
            if r.aggregate_kind == "total" and r.undrawn_amount is not None:
                disclosed_total = r.undrawn_amount
                break
        return agg, disclosed_total

    return [], None


def _link_to_parents(
    parser: "V2SOIParser",
    records: list[UnfundedRecord],
) -> None:
    """Populate `link_id` and `parent_row_id` on each record by matching
    against parser.investments via canonical company name."""
    name_to_link: dict[str, str] = {}
    for inv in parser.investments:
        name = inv.company_name or ""
        key = re.sub(r"\s+", " ", name).strip().casefold()
        if key and inv.link_id and key not in name_to_link:
            name_to_link[key] = inv.link_id
    for r in records:
        if not r.company_name:
            continue
        key = re.sub(r"\s+", " ", strip_footnotes(r.company_name)).strip().casefold()
        link = name_to_link.get(key)
        if link:
            r.link_id = link
            r.parent_row_id = link


def annotate_investments(
    parser: "V2SOIParser",
    records: list[UnfundedRecord],
) -> None:
    """For each per-row unfunded record that links to a parent SOI
    investment, set `unfunded_amount` and `is_unfunded_commitment` on the
    matching investment row. Multiple unfunded facilities for one
    company are summed onto the aggregate name match (a parent without a
    per-tranche match)."""
    if not records:
        return
    # Build company-name -> list[Investment] index.
    by_name: dict[str, list] = {}
    for inv in parser.investments:
        name = inv.company_name or ""
        key = re.sub(r"\s+", " ", strip_footnotes(name)).strip().casefold()
        if key:
            by_name.setdefault(key, []).append(inv)
    # Apply: when a company has exactly one investment, set the unfunded
    # amount on it; when multiple, sum onto the FIRST debt-like row.
    for r in records:
        if r.source != "soi_table" or not r.company_name:
            continue
        if r.undrawn_amount is None:
            continue
        key = re.sub(r"\s+", " ", strip_footnotes(r.company_name)).strip().casefold()
        invs = by_name.get(key, [])
        if not invs:
            continue
        target = invs[0]
        # Aggregate when the same company has multiple unfunded facilities.
        target.unfunded_amount = (target.unfunded_amount or 0) + r.undrawn_amount
        target.is_unfunded_commitment = True
        if "is_unfunded_commitment" not in target.footnote_flags:
            target.footnote_flags = sorted(
                set(target.footnote_flags) | {"is_unfunded_commitment"}
            )
