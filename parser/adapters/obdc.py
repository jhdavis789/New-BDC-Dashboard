"""OBDC V2 adapter — Blue Owl Capital Corp.

Reports in thousands. SDLP (Senior Direct Lending Program) is a JV
captured as a single SOI row in current periods.

OBDC-specific dialect:
- The header has "Ref. Rate" → "Cash" → "PIK" as three SEPARATE columns,
  but per OBDC's iXBRL tagging the "Cash" column adjacent to a "Ref. Rate"
  prefix actually carries the SPREAD over the reference rate (e.g. `4.25%`
  means SOFR+4.25%, tagged as `InvestmentBasisSpreadVariableRate`), NOT
  the all-in cash coupon. We override the schema label so the parser
  routes that column to `spread_text_raw` / `spread_bps`.

R6-CP-D2: Some OBDC 10-K filings (e.g. 2023-12-31) embed a Senior Loan
Fund (SLF) subsidiary SOI in the MD&A narrative BEFORE the main
consolidated SOI in the financial statements. The base find_soi_tables
walks document order and stops at the first 'Total Investments', which
on those periods captures the SLF subsidiary instead of the main SOI.
We override find_soi_tables to skip SLF tables (identified by the
caption "Senior Loan Fund" preceding the table).

R6-D1 fix: OBDC's SOI is split across ~20 small HTML tables. Affiliate
section banners ("Non-controlled/affiliated..." and
"Controlled/affiliated...") appear as the FINAL row of one table with
zero data rows following them. The data for that affiliate section
begins in the NEXT table. We override extract() to maintain a
`_pending_category` variable that carries the banner state across table
boundaries so the first data row of the next table is correctly tagged.

R6-D5 fix: OBDC footnote (25) = default pledged (all rows).
Footnote (26) = "Investment is not pledged..." (exception). The base
parser's is_pledged logic fires when the legend says "pledged as
collateral" — but that's footnote 25, which is the DEFAULT, not a
per-row override. After extract, we set ALL rows is_pledged=True and
flip only rows with marker '26' to is_pledged=False.

R8-D1 fix: OBDC 10-K filings use a DIFFERENT affiliate-marking structure
than 10-Q filings. In 10-Qs, affiliate banners appear inline in the
main SOI tables (and are handled by _extract_with_cross_table_banners).
In 10-Ks, the main SOI tables (30-48 doc indices) contain ALL investments
WITHOUT affiliate markers — affiliate info is in a SEPARATE SUBSCHEDULE
NOTE section appearing after the main SOI grand-total. The subschedule
tables (e.g. doc idx 85-88) re-list affiliate investments under
"Non-controlled/affiliated portfolio company investments" and
"Controlled/affiliated portfolio company investments" section banners.
After main extraction we scan those subschedule tables to build a company-
name → affiliation lookup and back-annotate the main SOI investments.

The subschedule also carries "not pledged" (fn26) markers on specific
company names that do NOT appear in the main SOI cells. We extract
those as well and back-annotate is_pledged=False on matching rows.
"""
from __future__ import annotations

import re
from typing import Optional

from bs4 import Tag

from .base import V2SOIParser, Investment, SectionTotal
from ._shared import (
    ColumnSchema,
    classify_row,
    is_header_row,
    is_total_like_text,
    normalize_label,
    parse_header_row,
    parse_number,
    parse_percent,
    LABEL_TO_FIELD,
)


_SLF_CAPTION_RE = re.compile(
    r"senior\s+loan\s+fund|"
    # Match 'SLF' only as a standalone program/caption title:
    # NOT 'SLF LLC' or 'SLF Fund' in a company name mid-sentence.
    # The negative lookahead avoids matching 'Blue Owl Credit SLF LLC'.
    r"\bSLF\b(?!\s+LLC\b|\s+Fund\b)|"
    r"joint\s+venture\s+portfolio",
    re.I,
)

# R6-D1: OBDC affiliate-section banner text patterns. These single-cell
# colspan rows appear at the TAIL of one table; data for the section is in
# the NEXT table. We detect them and carry the category state forward.
_OBDC_AFFILIATE_BANNER_RE = re.compile(
    r"^("
    r"non[-\s]?controlled\s*/?\s*affiliated\s+portfolio\s+company|"
    r"controlled\s*/?\s*affiliated\s+portfolio\s+company|"
    r"non[-\s]?controlled\s*[/,]\s*affiliated|"
    r"controlled\s*[/,]\s*affiliated|"
    r"non[-\s]?controlled\s+affiliated|"
    r"controlled\s+affiliated"
    r")",
    re.I,
)


class OBDCParser(V2SOIParser):
    TICKER = "OBDC"
    UNIT_MULTIPLIER = 1_000
    UNIT_DETECTION = True

    def find_soi_tables(self) -> list[Tag]:
        """Walk all tables and pick the canonical 2025 SOI run.

        OBDC files multiple SOIs in the same 10-K:
        - Item 1 Business: a compact summary view ("Section A").
        - Item 8 Financial Statements: the canonical SOI ("Section B"),
          which is what XBRL tags. Also contains a prior-period
          comparative SOI further down.

        Section A's last row is "Total" (not "Total Investments"), so it
        doesn't qualify as a grand-total run. Section B's last row is a
        strict "Total Investments" row whose FV matches the XBRL target
        exactly ($16,470,893k for 2025-12-31).

        We cannot rely on super().find_soi_tables() because base stops at
        the FIRST row matching its grand-total regex — which inadvertently
        matches "Total Investment Income" in the income statement that
        sits between Section A and Section B. So we walk self.tables
        directly, identify SOI-shaped tables, group into contiguous runs
        (gap > 2 breaks the run), and pick the latest run whose final
        table contains a STRICT "Total Investments" grand-total row. SLF
        / JV subsidiary runs and prior-period runs are filtered out by
        the existing caption-based guards.
        """
        import re as _re
        from ._shared import is_header_row, normalize_label

        if not self.soup:
            return []

        # Strict grand-total: matches "Total Investments" / "Total
        # Investment Portfolio" / "Total Portfolio Investments" but NOT
        # "Total Investment Income/Loss/Gain/Expense/...".
        STRICT_GT_RE = _re.compile(
            r"^total\s+(?:portfolio\s+)?investments?\b"
            r"(?!\s*[,]?\s*(?:income|loss|gain|expense|revenue|fees?|tax)\b)"
            r"|^total\s+investment\s+portfolio\b",
            _re.I,
        )

        # SOI-shape detector (mirrors base.py without the early-break).
        def _is_soi_table(table: Tag) -> bool:
            for r in table.find_all("tr")[:8]:
                if not is_header_row(r):
                    continue
                labels = [normalize_label(t) for _, _, t in self._row_spans(r) if t]
                has_fv = any(
                    "fair value" in l or "market value" in l or l == "value"
                    or "fairvalue" in l for l in labels
                )
                has_cost = any(
                    l == "cost" or "amortized cost" in l or l.startswith("cost")
                    or "cost basis" in l for l in labels
                )
                has_specific = any(
                    "rate" in l or "spread" in l or "maturity" in l
                    or "acquisition" in l or "principal" in l or "par" in l
                    or "shares" in l or "coupon" in l or "floor" in l
                    or "index" in l or l == "interest" or l.startswith("interest ")
                    for l in labels
                )
                if has_fv and has_cost and has_specific:
                    return True
            return False

        # Collect SOI-shaped tables, filtering SLF/JV preambles.
        soi_indices: list[int] = []
        for ti, t in enumerate(self.tables):
            if not _is_soi_table(t):
                continue
            if self._table_preceded_by_slf_caption(t):
                continue
            soi_indices.append(ti)
        if not soi_indices:
            return []

        # Group into contiguous runs (gap > 2 starts a new run).
        runs: list[list[int]] = []
        current: list[int] = []
        last = -10
        for ti in soi_indices:
            if not current or ti - last <= 2:
                current.append(ti)
            else:
                runs.append(current)
                current = [ti]
            last = ti
        if current:
            runs.append(current)

        # For each run, find the last "Total Investments" row (strict).
        def _has_strict_grand_total(run_indices: list[int]) -> bool:
            for ti in run_indices:
                for row in self.tables[ti].find_all("tr"):
                    texts = [t for _, _, t in self._row_spans(row) if t]
                    if texts and STRICT_GT_RE.match(texts[0]):
                        return True
            return False

        # Prefer the LATEST run with a strict grand-total. The intervening
        # comparative-period (e.g. 2024 in a 2025 10-K) is filed AFTER the
        # current-period SOI in OBDC's layout, so the LAST run is actually
        # prior-period. Use the prior-period caption guard to skip runs
        # whose preceding caption names a non-current year.
        cur_year = self.period_end[:4] if self.period_end else None
        candidate_runs: list[list[int]] = []
        for run in runs:
            if not _has_strict_grand_total(run):
                continue
            # Check the caption immediately preceding the first table in
            # this run; skip if it names a prior period.
            first_table = self.tables[run[0]]
            if self._caption_names_prior_period(first_table, cur_year):
                continue
            candidate_runs.append(run)

        if not candidate_runs:
            return []

        # Pick the LARGEST current-period run with a strict grand-total
        # (by total row count). 10-Q filings layer a small "Investments
        # by industry" table or unfunded-commitment schedule AFTER the
        # main SOI; picking the latest run would land on that tiny
        # tail run (~7 tables, ~160 rows) instead of the canonical SOI
        # (~26 tables, ~600 rows). 10-Ks happen to have the canonical
        # SOI be the latest run, so the old "latest" heuristic worked
        # there but misfires on 10-Q.
        def _run_rowcount(run: list[int]) -> int:
            return sum(len(self.tables[ti].find_all("tr")) for ti in run)
        best = max(candidate_runs, key=_run_rowcount)
        return [self.tables[ti] for ti in best]

    def _caption_names_prior_period(self, table: Tag, cur_year: Optional[str]) -> bool:
        """Return True if the narrative immediately preceding `table`
        contains an "As of <Month> <day>, <year>" caption whose year is
        NOT `cur_year`. Used to filter out the prior-period comparative
        SOI that OBDC prints AFTER the current-period SOI.
        """
        if not cur_year:
            return False
        as_of_re = re.compile(
            r"as\s+of\s+[A-Z][a-z]+\s+\d{1,2}\s*,?\s+(20\d{2})", re.I
        )
        bare_date_re = re.compile(
            r"^\s*(?:January|February|March|April|May|June|July|August"
            r"|September|October|November|December)\s+\d{1,2}\s*,?\s+(20\d{2})\s*$",
            re.I,
        )
        prev = table.find_previous(["p", "div", "h1", "h2", "h3", "h4"])
        scanned = 0
        # Walk back up to 40 narrative blocks; stop if we cross another table.
        while prev is not None and scanned < 40:
            if prev.name == "table":
                break
            text = prev.get_text(" ", strip=True).replace("\xa0", " ")
            if text:
                m = as_of_re.search(text)
                if m:
                    return m.group(1) != cur_year
                m = bare_date_re.match(text)
                if m:
                    return m.group(1) != cur_year
            prev = prev.find_previous(
                ["p", "div", "h1", "h2", "h3", "h4", "table"]
            )
            scanned += 1
        return False

    def _table_preceded_by_slf_caption(self, table: Tag) -> bool:
        """Return True if any narrative paragraph between the previous
        recognized SOI table and this one contains 'Senior Loan Fund' or
        equivalent subsidiary-portfolio caption.

        Guard: skip text blobs longer than 400 chars — those are inline
        iXBRL table-content divs, not captions. A genuine SLF caption like
        'Blue Owl Capital Corporation Senior Loan Fund' is <80 chars.
        """
        # Walk backward through siblings until we hit another <table>; if
        # any text along the way matches _SLF_CAPTION_RE, this table is
        # part of an SLF / JV subsidiary section.
        prev = table.find_previous(["p", "div", "h1", "h2", "h3", "h4"])
        scanned = 0
        while prev is not None and scanned < 8:
            if prev.name == "table":
                break
            text = prev.get_text(" ", strip=True)
            # Skip long text blobs that are inline iXBRL content, not captions.
            if text and len(text) <= 400 and _SLF_CAPTION_RE.search(text):
                return True
            prev = prev.find_previous(["p", "div", "h1", "h2", "h3", "h4", "table"])
            scanned += 1
        return False

    def extract(self):
        # Patch parse_header_row to remap the "Cash" column to spread when
        # we see OBDC's "Ref. Rate / Cash / PIK" three-column pattern.
        from . import base as base_mod
        orig = base_mod.parse_header_row

        def patched(row):
            schema = orig(row)
            return self._obdc_remap(schema)

        base_mod.parse_header_row = patched
        try:
            result = self._extract_with_cross_table_banners()
        finally:
            base_mod.parse_header_row = orig

        # R8-D1 fix: 10-K affiliate backfill from subschedule tables.
        # In 10-K filings OBDC places affiliate info in a separate NOTE
        # SUBSCHEDULE (post-grand-total tables). The main SOI rows carry no
        # affiliate markers. We scan the subschedule for NCA / Controlled
        # section banners and back-annotate investments by name lookup.
        #
        # Determine if this is a 10-K period (affiliate banners absent from
        # main SOI). For 10-Qs, _extract_with_cross_table_banners() already
        # assigned categories; for 10-Ks it will leave everything as None /
        # Non-controlled/non-affiliated. We apply the subschedule backfill
        # if no affiliate rows were found via the main SOI walk.
        n_affiliate_from_main = sum(
            1 for inv in self.investments
            if inv.category in ("Non-controlled affiliate", "Controlled affiliate")
        )
        # explicit_not_pledged: investment object ids explicitly marked not-pledged
        # by the subschedule backfill.  `is_pledged` defaults to False so we cannot
        # use `inv.is_pledged is False` as a guard — we need an explicit set.
        explicit_not_pledged: set[int] = set()
        if n_affiliate_from_main == 0 and self.investments:
            # Likely a 10-K. Scan subschedule tables for affiliate data.
            nca_keys, ctrl_keys, not_pledged_keys = (
                self._obdc_build_affiliate_lookup()
            )
            if nca_keys or ctrl_keys:
                for inv in self.investments:
                    if not inv.company_name_raw:
                        continue
                    key = self._obdc_normalize_company_key(inv.company_name_raw)
                    if key in ctrl_keys:
                        inv.category = "Controlled affiliate"
                        flags = set(inv.footnote_flags)
                        flags.add("is_controlled_affiliate")
                        flags.discard("is_non_controlled_affiliate")
                        inv.footnote_flags = sorted(flags)
                        inv.is_controlled_affiliate = True
                        inv.is_non_controlled_affiliate = False
                    elif key in nca_keys:
                        inv.category = "Non-controlled affiliate"
                        flags = set(inv.footnote_flags)
                        flags.add("is_non_controlled_affiliate")
                        flags.discard("is_controlled_affiliate")
                        inv.footnote_flags = sorted(flags)
                        inv.is_non_controlled_affiliate = True
                        inv.is_controlled_affiliate = False
                    # Apply not-pledged from subschedule fn26 markers.
                    if not_pledged_keys and key in not_pledged_keys:
                        inv.is_pledged = False
                        inv.footnote_flags = [f for f in inv.footnote_flags
                                               if f != "is_pledged"]
                        explicit_not_pledged.add(id(inv))

        # R6-D5 fix: is_pledged inversion.
        # Footnote (25) = "Unless otherwise indicated, all portfolio companies
        # are pledged as collateral" — this is the DEFAULT, not a per-row flag.
        # Footnote (26) = "Investment is not pledged as collateral" — exception.
        # The base pipeline fires is_pledged=True on marker 25 (because the
        # text matches "pledged as collateral"), giving the wrong result.
        # Correct: ALL rows default True; marker 26 rows → False.
        #
        # Note: for 10-K periods the subschedule backfill above already set
        # is_pledged=False on rows in not_pledged_keys (tracked in
        # explicit_not_pledged by object id). Here we handle the 10-Q case
        # (marker '26' in cell text) and default all other rows to True.
        # Derive not-pledged markers from the legend rather than hardcoding
        # them. OBDC re-uses footnote slot numbers across filings: at
        # 2025-12-31 fn(9) was the not-pledged exception, but at
        # 2026-03-31 fn(9) is the 3-month SOFR rate footnote and fn(26)
        # is the only not-pledged exception. Hardcoded {"9","26"} thus
        # flipped 167 rows / $7.9B to unpledged at Q1 2026 (parser pledged
        # 41.2% vs disclosed 92.7%). Scan the legend text and trust the
        # `is_not_pledged` controlled-vocab classifier from
        # adapters/_shared.py.
        from ._shared import classify_footnote_meaning, clean_text
        not_pledged_markers: set[str] = set()
        for marker, meaning in (self.footnote_legend or {}).items():
            mflags = classify_footnote_meaning(
                clean_text(meaning), period_end=self.period_end
            )
            if "is_not_pledged" in mflags:
                not_pledged_markers.add(marker)
        for inv in self.investments:
            if id(inv) in explicit_not_pledged:
                continue
            if any(m in inv.footnote_markers for m in not_pledged_markers):
                inv.is_pledged = False
                inv.footnote_flags = [f for f in inv.footnote_flags
                                       if f != "is_pledged"]
            else:
                inv.is_pledged = True
                if "is_pledged" not in inv.footnote_flags:
                    inv.footnote_flags = sorted(inv.footnote_flags + ["is_pledged"])

        # D-6 fix (re-audit 2026-05-15): is_level_3 on fn(4) rows.
        # OBDC SOI legend (4) = "These investments were valued using
        # unobservable inputs and are considered Level 3 investments."
        # The shared classify_footnote_meaning regex doesn't recognize the
        # phrasing, so OBDC's legend-extraction layer never wires fn(4) →
        # is_level_3. Worse: the legend extractor itself drops the (4)
        # entry on this filing (filed as D-7 below — out-of-scope for the
        # OBDC adapter). For OBDC we deterministically fire is_level_3 on
        # any row carrying fn(4). 351 rows expected.
        for inv in self.investments:
            if "4" in (inv.footnote_markers or []):
                inv.is_level_3 = True
                if "is_level_3" not in inv.footnote_flags:
                    inv.footnote_flags = sorted(set(inv.footnote_flags) | {"is_level_3"})

        # D-1 fix (spot-obdc 2026-05-16): SOFR rate-attribute routing.
        # OBDC's SOI uses a "Ref. Rate / Cash / PIK" three-column layout.
        # `_obdc_remap` above relabels "Cash" → "Spread" so cash spreads in
        # the Cash column route to spread_bps. On some rows the filer
        # renders the row with the Cash column EMPTY and a lone rate token
        # in either the (relabeled-)Spread or PIK column slot — the HTM
        # cell shape ALONE cannot distinguish:
        #   (A) distressed PIK-only conversion (was cash+PIK, now all PIK),
        #       e.g. Walker Edison 1L, National Dentex 1L/DDTL 10–12% — iXBRL
        #       tags ONLY `InvestmentInterestRatePaidInKind` on the row's
        #       context (no `BasisSpread`); pre-e1d4581 routing → pik was
        #       CORRECT.
        #   (B) filer rendered Cash column blank for a normal cash-spread row,
        #       e.g. National Dentex Revolver $10,817K @ 9.00% — iXBRL tags
        #       ONLY `InvestmentBasisSpreadVariableRate` (no `IRPK`);
        #       pre-e1d4581 routing → pik was WRONG.
        # The prior commit (e1d4581) used a row-shape heuristic that flipped
        # all 22 such S+ rows to `spread_bps`, which was wrong for 21 of 22
        # (cross-source verification 2026-05-16, `audit/PERFECTION_NOTES.md`,
        # section `OBDC e1d4581 cross-source verification`).
        #
        # Replacement: drive PIK-vs-spread routing from XBRL fact tagging.
        # Build (company, type-stem, principal) → {has_basis_spread, has_irpk}
        # from the iXBRL XML, then for every investment row consult the map:
        #   - BasisSpread present, IRPK absent → ensure spread_bps populated
        #     (and pik_rate_pct cleared if the parser had routed the value
        #     to pik).
        #   - IRPK present, BasisSpread absent → ensure pik_rate_pct populated
        #     (and spread_bps cleared if the parser had routed the value to
        #     spread under the now-reverted e1d4581 heuristic).
        #   - Both present → split-coupon; leave the existing parser routing
        #     alone (Granicus DDT pattern: spread_bps=300, pik_rate_pct=2.00).
        #   - Neither present (rate-less context or no XBRL match) → no
        #     change.
        # Scope guard: only adjust rows where reference_rate_text_raw is a
        # variable-rate marker (S+, SA+, E+, P+, B+, etc.) AND exactly one
        # of spread_bps / pik_rate_pct currently holds the row's single rate
        # token AND cash_rate_pct is None. Rows with cash_rate_pct populated
        # have an explicit cash + PIK split and are left alone.
        self._obdc_xbrl_route_pik_vs_spread()

        # D-SPLITCOUPON-CASH (recon-obdc 2026-05-16): on split-coupon
        # floaters where the parser populated both `spread_bps` and
        # `pik_rate_pct`, also stamp `cash_rate_pct` with the cash
        # portion (= spread_bps / 100). OBDC's filer convention for
        # these cells is `<RefRate> X.XX% Y.YY% <maturity>` (e.g. raw
        # text `S+ 2.50% 2.25%`), where X.XX is the basis-spread cash
        # portion and Y.YY is the PIK accrual. The parser already routes
        # X.XX → `spread_bps` and Y.YY → `pik_rate_pct` via OBDC's
        # `_obdc_remap` cash-column relabel, but `cash_rate_pct` was
        # left null. Downstream cash-vs-PIK income decomposition cannot
        # distinguish "PIK-only distressed conversion" from "split
        # coupon with cash component" using only `pik_rate_pct` — it
        # needs an explicit `cash_rate_pct` signal on the split-coupon
        # rows. 21 rows are affected. XBRL contract (A1-A5) does not
        # constrain cash_rate_pct, so this is contract-safe. Cross-BDC
        # convention (OTF / OTIC / BCRED / BXSL split-coupon floaters)
        # populates BOTH `spread_bps` and `cash_rate_pct` on these rows;
        # OBDC was the outlier.
        self._obdc_stamp_cash_rate_on_split_coupons()

        # D-LEGEND-34 (cim-obdc 2026-05-16): rescue fn(34) BOCSO from
        # <ix:footnote> narrative. OBDC files fn(34) as an inline-XBRL
        # `<ix:footnote id="fn-33">` element rather than the standard
        # `(N) <definition>` legend-table paragraph the base extractor
        # walks. The visual `(34)` marker sits in a `<span>` sibling that
        # immediately precedes the `<ix:footnote>` element inside the same
        # `<div>`. We post-process the DOM, find every such (span, footnote)
        # pair in the document tail, and back-fill any `footnote_legend`
        # entry the base extractor missed.
        self._obdc_rescue_ix_footnote_legend()

        # D-SUBTOTAL-MISS (cim-obdc 2026-05-16): T84.R22 leaf subtotal for
        # the Human-resource-support-services sub-section (Sunshine Software)
        # is shaped `['N/A', '75,162', '66,872', '0.9', '%']` — first cell
        # 'N/A' (placeholder where Ref-Rate would normally be) prevents the
        # base subtotal classifier from matching. The row is then mis-
        # classified as `data`, but is correctly caught and merged into the
        # preceding Sunshine row by the OBDC pagination guard above (see
        # lines around "OBDC pagination guard"). The data path therefore
        # does NOT emit a phantom investment — but `section_totals[]` is
        # left short one leaf subtotal, which breaks per-industry tie-out
        # in downstream consumers. This adapter-local pass scans every SOI
        # row whose first non-empty cell is 'N/A' followed by 2+ monetary-
        # column numerics and a trailing '%' percent-of-net-assets cell,
        # and appends a SectionTotal record. The (table_idx, row_idx) pair
        # is uniquified against the existing section_totals so re-extracts
        # are idempotent.
        self._obdc_capture_na_led_subtotals()

        return result

    # ── Partial revert of e1d4581 — XBRL-driven PIK/spread routing ─────────
    #
    # Replaces the row-shape heuristic from e1d4581. The shape "S+ with a
    # lone rate token and no separate Cash cell" maps onto two genuinely
    # different economic states (distressed PIK-only conversion vs. filer
    # rendering omitting the Cash column on a normal spread row), and the
    # only signal that disambiguates them is the iXBRL fact the filer
    # attached to the row's `InvestmentIdentifierAxis` context:
    # `InvestmentBasisSpreadVariableRate` → cash spread, vs.
    # `InvestmentInterestRatePaidInKind` → PIK.
    #
    # Verified 2026-05-16 — see PERFECTION_NOTES section "OBDC e1d4581
    # cross-source verification". 21 of 22 e1d4581-flipped rows are PIK-only
    # (revert); 1 is a genuine cash-spread row (keep flipped). The 3 SA+/E+
    # Hg held-out rows are PIK-only in XBRL and stay PIK under the new logic.

    def _obdc_xbrl_load_rate_lookup(
        self,
    ) -> "dict[tuple, dict[str, object]]":
        """Parse the filing's iXBRL XML once and return a lookup keyed by
        (company_name, normalized_type, principal_int) →
            {has_basis_spread: bool, has_irpk: bool,
             basis_spread_value: float | None, irpk_value: float | None,
             context_id: str}.

        Type normalization: strip a trailing enumerator (" 1", " 2", …) that
        OBDC appends to the `InvestmentIdentifierAxis` domain string to
        disambiguate multiple lots of the same instrument. Principal is
        rounded to the nearest 1,000 (XBRL uses thousands-precision).
        Company identifier is the first pipe-separated segment of the axis
        domain; when that segment is the literal "Non-Affiliated", the
        company is read from the SECOND segment (OBDC's alternate context
        shape).
        """
        import os as _os
        import re as _re
        import xml.etree.ElementTree as ET

        # Discover the iXBRL XML alongside the source HTM. The base parser
        # exposes `self.html_path`; the XBRL file is the same stem with
        # `_htm.xml` instead of `.htm`.
        html_path = getattr(self, "html_path", None)
        if not html_path:
            return {}
        xml_path = _re.sub(r"\.html?$", "_htm.xml", html_path, flags=_re.I)
        if not _os.path.exists(xml_path):
            return {}

        NS_INSTANCE = "{http://www.xbrl.org/2003/instance}"
        NS_XBRLDI = "{http://xbrl.org/2006/xbrldi}"
        # The us-gaap namespace year may change in future filings; match by
        # local-name suffix only.
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            return {}

        # Build ctx → identifier string from typedMember domains under
        # us-gaap:InvestmentIdentifierAxis.
        ctx_to_ident: dict[str, str] = {}
        for ctx in root.findall(f"{NS_INSTANCE}context"):
            cid = ctx.get("id")
            if not cid:
                continue
            for tm in ctx.iter(f"{NS_XBRLDI}typedMember"):
                if tm.get("dimension") != "us-gaap:InvestmentIdentifierAxis":
                    continue
                for child in tm:
                    txt = (child.text or "").strip()
                    if txt:
                        ctx_to_ident[cid] = txt
                        break
                break

        # Collect rate / principal facts per context.
        WANT = {
            "InvestmentBasisSpreadVariableRate",
            "InvestmentInterestRatePaidInKind",
            "InvestmentOwnedBalancePrincipalAmount",
            "InvestmentOwnedAtFairValue",
        }
        facts_by_ctx: dict[str, dict[str, float]] = {}
        for fact in root.iter():
            tag = fact.tag
            if "}" not in tag:
                continue
            uri, local = tag.split("}", 1)
            if not uri.startswith("{http://fasb.org/us-gaap"):
                continue
            if local not in WANT:
                continue
            ctx = fact.get("contextRef")
            if not ctx or ctx not in ctx_to_ident:
                continue
            raw = (fact.text or "").strip()
            if not raw:
                continue
            try:
                v = float(raw)
            except ValueError:
                continue
            facts_by_ctx.setdefault(ctx, {})[local] = v

        # Build the (company, type-stem, principal) key.
        def _split_ident(ident: str) -> tuple[Optional[str], Optional[str]]:
            parts = [p.strip() for p in ident.split("|")]
            if not parts:
                return None, None
            company = parts[0]
            itype = parts[1] if len(parts) > 1 else None
            # Some OBDC contexts start with "Non-Affiliated | <company> | …".
            if company == "Non-Affiliated" and len(parts) > 2:
                company = parts[1]
                itype = parts[2] if len(parts) > 2 else None
            if itype:
                itype = re.sub(r"\s+\d+$", "", itype).strip()
            return company, itype

        lookup: dict[tuple, dict[str, object]] = {}
        for cid, ident in ctx_to_ident.items():
            f = facts_by_ctx.get(cid, {})
            if "InvestmentOwnedBalancePrincipalAmount" not in f:
                continue
            company, itype = _split_ident(ident)
            if not company:
                continue
            principal = int(round(f["InvestmentOwnedBalancePrincipalAmount"]))
            key = (company, itype, principal)
            bs = f.get("InvestmentBasisSpreadVariableRate")
            irpk = f.get("InvestmentInterestRatePaidInKind")
            entry = {
                "has_basis_spread": bs is not None,
                "has_irpk": irpk is not None,
                "basis_spread_value": bs,
                "irpk_value": irpk,
                "context_id": cid,
            }
            # Conflict policy: if the same key resolves to multiple
            # contexts, prefer one with rate facts present over one without;
            # if both have rate facts, the OR of signals is recorded (which
            # in OBDC's 2025-12-31 filing never occurs — 559 unique keys,
            # 0 cross-context fact conflicts at audit time).
            prior = lookup.get(key)
            if prior is None:
                lookup[key] = entry
            else:
                if (
                    not (prior["has_basis_spread"] or prior["has_irpk"])
                    and (entry["has_basis_spread"] or entry["has_irpk"])
                ):
                    lookup[key] = entry
                else:
                    prior["has_basis_spread"] = (
                        prior["has_basis_spread"] or entry["has_basis_spread"]
                    )
                    prior["has_irpk"] = prior["has_irpk"] or entry["has_irpk"]
                    if prior["basis_spread_value"] is None:
                        prior["basis_spread_value"] = entry["basis_spread_value"]
                    if prior["irpk_value"] is None:
                        prior["irpk_value"] = entry["irpk_value"]
        return lookup

    def _obdc_xbrl_route_pik_vs_spread(self) -> None:
        """Apply XBRL-fact-driven PIK / spread classification to every row
        where the parser carries exactly one of spread_bps / pik_rate_pct
        and no cash_rate_pct. This is the replacement for the e1d4581
        row-shape heuristic.
        """
        lookup = self._obdc_xbrl_load_rate_lookup()
        if not lookup:
            return

        # Variable-rate markers we trust XBRL to disambiguate. We deliberately
        # do NOT include fixed-rate rows here — they don't carry a
        # reference-rate prefix and the XBRL never tags BasisSpread on them.
        VARIABLE_RATE_MARKERS = {"S+", "SA+", "E+", "P+", "B+", "SN+", "SF+"}

        for inv in self.investments:
            if inv.cash_rate_pct is not None:
                continue  # explicit cash+PIK split — leave alone
            if inv.reference_rate_text_raw not in VARIABLE_RATE_MARKERS:
                continue
            if inv.par_principal is None:
                continue

            principal_int = int(round(inv.par_principal))
            key = (
                inv.company_name,
                inv.investment_type_raw,
                principal_int,
            )
            entry = lookup.get(key)
            if entry is None:
                continue
            has_bs = bool(entry["has_basis_spread"])
            has_irpk = bool(entry["has_irpk"])

            # Case A: XBRL says spread only.
            if has_bs and not has_irpk:
                if inv.spread_bps is None and inv.pik_rate_pct is not None:
                    inv.spread_bps = inv.pik_rate_pct * 100.0
                    inv.pik_rate_pct = None
                    if not inv.spread_text_raw:
                        inv.spread_text_raw = f"{inv.spread_bps / 100:.2f}%"
                # If spread_bps already set (parser routed it correctly via
                # the column remap), nothing to do.
                continue

            # Case B: XBRL says PIK only.
            if has_irpk and not has_bs:
                if inv.pik_rate_pct is None and inv.spread_bps is not None:
                    # The e1d4581 heuristic mis-routed the lone rate token
                    # into spread_bps; restore it to pik_rate_pct.
                    inv.pik_rate_pct = inv.spread_bps / 100.0
                    inv.spread_bps = None
                    inv.spread_text_raw = None
                continue

            # Case C: both present → split coupon; leave parser routing as-is.
            # Case D: neither present → no signal; leave parser routing as-is.

    def _obdc_stamp_cash_rate_on_split_coupons(self) -> None:
        """For every variable-rate row where the parser carries BOTH
        `spread_bps` AND `pik_rate_pct` (split-coupon floater) AND
        `cash_rate_pct is None`, set `cash_rate_pct = spread_bps / 100`.

        Scope guard: only operate on rows where the reference rate is a
        variable-rate marker (S+, SA+, E+, etc.). Fixed-rate cash+PIK
        rows (`reference_rate_text_raw` in ("N/A", None)) follow a
        different code path and are not in scope for this fix. Idempotent:
        re-extract calls leave already-populated `cash_rate_pct` values
        alone.
        """
        VARIABLE_RATE_MARKERS = {"S+", "SA+", "E+", "P+", "B+", "SN+", "SF+"}
        for inv in self.investments:
            if inv.reference_rate_text_raw not in VARIABLE_RATE_MARKERS:
                continue
            if inv.spread_bps is None or inv.pik_rate_pct is None:
                continue
            if inv.cash_rate_pct is not None:
                continue
            inv.cash_rate_pct = inv.spread_bps / 100.0

    # ── D-LEGEND-34: rescue (N) markers filed as <ix:footnote> narrative ──

    def _obdc_rescue_ix_footnote_legend(self) -> None:
        """Walk `<ix:footnote>` elements in the DOM and populate any
        `footnote_legend` entry missing from the base extractor.

        OBDC's 2025-12-31 filing renders fn(34) as:

            <div ...>
              <span ...>(34)</span>
              <ix:footnote id="fn-33" ...>Blue Owl Cross-Strategy
              Opportunities LLC ("BOCSO") was formed to hold alternative
              credit assets, including asset-based finance ("ABF"). …
              </ix:footnote>
            </div>

        The base `_extract_footnote_legend` walks narrative paragraphs
        outside any `<table>` and pairs `(N)` markers to definition
        sentences via a vocab regex. The BOCSO definition does not
        contain any of the vocab tokens (no "portfolio company", "loan",
        "investment" in the singular, etc.), so the (34) marker is
        dropped even though it sits in a narrative `<div>`. Rather than
        relax the shared vocab regex (out-of-scope), this adapter-local
        pass walks `<ix:footnote>` elements directly and back-fills any
        marker the base extractor missed.

        Matching rule: for each `<ix:footnote>` element, examine the
        immediately-preceding sibling `<span>` (or nested span text). If
        that sibling's text is a bare `(N)` marker, set
        `footnote_legend[N]` to the cleaned footnote text — but only if
        the key is not already populated (the base extractor's pick wins
        on conflicts; we only rescue gaps).
        """
        if not self.soup:
            return
        legend = getattr(self, "footnote_legend", None)
        if legend is None:
            # `footnote_legend` is set by `_extract_footnote_legend`.
            # If somehow absent, initialize so callers don't trip.
            self.footnote_legend = {}
            legend = self.footnote_legend

        # Bare-marker pattern: `(N)` or `(abc)` with optional whitespace.
        marker_re = re.compile(r"^\s*\(([0-9a-z]{1,3})\)\s*$", re.I)

        ix_footnotes = self.soup.find_all("ix:footnote")
        for fn in ix_footnotes:
            # Look at preceding siblings until we find a non-empty element
            # / text node. The OBDC pattern: span with marker text is the
            # immediately-preceding sibling element.
            marker: Optional[str] = None
            prev = fn.previous_sibling
            scanned = 0
            while prev is not None and scanned < 4:
                # NavigableStrings and empty-text spans are skipped.
                if hasattr(prev, "get_text"):
                    text = prev.get_text(" ", strip=True)
                else:
                    text = str(prev).strip()
                if text:
                    m = marker_re.match(text)
                    if m:
                        marker = m.group(1)
                    break
                prev = prev.previous_sibling
                scanned += 1
            if marker is None:
                continue

            # Already populated by base extractor? Don't overwrite.
            if marker in legend and legend[marker]:
                continue

            body = fn.get_text(" ", strip=True)
            # Normalize whitespace and curly-quotes/nbsp.
            body = body.replace("\xa0", " ")
            body = re.sub(r"\s+", " ", body).strip()
            if not body:
                continue
            legend[marker] = body

    # ── D-SUBTOTAL-MISS: capture leaf subtotal rows shaped 'N/A, $, $, %, %' ──

    def _obdc_capture_na_led_subtotals(self) -> None:
        """Capture leaf-subtotal rows the base classifier rejects because
        the row's first non-empty cell is the literal `'N/A'` placeholder
        rather than a bare numeric.

        Pattern (T84.R22 in the OBDC 2025-12-31 filing):

            ['N/A', '75,162', '66,872', '0.9', '%']

        The base `classify_row` requires every non-empty cell to live in
        a monetary column for the `subtotal` verdict; the `'N/A'` cell
        sits at the Ref-Rate logical position and fails that gate. The
        row is then re-classified as `data` and (correctly) absorbed by
        the OBDC pagination guard into the preceding Sunshine Software
        row. The investment list is therefore unchanged — but
        `section_totals[]` is left missing one leaf subtotal, which
        breaks per-industry tie-out in downstream auditors / dashboards.

        Detection rule (deliberately strict to avoid false positives):
          - Row lives inside a SOI table (returned by `find_soi_tables`).
          - First non-empty cell text is exactly `'N/A'` (after whitespace
            strip, case-sensitive — OBDC uses uppercase).
          - The remaining non-empty cells, in order, parse to:
              numeric, numeric, numeric (or em-dash placeholder), then the
              literal `'%'` percent-sign cell.
          - The schema must position the two numeric cells in the
            amortized-cost and fair-value logical columns, and the third
            numeric / em-dash cell in the pct-of-net-assets logical
            column.

        Idempotency: dedupe against existing `section_totals[]` by
        `(_source_table_index, _source_row_index)`.
        """
        if not self.soup:
            return
        try:
            soi_tables = self.find_soi_tables()
        except Exception:
            return
        if not soi_tables:
            return

        # Existing (table_idx, row_idx) pairs in section_totals — used to
        # avoid double-emit on re-extract.
        existing_keys = {
            (st._source_table_index, st._source_row_index)
            for st in self.section_totals
        }

        from ._shared import (
            get_cell_text_for_field,
            parse_header_row as _parse_header_row,
            is_header_row as _is_header_row,
        )

        for table in soi_tables:
            try:
                table_doc_idx = self.tables.index(table)
            except ValueError:
                continue
            schema: Optional[ColumnSchema] = None
            for r_idx, row in enumerate(table.find_all("tr")):
                if _is_header_row(row):
                    schema = self._obdc_remap(_parse_header_row(row))
                    continue
                if schema is None:
                    continue
                spans = self._row_spans(row)
                non_empty = [(s, e, t) for s, e, t in spans if t]
                if len(non_empty) < 3:
                    continue
                first_text = non_empty[0][2].strip()
                if first_text != "N/A":
                    continue

                # Identify cost / fv / pct cells by schema field mapping.
                cost_text = get_cell_text_for_field(non_empty, schema, "amortized_cost_text")
                fv_text = get_cell_text_for_field(non_empty, schema, "fair_value_text")
                pct_text = get_cell_text_for_field(non_empty, schema, "pct_net_assets_text")
                if cost_text is None or fv_text is None:
                    continue
                cost_val = parse_number(cost_text)
                fv_val = parse_number(fv_text)
                if cost_val is None or fv_val is None:
                    continue

                # Require a percent-of-net-assets-shaped trailing cell.
                # Either the schema maps a value into pct_net_assets_text
                # (which parses as a number) OR the last non-empty cell is
                # the literal '%' sign.
                pct_val = parse_percent(pct_text) if pct_text else None
                last_text = non_empty[-1][2].strip()
                if pct_val is None and last_text != "%":
                    continue

                key = (table_doc_idx, r_idx)
                if key in existing_keys:
                    continue

                tot = SectionTotal(
                    label=first_text,  # 'N/A' — best available leaf label
                    amortized_cost=cost_val * self.unit_multiplier,
                    fair_value=fv_val * self.unit_multiplier,
                    pct_net_assets=pct_val,
                    _source_table_index=table_doc_idx,
                    _source_row_index=r_idx,
                )
                self.section_totals.append(tot)
                existing_keys.add(key)


    def _detect_affiliate_banner(self, row: Tag) -> Optional[str]:
        """Return the canonical category name if this row is an OBDC
        affiliate-section banner, else None."""
        cells = row.find_all(["td", "th"])
        texts = [c.get_text(" ", strip=True) for c in cells]
        non_empty = [t for t in texts if t]
        if not non_empty:
            return None
        first = non_empty[0]
        m = _OBDC_AFFILIATE_BANNER_RE.match(first)
        if not m:
            return None
        low = first.lower()
        if "controlled" in low and "non" in low:
            return "Non-controlled affiliate"
        if "controlled" in low:
            return "Controlled affiliate"
        return None

    def _extract_with_cross_table_banners(self) -> list:
        """Full OBDC extract that carries affiliate-section-banner state
        across table boundaries.

        OBDC's SOI is split across ~20 small HTML tables. Affiliate banners
        ("Non-controlled/affiliated portfolio company investments") appear as
        the FINAL row of one table; data for that section starts in the NEXT
        table. The base pipeline resets state at each table boundary.

        This method re-implements the base extract() loop with one addition:
        a `pending_category` variable that is set when a banner row ends a
        table with no data rows following it, and is applied to the first
        data row(s) of the next table.
        """
        import re as _re
        from ._shared import (
            classify_row as _classify_row,
            is_header_row as _is_header_row,
            parse_header_row as _parse_header_row,
        )

        assert self.soup is not None, "call load() first"
        tables = self.find_soi_tables()
        if not tables:
            raise RuntimeError(f"{self.TICKER}: no SOI tables found")
        self._main_soi_tables = tables

        current_industry: Optional[str] = None
        current_company_raw: Optional[str] = None
        current_company: Optional[str] = None
        current_business_desc: Optional[str] = None
        current_link_id: Optional[str] = None
        schema: Optional[ColumnSchema] = None

        # R6-D1: persists across table boundaries. When the last row of a
        # table was a banner with no data rows following, this holds the
        # pending category for the NEXT table's first data row.
        # `pending_category` is set whenever a banner fires. It is cleared
        # only when we move to a NEW TABLE after seeing data rows.
        pending_category: Optional[str] = None
        # Current active category for data-row assignment.
        active_category: Optional[str] = None
        # Track whether this table has emitted any data rows AFTER the most
        # recent banner seen in this table.
        data_rows_after_last_banner = 0
        last_banner_table_idx = -1

        for t_idx, table in enumerate(tables):
            table_index_in_doc = self.tables.index(table)
            rows = table.find_all("tr")

            # Apply pending_category from previous table's trailing banner.
            # A pending_category is "consumed" (cleared) once it has been
            # applied to at least one data row in the current table.
            if pending_category is not None:
                active_category = pending_category
                # Don't clear yet — clear only after first data row.

            for r_idx, row in enumerate(rows):
                if is_header_row(row):
                    schema = self._obdc_remap(parse_header_row(row))
                    continue

                # R6-D1: check for affiliate banner BEFORE classifying as
                # a generic "industry" / "skip" row.
                banner_cat = self._detect_affiliate_banner(row)
                if banner_cat is not None:
                    active_category = banner_cat
                    pending_category = banner_cat  # carry forward in case no data follows
                    last_banner_table_idx = t_idx
                    data_rows_after_last_banner = 0  # reset
                    continue

                kind = _classify_row(row, schema) if schema else "skip"
                if kind in ("spacer", "skip"):
                    continue
                if kind == "header":
                    continue
                if kind == "industry":
                    spans = self._row_spans(row)
                    for _, _, t in spans:
                        if t:
                            current_industry = _re.sub(
                                r"\s*\((?:continued|cont(?:'d)?|cont\.)\)\s*$",
                                "",
                                t,
                                flags=_re.I,
                            ).strip()
                            break
                    current_company_raw = None
                    current_company = None
                    current_business_desc = None
                    current_link_id = None
                    continue
                if kind in ("total", "subtotal"):
                    self._capture_total(row, schema, table_index_in_doc, r_idx, kind)
                    continue

                # Data row.
                if schema is None:
                    continue
                inv_data = self._parse_data_row(
                    row, schema,
                    current_industry=current_industry,
                    current_company_raw=current_company_raw,
                    current_company=current_company,
                    current_business_desc=current_business_desc,
                )
                if inv_data is None:
                    continue

                data_rows_after_last_banner += 1

                # Once we see a data row in a DIFFERENT table than the banner,
                # the pending carry-forward has been consumed.
                if pending_category is not None and t_idx != last_banner_table_idx:
                    pending_category = None

                # Stamp active_category onto data row. We store it in
                # inv_data["category"] so _build_investment picks it up
                # and _assign_categories() (which runs after footnote
                # processing) can still override via footnote_flags if needed.
                if active_category is not None and inv_data.get("category") is None:
                    inv_data["category"] = active_category

                if inv_data.get("_starts_new_company"):
                    current_company_raw = inv_data["company_name_raw"]
                    current_company = inv_data["company_name"]
                    current_business_desc = inv_data.get("business_description")
                    current_link_id = self._make_row_id(
                        table_index_in_doc, r_idx, current_company_raw or ""
                    )

                inv_data["link_id"] = current_link_id

                # OBDC pagination guard: when the filer splits a single
                # investment onto two HTML rows for layout, the second
                # arrives here with no _starts_new_company, no
                # investment_type_raw, no footnote markers, no shares,
                # and cost/FV exactly duplicating the most recent emitted
                # investment. The two known patterns:
                #
                #   2025 10-K (Sunshine Software / Cornerstone OnDemand,
                #   T84 r22): R22 has par=NONE, cost+FV duplicate R21,
                #   adds pct_net_assets. Drift +$66.9M if not caught.
                #
                #   2023 10-K (Olaplex, Saphilux, CHA Holding, Safe
                #   Fleet, Aruba, 8 phantom tranches): continuation rows
                #   REPEAT par_principal alongside cost+FV. Drift
                #   +$172.9M if not caught.
                #
                # Allow the guard to fire when par_principal is either
                # absent OR equal to the previous row's par_principal.
                if (
                    self.investments
                    and not inv_data.get("_starts_new_company")
                    and not inv_data.get("investment_type_raw")
                    and not inv_data.get("footnote_markers")
                    and not inv_data.get("shares")
                    and inv_data.get("fair_value") is not None
                    and inv_data.get("amortized_cost") is not None
                ):
                    prev = self.investments[-1]
                    par_now = inv_data.get("par_principal")
                    par_matches = (
                        par_now is None
                        or par_now == prev.par_principal
                    )
                    if (
                        par_matches
                        and prev.fair_value == inv_data.get("fair_value")
                        and prev.amortized_cost == inv_data.get("amortized_cost")
                    ):
                        if inv_data.get("pct_net_assets") is not None and prev.pct_net_assets is None:
                            prev.pct_net_assets = inv_data.get("pct_net_assets")
                        continue

                inv = self._build_investment(
                    inv_data, table_idx=table_index_in_doc, row_idx=r_idx
                )
                if inv is not None:
                    self.investments.append(inv)

        # Parse footnote legend from document tail
        self._extract_footnote_legend()
        self._apply_footnotes()
        self._propagate_company_level_flags()

        # R6-D1: Banner-derived category was pre-seeded into inv.category
        # during the extract loop. _assign_categories() will override it
        # with footnote-flag-derived values where applicable (controlled /
        # non-controlled affiliate footnote markers win over banner state).
        # For rows where footnote_flags don't fire, the banner-derived
        # category set above is preserved because _assign_categories() only
        # assigns "Non-controlled/non-affiliated" as a fallback when neither
        # flag is set — but since we stored the banner cat in inv.category
        # before this call, we need to protect it.
        # Solution: for rows that already have a non-unaffiliated category
        # set from banners, pre-seed the matching footnote_flags so that
        # _assign_categories() confirms the banner decision.
        for inv in self.investments:
            if inv.category in ("Non-controlled affiliate", "Controlled affiliate"):
                flags = set(inv.footnote_flags)
                if inv.category == "Controlled affiliate":
                    flags.add("is_controlled_affiliate")
                    flags.discard("is_non_controlled_affiliate")
                else:
                    flags.add("is_non_controlled_affiliate")
                    flags.discard("is_controlled_affiliate")
                inv.footnote_flags = sorted(flags)

        self._assign_categories()
        self._ingest_unfunded_schedule()
        self._ingest_spv_subschedules()
        # Per-facility unfunded-commitment record extraction (separate
        # top-level array, structured per the schema in `_unfunded.py`).
        self._extract_unfunded_records_v2()
        for inv in self.investments:
            inv.finalize()
        return self.investments

    # ── R8-D1: subschedule affiliate backfill ───────────────────────────────

    @staticmethod
    def _obdc_normalize_company_key(raw: str) -> str:
        """Strip footnote markers and address suffix for company-name matching.

        Main SOI has:  'Walker Edison Furniture Company LLC(9)—1553 West...'
        Subschedule has: 'Walker Edison Furniture Company LLC(3)(4)(9)(22)(24)'
        Both normalize to: 'Walker Edison Furniture Company LLC'
        """
        # Strip all parenthesized footnote markers (numeric)
        name = re.sub(r"\(\d+\)", "", raw)
        # Strip address after em-dash or double-hyphen
        name = re.split(r"\s*—|\s*—|\s*--", name)[0]
        # Strip trailing asterisk (OBDC subschedule sometimes appends '*')
        name = name.rstrip("* ").strip()
        # Collapse internal whitespace
        name = re.sub(r"\s{2,}", " ", name).strip()
        return name

    def _obdc_build_affiliate_lookup(
        self,
    ) -> tuple[set[str], set[str], set[str]]:
        """Scan subschedule tables AFTER the main SOI grand-total row for
        OBDC 10-K affiliate and pledge data.

        Returns three sets of normalized company-name keys:
          nca_keys    — Non-controlled affiliate investments
          ctrl_keys   — Controlled affiliate investments
          not_pledged_keys — Investments with 'not pledged' marker (fn26)

        10-K structure: the main SOI (tables 30-48) does NOT carry affiliate
        markers in company-name cells. Instead, a separate subschedule note
        (tables 85-88 in the 2025-12-31 10-K, possibly different indices in
        other years) re-lists investments under section banners:
          "Non-controlled/affiliated portfolio company investments"
          "Controlled/affiliated portfolio company investments"

        These subschedule tables use the SAME cross-table-banner pattern as
        the main 10-Q SOI: the NCA banner appears as the LAST row of one
        table, and the actual NCA data starts at the BEGINNING of the NEXT
        table. We carry `pending_cat` across table boundaries identically to
        `_extract_with_cross_table_banners`.

        We stop scanning once we see a "Total Investments" grand-total row
        (which ends the current-year subschedule), to avoid ingesting the
        prior-year comparative subschedule that appears later.
        """
        soi_tables = self.find_soi_tables()
        if not soi_tables:
            return set(), set(), set()
        last_soi_doc_idx = self.tables.index(soi_tables[-1])

        nca_keys: set[str] = set()
        ctrl_keys: set[str] = set()
        not_pledged_keys: set[str] = set()

        # Subschedule grand-total: match "Total Investments" or "Total Portfolio
        # Investments" but NOT "Total Investment Income" / "Total Investment
        # Fee" etc. (singular-form P&L lines). The `s` on investments is
        # required to distinguish plural "investments" from singular "investment"
        # which appears in income-statement row labels.
        _GRAND_TOTAL_RE = re.compile(
            r"^total\s+(?:portfolio\s+)?investments\b"
            r"|^total\s+controlled/affiliated\s+portfolio\s+company\s+investments\b",
            re.I,
        )
        _NOT_PLEDGED_MARKER_RE = re.compile(r"\(26\)")

        # Only process tables that have an SOI-like column header (Company +
        # FairValue columns). This filters out balance-sheet, income-statement,
        # and other non-SOI tables that happen to contain rows matching the
        # affiliate-banner or grand-total regexes.
        from ._shared import is_header_row, normalize_label

        def _has_soi_subschedule_header(table: Tag) -> bool:
            for row in table.find_all("tr")[:5]:
                if not is_header_row(row):
                    continue
                labels = [normalize_label(l) for _, _, l in self._row_spans(row) if l]
                has_fv = any("fair value" in l for l in labels)
                has_company = any("company" in l for l in labels)
                if has_fv and has_company:
                    return True
            return False

        pending_cat: Optional[str] = None   # carries across table boundary
        active_cat: Optional[str] = None    # current assignment category
        seen_subschedule_grand_total = False
        # Track whether the previous table was a valid subschedule table.
        # pending_cat only carries over between adjacent subschedule tables.
        prev_was_subschedule = False

        for t_idx in range(last_soi_doc_idx + 1, len(self.tables)):
            if seen_subschedule_grand_total:
                break

            t = self.tables[t_idx]
            rows = t.find_all("tr")

            is_subschedule = _has_soi_subschedule_header(t)

            # At each subschedule-table boundary, apply any pending category
            # from the prior subschedule table's trailing banner.
            # If this table is NOT a subschedule table, pending_cat is only
            # carried if it was set by an ADJACENT subschedule table (to handle
            # the cross-table-banner pattern within the subschedule run).
            if pending_cat is not None and (is_subschedule or prev_was_subschedule):
                active_cat = pending_cat

            if not is_subschedule:
                # Non-subschedule table. Do NOT update active_cat or extract
                # company names. But keep pending_cat alive for one more table
                # (handles very narrow gap between subschedule tables).
                prev_was_subschedule = False
                continue

            prev_was_subschedule = True

            for row in rows:
                cells = row.find_all(["td", "th"])
                texts = [c.get_text(" ", strip=True) for c in cells]
                if not texts:
                    continue
                first = texts[0]

                # Grand-total → stop (end of current-year subschedule).
                if _GRAND_TOTAL_RE.match(first):
                    seen_subschedule_grand_total = True
                    break

                # Affiliate-section banner.
                m = _OBDC_AFFILIATE_BANNER_RE.match(first)
                if m:
                    low = first.lower()
                    if "controlled" in low and "non" in low:
                        new_cat = "NCA"
                    elif "controlled" in low:
                        new_cat = "Controlled"
                    else:
                        new_cat = None
                    if new_cat is not None:
                        active_cat = new_cat
                        pending_cat = new_cat
                    continue

                # Data row: must have a non-empty investment-type in column 2.
                if len(texts) <= 2 or not texts[2]:
                    continue
                # Skip total rows and subheadings that lack an investment type.
                if first.lower().startswith("total"):
                    continue
                # Skip rows that look like section/type headers
                if not first or not first[0].isupper():
                    continue

                # At this point we have a company data row.
                # Consume pending_cat once we reach the first data row.
                if pending_cat is not None:
                    active_cat = pending_cat
                    pending_cat = None

                if active_cat is None:
                    continue  # pre-banner (non-affiliated section — skip)

                # Normalize name and record affiliation.
                key = self._obdc_normalize_company_key(first)
                if not key or len(key) < 4:
                    continue

                if active_cat == "NCA":
                    nca_keys.add(key)
                elif active_cat == "Controlled":
                    ctrl_keys.add(key)

                # Check for "not pledged" (fn26) marker in this row's company cell.
                if _NOT_PLEDGED_MARKER_RE.search(first):
                    not_pledged_keys.add(key)

        return nca_keys, ctrl_keys, not_pledged_keys

    @staticmethod
    def _obdc_remap(schema: ColumnSchema) -> ColumnSchema:
        """If the schema has Ref. Rate immediately followed by a 'Cash'
        column (mapped to cash_rate_pct) and then 'PIK' (mapped to
        pik_rate_pct), relabel the 'Cash' column as 'Spread' so it routes
        to spread_text. This is OBDC's specific column-naming convention
        per InvestmentBasisSpreadVariableRate iXBRL tagging.
        """
        # Find sequential ref-rate / cash / pik
        spans = list(schema.spans)
        ref_idx = cash_idx = pik_idx = None
        for i, (s, e, lbl) in enumerate(spans):
            n = normalize_label(lbl)
            fld = LABEL_TO_FIELD.get(n)
            if fld == "reference_rate_text" and ref_idx is None:
                ref_idx = i
            elif fld == "cash_rate_pct" and ref_idx is not None and cash_idx is None and i > ref_idx:
                cash_idx = i
            elif fld == "pik_rate_pct" and cash_idx is not None and pik_idx is None and i > cash_idx:
                pik_idx = i
        if ref_idx is not None and cash_idx is not None and pik_idx is not None:
            # Confirmed OBDC pattern. Relabel cash_idx span as Spread.
            s, e, _ = spans[cash_idx]
            spans[cash_idx] = (s, e, "Spread")
        return ColumnSchema(spans=spans, total_logical=schema.total_logical)
