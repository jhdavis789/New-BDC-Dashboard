"""
ARCC V2 adapter.

ARCC's 10-K SOI quirks:

  * SOI is paginated across 60-80 `<table>` elements in the filing.
  * Every table repeats the same header row: first cell is 'Company (1)',
    then 'Business Description', 'Investment', 'Coupon (3)', 'Reference (7)',
    'Spread (3)', 'Acquisition Date', 'Maturity Date', 'Shares/Units',
    'Principal', 'Amortized Cost', 'Fair Value', '% of Net Assets'.
  * Each logical column consumes colspan=3 in the header row. Data rows
    respect the same 66-logical-column layout via their own colspans.
  * Footnote (4) marks non-controlled-affiliate ownership; (5) marks
    controlled-affiliate. (13) marks non-qualifying asset. These are all
    structurally decoded from the footnote legend at the end of the SOI —
    we don't hardcode them.
  * Ivy Hill Asset Management, L.P. and Senior Direct Lending Program, LLC
    are JVs.

R6-CP-D2 (ARCC variant): Some 10-K periods (e.g. 2024-12-31) embed the
SDLP (Senior Direct Lending Program) loan portfolio as full SOI-shaped
tables in the MD&A section BEFORE the main consolidated SOI. The tables
are preceded by the caption "SDLP Loan Portfolio as of <date>". The base
parser's grand-total break fires inside SDLP and stops before reaching
the main SOI. We skip these via a caption-preceded filter.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from bs4 import Tag

from .base import V2SOIParser, Investment
from ._shared import classify_footnote_meaning, clean_text


# ── D-10: regex captions identifying the two ARCC affiliate-cashflow
#    supplementary tables that sit between the SOI grand total and Note 6.
#    Each caption appears twice in the 10-K (once for 2025, once for the
#    2024 comparative period); we scope to the period_end year to pick
#    the right one.
_ARCC_NCA_CAPTION_RE = re.compile(
    r"in\s+which\s+the\s+issuer\s+was\s+an\s+Affiliated\s+Person\s+of\s+the\s+Company",
    re.I,
)
_ARCC_CA_CAPTION_RE = re.compile(
    r"in\s+which\s+the\s+issuer\s+was\s+both\s+an\s+Affiliated\s+Person\s+"
    r"and\s+a\s+portfolio\s+company",
    re.I,
)


_SDLP_CAPTION_RE = re.compile(
    r"sdlp\s+loan\s+portfolio|"
    r"\bSDLP\b|"
    r"senior\s+direct\s+lending\s+program",
    re.I,
)


# D-9 sanity: legal numeric/alpha footnote markers ARCC actually uses.
# (1)-(30) plus the single-letter (a)-(z). Anything else (US, USA,
# Holding, SaaS, W, E, CP, UK, MFA, Prism, USU, ALTI, ...) is an upstream
# artifact from a company-name fragment leaking into the marker bucket.
_VALID_ARCC_MARKER_RE = re.compile(r"^(?:\d{1,3}|[A-Za-z])$")


# D-3: ARCC-local equity-type vocabulary the base classifier misses.
# 38 rows / $2,366M FV were left investment_type=None pre-fix; all are
# limited-partner / membership / series-units instruments. Match is
# case-insensitive against investment_type_raw, with company_name as
# fallback for the few rows that leave investment_type_raw blank.
#
# Note: "preferred membership units" is intentionally NOT included here —
# D-11.PREF (below) re-classifies any /preferred/i raw type to Preferred,
# which has priority over the Equity catch-all.
_ARCC_EQUITY_RES = [
    re.compile(r"limited\s+partner(?:ship)?\s+interests?", re.I),
    re.compile(r"limited\s+partnership\s+units?", re.I),
    re.compile(r"partnership\s+units?", re.I),
    re.compile(r"member(?:ship)?\s+(?:interest|units?)", re.I),
    re.compile(r"common\s+member\s+units?", re.I),
    re.compile(r"series\s+[A-Z](?:[-\d]+)?\s+units?", re.I),
    re.compile(r"series\s+[A-Z](?:[-\d]+)?\s+(?:convertible\s+)?shares?", re.I),
    re.compile(r"class\s+[A-Z]\s+(?:interest|units?)", re.I),
    re.compile(r"class\s+[A-Z]\s+limited\s+liability\s+company\s+interest", re.I),
    re.compile(r"loan\s+instrument\s+units?", re.I),
    re.compile(r"participation\s+rights?", re.I),
    re.compile(r"company\s+units?", re.I),
]


def _arcc_equity_match(text: Optional[str]) -> bool:
    if not text:
        return False
    return any(p.search(text) for p in _ARCC_EQUITY_RES)


# D-11.PREF: Preferred-equity classifier. The disclosed Portfolio
# Composition table (p.74 of the 2025 10-K) buckets $2,475M of FV as
# "Preferred equity" but the parser pre-fix bucket sums to $2,368.8M — a
# -$106.2M gap, offset by an equal +$105.9M over-allocation to "Other
# equity". Two structural causes were identified by the 2026-05-17
# breakdown reconciliation (`audit/_tmp/arcc_recon_2025-12-31.py`):
#
#   1. Raw types containing the word "preferred" that the ARCC equity
#      catch-all (`_ARCC_EQUITY_RES`, originally including
#      `preferred\s+membership\s+units?`) routed to Equity before the base
#      `_PREF_RE` could fire. Affects $3.2M FV (Pyramid-BMC).
#   2. Equity-type raw strings (Class A units / Series A units / etc.)
#      that carry a PIK coupon rate. These are economically preferred —
#      ARCC's 10-K "Preferred equity" bucket aggregates them as such even
#      though their SOI instrument label is plain "Class A units".
#      Affects ~$128M FV across ~10 rows.
#
# D-11.SUB: Senior-subordinated classifier. The disclosed bucket has
# $1,585M, parser Mezz-ex-SDLP has $1,545.2M — a -$39.8M gap, offset by
# +$39.6M over-allocation to First-lien. The likely cause is first-lien
# last-out / unitranche tranches that ARCC files in the SOI as "First lien
# senior secured loan" but rolls to "Senior subordinated" in the
# disclosed Portfolio Composition table. For ARCC 2025-12-31 NO 1L row's
# `investment_type_raw` (or `company_name`) actually contains
# /mezzanine|subordinated|last.out|second.out/ at the instrument level
# (the only company-name hit, "Diamond Mezzanine 24 LLC", is a fund-name
# false positive). The regex below is implemented defensively for future
# periods or for adapter-side promotion of unitranche-like rows, but on
# 2025-12-31 it is a no-op. The residual $40M crossover is therefore
# documented but not adapter-resolvable from the SOI text alone.
_ARCC_PREFERRED_RE = re.compile(r"preferred", re.I)
_ARCC_SUB_RE = re.compile(
    r"mezzanine|subordinated|last[-\s]?out|second[-\s]?out", re.I
)


class ARCCParser(V2SOIParser):
    TICKER = "ARCC"
    UNIT_MULTIPLIER = 1_000_000  # ARCC reports in $ millions
    UNIT_DETECTION = True

    # D-10: per-affiliate-company supplementary-cashflow tables captured
    # from the post-SOI / pre-Note-6 section. Populated by
    # `_arcc_extract_affiliate_activity` during extract().
    affiliate_company_activity: list[dict]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.affiliate_company_activity = []

    def extract(self) -> list[Investment]:
        """Run the base pipeline, then apply ARCC-specific post-fixes."""
        invs = super().extract()
        self._arcc_strip_stray_markers()
        self._arcc_propagate_affiliation()
        self._arcc_classify_equity_types()
        # D-11.PREF / D-11.SUB must run AFTER `_arcc_classify_equity_types`
        # so we can re-route any of the catch-all-Equity rows whose raw
        # text actually identifies them as preferred or subordinated.
        self._arcc_classify_preferred()
        self._arcc_classify_subordinated()
        self._arcc_set_floor_sentinel()
        self._arcc_rebuild_footnote_legend()
        self._arcc_extract_affiliate_activity()
        # Re-derive boolean projections after we mutated flags/markers.
        for inv in invs:
            inv.finalize()
        # Sidecar: persist affiliate activity alongside the main JSON so
        # downstream consumers can see it without changes to run_adapter.
        self._arcc_write_affiliate_sidecar()
        return invs

    # ── D-3: ARCC equity-type classifier ────────────────────────────────
    def _arcc_classify_equity_types(self) -> None:
        """For rows the base parser left `investment_type=None`, match
        ARCC's equity vocabulary (limited-partner / membership / series-
        units / class-interests / participation-rights / loan-instrument-
        units / convertible-shares) and set `investment_type='Equity'`.

        Source field priority: `investment_type_raw`, then `company_name`
        (some equity stubs leave investment_type_raw blank with the
        instrument name embedded in the company column)."""
        for inv in self.investments:
            if inv.investment_type is not None:
                continue
            if _arcc_equity_match(inv.investment_type_raw) or \
               _arcc_equity_match(inv.company_name):
                inv.investment_type = "Equity"
                if inv.investment_category is None:
                    inv.investment_category = "Equity"

    # ── D-11.PREF: rescue Preferred-equity rows mis-bucketed as Equity ──
    def _arcc_classify_preferred(self) -> None:
        """Re-classify any row whose `investment_type_raw` contains
        `preferred` (case-insensitive) AND was bucketed by the base or
        ARCC-local classifier as `Equity` or `Warrant` to `Preferred`.

        Also re-classify Equity rows that carry a PIK coupon rate: ARCC's
        disclosed Portfolio Composition table aggregates equity-form
        instruments with PIK coupons (e.g. "Class A units 8.00% PIK") into
        the "Preferred equity" bucket even though the SOI instrument label
        is not literally "preferred". A non-zero `pik_rate_pct` on an
        Equity row is the discriminator — common-equity rows carry no
        coupon at all.

        Disclosed target: Preferred equity bucket $2,475M (10-K p.74,
        Portfolio Composition table).
        Pre-fix parser:   $2,368.8M  (gap -$106.2M, offset by Other equity
                                      +$105.9M).

        Read-only side effect on `investment_type` field; `investment_type_raw`
        is preserved untouched."""
        for inv in self.investments:
            raw = inv.investment_type_raw or ""
            cur = inv.investment_type
            if cur not in ("Equity", "Warrant"):
                continue
            is_pref_raw = bool(_ARCC_PREFERRED_RE.search(raw))
            has_pik = bool(inv.pik_rate_pct)
            if is_pref_raw or has_pik:
                inv.investment_type = "Preferred"

    # ── D-11.SUB: rescue Senior-subordinated rows mis-bucketed as 1L ────
    def _arcc_classify_subordinated(self) -> None:
        """Re-classify 1L rows whose `investment_type_raw` carries an
        explicit subordinated/last-out/second-out token to `Mezz`.

        Disclosed target: Senior subordinated loans $1,585M.
        Pre-fix parser:   $1,545.2M (Mezz ex-SDLP). Gap -$39.8M offset by
                          First lien +$39.6M.

        For ARCC 2025-12-31 NO row's raw type matches this pattern — the
        $40M crossover is structurally caused by ARCC's filing convention
        of labeling first-lien-last-out / unitranche tranches as plain
        "First lien senior secured loan" in the SOI while rolling them to
        "Senior subordinated" in the disclosed Portfolio Composition
        table. The rule is implemented defensively here in case a future
        period's SOI uses more discriminating raw text. The match
        explicitly skips the `company_name` field because "Diamond
        Mezzanine 24 LLC" (a fund issuer name) is a false-positive
        match — only the instrument-level `investment_type_raw` is
        consulted."""
        for inv in self.investments:
            if inv.investment_type != "1L":
                continue
            raw = inv.investment_type_raw or ""
            if _ARCC_SUB_RE.search(raw):
                inv.investment_type = "Mezz"

    # ── D-1: company-level affiliation propagation via (4) / (5) ────────
    def _arcc_propagate_affiliation(self) -> None:
        """ARCC discloses affiliation via a per-company footnote marker on
        the FIRST tranche only. fn(4) → Non-controlled Affiliated Person
        (5%-25% ownership). fn(5) → both Affiliated Person AND Control
        (>25% ownership / control).

        The base parser correctly extracts (4)/(5) on the row that carries
        them, but the legend-derived marker→flag mapping in `_apply_footnotes`
        doesn't fire because ARCC's footnote legend is cross-period
        contaminated (D-2, orchestrator scope) — (4)/(5) text comes from
        the MD&A SDLP table, not the 2025 SOI definitions. So we patch
        on raw markers instead.

        Company-grouping key: union over `link_id` AND canonical
        `company_name` — any tranche whose either key matches a row that
        carried the marker gets the flag and category.

        Targets (from audit): NCA $600M / 17 companies; CA $4,013M /
        19 companies. Observed after fix: $600.2M / 16 companies and
        $4,013.4M / 18 companies — exact $-match to the disclosed
        per-bucket subtotal, with the one-company gap reflecting two
        affiliate companies grouped under the same primary issuer in the
        SOI (parser's company-name canonicalization)."""
        company_has_4: set[str] = set()
        company_has_5: set[str] = set()
        link_has_4: set[str] = set()
        link_has_5: set[str] = set()
        for inv in self.investments:
            markers = inv.footnote_markers or []
            name = (inv.company_name or "").strip()
            link = inv.link_id or ""
            if "4" in markers:
                if name:
                    company_has_4.add(name)
                if link:
                    link_has_4.add(link)
            if "5" in markers:
                if name:
                    company_has_5.add(name)
                if link:
                    link_has_5.add(link)
        for inv in self.investments:
            name = (inv.company_name or "").strip()
            link = inv.link_id or ""
            in_4 = (name and name in company_has_4) or (link and link in link_has_4)
            in_5 = (name and name in company_has_5) or (link and link in link_has_5)
            if not (in_4 or in_5):
                continue
            flags = set(inv.footnote_flags or [])
            # Controlled wins if both fire (parity with base RC-I logic).
            if in_5:
                flags.add("is_controlled_affiliate")
                flags.discard("is_non_controlled_affiliate")
                inv.category = "Controlled affiliate"
            else:
                flags.add("is_non_controlled_affiliate")
                inv.category = "Non-controlled affiliate"
            inv.footnote_flags = sorted(flags)

    # ── D-4: fn(9) floor sentinel ───────────────────────────────────────
    def _arcc_set_floor_sentinel(self) -> None:
        """Marker (9) in the ARCC SOI denotes "interest rate floor
        feature" but ARCC does not disclose the floor value numerically.
        Use `floor_pct = 0.0` as a sentinel meaning "has floor" so the
        signal isn't silently dropped. Don't overwrite an existing
        numeric value (defensive — ARCC never sets one, but be safe)."""
        for inv in self.investments:
            markers = inv.footnote_markers or []
            if "9" in markers and inv.floor_pct is None:
                inv.floor_pct = 0.0

    # ── D-10: per-affiliate cashflow supplementary tables ───────────────
    def _arcc_extract_affiliate_activity(self) -> None:
        """ARCC appends two supplementary tables to the SOI (between
        "Total investments" and Note 6) that disclose per-affiliate-company
        cashflow detail: Purchases / Redemptions / Sales / Interest /
        Capital-structuring / Dividend / Other-income / Realized G/L /
        Unrealized G/L / ending Fair Value.

        Table fn(4): Affiliated Person — non-controlled affiliates (NCA).
        Table fn(5): Affiliated Person and Control — controlled (CA).

        Each caption appears twice in the 10-K (current period + prior-year
        comparative). We scope to `self.period_end`'s year so we only
        capture the current-period table.

        Cross-validates the D-1 affiliation propagation: ending_fv sums
        for NCA = $600M; CA = $4,013M per the 2025-12-31 disclosure.

        Populates `self.affiliate_company_activity` with dicts shaped:
            {company, affiliation_class ("NCA"|"CA"),
             beginning_fv (None — not disclosed by ARCC),
             gross_additions, gross_reductions, sales_cost,
             interest_income, cap_struct_fees, dividend_income,
             other_income, realized_gl, unrealized_gl, ending_fv}
        """
        if self.soup is None:
            return
        year = (self.period_end or "")[:4]
        if not year:
            return
        for klass, pattern in (
            ("NCA", _ARCC_NCA_CAPTION_RE),
            ("CA", _ARCC_CA_CAPTION_RE),
        ):
            div = self._arcc_find_supplementary_caption(pattern, year)
            if div is None:
                continue
            tbl = div.find_next("table")
            if tbl is None:
                continue
            rows = self._arcc_parse_affiliate_table(tbl, klass)
            self.affiliate_company_activity.extend(rows)

    def _arcc_find_supplementary_caption(self, pattern: re.Pattern, year: str):
        """Find the narrative `<div>` whose text contains the supplementary-
        table caption AND the target year. ARCC's caption text is a single
        sentence ~500 chars long; we constrain on text-length to avoid
        matching the outer body wrapper. We pick the first match (current
        period appears in document order before the prior-year copy)."""
        assert self.soup is not None
        for div in self.soup.find_all("div"):
            t = div.get_text(" ", strip=True)
            if len(t) > 800:
                continue
            if not pattern.search(t):
                continue
            if year not in t:
                continue
            return div
        return None

    @staticmethod
    def _arcc_parse_amount(s: str) -> float:
        """Parse a numeric cell from the supplementary table. Em-dash and
        empty cells → 0.0. Parenthesized values → negative. Stray currency
        symbols are stripped."""
        s = (s or "").strip()
        if s in ("", "—", "-", "$"):
            return 0.0
        neg = False
        if s.startswith("(") and s.endswith(")"):
            neg = True
            s = s[1:-1].strip()
        s = s.replace(",", "").replace("$", "").strip()
        try:
            v = float(s)
            return -v if neg else v
        except ValueError:
            return 0.0

    def _arcc_parse_affiliate_table(
        self, tbl: Tag, klass: str
    ) -> list[dict]:
        """Walk rows of one supplementary table; emit one dict per company.

        Skip:
          * Empty rows (no non-empty cells).
          * Period-header rows ("For the Year Ended...", "As of...").
          * Column-header rows (start with "(in millions) Company").
          * The bottom totals row (first non-empty cell is the bare "$"
            currency anchor).

        Each data row contains 10 numeric amounts in fixed column order.
        Extracted as the LAST 10 numerics in the row (the leading cells
        are colspan-paded for the company-name column)."""
        out: list[dict] = []
        for r in tbl.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", c.get_text(" ", strip=True)).strip()
                for c in r.find_all(["td", "th"])
            ]
            nonempty = [c for c in cells if c]
            if len(nonempty) < 2:
                continue
            first = nonempty[0]
            if first == "$":
                # bottom totals row
                continue
            if first.lower().startswith("for the year") or first.lower().startswith("as of"):
                continue
            if "in millions" in first.lower() and "company" in first.lower():
                continue
            nums: list[float] = []
            for c in cells:
                c2 = c.strip()
                if c2 in ("", "$"):
                    continue
                nums.append(self._arcc_parse_amount(c2))
            if len(nums) < 10:
                continue
            vals = nums[-10:]
            out.append({
                "company": first,
                "affiliation_class": klass,
                # ARCC's supplementary table does not disclose beginning_fv
                # as a column; populate None to keep the schema honest.
                "beginning_fv": None,
                "gross_additions": vals[0],
                "gross_reductions": vals[1],
                "sales_cost": vals[2],
                "interest_income": vals[3],
                "cap_struct_fees": vals[4],
                "dividend_income": vals[5],
                "other_income": vals[6],
                "realized_gl": vals[7],
                "unrealized_gl": vals[8],
                "ending_fv": vals[9],
            })
        return out

    def _arcc_write_affiliate_sidecar(self) -> None:
        """Persist the affiliate-cashflow table to a JSON sidecar next to
        the main parser output. Path:
        `<HERE>/out/v2_raw/ARCC/ARCC_<period>_affiliate_activity.json`
        Silently no-op if the output dir doesn't exist or the list is
        empty (e.g. the period predates the supplementary tables)."""
        if not self.affiliate_company_activity:
            return
        # adapters/arcc.py → PARSER_V2/
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(here, "out", "v2_raw", "ARCC")
        if not os.path.isdir(out_dir):
            return
        out_path = os.path.join(
            out_dir, f"ARCC_{self.period_end}_affiliate_activity.json"
        )
        nca = [r for r in self.affiliate_company_activity if r["affiliation_class"] == "NCA"]
        ca = [r for r in self.affiliate_company_activity if r["affiliation_class"] == "CA"]
        payload = {
            "ticker": "ARCC",
            "period_end": self.period_end,
            "n_rows": len(self.affiliate_company_activity),
            "n_nca": len(nca),
            "n_ca": len(ca),
            "nca_ending_fv_total": round(sum(r["ending_fv"] for r in nca), 4),
            "ca_ending_fv_total": round(sum(r["ending_fv"] for r in ca), 4),
            "rows": self.affiliate_company_activity,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

    # ── D-2: period-scoped footnote legend ──────────────────────────────
    def _arcc_rebuild_footnote_legend(self) -> None:
        """Re-extract `footnote_legend` from the current-period SOI legend
        block ONLY.

        Defect: ARCC 10-Ks always embed the prior-year comparative SOI
        (with its own full legend block, re-using markers 1, 10, 13, 14,
        17, 18, etc.) immediately after the current-period SOI. Some of
        the prior-year legend texts don't carry a year token, and others
        do but tie-break the same as the current-period definition, so
        the base extractor's `_more_definitional` selection lands on the
        wrong period for ~6 markers per filing. The legend extractor
        accepts paragraphs from the entire post-SOI narrative window,
        which spans both periods plus Note 4 ("The amortized cost
        represents...") and other downstream sections that share the same
        `(N)` marker syntax.

        Fix: walk the same narrative stream but stop at the first
        prior-period `CONSOLIDATED SCHEDULE OF INVESTMENTS` heading or
        any out-of-period-year SOI section break. This gives us a clean
        single-period window — the legend authored for `self.period_end`
        only.

        Strictly ARCC-local: replaces `self.footnote_legend` only when we
        can identify a valid period-scoped window AND the rebuilt legend
        passes a non-regression guard (no fewer keys than the base, and
        no key's meaning replaced by something that loses a controlled-
        vocab flag the base entry produced)."""
        if self.soup is None:
            return
        soi_tables = self.find_soi_tables()
        if not soi_tables:
            return
        period_year = (self.period_end or "")[:4]
        if not period_year:
            return

        body = self.soup.find("body") or self.soup
        all_strings = list(body.find_all(string=True))
        seq = {id(s): i for i, s in enumerate(all_strings)}

        first_soi = soi_tables[0]
        last_soi = soi_tables[-1]
        first_seqs = [seq[id(s)] for s in first_soi.find_all(string=True) if id(s) in seq]
        last_seqs = [seq[id(s)] for s in last_soi.find_all(string=True) if id(s) in seq]
        start_seq = min(first_seqs) if first_seqs else 0
        after_last_soi_seq = (max(last_seqs) + 1) if last_seqs else start_seq

        # Compose the next-period SOI banner pattern. ARCC's comparative
        # SOI is preceded by a "CONSOLIDATED SCHEDULE OF INVESTMENTS"
        # heading followed (after up to ~6 nodes) by "As of December
        # <day>, <year>". The prior-period year is period_year-1. Stop at
        # ANY out-of-period As-of-date heading.
        try:
            py = int(period_year)
        except ValueError:
            return
        # Build a regex for "As of <Month> <day>, <year>" where year != period_year.
        # Most filings use December 31; allow Mar/Jun/Sep for 10-Q periods.
        out_of_period_as_of_re = re.compile(
            r"^\s*As\s+of\s+(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+\d{1,2},?\s+"
            r"(\d{4})\b",
            re.I,
        )
        section_heading_re = re.compile(
            r"^(Item\s+\d+[A-Z]?\.|PART\s+[IVX]+\b|Exhibit\s+\d+)",
            re.I,
        )
        # Walk forward from after_last_soi_seq, looking for either:
        #   (a) the "CONSOLIDATED SCHEDULE OF INVESTMENTS" banner — its
        #       subsequent "As of" node tells us which year follows;
        #   (b) a generic Item/PART/Exhibit section heading.
        end_seq = len(all_strings)
        i = after_last_soi_seq + 1
        while i < len(all_strings):
            s = all_strings[i]
            if s.find_parent("table") is None:
                txt = str(s).replace("\xa0", " ").strip()
                # Generic section heading: stop unconditionally.
                if section_heading_re.match(txt):
                    end_seq = i
                    break
                # SOI banner: look ahead a few nodes for an "As of <date>,
                # YYYY". If YYYY != period_year, this is the comparative
                # SOI — stop at the BANNER, not the "As of" line.
                if "CONSOLIDATED SCHEDULE OF INVESTMENTS" in txt.upper():
                    found_year = None
                    look = i + 1
                    look_end = min(i + 25, len(all_strings))
                    while look < look_end:
                        s2 = all_strings[look]
                        if s2.find_parent("table") is None:
                            t2 = str(s2).replace("\xa0", " ").strip()
                            mm = out_of_period_as_of_re.match(t2)
                            if mm:
                                found_year = mm.group(1)
                                break
                        look += 1
                    if found_year and found_year != period_year:
                        end_seq = i
                        break
                # Direct "As of <Month> <day>, YYYY" with non-period year
                # — also a hard stop (some filings drop the banner).
                mm = out_of_period_as_of_re.match(txt)
                if mm and mm.group(1) != period_year:
                    end_seq = i
                    break
            i += 1

        # Build narrative paragraphs in [start_seq, end_seq).
        parts: list[str] = []
        for j in range(start_seq, end_seq):
            s = all_strings[j]
            if s.find_parent("table") is not None:
                continue
            txt = str(s).replace("\xa0", " ")
            if not txt.strip():
                continue
            parts.append(txt.strip())
            parts.append("\n")
        narrative = " ".join(parts)
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in narrative.split("\n")]
        paragraphs = [p for p in paragraphs if p]
        if not paragraphs:
            return

        # Replicated marker / vocab regexes from base._extract_footnote_legend.
        marker_inline_re = re.compile(
            r"^(?:\(([0-9a-z]{1,3})\)|(\d{1,3})\.|([a-z]{1,3})\.)\s+(\S.*)"
        )
        marker_only_re = re.compile(
            r"^(?:\(([0-9a-z]{1,3})\)|(\d{1,3})\.|([a-z]{1,3})\.)$"
        )
        legend_vocab_re = re.compile(
            r"\b(portfolio\s+company|loan|investment|pledged|qualifying\s+asset|"
            r"non[-\s]?accrual|interest\s+rate|variable\s+rate|fixed\s+rate|"
            r"affiliated\s+person|Section\s+55|SOFR|LIBOR|delayed[-\s]?draw|"
            r"revolving|letter\s+of\s+credit|senior\s+secured|unfunded|"
            r"first\s+lien|second\s+lien|unsecured|subordinated|"
            r"Investment\s+Company\s+Act|as\s+defined|"
            r"security|securities|held\s+within|collateraliz|"
            r"non[-\s]?income|level\s*[123]|fair\s+value\s+hierarch|"
            r"unobservable|cost\s+represents|original\s+cost|"
            r"interest\s+rate\s+floor|exempt\s+from\s+registration|"
            r"securities\s+act\s+of\s+1933|restricted\s+(?:security|securities)|"
            r"unsettled|debt\s+securitization|credit\s+facility|"
            r"clo\b|joint\s+venture|securitization|"
            r"subsidiary|cash\s+equivalent|money\s+market|"
            r"amortiz|premium|discount|warrant|preferred|common\s+(stock|equity)|"
            # ARCC-specific legend vocabulary not covered above. The 2025
            # SOI legend uses each of these phrases in at least one entry:
            #   fn(1)  "does not 'Control'" / "Investment Company Act"
            #   fn(10) "excess cash flow from the SDLP"
            #   fn(13) "commitments to fund various revolving and delayed
            #          draw senior secured and subordinated loans"
            #   fn(14) "agreements to fund equity investments"
            #   fn(15) "commitments to co-invest in the SDLP"
            #   fn(16) "unobservable inputs that are significant"
            #   fn(17) "estimated net unrealized gain for federal tax
            #          purposes"
            r"excess\s+cash\s+flow|commitments\s+to\s+(?:fund|co-invest)|"
            r"does\s+not\s+.?control|"
            r"agreements\s+to\s+fund|fund\s+equity|equity\s+investment|"
            r"net\s+unrealized|federal\s+tax|tax\s+purposes|"
            # ARCC fn(7) "Reference" rate vocabulary.
            r"reference\s+rate|adjustment|base\s+rate|stated\s+spread)\b",
            re.I,
        )
        DEFINITIONAL_LEAD_RE = re.compile(
            r"^\s*(?:loan|investment|position|asset|securities?)\s+(?:was|is|are)\b"
            r"|^\s*non[-\s]?accrual\b"
            r"|^\s*classified\s+as\b"
            r"|^\s*denotes\b"
            r"|^\s*all\s+or\s+a\s+portion\b"
            r"|^\s*these\s+investments\b",
            re.I,
        )
        cur_year_re = re.compile(rf"\b{period_year}\b")
        prior_year_re = re.compile(rf"\b{py - 1}\b")

        def _cites_current(s: str) -> bool:
            return bool(cur_year_re.search(s))

        def _cites_prior_only(s: str) -> bool:
            return bool(prior_year_re.search(s) and not cur_year_re.search(s))

        def _produces_flag(s: str) -> bool:
            return bool(classify_footnote_meaning(
                clean_text(s), period_end=self.period_end
            ))

        def _more_definitional(new: str, existing: str) -> bool:
            new_def = bool(DEFINITIONAL_LEAD_RE.match(new))
            old_def = bool(DEFINITIONAL_LEAD_RE.match(existing))
            new_current = _cites_current(new)
            old_current = _cites_current(existing)
            new_prior = _cites_prior_only(new)
            old_prior = _cites_prior_only(existing)
            new_flagged = _produces_flag(new)
            old_flagged = _produces_flag(existing)
            if new_flagged and not old_flagged:
                return True
            if old_flagged and not new_flagged:
                return False
            if new_def and not old_def:
                return True
            if not new_def and old_def:
                return False
            if new_current and old_prior:
                return True
            if old_current and new_prior:
                return False
            if new_current and not old_current:
                return True
            if old_current and not new_current:
                return False
            return len(new) < 400 and len(existing) > 400

        legend: dict[str, str] = {}
        for i_para, p in enumerate(paragraphs):
            marker: Optional[str] = None
            meaning: Optional[str] = None
            m = marker_inline_re.match(p)
            if m:
                marker = m.group(1) or m.group(2) or m.group(3)
                meaning = m.group(4).strip()
            else:
                m2 = marker_only_re.match(p)
                if m2 and i_para + 1 < len(paragraphs):
                    marker = m2.group(1) or m2.group(2) or m2.group(3)
                    meaning = paragraphs[i_para + 1].strip()
            if not marker or not meaning:
                continue
            if len(meaning) < 15:
                continue
            if not legend_vocab_re.search(meaning):
                continue
            if len(meaning) > 1000:
                cut = meaning.rfind(". ", 0, 1000)
                meaning = meaning[: cut + 1] if cut > 200 else meaning[:1000]
            if marker in legend:
                if _more_definitional(meaning, legend[marker]):
                    legend[marker] = meaning
                continue
            legend[marker] = meaning

        # Coverage guard: keep a base-extracted marker only if (a) the
        # period-scoped scan didn't see it AND (b) the base meaning isn't
        # demonstrably from the prior-period block. Markers like (18)
        # don't exist in the 2025 legend at all — the base picked them up
        # from the 2024 comparative SOI block. Dropping those is the
        # correct period-scoped behavior. We deliberately do NOT keep
        # base on flag-loss: the entire point of this rebuild is that
        # base entries for cross-period markers were classified into
        # flags that belong to the WRONG period (e.g. base fn(1) flagged
        # is_pik from "amortized cost represents..."); those flags were
        # the regression, not a property to preserve.
        # Compute the highest numeric marker the period-scoped scan saw.
        # Any base-only marker with a strictly higher number is necessarily
        # from the prior-period block (the prior-period legend recycles 1..N
        # but extends further when the prior filing had more disclosures).
        max_numeric = 0
        for k in legend.keys():
            try:
                v = int(k)
                if v > max_numeric:
                    max_numeric = v
            except ValueError:
                pass
        base_legend = dict(self.footnote_legend)
        for marker, base_meaning in base_legend.items():
            if marker in legend:
                continue
            if _cites_prior_only(base_meaning):
                # Base meaning is unambiguously the prior-period entry —
                # drop it rather than carry the wrong text forward.
                continue
            try:
                mnum = int(marker)
            except ValueError:
                mnum = None
            if mnum is not None and max_numeric and mnum > max_numeric:
                # Out-of-range numeric marker: the current-period legend
                # doesn't define this number, so the base entry came from
                # the prior-period block. Drop it.
                continue
            legend[marker] = base_meaning

        self.footnote_legend = legend
        # Re-run the marker→flag application now that the legend is
        # period-correct. _apply_footnotes uses self.footnote_legend to
        # decode markers on every investment row.
        self._apply_footnotes()

    # ── D-9: stray-marker cleanup ───────────────────────────────────────
    def _arcc_strip_stray_markers(self) -> None:
        """Remove footnote_markers entries that aren't valid ARCC tokens.
        Valid: numeric 1-3 digits, or single ASCII letter. Anything else
        (US, USA, Holding, SaaS, W, E, CP, UK, MFA, Prism, USU, ALTI, ...)
        is a name-fragment leak from upstream stripping."""
        for inv in self.investments:
            if not inv.footnote_markers:
                continue
            cleaned = [m for m in inv.footnote_markers
                       if _VALID_ARCC_MARKER_RE.match(m or "")]
            if len(cleaned) != len(inv.footnote_markers):
                inv.footnote_markers = cleaned

    def find_soi_tables(self) -> list[Tag]:
        """Skip tables preceded by an "SDLP Loan Portfolio" caption — those
        are the JV sub-portfolio, not the main consolidated SOI."""
        all_soi = super().find_soi_tables()
        out = []
        skipped_any = False
        for table in all_soi:
            if self._table_preceded_by_sdlp_caption(table):
                skipped_any = True
                continue
            out.append(table)
        # If we skipped some (i.e., this is a 10-K with the SDLP detour),
        # the base loop may have stopped at the SDLP "Total" row before
        # reaching main SOI. Re-scan from scratch with SDLP filter applied.
        if skipped_any and len(out) < 5:
            return self._rescan_skipping_sdlp()
        return out

    def _rescan_skipping_sdlp(self) -> list[Tag]:
        """Re-walk ALL tables, skip SDLP, and stop at the main-SOI grand
        total — bypasses the base parser's early-exit at the SDLP
        sub-schedule's own grand-total row."""
        from ._shared import is_header_row, normalize_label
        out = []
        seen_main_total = False
        last_accepted_idx = -1

        def _is_soi(table: Tag) -> bool:
            rows = table.find_all("tr")
            for row in rows[:8]:
                if not is_header_row(row):
                    continue
                labels = [normalize_label(t) for _, _, t in self._row_spans(row) if t]
                has_fv = any("fair value" in l or "market value" in l for l in labels)
                has_cost = any(
                    l == "cost" or "amortized cost" in l or "amortizedcost" in l
                    for l in labels
                )
                has_specific = any(
                    "maturity" in l or "principal" in l or "shares" in l
                    or "spread" in l or "coupon" in l or "reference" in l
                    for l in labels
                )
                if has_fv and has_cost and has_specific:
                    return True
            return False

        for ti, table in enumerate(self.tables):
            if seen_main_total:
                break
            if not _is_soi(table):
                continue
            if self._table_preceded_by_sdlp_caption(table):
                continue
            if self._table_preceding_caption_mentions_prior_period(table):
                continue
            out.append(table)
            last_accepted_idx = ti
            for row in table.find_all("tr"):
                texts = [t for _, _, t in self._row_spans(row) if t]
                if texts and re.match(r"^total\s+investments?\b", texts[0], re.I):
                    seen_main_total = True
                    break
        return out

    def _table_preceded_by_sdlp_caption(self, table: Tag) -> bool:
        """Return True if any narrative paragraph in the 8 elements
        preceding this table contains an SDLP / Senior Direct Lending
        Program caption."""
        prev = table.find_previous(["p", "div", "h1", "h2", "h3", "h4"])
        scanned = 0
        while prev is not None and scanned < 8:
            if prev.name == "table":
                break
            text = prev.get_text(" ", strip=True)
            if text and len(text) <= 400 and _SDLP_CAPTION_RE.search(text):
                return True
            prev = prev.find_previous(["p", "div", "h1", "h2", "h3", "h4", "table"])
            scanned += 1
        return False
