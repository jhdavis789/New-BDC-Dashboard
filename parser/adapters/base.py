"""
V2 parser base class. Structural-only — no positional hardcoding.

Every adapter subclasses `V2SOIParser` and implements, at minimum:
    * `TICKER`
    * (optional) structural dialect overrides

The base pipeline:
    1. load HTML → BeautifulSoup
    2. locate SOI tables structurally (tables whose first non-empty header
       row matches a Company/Issuer/Portfolio-company label AND contains
       a Fair Value column header)
    3. within each table, parse the header row into a ColumnSchema (no
       cell-index assumption)
    4. classify each row by its relation to the schema and its non-empty
       content distribution
    5. emit Investment records with every field populated from the schema
    6. detect SPV sub-schedules as distinct table groups (configurable
       detector, but the default is: SOI header whose caption text names a
       subsidiary/SPV)
    7. parse post-SOI footnote legend → {marker: meaning} + controlled vocab
       flags; attach to each investment's `footnotes`
    8. emit unfunded-commitment rows as separate `Investment` records of
       `investment_category='Unfunded Commitment'` linked to parent via
       `link_id`
    9. capture grand-total and subtotal values from 'total' rows for later
       reconciliation against XBRL
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from bs4 import BeautifulSoup, Tag

from ._shared import (
    ColumnSchema,
    classify_row,
    classify_footnote_meaning,
    clean_text,
    currency_of_reference_rate,
    extract_footnote_markers,
    get_cell_text_for_field,
    infer_currency_from_cell,
    is_header_row,
    is_total_like_text,
    normalize_label,
    parse_basis_points,
    parse_date,
    parse_footnote_legend,
    parse_header_row,
    parse_money_number,
    parse_number,
    parse_percent,
    parse_rate_cell,
    split_ref_and_spread,
    strip_footnotes,
    tokenize_reference_rate,
    _RATE_RESET_CODES,
)


# ── Investment record ───────────────────────────────────────────────────────


@dataclass
class Investment:
    """V2 investment record. All numeric amounts in raw USD (not $K or $M).
    Unit scaling is applied inside the adapter based on the filing's units."""

    # Lineage
    row_id: str
    ticker: str
    period_end: str
    accession: Optional[str] = None
    parent_entity: Optional[str] = None  # For SPV child rows, the SPV name.
    parent_row_id: Optional[str] = None
    link_id: Optional[str] = None        # Groups funded + unfunded sibling rows.

    # Identification
    company_name_raw: str = ""
    company_name: str = ""   # footnote-stripped, address-stripped
    # Canonical borrower (portfolio company) for parent-rollup. When the
    # filer writes "<sub-entity>, LLC (<Parent Co.>)", the underlying credit
    # exposure is to <Parent Co.>. Adapters populate this in finalize() when
    # they recognise the dialect; otherwise it stays None and consumers fall
    # back to company_name. Used by the dashboard to count non-accrual
    # borrowers without double-counting sub-entities of the same parent.
    canonical_borrower_name: Optional[str] = None
    industry: Optional[str] = None
    category: Optional[str] = None   # Non-controlled/non-affiliated, etc.
    business_description: Optional[str] = None

    # Classification
    investment_type_raw: Optional[str] = None
    investment_type: Optional[str] = None     # normalized: 1L, 2L, Mezz, Unsec, Equity, Preferred, Warrant, JV, Revolver, DDT
    investment_category: Optional[str] = None  # Debt, Equity, Unfunded Commitment, Derivative

    # Rates
    reference_rate: Optional[str] = None       # SOFR, EURIBOR, Prime, Fixed
    reference_rate_text_raw: Optional[str] = None
    spread_bps: Optional[float] = None
    spread_text_raw: Optional[str] = None
    cash_rate_pct: Optional[float] = None
    pik_rate_pct: Optional[float] = None
    pik_rate_max_pct: Optional[float] = None
    pik_type: Optional[str] = None             # "toggle" | "mandatory" | "accrued"
    floor_pct: Optional[float] = None

    # Dates
    maturity_date: Optional[str] = None
    acquisition_date: Optional[str] = None

    # Amounts
    par_principal: Optional[float] = None
    amortized_cost: Optional[float] = None
    fair_value: Optional[float] = None
    shares: Optional[float] = None
    pct_net_assets: Optional[float] = None
    unfunded_amount: Optional[float] = None   # Non-null for unfunded sibling rows

    # Currency
    currency: str = "USD"

    # Geography (ISO 3166 alpha-2 country code + macro region)
    # Populated in finalize() from currency + company-name suffix heuristics.
    # Used for US-vs-international portfolio analytics.
    country: Optional[str] = None              # "US", "GB", "AU", "CA", ...
    geographic_region: Optional[str] = None    # "Americas", "EMEA", "APAC", "Other"

    # Footnotes
    footnote_markers: list[str] = field(default_factory=list)
    footnote_meanings: dict[str, str] = field(default_factory=dict)
    footnote_flags: list[str] = field(default_factory=list)   # controlled vocab

    # Convenience boolean projections of footnote_flags (populated at emit).
    is_non_accrual: bool = False
    is_non_qualifying_asset: bool = False
    is_qualifying_asset: bool = False
    is_level_3: bool = False
    is_unfunded_commitment: bool = False
    is_controlled_affiliate: bool = False
    is_non_controlled_affiliate: bool = False
    is_pledged: bool = False
    # Per-facility pledged tracking. When the filing distinguishes multiple
    # secured-debt credit facilities (GBDC 7 facilities via pledge symbols;
    # FSK 9 facilities via fn(f/i/j/k/l/m/p/s/t/u/v); HTGC 2031 ABL Notes;
    # GCRED 4 facilities via */^/&/#), this list names each facility this
    # specific row is pledged to. Empty list means either not pledged or
    # the filer doesn't disclose per-facility granularity (ARCC single-pool
    # fn(2), OBDC single-pool fn(25) — for these is_pledged carries the
    # full signal).
    pledged_facilities: list[str] = field(default_factory=list)
    # Cash-and-cash-equivalents row (BXSL and other filers list these IN
    # the SOI as separate rows outside the investment portfolio total).
    is_cash_equivalent: bool = False

    # Parser debug lineage — which table row produced this record
    _source_table_index: Optional[int] = None
    _source_row_index: Optional[int] = None

    def finalize(self) -> None:
        """Populate derived boolean flags from footnote_flags list."""
        flags = set(self.footnote_flags)
        self.is_non_accrual = "is_non_accrual" in flags
        self.is_non_qualifying_asset = "is_non_qualifying_asset" in flags
        self.is_qualifying_asset = "is_qualifying_asset" in flags
        self.is_level_3 = "is_level_3" in flags
        self.is_unfunded_commitment = "is_unfunded_commitment" in flags
        self.is_controlled_affiliate = "is_controlled_affiliate" in flags
        self.is_non_controlled_affiliate = "is_non_controlled_affiliate" in flags
        self.is_pledged = "is_pledged" in flags
        # R6-D9 / Fix-2: is_unfunded_commitment must also be set when the
        # investment_category is already 'Unfunded Commitment' (set by the
        # base parser from investment_type_raw containing "unfunded") or when
        # investment_type_raw itself mentions "unfunded".  The footnote-flag
        # path only fires when a legend entry explicitly describes commitments,
        # but many filers (ASIF, MFIC, NMFC, OTF) put "Unfunded commitment"
        # directly in the investment_type_raw column.
        import re as _re
        if not self.is_unfunded_commitment:
            if self.investment_category == "Unfunded Commitment":
                self.is_unfunded_commitment = True
            elif self.investment_type_raw and _re.search(
                r"\bunfunded\b", self.investment_type_raw, _re.I
            ):
                self.is_unfunded_commitment = True
            elif self.company_name and _re.search(
                r"\bunfunded\b", self.company_name, _re.I
            ):
                # FSK-style: "Unfunded Loan Commitments" / "Unfunded Debt
                # Commitments" appear as aggregate adjustment rows with no
                # investment_type_raw populated.
                self.is_unfunded_commitment = True
        # R6-D9 / Fleet-wide structural rule (CHECK-12 fix): a row with
        # non-positive fair_value AND non-positive amortized_cost on a
        # NON-equity instrument is, by economic identity, an unfunded
        # commitment carrying negative OID amortization.  A drawn debt
        # position cannot have FV <= 0 when amortized_cost <= 0 — there is
        # nothing on the books being marked down.  This catches the
        # ANTARES / GBDC / GSBD / TCPC / BCSF / CCAP / MSDL / NMFC / SPCC
        # / BDEBT / FSK / MFIC / MAIN / OCREDIT pattern where filers list
        # unfunded revolvers/DDTs inline within the main SOI body without
        # an explicit "Unfunded" label or per-row legend marker.
        # Both FV<0 (negative OID accrual marked at negative valuation)
        # AND FV=0 with a populated par + non-positive cost (unfunded
        # commitment with em-dash in the FV column, e.g. OCREDIT's
        # Mantech r11/r12) are economic equivalents.
        # Excludes equity (FV markdowns below cost are normal there).
        if (not self.is_unfunded_commitment
                and self.fair_value is not None
                and self.fair_value <= 0
                and self.investment_category not in (
                    "Equity", "Cash Equivalent", "Derivative")
                and self.investment_type not in (
                    "Equity", "Preferred", "Warrant", "JV")
                and self.amortized_cost is not None
                and self.amortized_cost <= 0
                # Only fire when BOTH FV and cost are non-positive AND
                # at least one is strictly negative (a true zero/zero row
                # is usually a placeholder section heading, not an
                # unfunded commitment).
                and (self.fair_value < 0 or self.amortized_cost < 0)):
            # Guard against legitimate derivatives (e.g. "Total return swap").
            inv_raw_lc = (self.investment_type_raw or "").lower()
            if not _re.search(
                r"\b(swap|forward contract|cdo|cdx|hedge contract|notional|"
                r"futures? contract|option contract|credit default)\b",
                inv_raw_lc,
            ):
                self.is_unfunded_commitment = True
                if self.investment_category in (None, "Debt"):
                    self.investment_category = "Unfunded Commitment"
        # Companion rule: revolver/delayed-draw/undrawn instrument with
        # negative fair_value (regardless of amortized_cost sign) is
        # always an unfunded commitment.
        if (not self.is_unfunded_commitment
                and self.fair_value is not None
                and self.fair_value < 0
                and self.investment_type_raw
                and _re.search(
                    r"\b(revolver|revolving|delayed[-\s]?draw|undrawn|"
                    r"\bDDT\b|delayed\s+funding)\b",
                    self.investment_type_raw, _re.I)):
            self.is_unfunded_commitment = True
            if self.investment_category in (None, "Debt"):
                self.investment_category = "Unfunded Commitment"
        # All-zero placeholder rule: a debt-classified row with par == 0,
        # cost == 0, AND fv == 0 has no economic substance except as a
        # placeholder for an unfunded commitment. Caught the OSCF Q1 2026
        # US WorldMeds Ventures all-dash rows (footnote 9 = "Investment
        # has undrawn commitments") that the negative-OID economic-identity
        # detector skips because zero/zero/zero rows are also typical of
        # section-heading placeholders. Restrict to (a) debt category and
        # (b) at least one non-(5)(8) footnote marker — section-heading
        # placeholders carry no per-row footnotes.
        if (not self.is_unfunded_commitment
                and self.fair_value == 0
                and self.amortized_cost == 0
                and (self.par_principal == 0 or self.par_principal is None)
                and self.investment_category in (None, "Debt")
                and self.investment_type not in (
                    "Equity", "Preferred", "Warrant", "JV")
                and self.footnote_markers
                and any(m not in ("5", "8") for m in self.footnote_markers)):
            inv_raw_lc = (self.investment_type_raw or "").lower()
            if not _re.search(
                r"\b(swap|forward contract|cdo|cdx|hedge contract|notional|"
                r"futures? contract|option contract|credit default)\b",
                inv_raw_lc,
            ):
                self.is_unfunded_commitment = True
                if self.investment_category in (None, "Debt"):
                    self.investment_category = "Unfunded Commitment"
        # Cash-equivalent detection: company name contains "Cash" and
        # "Equivalents" / "Equivalent" or is a treasury / money-market
        # instrument listed as a non-loan.
        name = (self.company_name or "") + " " + (self.investment_type_raw or "")
        if _re.search(r"\bcash\s+(and\s+)?(cash\s+)?equivalents?\b", name, _re.I):
            # Subtotal guard: rows like "Cash and Cash Equivalents and
            # Restricted Cash (5.7% of net assets)" or "Total Cash and Cash
            # Equivalents" are SECTION TOTALS, not individual MMF instruments.
            # Mark them as Subtotal so they don't double-count.
            if _re.search(
                r"%\s+of\s+net\s+assets|^\s*total\s+(cash|investments|portfolio)\b",
                (self.company_name or ""), _re.I,
            ):
                self.is_cash_equivalent = False
                self.investment_category = "Subtotal"
            else:
                self.is_cash_equivalent = True
        # Money-market funds and treasury MMFs sit in cash-equivalent
        # subsections in the SOI (GBDC's BlackRock T-Fund, BXSL's State
        # Street Inst MMF / BlackRock ICS US Treasury Fund). The base
        # category-setter below treats these as Equity by default because
        # they have no investment_type_raw. Identify by name.
        elif _re.search(
            r"\b(money\s+market\s+(fund|portfolio)|treasury\s+(fund|portfolio)|"
            r"institutional\s+(?:u\.?s\.?\s+)?(?:gov(?:ernmen)?t|treasury)\s+(?:mmf|"
            r"money\s+market)|government\s+money\s+market|"
            r"\bMMF\b|institutional\s+liquidity|ICS\s+us\s+treasury|"
            # Broader MMF family — OSCF has BNY Mellon Short Term Investment
            # Fund + Goldman Sachs FS Treasury Obligations + Dreyfus Treasury
            # Cash Management in investments[] without is_cash_equivalent fire.
            r"short[-\s]term\s+investment\s+fund|treasury\s+obligations\s+fund|"
            r"treasury\s+cash\s+management|cash\s+management\s+fund|"
            r"liquid\s+reserves?|liquid\s+assets?\s+fund|treasury\s+plus\s+fund|"
            r"government\s+(?:obligations|securities)\s+fund|"
            r"\bGSAM\b\s+treasury|\bFS\s+treasury\b)",
            (self.company_name or ""), _re.I,
        ):
            # Exclude section-subtotal rows: a row whose name contains
            # "(X% of net assets)" or starts with "Total Cash and Cash
            # Equivalents" is a section subtotal, NOT an individual MMF
            # instrument. OSCF "Cash and Cash Equivalents and Restricted
            # Cash (5.7% of net assets)" — drop from investments to avoid
            # double-counting with the 4 individual MMF rows.
            if _re.search(
                r"%\s+of\s+net\s+assets|^\s*total\s+(cash|investments|portfolio)\b",
                (self.company_name or ""), _re.I,
            ):
                # Mark as subtotal: keep is_cash_equivalent=False so the
                # downstream filter rejects it from row sums. The caller
                # can choose to move it to section_totals separately.
                self.is_cash_equivalent = False
                self.investment_category = "Subtotal"
            else:
                self.is_cash_equivalent = True
        if self.investment_category is None:
            if self.is_cash_equivalent:
                self.investment_category = "Cash Equivalent"
            elif self.investment_type in ("Equity", "Preferred", "Warrant", "JV"):
                self.investment_category = "Equity"
            elif self.unfunded_amount is not None and not (
                # Funded portion present → it's a debt instrument with
                # a partial unfunded commitment, not a pure unfunded
                # commitment line item. OTF audit caught 126 such rows
                # (~$8.9B funded FV) wrongly tagged Unfunded.
                (self.par_principal is not None and self.par_principal > 0)
                or (self.fair_value is not None and self.fair_value > 0)
            ):
                self.investment_category = "Unfunded Commitment"
            else:
                self.investment_category = "Debt"
        # is_qualifying_asset = complement of is_non_qualifying_asset for
        # any real (non-cash-equivalent) investment carrying actual value.
        # Section 55(a) of the 1940 Act defines the qualifying-asset bucket;
        # a BDC's non-qualifying investments are the explicit exception, so
        # everything that isn't NQA and isn't a cash-equivalent line is by
        # construction a qualifying asset. Cross-BDC pattern #4 in audit.
        if (not self.is_qualifying_asset
                and not self.is_non_qualifying_asset
                and not self.is_cash_equivalent
                and self.investment_category not in ("Cash Equivalent", "Derivative")
                and (
                    (self.fair_value is not None and self.fair_value != 0)
                    or (self.amortized_cost is not None and self.amortized_cost != 0)
                    or (self.par_principal is not None and self.par_principal != 0)
                )):
            self.is_qualifying_asset = True

        # ── Geographic inference: country (ISO 3166 alpha-2) + region ────────
        # Two signals: (1) company-name suffix / corporate-form token, which is
        # the STRONGER signal (a Pty Limited issuer is Australian even if the
        # bond happens to be USD-denominated), and (2) currency, which is a
        # fallback when no name signal is present.
        if self.country is None:
            name = self.company_name or self.company_name_raw or ""
            # Name-suffix patterns — checked in priority order (most specific
            # first). Parenthesised geographic suffix beats corporate-form
            # tokens because filers use "(Australia)" / "(UK)" as deliberate
            # disambiguation when the corporate form is ambiguous.
            country: Optional[str] = None
            paren_map = {
                r"\((?:USA|U\.S\.|U\.S\.A\.|United\s+States)\)": "US",
                r"\((?:UK|U\.K\.|United\s+Kingdom|England|Scotland)\)": "GB",
                r"\(Australia\)": "AU",
                r"\(Canada\)": "CA",
                r"\(Cayman(?:\s+Islands)?\)": "KY",
                r"\(Bermuda\)": "BM",
                r"\(Ireland\)": "IE",
                r"\(Luxembourg\)": "LU",
                r"\(Netherlands\)": "NL",
                r"\(Germany\)": "DE",
                r"\(France\)": "FR",
                r"\(Jersey\)": "JE",
                r"\(Guernsey\)": "GG",
                r"\(Switzerland\)": "CH",
                r"\(Japan\)": "JP",
                r"\(Singapore\)": "SG",
                r"\(Hong\s*Kong\)": "HK",
                r"\(New\s*Zealand\)": "NZ",
                r"\(India\)": "IN",
                r"\(Brazil\)": "BR",
                r"\(Mexico\)": "MX",
                r"\(Spain\)": "ES",
                r"\(Italy\)": "IT",
                r"\(Sweden\)": "SE",
                r"\(Norway\)": "NO",
                r"\(Denmark\)": "DK",
                r"\(Finland\)": "FI",
            }
            for pat, cc in paren_map.items():
                if _re.search(pat, name, _re.I):
                    country = cc
                    break
            if country is None:
                # Corporate-form tokens. These are weaker than parenthesised
                # geographic suffixes because forms like "S.A." are shared
                # across many Romance-language jurisdictions; we only assign
                # FR for S.A. when no stronger signal exists, and we prefer
                # currency over form when both disagree.
                form_patterns = [
                    (r"\bPty\.?\s+(?:Ltd|Limited)\b", "AU"),
                    (r"\bProprietary\s+Limited\b", "AU"),
                    (r"\bB\.?V\.?\b", "NL"),
                    (r"\bGmbH\b", "DE"),
                    (r"\bAG\b(?:\s|$|,)", "DE"),
                    (r"\bS\.?p\.?A\.?\b", "IT"),
                    (r"\bS\.?L\.?\b(?:\s|$|,)", "ES"),
                    (r"\bAB\b(?:\s|$|,)", "SE"),
                    (r"\bA[/-]?S\b(?:\s|$|,)", "DK"),
                    (r"\bASA\b", "NO"),
                    (r"\bOy\b", "FI"),
                    (r"\bUnlimited\s+Company\b", "IE"),
                    (r"\bDAC\b\b", "IE"),
                    (r"\bplc\b", "GB"),
                    (r"\bPLC\b", "GB"),
                    (r"\bS\.?A\.?R\.?L\.?\b", "LU"),
                    (r"\bS\.?\s*A\.?\b(?:\s|$|,)", "FR"),
                    (r"\bSAS\b", "FR"),
                    (r"\bK\.?K\.?\b(?:\s|$|,)", "JP"),
                    (r"\bPte\.?\s+Ltd\.?\b", "SG"),
                ]
                for pat, cc in form_patterns:
                    if _re.search(pat, name, _re.I):
                        country = cc
                        break
                # Direct country-name tokens (when corporate form fails).
                # Catches "SumUp Holdings Luxembourg" → LU, etc.
                if country is None:
                    name_country = [
                        (r"\bLuxembourg\b", "LU"),
                        (r"\bNetherlands\b", "NL"),
                        (r"\bDeutschland\b", "DE"),
                        (r"\bSverige\b", "SE"),
                        (r"\bDanmark\b", "DK"),
                    ]
                    for pat, cc in name_country:
                        if _re.search(pat, name, _re.I):
                            country = cc
                            break
            if country is None:
                # Currency fallback. Most rows are USD → US. Non-USD currencies
                # map to their natural home country.
                cur_map = {
                    "USD": "US",
                    "EUR": "EU",  # generic eurozone — region still EMEA
                    "GBP": "GB",
                    "AUD": "AU",
                    "CAD": "CA",
                    "CHF": "CH",
                    "JPY": "JP",
                    "NZD": "NZ",
                    "SEK": "SE",
                    "NOK": "NO",
                    "DKK": "DK",
                    "MXN": "MX",
                    "BRL": "BR",
                    "HKD": "HK",
                    "SGD": "SG",
                    "INR": "IN",
                    "CNY": "CN",
                    "KRW": "KR",
                }
                country = cur_map.get(self.currency or "USD", "US")
            self.country = country

        if self.geographic_region is None and self.country:
            americas = {"US", "CA", "BR", "MX", "AR", "CL", "PE"}
            emea = {
                "GB", "IE", "FR", "DE", "NL", "LU", "ES", "IT", "SE", "NO",
                "DK", "FI", "BE", "CH", "AT", "PL", "CZ", "HU", "RO", "BG",
                "GR", "PT", "ZA", "AE", "SA", "EG", "MA", "JE", "GG", "KY",
                "BM", "EU",
            }
            apac = {
                "JP", "CN", "KR", "AU", "NZ", "IN", "SG", "HK", "TW", "ID",
                "MY", "PH", "TH", "VN",
            }
            if self.country in americas:
                self.geographic_region = "Americas"
            elif self.country in emea:
                self.geographic_region = "EMEA"
            elif self.country in apac:
                self.geographic_region = "APAC"
            else:
                self.geographic_region = "Other"


# ── Captured totals for reconciliation ──────────────────────────────────────


@dataclass
class SectionTotal:
    label: str
    amortized_cost: Optional[float] = None
    fair_value: Optional[float] = None
    pct_net_assets: Optional[float] = None
    _source_table_index: Optional[int] = None
    _source_row_index: Optional[int] = None


# ── Base parser ─────────────────────────────────────────────────────────────


class V2SOIParser:
    TICKER: str = ""
    # Subclasses set these; base class infers everything it can.
    UNIT_MULTIPLIER: float = 1.0   # 1 for raw, 1_000 for $K, 1_000_000 for $M
    UNIT_DETECTION: bool = True    # If True, adapter auto-detects unit from filing text.

    # Grand-total stop sentinel used by extract() to terminate row processing.
    # Subclasses with multiple "Total <X>" subtotals where the section subtotal
    # text matches the base regex (e.g. GCRED's "Total investments" debt-only
    # subtotal preceding the money-market-funds section + "Total investments
    # and money market funds" grand total) override this with a stricter
    # pattern.
    #
    # Default sentinel:
    #   - "Total Investments" / "Total Portfolio Investments" / "Total
    #     Investment Portfolio"
    #   - Excludes "Total Investments in <X>" subtotals via a negative
    #     lookahead (LRFC/LAPCF-style affiliated/JV section subtotals that
    #     sit inside the same table as the NCNA section).
    #   - Excludes "Total Investments and Money Market Funds" only when the
    #     subclass needs the section subtotal NOT to fire — see GCRED.
    GRAND_TOTAL_SENTINEL_RE = re.compile(
        r"^total\s+(?:portfolio\s+)?investments?(?!\s+in\b)\b"
        r"|^total\s+investment\s+portfolio\b",
        re.I,
    )

    def __init__(self, html_path: str, period_end: str, accession: Optional[str] = None):
        self.html_path = html_path
        self.period_end = period_end
        self.accession = accession
        self.soup: Optional[BeautifulSoup] = None
        self.tables: list[Tag] = []
        self.investments: list[Investment] = []
        self.section_totals: list[SectionTotal] = []
        self.footnote_legend: dict[str, str] = {}
        self.unit_multiplier: float = self.UNIT_MULTIPLIER
        # Per-facility unfunded-commitment records (Schedule of Unfunded
        # Commitments). Populated by _extract_unfunded_records_v2() at the
        # end of extract(). Separate from the inline `Unfunded Commitment`
        # Investment rows that the legacy `_ingest_unfunded_schedule()`
        # method emits — that path remains for ARCC-style filings whose
        # SOI semantics merge unfunded into the same row stream.
        self.unfunded_commitments: list = []
        # Disclosed total $ (from filing's own "Total Unfunded Commitments"
        # row or narrative aggregate). Used for reconciliation tests.
        self.unfunded_disclosed_total: Optional[float] = None

    # ── Load ────────────────────────────────────────────────────────────

    def load(self) -> None:
        with open(self.html_path, "rb") as f:
            content = f.read()
        # lxml handles giant inline-XBRL documents reliably
        self.soup = BeautifulSoup(content, "lxml")
        self.tables = self.soup.find_all("table")
        # Replace <br/> WITHIN TABLE CELLS with a space so multi-word
        # labels like "Maturity<br/>Date" → "Maturity Date" survive
        # `get_text()`. Scoped to table descendants only — doing it
        # globally on a 14-24MB iXBRL document is prohibitively slow.
        for table in self.tables:
            for br in table.find_all("br"):
                br.replace_with(" ")
        if self.UNIT_DETECTION:
            self._detect_units()

    def _detect_units(self) -> None:
        """Determine the unit multiplier in this priority order:

        1. **XBRL fv_decimals_modal** (authoritative). The XBRL `decimals`
           attribute on `us-gaap:InvestmentOwnedAtFairValue` facts encodes
           the reporting scale: "0" = raw $, "-3" = thousands, "-6" =
           millions. This is the SEC-validated ground truth and overrides
           caption text. BDEBT's $4500B blowup happened because the parser
           trusted a "$ in thousands" filename hint when the actual XBRL
           said decimals="0" (raw $). One field, one fix.
        2. **Caption text** ("(in thousands)" / "(in millions)") near any
           SOI heading — used when no XBRL extract exists for this period.
        3. **Fallback to 1.0** (raw $) when neither is available — pre-XBRL
           filings without explicit unit captions.
        """
        # --- 1. XBRL schema-aware override ---
        xbrl_mult = self._unit_from_xbrl()
        if xbrl_mult is not None:
            self.unit_multiplier = xbrl_mult
            return

        # --- 2. Caption-based detection ---
        import re as _re
        text = self.soup.get_text(" ", strip=True) if self.soup else ""
        # Some filers (ARCC 10-Q) write "consolidated SCHEDULES of investments"
        # (plural). Accept both singular and plural.
        soi_re = _re.compile(r"schedules?\s+of\s+investments", _re.I)

        def _window_multiplier(window: str) -> "Optional[float]":
            lw = window.lower()
            # ARCC Q1 10-Qs write "(dollar amounts in millions)" — variations of
            # "amounts in millions" with the unit-of-measure phrase split across
            # the parenthetical need to match too.
            if (
                "(in millions" in lw
                or "dollars in millions" in lw
                or "amounts in millions" in lw
                or "dollar amounts in millions" in lw
                or "in millions, except" in lw
            ):
                return 1_000_000.0
            if (
                "(in thousands" in lw
                or "dollars in thousands" in lw
                or "amounts in thousands" in lw
                or "dollar amounts in thousands" in lw
                or "in thousands, except" in lw
            ):
                return 1_000.0
            return None

        for m in soi_re.finditer(text):
            window = text[max(0, m.start() - 200): m.start() + 800]
            mult = _window_multiplier(window)
            if mult is not None:
                self.unit_multiplier = mult
                return

        # --- 3. Default raw $ ---
        self.unit_multiplier = 1.0

    def _unit_from_xbrl(self) -> "Optional[float]":
        """Read the XBRL extract for this ticker/period and return the
        multiplier implied by the modal `decimals` attribute on FairValue
        facts. Returns None when no extract is on disk.
        """
        import json as _json
        import os as _os
        if not self.TICKER or not self.period_end:
            return None
        # PARSER_V2/out/xbrl/<TICKER>_<period>.json — relative to the
        # adapter package's parent (PARSER_V2 root).
        here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        path = _os.path.join(here, "out", "xbrl", f"{self.TICKER}_{self.period_end}.json")
        if not _os.path.exists(path):
            return None
        try:
            with open(path) as f:
                d = _json.load(f)
        except (OSError, ValueError):
            return None
        modal = d.get("fv_decimals_modal")
        if modal is None:
            return None
        # XBRL decimals encodes rounding precision. Map to the implied
        # caption scale used in the matching HTML table:
        #   0      raw $ (BDEBT, pre-2010 filings)         → 1
        #   -1..-3 $K precision ("(in thousands)")         → 1_000
        #   -4..-6 $M precision ("(in millions)")          → 1_000_000
        #   INF / positive → exact, ambiguous → fall through to caption.
        try:
            n = int(modal)
        except (TypeError, ValueError):
            return None
        if n >= 0:
            return 1.0
        if -3 <= n <= -1:
            return 1_000.0
        if -6 <= n <= -4:
            return 1_000_000.0
        return None

    # ── SOI table discovery (structural) ────────────────────────────────

    def find_soi_tables(self) -> list[Tag]:
        """Default structural detector:

        A table qualifies as an SOI table iff within its first ~8 rows there
        exists a header row that also contains 'Fair Value' AND
        'Cost'/'Amortized Cost'. We collect all such tables until we hit a
        true "Total Investments" grand-total row, then stop. Headerless
        continuation tables (R6-CP-D3, e.g. TSLX paginated 10-Q) are
        accepted IFF they immediately follow an accepted SOI table AND
        contain data rows.

        Period-skip via _table_preceding_caption_mentions_prior_period
        (with R6-CP-D1 audit-list disambiguation).

        R6 cross-period fix: when MULTIPLE runs of SOI tables exist
        (10-K with comparative-year SOI for both current and prior year),
        pick the run preceded by an "As of <current_year>" caption. This
        eliminates the comparative-year double-ingestion that affected
        KBDC, MSDL, TCPC, TPVG, TRIN, OCSL, MAIN, OSCF (R6-SMALL-D1 /
        R6-D4 / R6-OSCF-AV02).
        """
        assert self.soup is not None, "call load() first"
        out: list[Tag] = []
        seen_grand_total = False
        last_accepted_idx = -1

        def _has_soi_fv(labels: list[str]) -> bool:
            return any(
                "fair value" in l or "market value" in l or l == "value"
                or "fairvalue" in l
                for l in labels
            )

        def _has_soi_cost(labels: list[str]) -> bool:
            return any(
                l == "cost" or "amortized cost" in l or l == "amortizedcost"
                or "cost basis" in l
                or l.startswith("cost")   # also matches "cost12,17" (CSWC footnote-suffixed)
                for l in labels
            )

        def _has_soi_specific(labels: list[str]) -> bool:
            return any(
                "rate" in l or "spread" in l or "maturity" in l
                or "acquisition" in l or "principal" in l
                or "par" in l or "shares" in l or "coupon" in l
                or "floor" in l or "index" in l
                or l == "interest" or l.startswith("interest ")
                for l in labels
            )

        def _is_soi_header(row: Tag, sibling_labels: Optional[list[str]] = None) -> bool:
            if not is_header_row(row):
                return False
            labels = [normalize_label(t) for _, _, t in self._row_spans(row) if t]
            has_fv = _has_soi_fv(labels)
            has_cost = _has_soi_cost(labels)
            # R6-CP-D6: reject portfolio-summary / roll-forward tables that
            # happen to have both a "Cost" and "Fair Value" column but are NOT
            # actual per-investment SOI tables. These appear in 10-K filings
            # as aggregate-by-type or aggregate-by-industry summaries (e.g.
            # RWAY "Investments | Cost | Fair Value | % of Total Portfolio"
            # table, PSEC "Type of Investment | Cost | % of Portfolio | Fair
            # Value..." table). They lack any SOI-specific columns such as
            # rate/spread, maturity date, acquisition date, par/principal, or
            # shares — so we require at least one such column.
            #
            # Also accepts "interest" as an SOI-specific indicator: early BDC
            # 10-Q filings (pre-2009) use multi-row headers where "Initial
            # Acquisition Date" is split across rows.  The bottom row only
            # contains "Date" (not "acquisition date"), so "acquisition" is
            # absent from the single-row check.  But these same filings have
            # "Interest" as a distinct column header, which is equally specific
            # to loan SOI tables (income-statement tables don't start with
            # "Company" and wouldn't pass is_header_row above).
            has_soi_specific = _has_soi_specific(labels)

            # TC0-AV09: pre-2016 split-row headers.  Some early-era 10-Qs
            # (FSK 2011-Q1, GBDC 2011/2012 quarterlies, PSEC 2010/2011
            # quarterlies) place "Principal", "Amortized", "Fair" in the row
            # IMMEDIATELY ABOVE the column-label row, with the column-label
            # row carrying only "Amount(b)", "Cost", "Value(c)".  Each row
            # alone is insufficient — the column-label row is the structural
            # header (passes is_header_row) but lacks SOI-specific labels.
            # Resolution: if the candidate header row passes has_fv & has_cost
            # but not has_soi_specific, accept additional SOI-specific labels
            # provided by sibling rows above it in the same table.
            if has_fv and has_cost and not has_soi_specific and sibling_labels:
                if _has_soi_specific(sibling_labels):
                    has_soi_specific = True
            return has_fv and has_cost and has_soi_specific

        # R6-CP-D5: grand-total row detector.
        # Must match the portfolio-level "Total Investments" row regardless
        # of whether the filer says "Total Investments", "Total Portfolio
        # Investments", or "Total Investment Portfolio". The row signals that
        # the current-period SOI is complete and prevents the parser from
        # continuing into comparative-period tables or sub-fund schedules.
        #
        # Negative lookahead excludes affiliation subtotals (BXSL/BCRED
        # pattern): "Total Investments - non-controlled/non-affiliated"
        # is a SUBTOTAL by affiliation, NOT the canonical grand-total.
        # Without this guard the walk stops at the per-affiliation
        # subtotal and the captured grand-total row is wrong.
        #
        # Permissive on income/loss/etc.: tightening THAT prefix caused
        # widespread prior-period over-walking (OTF doubled, BCRED +500
        # rows, etc.) so it stays in.
        _grand_total_re = re.compile(
            r"^total\s+(?:portfolio\s+)?investments?\b"
            r"(?!\s*[-—,]\s*(?:non[-\s]?controlled|controlled|affiliated|unaffiliated))"
            r"|^total\s+investment\s+portfolio\b",
            re.I
        )

        def _is_grand_total_row(text: str) -> bool:
            return bool(_grand_total_re.match(text))

        def _looks_like_continuation(table: Tag) -> bool:
            """Headerless table whose first non-empty row is a data row.

            R6-CP-D3: paginated SOI continuation. Conservative: only when
            (a) at least 4 non-empty cells, (b) >=1 numeric cell, AND
            (c) the table HAS column shape consistent with SOI (>=6
            distinct columns).

            R6-CP-D3b: scan ALL first-5 rows, not just the first qualifying
            one. TSLX page-break tables start with a 4-cell subtotal row
            (prev-section total) followed by a 1-cell industry header, then
            a 10-cell data row. The old code returned False prematurely on
            the 4-cell subtotal row before ever seeing the data row.
            """
            from ._shared import parse_number
            rows = table.find_all("tr")
            for r in rows[:5]:
                spans = self._row_spans(r)
                non_empty = [s for s in spans if s[2]]
                if len(non_empty) < 4:
                    continue
                num_cells = sum(1 for _, _, t in non_empty if parse_number(t) is not None)
                # Must have at least one money-magnitude number
                has_money = any(
                    parse_number(t) is not None
                    and abs(parse_number(t) or 0) >= 100
                    for _, _, t in non_empty
                )
                if num_cells >= 2 and has_money and len(non_empty) >= 6:
                    return True
                # Do NOT return False here — keep scanning remaining rows.
                # A subtotal row with >=4 but <6 non-empty cells should not
                # short-circuit the check for subsequent data rows.
            return False

        # R6-SMALL-D1 sticky-prior-period: once a table is identified as
        # prior-period, every subsequent table is also prior-period until
        # an "As of <current_year>" caption is found in the intervening
        # narrative. Captions don't repeat per-table in some 10-Ks.
        sticky_prior_period = False
        last_caption_year: Optional[str] = None
        cur_year = self.period_end[:4] if self.period_end else None
        as_of_re = re.compile(
            r"As\s+of\s+[A-Z][a-z]+\s+\d{1,2},?\s+(20\d{2})", re.I
        )
        # R6-CP-D5: some filers use bare date headings without "As of" prefix
        # (e.g. "December 31, 2024" or "September 30, 2025") as SOI title rows.
        bare_date_re = re.compile(
            r"(?:January|February|March|April|May|June|July|August|September"
            r"|October|November|December)\s+\d{1,2},?\s+(20\d{2})", re.I
        )
        audit_list_year_re = re.compile(r"(20\d{2})\s*,\s*(20\d{2})")

        def _scan_intervening_caption(prev_idx: int, this_idx: int) -> Optional[str]:
            """Look for an "As of <date>" or bare date-heading caption in the
            narrative between tables prev_idx and this_idx. Returns the year
            string if found.

            R6-CP-D5: some filers (MSDL 10-K, OCSL 10-Q) use bare date
            headings ("December 31, 2024") without an "As of" prefix for the
            comparative-period SOI title. Detect those as well.

            R6-CP-D6b: For the pre-first-table case (prev_idx < 0) the old
            code used `find_previous` for the initial element then switched to
            `find_next` in the loop — `find_previous` lands INSIDE the prior
            SOI table's cells, and `find_next` then walks through those cells
            without ever reaching the narrative heading (e.g. BXSL's
            "September 30, 2024" bare-date heading sits between the prior
            table and this_table in the DOM). Fix: use `find_previous`
            consistently for the pre-first-table backward scan, skipping any
            element that is a descendant of a <table> other than this_table.

            R10-CP-D1: For the forward scan (prev_idx >= 0), `find_next` from
            a <table> element traverses in DFS pre-order, so the first p/div/h
            elements found are INSIDE prev_table's own cells — not the
            inter-table narrative.  These cell elements exhaust the scanned<30
            budget before reaching the narrative heading (e.g. "March 31,
            2025") after prev_table.  Fix: skip any element whose
            find_parent("table") is prev_table; use free iterations (no
            scanned++) so the 30-element budget counts only true inter-table
            narrative elements.
            """
            # Cover-page boilerplate keywords: a short bare-date text whose
            # ancestor block contains any of these is a filing-header date
            # (e.g. GBDC 10-Q "For the quarterly period ended June 30, 2012"),
            # NOT an SOI period caption.  Detected by checking the nearest
            # non-table ancestor's text for these keywords.  Only applied to
            # short bare-date texts (<=30 chars) to avoid suppressing longer
            # narrative that happens to mention cover-page context.
            _COVER_PAGE_RE = re.compile(
                r"securities\s+and\s+exchange\s+commission"
                r"|form\s+10-[kq]\b"
                r"|quarterly\s+report.*pursuant"
                r"|annual\s+report.*pursuant",
                re.I,
            )

            def _is_in_cover_page_block(elem) -> bool:
                """Return True if elem is inside an ancestor block that looks
                like SEC filing cover-page boilerplate."""
                parent = elem.parent
                depth = 0
                while parent is not None and depth < 4:
                    if parent.find_parent("table") is None:
                        ptext = parent.get_text(" ", strip=True)
                        if _COVER_PAGE_RE.search(ptext[:300]):
                            return True
                    parent = parent.parent
                    depth += 1
                return False

            def _text_or_none(elem) -> Optional[str]:
                """Return stripped text only if elem is outside any <table>."""
                if elem.find_parent("table") is not None:
                    return None
                return elem.get_text(" ", strip=True) or None

            def _check_text(text: str, elem=None) -> Optional[str]:
                """Return year if text contains a valid As-of or bare-date match.

                For long prose text (>200 chars), an 'As of' match is only
                trusted as a SOI period heading when 'schedule of investments'
                appears within 120 chars preceding the match.  This prevents
                footnote body text (e.g. "… as of June 30, 2006 the company
                was not in default…") from being mistaken for an SOI caption
                when the actual SOI heading for the comparative period appears
                further along in the same navigation window.

                R10-CP-D2: bare-date matches (<=30 chars) inside cover-page
                boilerplate (SEC header, Form 10-Q/K cover) are skipped.
                These appear in GBDC 10-Q filings as the prior-fiscal-year
                end date on the cover page (e.g. "September 30, 2011") and
                cause false sticky_prior_period detection when they precede
                the SOI table in the backward scan.
                """
                soi_nearby_re = re.compile(r"schedule\s+of\s+investments", re.I)
                for m in as_of_re.finditer(text):
                    tail = text[m.end():m.end() + 60]
                    if audit_list_year_re.match(tail):
                        continue
                    if len(text) > 200:
                        # Only trust if preceded by "schedule of investments"
                        # within 120 chars
                        preamble = text[max(0, m.start() - 120): m.start()]
                        if not soi_nearby_re.search(preamble):
                            continue
                    return m.group(1)
                if len(text) <= 30:
                    # R10-CP-D2: skip bare dates inside cover-page blocks
                    if elem is not None and _is_in_cover_page_block(elem):
                        return None
                    for m in bare_date_re.finditer(text):
                        tail = text[m.end():m.end() + 20]
                        if audit_list_year_re.match(tail):
                            continue
                        return m.group(1)
                return None

            if prev_idx < 0:
                # Before the first table — scan BACKWARD from this_table,
                # ignoring any element that lives inside another table.
                p = self.tables[this_idx].find_previous(
                    ["p", "div", "h1", "h2", "h3", "h4"]
                )
                scanned = 0
                while p is not None and scanned < 50:
                    text = _text_or_none(p)
                    if text:
                        # R10-CP-D2: pass elem so cover-page guard can fire
                        yr = _check_text(text, elem=p)
                        if yr is not None:
                            return yr
                    p = p.find_previous(["p", "div", "h1", "h2", "h3", "h4"])
                    scanned += 1
                return None

            # prev_idx >= 0: scan FORWARD from end of prev_table to this_table.
            #
            # R10-CP-D1: `find_next` from a <table> element traverses in DFS
            # pre-order, so the first p/div/h elements found are INSIDE
            # prev_table's own cells — not the inter-table narrative.  These
            # cell elements exhaust the scanned<30 budget before the code ever
            # reaches the narrative heading (e.g. "March 31, 2025") that sits
            # AFTER prev_table in the DOM.  Fix: skip any element whose
            # find_parent("table") is prev_table.
            prev_table = self.tables[prev_idx]
            p = prev_table.find_next(["p", "div", "h1", "h2", "h3", "h4", "table"])
            scanned = 0
            while p is not None and scanned < 30:
                if p.name == "table":
                    if p is self.tables[this_idx]:
                        break
                    # Hit another table before reaching this_idx
                    p = p.find_next(["p", "div", "h1", "h2", "h3", "h4", "table"])
                    scanned += 1
                    continue
                # Skip elements that are descendants of prev_table
                if p.find_parent("table") is prev_table:
                    p = p.find_next(["p", "div", "h1", "h2", "h3", "h4", "table"])
                    # Do NOT increment scanned — these are free skips, not real
                    # inter-table narrative elements.
                    continue
                text = p.get_text(" ", strip=True) or ""
                yr = _check_text(text)
                if yr is not None:
                    return yr
                p = p.find_next(["p", "div", "h1", "h2", "h3", "h4", "table"])
                scanned += 1
            return None

        for ti, table in enumerate(self.tables):
            if seen_grand_total:
                break
            rows = table.find_all("tr")
            # TC0-AV09: collect labels from rows above each candidate header
            # so a split header (top row "Principal/Amortized/Fair", bottom
            # row "Amount/Cost/Value") can be recognized as SOI-shaped.
            is_soi = False
            sibling_labels_acc: list[str] = []
            for ri, r in enumerate(rows[:8]):
                if _is_soi_header(r, sibling_labels=sibling_labels_acc):
                    is_soi = True
                    break
                # Otherwise, accumulate this row's labels for the next
                # candidate. Only labels from rows in the same table that
                # PRECEDE the candidate are considered.
                row_labels = [
                    normalize_label(t) for _, _, t in self._row_spans(r) if t
                ]
                sibling_labels_acc.extend(row_labels)
            if not is_soi:
                if (last_accepted_idx >= 0 and ti - last_accepted_idx == 1
                        and not seen_grand_total
                        and _looks_like_continuation(table)):
                    out.append(table)
                    last_accepted_idx = ti
                    for row in rows:
                        texts = [t for _, _, t in self._row_spans(row) if t]
                        if texts and _is_grand_total_row(texts[0]):
                            seen_grand_total = True
                            break
                continue

            # Update sticky-prior-period state from intervening narrative.
            cap_year = _scan_intervening_caption(last_accepted_idx, ti)
            if cap_year is not None:
                last_caption_year = cap_year
                sticky_prior_period = (cur_year is not None and cap_year != cur_year)

            # Per-table prior-period check (existing logic, with audit-list
            # disambiguator already applied via finditer).
            #
            # R6-CP-D6 guard: when _scan_intervening_caption already found
            # the CURRENT year in the forward-scan window, that confirmation
            # is authoritative — skip the per-table backward scan.  The
            # backward scan (_table_preceding_caption_mentions_prior_period)
            # is prone to two false-positive sources that the forward scan
            # does not have:
            #   (a) Cover-page "As of [filing date], the Registrant had X
            #       shares…" split across multiple DOM text nodes — "shares"
            #       lands in a sibling node so cover_page_re never fires.
            #       This makes GBDC 10-Q tables (where the cover page names
            #       the next fiscal quarter) appear as prior-period.
            #   (b) Boilerplate 10-Q notes referencing the prior annual
            #       filing ("our Annual Report … for the year ended
            #       December 31, YYYY") hit the "last-resort broad December
            #       scan" and return True even when a bare-date heading like
            #       "September 30, 2024" is sitting right above the SOI.
            #       This drops BXSL Q3/Q2 tables.
            # Only fall back to the per-table check when the forward scan
            # produced no result (cap_year is None).
            if sticky_prior_period:
                continue
            if cap_year is None:
                per_table_prior = self._table_preceding_caption_mentions_prior_period(table)
                if per_table_prior:
                    continue
            out.append(table)
            last_accepted_idx = ti
            for row in rows:
                texts = [t for _, _, t in self._row_spans(row) if t]
                if texts and _is_grand_total_row(texts[0]):
                    seen_grand_total = True
                    break
        return out

    # ── Row walking ─────────────────────────────────────────────────────

    @staticmethod
    def _row_spans(row: Tag) -> list[tuple[int, int, str]]:
        from ._shared import row_cell_spans
        return row_cell_spans(row)

    def _make_row_id(self, table_idx: int, row_idx: int, company: str) -> str:
        h = hashlib.sha256(
            f"{self.TICKER}|{self.period_end}|{table_idx}|{row_idx}|{company}".encode()
        ).hexdigest()[:12]
        return h

    # ── Main extract ────────────────────────────────────────────────────

    def extract(self) -> list[Investment]:
        """Run the full pipeline. Populates `self.investments` and
        `self.section_totals`. Returns `self.investments`."""
        assert self.soup is not None, "call load() first"
        tables = self.find_soi_tables()
        if not tables:
            raise RuntimeError(f"{self.TICKER}: no SOI tables found")
        self._main_soi_tables = tables  # for unfunded detector

        current_industry: Optional[str] = None
        current_company_raw: Optional[str] = None
        current_company: Optional[str] = None
        current_business_desc: Optional[str] = None
        current_link_id: Optional[str] = None
        schema: Optional[ColumnSchema] = None

        # Grand-total pattern — used to stop processing rows after the
        # portfolio-level grand total fires, preventing post-SOI net-assets
        # summary rows (e.g. PFLT "Liabilities in Excess of Other Assets")
        # from being ingested as investment rows.
        # See GRAND_TOTAL_SENTINEL_RE class attribute for the docstring on
        # the default pattern and how subclasses override it.
        _extract_grand_total_re = self.GRAND_TOTAL_SENTINEL_RE

        for t_idx, table in enumerate(tables):
            table_index_in_doc = self.tables.index(table)
            rows = table.find_all("tr")
            extract_done = False  # stop flag for post-grand-total rows
            for r_idx, row in enumerate(rows):
                if extract_done:
                    break
                # Refresh schema whenever a header row appears (filings
                # repeat it on every page).
                if is_header_row(row):
                    schema = parse_header_row(row)
                    continue

                kind = classify_row(row, schema) if schema else "skip"
                if kind in ("spacer", "skip"):
                    continue
                if kind == "header":
                    continue
                if kind == "industry":
                    spans = self._row_spans(row)
                    for _, _, t in spans:
                        if t:
                            # Strip "(continued)" suffix used by filers
                            # when an industry section spans multiple
                            # paginated tables.
                            current_industry = re.sub(
                                r"\s*\((?:continued|cont(?:'d)?|cont\.)\)\s*$",
                                "",
                                t,
                                flags=re.I,
                            ).strip()
                            break
                    # New industry section -> forget prior company continuation
                    current_company_raw = None
                    current_company = None
                    current_business_desc = None
                    current_link_id = None
                    continue
                if kind in ("total", "subtotal"):
                    self._capture_total(row, schema, table_index_in_doc, r_idx, kind)
                    # Stop extracting data rows once the portfolio grand total
                    # is seen.  Post-grand-total rows (e.g. "Liabilities in
                    # Excess of Other Assets", "Members' Equity") are net-
                    # asset summary rows, NOT investments.
                    spans = self._row_spans(row)
                    row_text = next((t for _, _, t in spans if t), "")
                    if _extract_grand_total_re.match(row_text):
                        extract_done = True
                    continue

                # Data row
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

                # Track company continuation
                if inv_data.get("_starts_new_company"):
                    current_company_raw = inv_data["company_name_raw"]
                    current_company = inv_data["company_name"]
                    current_business_desc = inv_data.get("business_description")
                    current_link_id = self._make_row_id(
                        table_index_in_doc, r_idx, current_company_raw or ""
                    )

                inv_data["link_id"] = current_link_id
                inv = self._build_investment(
                    inv_data, table_idx=table_index_in_doc, row_idx=r_idx
                )
                if inv is not None:
                    self.investments.append(inv)

        # Parse footnote legend from document tail
        self._extract_footnote_legend()
        self._apply_footnotes()
        self._propagate_company_level_flags()
        self._assign_categories()
        # Ingest supplementary schedules (unfunded commitments, SPV children)
        self._ingest_unfunded_schedule()
        self._ingest_spv_subschedules()
        # Extract per-facility unfunded-commitment records into a separate
        # top-level array. Runs in parallel with the legacy ingester above
        # — emits structured records (with commitment type, expiration
        # date, etc.) rather than synthetic Investment rows.
        self._extract_unfunded_records_v2()
        for inv in self.investments:
            inv.finalize()
        # Opt-in section-banner propagation. Adapters set
        # USE_BANNER_PROPAGATION=True (and optionally override
        # BANNER_PATTERNS) when the filer prints seniority section
        # banners as standalone rows (e.g. CGBD, CION, OCREDIT, HCSF,
        # ABPCIC). The helper fills investment_type_raw on rows the
        # base extractor left blank.
        if getattr(self, "USE_BANNER_PROPAGATION", False):
            from ._shared import propagate_section_banners
            propagate_section_banners(
                self, patterns=getattr(self, "BANNER_PATTERNS", None)
            )
        # Opt-in affiliation-section propagation. Adapters set
        # USE_AFFILIATION_PROPAGATION=True when the filer uses
        # section-header rows like "Non-controlled/affiliated company
        # investments" to partition the SOI. Default-off so adapters
        # with custom affiliation logic (HLEND, ANTARES, FSK) aren't
        # disturbed.
        if getattr(self, "USE_AFFILIATION_PROPAGATION", False):
            from ._shared import propagate_affiliation_banners
            propagate_affiliation_banners(self)
        return self.investments

    # ── Schedule of Unfunded Commitments — structured records ──────────
    def _extract_unfunded_records_v2(self) -> None:
        """Populate `self.unfunded_commitments` and
        `self.unfunded_disclosed_total` from the Schedule of Unfunded
        Commitments (or, for filers with no per-facility table, from the
        notes-to-financial-statements aggregate).

        The detection logic lives in `_unfunded.py` and is purely
        structural — it keys off table-header tokens such as
        'Commitment Expiration Date', 'Unfunded Commitment', or
        'Category / Company' / 'Commitment Amount', plus the parser's
        existing prior-period detector for current-vs-comparative
        guarding.
        """
        from . import _unfunded as _u
        records, total = _u.extract_unfunded_commitments(self)
        self.unfunded_commitments = records
        self.unfunded_disclosed_total = total
        # Back-annotate matching Investment rows with `unfunded_amount`
        # and `is_unfunded_commitment`.
        _u.annotate_investments(self, records)

    # ── Post-processing: propagate company-level flags ─────────────────

    # Flags that apply to EVERY row of a company, not just the row that
    # carries the marker. Control/affiliation and non-qualifying-asset
    # are properties of the borrower, not of the specific tranche.
    _COMPANY_LEVEL_FLAGS: tuple[str, ...] = (
        "is_controlled_affiliate",
        "is_non_controlled_affiliate",
        "is_non_qualifying_asset",
        "is_sdlp_co_investment",
    )

    def _propagate_company_level_flags(self) -> None:
        """If any row of company X carries a company-level flag, copy the
        flag to every row of X (identified by link_id, falling back to the
        canonical company name)."""
        by_group: dict[str, set[str]] = {}
        for inv in self.investments:
            key = inv.link_id or inv.company_name
            if not key:
                continue
            acc = by_group.setdefault(key, set())
            for f in inv.footnote_flags:
                if f in self._COMPANY_LEVEL_FLAGS:
                    acc.add(f)
        for inv in self.investments:
            key = inv.link_id or inv.company_name
            if not key:
                continue
            group_flags = by_group.get(key, set())
            if not group_flags:
                continue
            new_flags = set(inv.footnote_flags) | group_flags
            inv.footnote_flags = sorted(new_flags)

    def _assign_categories(self) -> None:
        """Derive the `category` field (Non-controlled/non-affiliated,
        Non-controlled affiliate, Controlled affiliate) from the flags.

        RC-I: when BOTH `is_controlled_affiliate` and
        `is_non_controlled_affiliate` are set on the same row (possible when
        a filer's legend phrasing triggers both regex rules disjunctively),
        prefer Controlled and strip the weaker flag so the boolean
        projections in finalize() aren't both True.
        """
        for inv in self.investments:
            flags = set(inv.footnote_flags)
            if "is_controlled_affiliate" in flags:
                inv.category = "Controlled affiliate"
                # RC-I: strip the conflicting weaker flag if both are set.
                if "is_non_controlled_affiliate" in flags:
                    flags.discard("is_non_controlled_affiliate")
                    inv.footnote_flags = sorted(flags)
            elif "is_non_controlled_affiliate" in flags:
                inv.category = "Non-controlled affiliate"
            else:
                inv.category = "Non-controlled/non-affiliated"

    # ── Supplementary schedules ─────────────────────────────────────────

    # Structural signature of an "unfunded commitments" table: the header
    # row contains both "Total revolving and delayed draw" and either
    # "unfunded commitments" or "funded commitments".
    _UNFUNDED_HEADER_RE = re.compile(
        r"total\s+(?:revolving\s+and\s+delayed[-\s]?draw|"
        r"revolving\s+loan\s+and\s+delayed[-\s]?draw)",
        re.I,
    )
    _UNFUNDED_COLUMN_HINTS = {
        "portfolio company": "company_name",
        "issuer": "company_name",
        "total revolving and delayed draw loan commitments": "total_commitment_text",
        "total revolving and delayed draw commitments": "total_commitment_text",
        "less: funded commitments": "funded_text",
        "funded commitments": "funded_text",
        "total unfunded commitments": "unfunded_text",
        "less: commitments substantially at discretion of the company":
            "discretion_text",
        "less: unavailable commitments due to borrowing base or other covenant restrictions":
            "unavailable_text",
        "total net unfunded revolving and delayed draw commitments":
            "net_unfunded_text",
        "net unfunded revolving and delayed draw commitments":
            "net_unfunded_text",
    }

    def find_unfunded_tables(self) -> list[Tag]:
        """Locate supplementary unfunded-commitments tables.

        Structural: any table whose first few rows include a header whose
        first non-empty cell matches `portfolio company` / `issuer` AND
        whose header row contains `total revolving and delayed draw...`.

        We only scan tables that appear *after* the last main-SOI table to
        avoid re-parsing SOI rows.
        """
        assert self.soup is not None
        soi_tables = getattr(self, "_main_soi_tables", None) or self.find_soi_tables()
        if not soi_tables:
            return []
        last_soi_idx = max(self.tables.index(t) for t in soi_tables)
        out: list[Tag] = []
        for i in range(last_soi_idx + 1, len(self.tables)):
            t = self.tables[i]
            rows = t.find_all("tr")
            # Scan up to first 6 rows; header may be preceded by a caption row
            for row in rows[:6]:
                labels = [normalize_label(txt)
                          for _, _, txt in self._row_spans(row) if txt]
                label_text = " ".join(labels)
                if not self._UNFUNDED_HEADER_RE.search(label_text):
                    continue
                # Substring match — some filers prefix column labels with
                # the unit caption, producing "(in millions)portfolio company".
                has_portfolio_company = any(
                    "portfolio company" in l or l == "issuer" or l.endswith(" issuer")
                    for l in labels
                )
                if not has_portfolio_company:
                    continue
                # Period filter: skip prior-period comparative tables. We
                # look at the narrative text IMMEDIATELY BEFORE this table
                # for an "As of <date>" caption. If the caption names a
                # period other than the current one, skip.
                if self._table_preceding_caption_mentions_prior_period(t):
                    break
                out.append(t)
                break
        return out

    def _table_preceding_caption_mentions_prior_period(self, table: Tag) -> bool:
        """Return True if the most-recent 'As of <Month Day, YYYY>' caption
        preceding this table names a year other than `self.period_end`'s.

        We walk backward through ALL text nodes in the document that live
        OUTSIDE any <table> (i.e., narrative) until we find an 'As of <date>'
        phrase. Tables of the unfunded schedule form contiguous runs, so
        only the first table in each run has its own caption; continuation
        tables are labeled by the caption that precedes the whole run.
        """
        from bs4 import NavigableString
        cur_year = self.period_end[:4] if self.period_end else None
        if not cur_year:
            return False
        # Build index of text nodes (cached per parser instance)
        if not hasattr(self, "_narrative_text_seq"):
            body = self.soup.find("body") or self.soup
            seq: list[tuple[int, str]] = []
            for i, s in enumerate(body.find_all(string=True)):
                if s.find_parent("table") is not None:
                    continue
                txt = str(s).strip()
                if txt:
                    seq.append((i, txt))
            self._narrative_text_seq = seq
            self._narrative_seq_by_id = {i: idx for idx, (i, _) in enumerate(seq)}
        seq = self._narrative_text_seq
        # Determine this table's sequence index: use the seq index of the
        # first text node inside the table (or the table's position in the
        # full-document traversal).
        all_text_nodes = list((self.soup.find("body") or self.soup).find_all(string=True))
        table_text_node_idxs = [
            i for i, s in enumerate(all_text_nodes)
            if s.find_parent("table") is table
        ]
        if not table_text_node_idxs:
            return False
        table_first_seq = min(table_text_node_idxs)
        # Walk backward through narrative-only text until we find "As of <date>"
        as_of_re = re.compile(r"As\s+of\s+[A-Z][a-z]+\s+\d{1,2},?\s+(20\d{2})", re.I)
        december_re = re.compile(r"December\s+31,?\s+(20\d{2})", re.I)
        # R6-CP-D1: auditor opinions list multiple historical years like
        # "as of March 31, 2022, 2021, 2020, 2019..." — the regex above
        # matches the FIRST date and falsely tags the table as prior-period.
        # An "audit-list" pattern is signalled by the year being immediately
        # followed by a comma + another 4-digit year.
        audit_list_year_re = re.compile(r"(20\d{2})\s*,\s*(20\d{2})")
        # R6-CP-D1b: certain "As of <date>" phrases in non-SOI contexts must
        # NOT be treated as SOI period indicators:
        #   (1) Cover-page boilerplate: "X shares outstanding as of <date>"
        #   (2) Auditor's report: "We have (previously) audited...as of <date>"
        # We detect these by checking the 120-char context around the match.
        cover_page_re = re.compile(
            r"\bshares?\b|\boutstanding\b|\brecord\b|\bpriced\b|\bregistered\b",
            re.I
        )
        auditor_re = re.compile(
            r"\bwe\s+have\s+(?:previously\s+)?audited\b"
            r"|\bour\s+audits?\s+included\b"
            r"|\bpublic\s+company\s+accounting\b"
            r"|\bPCAOB\b"
            r"|\bin\s+our\s+opinion\b"
            r"|\bindependent\s+registered\s+public\s+accounting\b"
            r"|\bconfirmation\s+of\s+investments\s+owned\b",
            re.I
        )
        acc: list[str] = []
        chars_accumulated = 0
        for i, s in reversed(seq):
            if i >= table_first_seq:
                continue
            # R6-CP-D1d: a short bare-date heading immediately preceding the
            # table (e.g. "December 31, 2024" sitting between the table-of-
            # contents stub and the SOI table itself) is the AUTHORITATIVE
            # period marker.  This must be checked BEFORE walking further
            # back to look for "As of <date>" body text — those distant
            # narrative mentions are not table-period headings.
            #
            # ANTARES 10-K has prior-year SOI tables (Dec 31, 2024) preceded
            # immediately by a "December 31, 2024" heading; further back is
            # an "As of December 31, 2025" sentence introducing the unfunded
            # commitments schedule (which sits between the 2025 SOI and the
            # 2024 comparative SOI).  Without this proximity check the loop
            # finds the 2025 As-of mention first, R6-CP-D1c then triggers
            # `return False`, and the prior-year section is wrongly
            # classified as current-period.
            if len(s) <= 40:
                m_bare = december_re.match(s.strip())
                if m_bare:
                    return m_bare.group(1) != cur_year
            # Collect all valid As-of matches in this chunk. The LAST one in
            # the chunk is the most recent (appears latest in the source text
            # = closest to the current-period SOI heading). Walking backward
            # means this chunk is closer to the table than all prior chunks.
            chunk_matches = []
            for m in as_of_re.finditer(s):
                tail = s[m.end():m.end() + 60]
                if audit_list_year_re.match(tail):
                    continue
                # Skip cover-page boilerplate and auditor's report
                context = s[max(0, m.start() - 120):m.end() + 120]
                if cover_page_re.search(context) or auditor_re.search(s):
                    continue
                chunk_matches.append(m.group(1))
            if chunk_matches:
                # R6-CP-D1c: comparative paragraphs mention current year
                # AND prior year in the same text node (e.g. "As of December
                # 31, 2024, we had no loans on non-accrual status. As of
                # December 31, 2023, we had..."). If the current year appears
                # anywhere in the chunk, this is NOT a prior-period heading —
                # it is body text comparing the two periods. Only classify as
                # prior-period when ALL matches in the chunk are prior-year.
                if cur_year and cur_year in chunk_matches:
                    return False
                # The last valid match in the chunk is the most recent.
                return chunk_matches[-1] != cur_year
            acc.append(s)
            chars_accumulated += len(s)
            if chars_accumulated > 6000:
                break
        # Fallback: look for bare "Month DD, YYYY" heading (SOI title)
        # in the narrow pre-table window. Only trust spans <= 40 chars to
        # avoid false hits in body text.  This is mostly a safety-net now
        # that the inline check above catches proximity matches; it still
        # runs in case the table sits beyond the 6000-char accumulator
        # window.
        for i, s in reversed(seq):
            if i >= table_first_seq:
                continue
            if len(s) <= 40:
                m2 = december_re.match(s.strip())
                if m2:
                    return m2.group(1) != cur_year
        # Last resort: broad December scan across full accumulated text
        joined = " ".join(reversed(acc))
        dates = december_re.findall(joined)
        if dates:
            return dates[-1] != cur_year
        return False

    def _ingest_unfunded_schedule(self) -> None:
        """Parse unfunded-commitment tables and emit one Investment per row
        with `investment_category = 'Unfunded Commitment'`. Link to parent
        SOI row(s) via `link_id` lookup by canonical company name.

        Dollar amounts in the unfunded schedule are in the same unit as
        the main SOI (ARCC: millions)."""
        unfunded_tables = self.find_unfunded_tables()
        if not unfunded_tables:
            return

        # Build a canonical-name -> link_id map from existing SOI rows for
        # parent linking. Use the company_name (footnote-stripped form).
        soi_name_to_link: dict[str, str] = {}
        for inv in self.investments:
            name = inv.company_name or ""
            key = re.sub(r"\s+", " ", name).strip().casefold()
            if key and key not in soi_name_to_link and inv.link_id:
                soi_name_to_link[key] = inv.link_id

        for t_idx_local, t in enumerate(unfunded_tables):
            table_doc_idx = self.tables.index(t)
            # Find the header row for this table (structural — first row
            # whose labels include 'portfolio company' or 'issuer', via
            # substring match to tolerate caption-prefixed labels).
            rows = t.find_all("tr")
            schema: Optional[ColumnSchema] = None
            for row in rows[:8]:
                labels = [normalize_label(txt)
                          for _, _, txt in self._row_spans(row) if txt]
                if any("portfolio company" in l or l == "issuer" for l in labels):
                    schema = ColumnSchema(spans=[], total_logical=0)
                    running = 0
                    for cell in row.find_all(["td", "th"]):
                        cs = int(cell.get("colspan", 1) or 1)
                        label = clean_text(cell.get_text())
                        # Strip leading "(in millions)" / "(in thousands)"
                        # caption prefix that some filers merge into the
                        # first column header.
                        label = re.sub(
                            r"^\(in\s+(millions|thousands|billions)\)\s*",
                            "",
                            label,
                            flags=re.I,
                        ).strip()
                        schema.spans.append((running, running + cs, label))
                        running += cs
                    schema.total_logical = running
                    break
            if schema is None:
                continue

            # For each data row of this table, extract fields by schema
            for r_idx, row in enumerate(rows):
                spans = self._row_spans(row)
                non_empty = [s for s in spans if s[2]]
                if not non_empty:
                    continue
                first_text = non_empty[0][2]
                # Skip header repetitions and grand-total / caption rows
                if normalize_label(first_text) in ("portfolio company", "issuer"):
                    continue
                if is_total_like_text(first_text):
                    # Total row — could reconcile but skip as a data row
                    continue

                # Resolve each span to our field names via hint-map
                company = None
                amounts: dict[str, Optional[float]] = {}
                for start, end, text in spans:
                    if not text:
                        continue
                    label = None
                    for s_lbl_start, s_lbl_end, s_label in schema.spans:
                        if s_lbl_start <= start < s_lbl_end:
                            label = normalize_label(s_label)
                            break
                    if not label:
                        continue
                    field_name = self._UNFUNDED_COLUMN_HINTS.get(label)
                    if field_name == "company_name":
                        company = text
                    elif field_name:
                        amounts.setdefault(field_name, parse_number(text))

                if not company:
                    continue
                # Scale amounts
                def sc(x):
                    return None if x is None else x * self.unit_multiplier
                unfunded = sc(amounts.get("unfunded_text"))
                net_unf = sc(amounts.get("net_unfunded_text"))
                total = sc(amounts.get("total_commitment_text"))
                funded = sc(amounts.get("funded_text"))
                if unfunded is None and net_unf is None:
                    continue

                # Parent linking by canonical name match
                key = re.sub(r"\s+", " ", strip_footnotes(company)).strip().casefold()
                parent_link = soi_name_to_link.get(key)

                row_id = self._make_row_id(
                    table_doc_idx, r_idx, f"UNFUNDED|{company}"
                )
                inv = Investment(
                    row_id=row_id,
                    ticker=self.TICKER,
                    period_end=self.period_end,
                    accession=self.accession,
                    link_id=parent_link,
                    parent_row_id=parent_link,
                    parent_entity=None,
                    company_name_raw=company,
                    company_name=strip_footnotes(company),
                    industry=None,
                    category=None,
                    investment_type="Unfunded",
                    investment_type_raw="Unfunded commitment",
                    investment_category="Unfunded Commitment",
                    par_principal=total,
                    unfunded_amount=unfunded if unfunded is not None else net_unf,
                    amortized_cost=None,
                    fair_value=None,
                    currency="USD",
                    _source_table_index=table_doc_idx,
                    _source_row_index=r_idx,
                )
                self.investments.append(inv)

    def find_spv_subschedule_tables(self) -> list[tuple[str, Tag]]:
        """Default: return []. BDCs with genuine per-holding SPV sub-schedules
        (OBDC's SDLP, GBDC's GJV, etc.) override this to return
        [(parent_entity_name, table), ...]. ARCC does NOT have per-holding
        SPV sub-schedules — its JVs are single SOI line items."""
        return []

    def _ingest_spv_subschedules(self) -> None:
        """For each (parent_entity, table) returned by find_spv_subschedule_tables,
        parse the table like a mini-SOI and emit child rows with
        parent_entity populated."""
        pairs = self.find_spv_subschedule_tables()
        if not pairs:
            return
        for parent_entity, table in pairs:
            table_doc_idx = self.tables.index(table)
            # Find the parent row's link_id for parent_row_id linkage
            parent_link_id = None
            for inv in self.investments:
                if inv.company_name and parent_entity.casefold() in inv.company_name.casefold():
                    parent_link_id = inv.link_id
                    break
            rows = table.find_all("tr")
            # Find header row via structural detector (same as main SOI)
            schema: Optional[ColumnSchema] = None
            schema_row_idx = -1
            for ri, row in enumerate(rows[:10]):
                if is_header_row(row):
                    labels = {normalize_label(t) for _, _, t in self._row_spans(row) if t}
                    if "fair value" in labels:
                        schema = parse_header_row(row)
                        schema_row_idx = ri
                        break
            if schema is None:
                continue
            # Parse data rows using schema
            current_industry: Optional[str] = None
            current_company: Optional[str] = None
            current_link: Optional[str] = None
            for ri, row in enumerate(rows):
                if ri <= schema_row_idx:
                    continue
                kind = classify_row(row, schema)
                if kind in ("spacer", "skip", "header"):
                    continue
                if kind == "industry":
                    spans = self._row_spans(row)
                    for _, _, t in spans:
                        if t:
                            current_industry = t
                            break
                    continue
                if kind in ("total", "subtotal"):
                    continue
                data = self._parse_data_row(
                    row, schema,
                    current_industry=current_industry,
                    current_company_raw=current_company,
                    current_company=current_company,
                    current_business_desc=None,
                )
                if data is None:
                    continue
                if data.get("_starts_new_company"):
                    current_company = data.get("company_name")
                    current_link = self._make_row_id(
                        table_doc_idx, ri, f"SPV|{parent_entity}|{current_company}"
                    )
                inv = self._build_investment(
                    data, table_idx=table_doc_idx, row_idx=ri
                )
                if inv is None:
                    continue
                inv.parent_entity = parent_entity
                inv.parent_row_id = parent_link_id
                inv.link_id = current_link
                self.investments.append(inv)

    # ── Per-row extraction ──────────────────────────────────────────────

    # RC-A: investment types that indicate the "par/units" column value is
    # actually a SHARE COUNT, not dollar principal. Must be checked before
    # we scale the column's numeric value by unit_multiplier.
    _EQUITY_TYPE_RE = re.compile(
        r"\b(common\s+stock|common\s+shares?|common\s+units?|common\s+member\s+units?|"
        r"preferred\s+(stock|shares?|units?|equity|member\s+units?)|"
        r"class\s+[a-z](-\d)?\s+(shares?|units?|common|preferred|interests?|stock)|"
        r"series\s+[a-z0-9]+\s+(shares?|units?|preferred|stock|interests?)|"
        r"ordinary\s+shares?|"
        r"membership\s+(interests?|units?)|partnership\s+(interests?|units?)|"
        r"limited\s+partner(?:ship)?\s+interests?|"
        r"general\s+partner\s+interests?|"
        r"llc\s+(interests?|units?)|lp\s+(interests?|units?)|"
        r"member\s+(interests?|units?)|company\s+(interests?|units?)|"
        r"warrant|warrants|"
        r"equity|equities|"
        r"participation\s+interests?|"
        r"escrow|co-?investment|"
        r"subordinated\s+certificate|"
        r"trust\s+preferred|"
        r"abf\s+equity|"
        r"convertible\s+preferred)\b",
        re.I,
    )

    def _is_equity_like(self, inv_type_raw: Optional[str],
                       inv_type_norm: Optional[str],
                       company_name: Optional[str] = None) -> bool:
        """Return True if this row's value in the Par/Units column should
        be routed to `shares` (unscaled), not `par_principal` (scaled).

        FSK encodes the investment type as a comma-suffix on the company
        name (e.g. "Cubic Corp, Preferred Equity"), and several rows have
        no separate investment_type column at all. As a fallback, scan the
        company-name suffix after the LAST comma for equity-like tokens.
        """
        if inv_type_norm in ("Equity", "Preferred", "Warrant", "JV"):
            return True
        if inv_type_raw and self._EQUITY_TYPE_RE.search(inv_type_raw):
            return True
        if company_name:
            # FSK pattern: "<Issuer>, <InvestmentTypeSuffix>"
            # BXSL pattern: "<Issuer> - <InvestmentTypeSuffix>"
            # Take the suffix after the LAST `,` or ` - ` and scan it.
            # This avoids false positives on entity names containing
            # tokens like "Common" or "Equity" earlier in the name.
            suffix_candidates: list[str] = []
            if "," in company_name:
                suffix_candidates.append(company_name.rsplit(",", 1)[-1].strip())
            # Match " - " or " – " (em/en dash with spaces) — entity names
            # like "Pizza Hut - The Original" don't typically have this
            # exact spacing for a non-suffix.
            for sep in (" - ", " – ", " — "):
                if sep in company_name:
                    suffix_candidates.append(company_name.rsplit(sep, 1)[-1].strip())
            for sfx in suffix_candidates:
                if self._EQUITY_TYPE_RE.search(sfx):
                    return True
        return False

    def _parse_data_row(
        self,
        row: Tag,
        schema: ColumnSchema,
        current_industry: Optional[str],
        current_company_raw: Optional[str],
        current_company: Optional[str],
        current_business_desc: Optional[str],
    ) -> Optional[dict]:
        spans = self._row_spans(row)

        # Pull each field by schema
        def fld(name: str) -> Optional[str]:
            return get_cell_text_for_field(spans, schema, name)

        # RC-L: some filers (MFIC) split accounting-negative numbers across
        # adjacent <td> cells: `( 5` lands in the money column's logical
        # range, but `)` is in the next "spacer" column. Detect this and
        # merge the closing paren into the field text. We look for the
        # IMMEDIATELY-NEXT non-empty cell and append it iff:
        #   (a) the field text ends with an unmatched `(` or has unmatched
        #       open paren count > close paren count
        #   (b) that next cell is just `)` (or `) (footnote)` patterns)
        money_field_names = (
            "amortized_cost_text",
            "fair_value_text",
            "par_text",
            "par_or_shares_text",
            "shares_text",
        )

        def fld_money(name: str) -> Optional[str]:
            text = get_cell_text_for_field(spans, schema, name)
            if text is None:
                return None
            if text.count("(") <= text.count(")"):
                return text
            # Find the LAST cell mapped to this field, then look at the
            # next non-empty cell in row order.
            field_end = -1
            for s_start, s_end, t in spans:
                if t and schema.field_for_logical(s_start) == name:
                    field_end = s_end
            if field_end < 0:
                return text
            # Look forward for the closing paren cell
            for s_start, s_end, t in spans:
                if s_start < field_end:
                    continue
                if not t:
                    continue
                # Allow trailing `)` possibly followed by footnotes like `) (1)`
                if t.startswith(")"):
                    return text + " " + t
                # First non-empty non-`)` cell ends the search
                break
            return text

        company_raw = fld("company_name")
        inv_type = fld("investment_type_raw")
        biz = fld("business_description")
        coupon = fld("cash_rate_pct")
        pik = fld("pik_rate_pct")
        ref = fld("reference_rate_text")
        ref_and_spread = fld("reference_rate_and_spread_text")
        spread = fld("spread_text")
        floor_t = fld("floor_text")           # FSK-style dedicated column
        acq = fld("acquisition_date")
        mat = fld("maturity_date")
        shares_t = fld_money("shares_text")
        par_only_t = fld_money("par_text")
        par_or_shares_t = fld_money("par_or_shares_text")
        cost_t = fld_money("amortized_cost_text")
        fv_t = fld_money("fair_value_text")
        pct_t = fld("pct_net_assets_text")
        # RC-E: per-row industry cell (FSK has one). Prefer it over
        # current_industry state (which comes from section banners and
        # gets polluted by non-industry banners).
        industry_row = fld("industry")

        starts_new_company = bool(company_raw and company_raw not in ("$", "€", "£"))

        if not starts_new_company and not (
            inv_type or par_only_t or par_or_shares_t or fv_t or cost_t or shares_t
        ):
            # Not a continuation and not a new row — skip.
            return None

        if starts_new_company:
            company_raw_eff = company_raw
            company_clean = strip_footnotes(company_raw)
            # Strip address after em-dash
            company_clean = re.split(r"[——]", company_clean)[0].strip()
        else:
            company_raw_eff = current_company_raw or ""
            company_clean = current_company or ""

        currency = infer_currency_from_cell(par_only_t or par_or_shares_t or "") or "USD"

        # RC-B: reference rate + spread extraction, in order of preference:
        #   1. combined "Reference Rate and Spread" column (BXSL style)
        #   2. separate ref column + separate spread column (ARCC style)
        #   3. reference rate inline in coupon cell (ADS/FSK style)
        ref_norm: Optional[str] = None
        spread_bps: Optional[float] = None
        if ref_and_spread:
            rn, sbp = split_ref_and_spread(ref_and_spread)
            ref_norm = rn
            spread_bps = sbp
        if ref_norm is None and ref:
            rn = tokenize_reference_rate(ref)
            if rn:
                ref_norm = rn
        if spread is not None:
            # Spread cell may be bare "4.75%" (ARCC) OR combined "SF + 5.00%" (GBDC)
            if re.match(r"^\s*[A-Z]{1,6}\s*\+", spread):
                rn, sbp = split_ref_and_spread(spread)
                if ref_norm is None and rn:
                    ref_norm = rn
                if spread_bps is None and sbp is not None:
                    spread_bps = sbp
            elif spread_bps is None:
                spread_bps = parse_basis_points(spread)

        # RC-B (FSK/ADS style): the "Rate" / coupon cell may actually be a
        # combined ref+spread cell (e.g. "SF + 6.0% (3.5% PIK)"), not a
        # total coupon. If it starts with a ref-rate abbreviation + `+`,
        # parse it as ref+spread and clear the cash_rate.
        # EURIBOR is 7 chars, STIBOR/TIBOR/HIBOR/SARON/AONIA fit; allow up
        # to 10 to be future-proof for new ARR tokens. Validation comes via
        # the parsing path (split_ref_and_spread) — false positives there
        # are harmless because they yield ref_norm=None.
        coupon_is_combined = bool(
            coupon and re.match(r"^\s*[A-Z]{1,10}\s*\+", coupon)
        )
        if coupon_is_combined:
            rn, sbp = split_ref_and_spread(coupon)
            if ref_norm is None and rn:
                ref_norm = rn
            if spread_bps is None and sbp is not None:
                spread_bps = sbp
        # Also treat the dedicated reference_rate_and_spread cell as a
        # candidate for PIK extraction. OBDC pre-2025 prints
        # "S+ 6.75% (0.75% PIK)" in its "Interest" column (mapped to
        # ref_and_spread); the PIK extraction below needs the same
        # cell-source contract as `coupon_is_combined` to find the
        # PIK annotation.
        ref_and_spread_has_pik = bool(
            ref_and_spread and re.search(r"\bPIK\b", ref_and_spread, re.I)
        )

        # RC-B: if we STILL don't have a reference rate, try to find one in
        # the coupon cell text (filers sometimes inline the rate token).
        if ref_norm is None and coupon:
            rn = tokenize_reference_rate(coupon)
            if rn:
                ref_norm = rn
                if spread_bps is None:
                    _rn, sbp = split_ref_and_spread(coupon)
                    if sbp is not None:
                        spread_bps = sbp

        # Normalize display form
        if ref_norm == "PRIME":
            ref_norm = "Prime"

        # RC-B (fix): do NOT default to "Fixed" just because there's a
        # coupon number. Only label "Fixed" when explicitly indicated.
        if ref_norm is None and coupon and re.search(r"\bfixed\b", coupon, re.I):
            ref_norm = "Fixed"

        if currency == "USD" and ref_norm:
            currency = currency_of_reference_rate(ref_norm)

        # GBP-currency-context disambiguation: a bare "S +" prefix maps
        # to SOFR by default, but in a GBP context it means SONIA. Same
        # for AUD context where "S +" can mean BBSW. Apply only when
        # currency was determined upstream (e.g. by ISO-code matching
        # in infer_currency_from_cell) and conflicts with ref_norm.
        # Observed at 2026-03-31: BCRED Riser GBP tranche, BXSL Denali
        # Bidco GBP tranche.
        if ref_norm == "SOFR" and currency and currency != "USD":
            _ccy_ref = {
                "GBP": "SONIA",
                "AUD": "BBSW",
                "EUR": "EURIBOR",
                "CAD": "CORRA",
                "CHF": "SARON",
                "JPY": "TONA",
            }
            mapped = _ccy_ref.get(currency)
            if mapped:
                ref_norm = mapped

        # Floor: prefer explicit column if present; inline extraction below
        # handles "0.50% Floor" annotations inside combined rate cells.
        floor_pct_extracted: Optional[float] = parse_percent(floor_t) if floor_t else None

        # RC-C: rate-cell parsing for PIK-annotated coupons.
        # If coupon is combined ref+spread (FSK/ADS), cash_pct is None and
        # we extract PIK + floor from annotations inside the cell.
        # Same logic applies when PIK is annotated in the dedicated
        # reference_rate_and_spread cell (OBDC pre-2025 "Interest" column).
        if coupon_is_combined or ref_and_spread_has_pik:
            # Source cell for PIK regex: prefer coupon when combined,
            # otherwise the ref_and_spread cell.
            pik_source = coupon if coupon_is_combined else (ref_and_spread or "")
            cash_pct = None
            pik_pct_from_cash = None
            # FSK pattern: "(current% PIK / max% PIK)" — take MAX (rightmost).
            mpair = re.search(
                r"\(\s*([\d.]+)\s*%\s*PIK\s*/\s*([\d.]+)\s*%\s*PIK\s*\)",
                pik_source,
                re.I,
            )
            if mpair:
                # FSK convention: left=current/elected, right=max. Filer's
                # SOI shows max as the meaningful PIK economics. Take MAX
                # (group 2). Also store left as pik_rate_max_pct=None and
                # use right as primary pik_rate.
                try:
                    pik_pct_from_cash = float(mpair.group(2))
                except ValueError:
                    pass
            else:
                for pat in (
                    r"\(\s*([\d.]+)\s*%\s*PIK",          # ARCC "(X% PIK)"
                    r"plus\s+([\d.]+)\s*%\s*PIK",        # ADS "plus X% PIK"
                    r"([\d.]+)\s*%\s*PIK\b",             # FSK inline
                    r"([\d.]+)\s*%\s*/\s*PIK",           # NMFC
                ):
                    mpik = re.search(pat, pik_source, re.I)
                    if mpik:
                        try:
                            pik_pct_from_cash = float(mpik.group(1))
                        except ValueError:
                            pass
                        break
            # Floor inline: "0.50% Floor"
            if floor_pct_extracted is None and coupon:
                mf = re.search(r"([\d.]+)\s*%\s*Floor", coupon, re.I)
                if mf:
                    try:
                        floor_pct_extracted = float(mf.group(1))
                    except ValueError:
                        pass
        else:
            cash_pct, pik_pct_from_cash = parse_rate_cell(coupon)
        pik_pct_explicit, _ = parse_rate_cell(pik)
        pik_pct = pik_pct_from_cash if pik_pct_from_cash is not None else pik_pct_explicit

        mat_iso = parse_date(mat)
        acq_iso = parse_date(acq)

        # Use parse_money_number for money columns so trailing footnote
        # markers ("( 5 ) (a)") don't break number extraction. Bare "(5)"
        # in a money column is an accounting NEGATIVE, not a footnote.
        par_num = parse_money_number(par_only_t) if par_only_t else None
        combined_par_num = parse_money_number(par_or_shares_t) if par_or_shares_t else None
        cost_num = parse_money_number(cost_t)
        fv_num = parse_money_number(fv_t)
        shares_num_explicit = parse_money_number(shares_t) if shares_t else None
        pct_num = parse_percent(pct_t)

        # Unit scale for money fields
        def scale(x: Optional[float]) -> Optional[float]:
            return None if x is None else x * self.unit_multiplier

        inv_type_norm = self._normalize_investment_type(inv_type or "")

        # RC-A: for the combined "Par / Units" column, route to shares
        # (unscaled) OR par_principal (scaled) based on whether the row is
        # equity-like. If both `par_text` (debt-only) AND
        # `par_or_shares_text` (combined) are present in the schema, prefer
        # `par_text` for principal. Explicit "shares" column always wins
        # for share count.
        is_equity = self._is_equity_like(
            inv_type, inv_type_norm, company_name=company_clean
        )
        par_principal_out: Optional[float] = None
        shares_out: Optional[float] = None
        if par_num is not None:
            par_principal_out = scale(par_num)
        if combined_par_num is not None:
            if is_equity:
                shares_out = combined_par_num  # unscaled share count
            else:
                # Debt tranche in a combined column — treat as principal.
                if par_principal_out is None:
                    par_principal_out = scale(combined_par_num)
        if shares_num_explicit is not None:
            shares_out = shares_num_explicit  # explicit shares column wins
        # Unfunded commitments: investment type / row description mentions it.
        # Distinguish "literally unfunded" (zero/negative funded amount) from
        # delayed-draw term loans that ARE drawn — both contain "delayed draw"
        # in the type, but the latter is a normal funded debt instrument
        # with an unfunded commitment portion.
        is_unfunded = False
        if inv_type:
            literal_unfunded = re.search(r"\bunfunded\b|\bundrawn\b", inv_type, re.I)
            delayed_draw = re.search(r"delayed[-\s]?draw", inv_type, re.I)
            if literal_unfunded:
                is_unfunded = True
            elif delayed_draw:
                # Treat as unfunded ONLY if there's no funded portion.
                # If par or fair value are positive, this is a funded
                # DDTL with a remaining unfunded commitment — it belongs
                # in Debt, not Unfunded.
                cost_val = scale(cost_num) if cost_num is not None else None
                fv_val = scale(fv_num) if fv_num is not None else None
                par_val = par_principal_out
                has_funded = (
                    (par_val is not None and par_val > 0)
                    or (fv_val is not None and fv_val > 0)
                    or (cost_val is not None and cost_val > 0)
                )
                if not has_funded:
                    is_unfunded = True

        # Footnote markers may appear in ANY cell of the row — most commonly
        # in the Fair Value / amount cells where filings render them as
        # '(2)(9)' etc. RC-D: pass is_money_column=True for cells in
        # monetary columns so single "(N)" cells are correctly treated
        # as negative dollar amounts, not footnote markers.
        money_fields = {
            "amortized_cost_text", "fair_value_text", "par_text",
            "par_or_shares_text", "shares_text", "pct_net_assets_text",
            "cash_rate_pct", "pik_rate_pct", "spread_text",
            "reference_rate_and_spread_text",
        }
        all_markers: list[str] = []
        for s_start, s_end, cell_text in spans:
            if not cell_text:
                continue
            fld = schema.field_for_logical(s_start) if schema else None
            in_money = fld in money_fields
            all_markers.extend(extract_footnote_markers(cell_text, is_money_column=in_money))
        # De-dup preserving order
        seen: set[str] = set()
        markers: list[str] = []
        for m in all_markers:
            if m not in seen:
                seen.add(m)
                markers.append(m)

        # Floor: use extracted value (handled above in combined-rate branch)
        floor_pct = floor_pct_extracted

        # RC-E: industry — prefer per-row industry column (FSK, MFIC sub),
        # fall back to section-banner state. Strip "(continued)".
        industry_final = industry_row or current_industry
        if industry_final:
            industry_final = re.sub(
                r"\s*\((?:continued|cont(?:'d)?|cont\.)\)\s*$",
                "",
                industry_final,
                flags=re.I,
            ).strip()

        return {
            "_starts_new_company": starts_new_company,
            "company_name_raw": company_raw_eff,
            "company_name": company_clean,
            "industry": industry_final,
            "business_description": (biz if starts_new_company else current_business_desc) or None,
            "investment_type_raw": inv_type,
            "investment_type": inv_type_norm,
            "reference_rate": ref_norm,
            "reference_rate_text_raw": (
                ref or ref_and_spread or (coupon if coupon_is_combined else None)
                # Synthesize when filer puts ref + spread in SEPARATE
                # columns (GBDC, ARCC) — reconstruct the canonical
                # "{REF}+{spread%}" form so downstream consumers always
                # have the raw text.
                or (
                    f"{ref_norm}+{spread_bps / 100:.3f}%"
                    if ref_norm and spread_bps is not None
                    else None
                )
            ),
            "spread_bps": spread_bps,
            "spread_text_raw": (
                spread or ref_and_spread or (coupon if coupon_is_combined else None)
            ),
            "cash_rate_pct": cash_pct,
            "pik_rate_pct": pik_pct,
            "floor_pct": floor_pct,
            "maturity_date": mat_iso,
            "acquisition_date": acq_iso,
            "par_principal": par_principal_out,
            "amortized_cost": scale(cost_num),
            "fair_value": scale(fv_num),
            "shares": shares_out,
            "pct_net_assets": pct_num,
            "unfunded_amount": scale(par_num if par_num is not None else combined_par_num) if is_unfunded else None,
            "currency": currency,
            "footnote_markers": markers,
            "investment_category": "Unfunded Commitment" if is_unfunded else None,
        }

    # ── Row → Investment ────────────────────────────────────────────────

    def _build_investment(
        self, data: dict, table_idx: int, row_idx: int
    ) -> Optional[Investment]:
        # Skip rows that have no useful content. The drop condition has
        # to balance two failure modes:
        #   (a) under-drop -> ingest stray narrative / banner rows as
        #       investments;
        #   (b) over-drop -> lose fully-written-down non-accrual loans
        #       (par populated, FV=AC=$0 from em-dash).
        # The legend is not yet populated at this call site, so we
        # cannot use marker semantics. Instead: retain any row that has
        # a populated par_principal AND at least one footnote marker —
        # the marker is a strong signal of a real position (legend pass
        # will later classify NA / unfunded / etc.). Drop only when
        # ALL of FV/cost/shares/unfunded/par/type are missing AND there
        # are no footnote markers.
        markers = data.get("footnote_markers") or []
        has_par = data.get("par_principal") not in (None, 0, 0.0)
        retain_for_markers = bool(markers) and has_par
        if (
            data.get("fair_value") is None
            and data.get("amortized_cost") is None
            and data.get("shares") is None
            and data.get("unfunded_amount") is None
            and not data.get("investment_type_raw")
            and not retain_for_markers
        ):
            return None
        row_id = self._make_row_id(
            table_idx, row_idx, data.get("company_name_raw") or ""
        )
        inv = Investment(
            row_id=row_id,
            ticker=self.TICKER,
            period_end=self.period_end,
            accession=self.accession,
            parent_entity=data.get("parent_entity"),
            parent_row_id=data.get("parent_row_id"),
            link_id=data.get("link_id"),
            company_name_raw=data.get("company_name_raw") or "",
            company_name=data.get("company_name") or "",
            industry=data.get("industry"),
            category=data.get("category"),
            business_description=data.get("business_description"),
            investment_type_raw=data.get("investment_type_raw"),
            investment_type=data.get("investment_type"),
            investment_category=data.get("investment_category"),
            reference_rate=data.get("reference_rate"),
            reference_rate_text_raw=data.get("reference_rate_text_raw"),
            spread_bps=data.get("spread_bps"),
            spread_text_raw=data.get("spread_text_raw"),
            cash_rate_pct=data.get("cash_rate_pct"),
            pik_rate_pct=data.get("pik_rate_pct"),
            pik_rate_max_pct=data.get("pik_rate_max_pct"),
            pik_type=data.get("pik_type"),
            floor_pct=data.get("floor_pct"),
            maturity_date=data.get("maturity_date"),
            acquisition_date=data.get("acquisition_date"),
            par_principal=data.get("par_principal"),
            amortized_cost=data.get("amortized_cost"),
            fair_value=data.get("fair_value"),
            shares=data.get("shares"),
            pct_net_assets=data.get("pct_net_assets"),
            unfunded_amount=data.get("unfunded_amount"),
            currency=data.get("currency") or "USD",
            footnote_markers=list(data.get("footnote_markers") or []),
            _source_table_index=table_idx,
            _source_row_index=row_idx,
        )
        return inv

    # ── Totals ─────────────────────────────────────────────────────────

    def _capture_total(self, row: Tag, schema: Optional[ColumnSchema],
                       table_idx: int, row_idx: int, kind: str) -> None:
        if schema is None:
            return
        spans = self._row_spans(row)
        label = next((t for _, _, t in spans if t), None) or kind
        cost = get_cell_text_for_field(spans, schema, "amortized_cost_text")
        fv = get_cell_text_for_field(spans, schema, "fair_value_text")
        pct = get_cell_text_for_field(spans, schema, "pct_net_assets_text")
        tot = SectionTotal(
            label=label,
            amortized_cost=(parse_number(cost) * self.unit_multiplier
                            if parse_number(cost) is not None else None),
            fair_value=(parse_number(fv) * self.unit_multiplier
                        if parse_number(fv) is not None else None),
            pct_net_assets=parse_percent(pct),
            _source_table_index=table_idx,
            _source_row_index=row_idx,
        )
        self.section_totals.append(tot)

    # ── Investment-type normalization ───────────────────────────────────

    _FIRST_LIEN_RE = re.compile(
        r"first[\s-]lien|1st\s+lien\b|senior\s+secured\s+first[\s-]lien|"
        r"unitranche|\bone\s+stop\b|"
        r"^\s*senior\s+secured\b",  # GBDC: "Senior secured" bare
        re.I,
    )
    _SECOND_LIEN_RE = re.compile(r"second[\s-]lien|2nd\s+lien\b|last[\s-]out", re.I)
    _UNSEC_RE = re.compile(
        r"unsecured|senior\s+notes?\b|subordinated\s+notes?\b|"
        r"senior\s+convertible\s+notes?\b|"
        r"structured\s+finance\s+obligation|trust\s+claim|collateralized\s+financing",
        re.I,
    )
    _MEZZ_RE = re.compile(
        r"mezzanine|subordinated\s+loan|sub[-\s]?debt|"
        r"^\s*subordinated\b",  # NMFC: "Subordinated (Tranche A)" bare
        re.I,
    )
    _REVOLVER_RE = re.compile(r"revolver|revolving", re.I)
    _DDT_RE = re.compile(r"delayed[-\s]?draw", re.I)
    _PREF_RE = re.compile(r"\bpref(?:\.|erred)?\s+(equity|stock|shares?|units?)", re.I)
    _COMMON_RE = re.compile(
        r"common\s+(stock|equity|units?|shares?)|"
        r"class\s+[a-z](-?\d)?\s+(common|shares?|units?)|"
        r"class\s+[a-z](-?\d)?\s+\w+\s+(shares?|units?|stock)",  # "Class B redeemable shares"
        re.I,
    )
    _WARRANT_RE = re.compile(r"warrant", re.I)
    _EQUITY_RE = re.compile(
        r"\bequity\b|partnership\s+interests?|membership\s+interests?|"
        r"\b(llc|llp|lp|lp/lll|ltd)\s+(interests?|units?|shares?)\b|"
        r"limited\s+partner(?:ship)?\s+interests?|"
        r"\bpartnership\s+units?\b|"
        r"\bseries\s+[A-Z](?:-?\d+)?\s+units?\b|"
        r"\bclass\s+[A-Z](?:-?\d+)?\s+interests?\b|"
        r"ordinary\s+shares?|participation\s+interests?|"
        r"escrow\b|co[-\s]?invest|trust\s+preferred",
        re.I,
    )
    _JV_RE = re.compile(r"joint\s+venture|joint\s*venture", re.I)
    _ABF_RE = re.compile(
        r"\basset[\s-]?based\s+(?:finance|lending)\b|\bABF\b|"
        r"\bspecialty\s+finance\s+(?:debt|loan)",
        re.I,
    )

    def _normalize_investment_type(self, raw: str) -> Optional[str]:
        if not raw:
            return None
        s = raw
        if self._JV_RE.search(s):
            return "JV"
        if self._ABF_RE.search(s):
            return "ABF"
        if self._DDT_RE.search(s):
            return "DDT"
        if self._REVOLVER_RE.search(s):
            return "Revolver"
        if self._FIRST_LIEN_RE.search(s):
            return "1L"
        if self._SECOND_LIEN_RE.search(s):
            return "2L"
        if self._UNSEC_RE.search(s):
            return "Unsec"
        # Mezz check AFTER Unsec so that "Subordinated notes" → Unsec, but
        # bare "Subordinated" (NMFC) → Mezz.
        if self._MEZZ_RE.search(s):
            return "Mezz"
        if self._WARRANT_RE.search(s):
            return "Warrant"
        if self._PREF_RE.search(s):
            return "Preferred"
        if self._COMMON_RE.search(s):
            return "Equity"
        if self._EQUITY_RE.search(s):
            return "Equity"
        return None

    # ── Footnote legend ─────────────────────────────────────────────────

    def _extract_footnote_legend(self) -> None:
        """Structurally locate the footnote legend.

        The legend is authored as narrative paragraphs (not table cells).
        Every entry has the form: a paragraph whose text starts with `(N)`
        or `(abc)` followed by a definition sentence. We build a narrative-
        only text stream by walking the DOM and skipping nodes that are
        inside a `<table>`. Every `(N)` that opens a narrative paragraph in
        the SOI-tail region is a legend entry; table-internal `(N)` markers
        (which are row footnotes pointing BACK to the legend) are excluded.

        Scope the scan to the SOI-tail region: starting from the last SOI
        table, collect narrative text until we hit a major heading or the
        body starts discussing a different primary section.
        """
        if not self.soup:
            return
        soi_tables = self.find_soi_tables()
        if not soi_tables:
            return

        # Strategy: build a text stream of text nodes that live OUTSIDE
        # any <table>, restricted to the SOI-tail region — i.e., text that
        # appears *after* the first SOI table and *before* the next major
        # top-level section break. This prevents us from picking up
        # legend-lookalike enumerations from other 10-K sections (Item 7
        # MD&A, investment-adviser agreement criteria, etc.) which each have
        # their own (1), (2), ... enumerations.
        first_soi = soi_tables[0]
        # Identify the char offset in the DOM traversal order.
        # Simpler, more robust: assign each text node a sequence number,
        # and keep only those whose seq is > seq_of(first_soi_first_text)
        # and < seq_of(first_non_soi_section_heading_after_soi).
        from bs4 import NavigableString
        body = self.soup.find("body") or self.soup
        all_strings = list(body.find_all(string=True))
        # Build a seq map: string node -> index
        seq = {id(s): i for i, s in enumerate(all_strings)}
        # Find the min seq of any text inside the FIRST SOI table.
        soi_string_seqs = [
            seq[id(s)] for s in first_soi.find_all(string=True)
            if id(s) in seq
        ]
        start_seq = min(soi_string_seqs) if soi_string_seqs else 0

        # End marker: the first text node whose content looks like a
        # post-SOI section heading (Item X, PART Y, Exhibit Z) that comes
        # AFTER the last SOI table. Only consider nodes that are not inside
        # a table (so we don't trip on per-row text).
        last_soi = soi_tables[-1]
        last_soi_seqs = [
            seq[id(s)] for s in last_soi.find_all(string=True)
            if id(s) in seq
        ]
        after_last_soi_seq = (max(last_soi_seqs) + 1) if last_soi_seqs else start_seq
        section_heading_re = re.compile(
            r"^(Item\s+\d+[A-Z]?\.|PART\s+[IVX]+\b|Exhibit\s+\d+)",
            re.I,
        )
        end_seq = len(all_strings)
        for i, s in enumerate(all_strings):
            if i <= after_last_soi_seq:
                continue
            if s.find_parent("table") is not None:
                continue
            txt = str(s).replace("\xa0", " ").strip()
            if section_heading_re.match(txt):
                end_seq = i
                break

        # Build narrative stream from text nodes in [start_seq, end_seq)
        # that are OUTSIDE any table.
        parts: list[str] = []
        for i, s in enumerate(all_strings):
            if i < start_seq or i >= end_seq:
                continue
            if s.find_parent("table") is not None:
                continue
            txt = str(s).replace("\xa0", " ")
            if not txt.strip():
                continue
            parts.append(txt.strip())
            parts.append("\n")
        narrative = " ".join(parts)
        # Normalize whitespace per paragraph
        paragraphs = [re.sub(r"\s+", " ", p).strip()
                      for p in narrative.split("\n")]
        paragraphs = [p for p in paragraphs if p]

        # In-table legend cells: some filers (KKRDL, KBDC, and the
        # presentation pattern they share with other 10-Ks) put the
        # footnote legend INSIDE a table at the end of the SOI section
        # — typically as a wide colspan cell at the foot of the table,
        # or as a separate small table immediately after the SOI body.
        # The narrative-stream loop above skips these. To recover them,
        # scan each <tr> inside the SOI tables (which find_soi_tables
        # returns), find rows whose first non-empty cell text starts
        # with a marker pattern AND whose content matches the legend
        # vocabulary filter, and add those as candidate paragraphs.
        # The downstream marker_inline_re / marker_only_re + legend_vocab_re
        # filters do the actual classification.
        legend_marker_prefix = re.compile(r"^\s*(?:\([0-9a-z]{1,3}\)|\d{1,3}\.)\s+\S")
        soi_tables_for_legend = list(self.find_soi_tables())
        # Also scan nearby small tables AFTER the last SOI table — KKRDL,
        # KBDC, and similar filers place the legend in a separate small
        # table immediately following the SOI section.
        if soi_tables_for_legend:
            last_idx = self.tables.index(soi_tables_for_legend[-1])
            for ti in range(last_idx + 1, min(last_idx + 30, len(self.tables))):
                t = self.tables[ti]
                # Skip large tables (likely income statement / balance
                # sheet / unrelated content). A legend table is small.
                if len(t.find_all("tr")) > 40:
                    continue
                soi_tables_for_legend.append(t)
        for tbl in soi_tables_for_legend:
            for tr in tbl.find_all("tr"):
                cells = [c.get_text(" ", strip=True).replace("\xa0", " ")
                         for c in tr.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if not cells:
                    continue
                # A legend row typically has ONE wide cell with the
                # full definition, or two cells: marker + definition.
                if len(cells) == 1 and legend_marker_prefix.match(cells[0]):
                    paragraphs.append(re.sub(r"\s+", " ", cells[0]).strip())
                elif len(cells) == 2 and re.match(
                    r"^\s*(?:\([0-9a-z]{1,3}\)|\d{1,3}\.)\s*$", cells[0]
                ):
                    paragraphs.append(cells[0].strip())
                    paragraphs.append(cells[1].strip())

        # A legend entry is a paragraph that begins with "(N) " followed by
        # a definitional sentence, OR a "(N)" on its own followed by the
        # definition on the next paragraph.
        #
        # Some filers (ANTARES, CION, possibly others) use period-suffixed
        # markers instead of parens:
        #   ANTARES: "17. Loan is on non-accrual status."
        #   CION:    "q. Investment or a portion thereof was on non-accrual…"
        # The legend_vocab_re downstream filter prevents this from
        # over-matching narrative numbered/lettered lists (only paragraphs
        # whose meaning contains SOI-specific vocabulary are kept).
        marker_inline_re = re.compile(
            r"^(?:\(([0-9a-z]{1,3})\)|(\d{1,3})\.|([a-z]{1,3})\.)\s+(\S.*)"
        )
        marker_only_re = re.compile(
            r"^(?:\(([0-9a-z]{1,3})\)|(\d{1,3})\.|([a-z]{1,3})\.)$"
        )
        # SOI-legend vocabulary filter: require the meaning to mention
        # SOI-relevant terms, which filters out unrelated markers in
        # other sections of the 10-K.
        legend_vocab_re = re.compile(
            r"\b(portfolio\s+company|loan|investment|pledged|qualifying\s+asset|"
            r"non[-\s]?accrual|interest\s+rate|variable\s+rate|fixed\s+rate|"
            r"affiliated\s+person|Section\s+55|SOFR|LIBOR|delayed[-\s]?draw|"
            r"revolving|letter\s+of\s+credit|senior\s+secured|unfunded|"
            r"first\s+lien|second\s+lien|unsecured|subordinated|"
            r"Investment\s+Company\s+Act|as\s+defined|"
            # Cross-BDC pattern #7 expansion. FSK fn(l)/fn(m) "Security held
            # within KKR-FSK CLO 2 LLC", fn(y) "Security is non-income
            # producing", fn(aa) "Level 1 or 2 in fair value hierarchy",
            # fn(ab) "Position unsettled". BXSL fn(3) "cost represents the
            # original cost", fn(4) "valued using unobservable inputs", fn(8)
            # "no interest rate floors", fn(19) "exempt from registration".
            # HLEND fn(22) restricted-security under 1933 Act. Each text
            # lacks the original vocab tokens but is unambiguously a legend
            # entry — broaden to catch them.
            r"security|securities|held\s+within|collateraliz|"
            r"non[-\s]?income|level\s*[123]|fair\s+value\s+hierarch|"
            r"unobservable|cost\s+represents|original\s+cost|"
            r"interest\s+rate\s+floor|exempt\s+from\s+registration|"
            r"securities\s+act\s+of\s+1933|restricted\s+(?:security|securities)|"
            r"unsettled|debt\s+securitization|credit\s+facility|"
            r"clo\b|joint\s+venture|securitization|"
            r"subsidiary|cash\s+equivalent|money\s+market|"
            r"amortiz|premium|discount|warrant|preferred|common\s+(stock|equity))\b",
            re.I,
        )
        # Prefer the MORE DEFINITIONAL match per marker. Some markers
        # appear twice — once as a section-prefix in narrative text
        # ("(30) The following shows the composition...") and once as
        # the actual footnote definition ("(30) Loan was on non-accrual
        # status..."). Without disambiguation the first-wins rule picks
        # the narrative.
        #
        # A match is more definitional when EITHER:
        #   (a) it starts with a per-row assertion phrase ("Loan was",
        #       "Investment was", "Position was", "Asset is", "Non-
        #       accrual", "Classified as", "Denotes", "All or a portion",
        #       "Investment is", "These investments", "Includes"); OR
        #   (b) it is shorter (< 400 chars) AND the current stored
        #       meaning is long-form narrative (> 400 chars).
        DEFINITIONAL_LEAD_RE = re.compile(
            r"^\s*(?:loan|investment|position|asset|securities?)\s+(?:was|is|are)\b"
            r"|^\s*non[-\s]?accrual\b"
            r"|^\s*classified\s+as\b"
            r"|^\s*denotes\b"
            r"|^\s*all\s+or\s+a\s+portion\b"
            r"|^\s*these\s+investments\b",
            re.I,
        )

        # Cross-period scoping (pattern #10). When the same 10-K contains the
        # current SOI plus the prior-period comparative SOI, the legend
        # extractor can land on prior-period text for markers that recycled
        # numbers (e.g. ARCC fn(10) "non-accrual as of December 31, 2024"
        # vs the current-period (10) "SDLP excess cash flow"). When period_end
        # is set, prefer entries that explicitly cite the current year over
        # those citing the prior year.
        period_year = None
        prior_year = None
        if self.period_end and len(self.period_end) >= 4:
            try:
                py = int(self.period_end[:4])
                period_year = str(py)
                prior_year = str(py - 1)
            except ValueError:
                pass
        current_year_re = re.compile(rf"\b{period_year}\b") if period_year else None
        prior_year_re = re.compile(rf"\b{prior_year}\b") if prior_year else None

        def _cites_current(s: str) -> bool:
            return bool(current_year_re and current_year_re.search(s))

        def _cites_prior_only(s: str) -> bool:
            return bool(
                prior_year_re and prior_year_re.search(s)
                and not (current_year_re and current_year_re.search(s))
            )

        def _produces_flag(s: str) -> bool:
            # A meaning that classifies to ANY controlled-vocab flag is by
            # construction a legend entry. Distinguishes pledge / NQA /
            # non-accrual / Level-3 / affiliation definitions from narrative
            # paragraphs that happen to land in the legend region.
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
            # Highest priority: a meaning that resolves to a controlled-vocab
            # flag beats a meaning that doesn't. ARCC fn(2) regression: the
            # narrative SDLP-unitranche paragraph ("First lien senior secured
            # loans include certain loans that the SDLP classifies...") was
            # winning over the actual pledged-collateral legend entry ("All
            # or a portion of these debt investments are pledged as
            # collateral..."), zeroing out ARCC's pledged-FV metric.
            if new_flagged and not old_flagged:
                return True
            if old_flagged and not new_flagged:
                return False
            # Definitional status: a definitional lead-in ("Loan was...",
            # "These investments...", "Denotes...") is structurally a legend
            # entry; a non-definitional paragraph that happens to cite the
            # current year (balance-sheet narrative or MD&A excerpt) is not.
            if new_def and not old_def:
                return True
            if not new_def and old_def:
                return False
            # Both same definitional status: prefer current-period over
            # prior-only.
            if new_current and old_prior:
                return True
            if old_current and new_prior:
                return False
            if new_current and not old_current:
                return True
            if old_current and not new_current:
                return False
            # Same definitional status, same period: prefer shorter when
            # existing is long.
            return len(new) < 400 and len(existing) > 400

        legend: dict[str, str] = {}
        for i, p in enumerate(paragraphs):
            marker: Optional[str] = None
            meaning: Optional[str] = None
            m = marker_inline_re.match(p)
            if m:
                # group(1) = parens form '(N)/(abc)' ; group(2) = digit-dot 'N.' ;
                # group(3) = letter-dot 'q.'.
                marker = m.group(1) or m.group(2) or m.group(3)
                meaning = m.group(4).strip()
            else:
                m2 = marker_only_re.match(p)
                if m2 and i + 1 < len(paragraphs):
                    marker = m2.group(1) or m2.group(2) or m2.group(3)
                    meaning = paragraphs[i + 1].strip()
            if not marker or not meaning:
                continue
            if len(meaning) < 15:
                continue
            if not legend_vocab_re.search(meaning):
                continue
            # Cap meaning length to the first ~1000 chars — we only need the
            # opening definition sentence, not a full paragraph dump.
            if len(meaning) > 1000:
                cut = meaning.rfind(". ", 0, 1000)
                meaning = meaning[: cut + 1] if cut > 200 else meaning[:1000]
            if marker in legend:
                if _more_definitional(meaning, legend[marker]):
                    legend[marker] = meaning
                continue
            legend[marker] = meaning

        self.footnote_legend = legend

    def _apply_footnotes(self) -> None:
        # TC3-ARCC-RESIDUAL (post-legend pass): single uppercase letters in
        # _RATE_RESET_CODES that are not keys in the footnote legend are
        # rate-reset frequency codes or corporate-name qualifiers extracted from
        # non-rate cells.  Strip them now that we have the full legend available.
        # (During _build_investment the legend is not yet built, so this cleanup
        # cannot happen there.)
        for inv in self.investments:
            inv.footnote_markers = [
                m for m in inv.footnote_markers
                if not (
                    len(m) == 1
                    and m.upper() in _RATE_RESET_CODES
                    and m not in self.footnote_legend
                )
            ]
        for inv in self.investments:
            meanings: dict[str, str] = {}
            flags: list[str] = []
            for m in inv.footnote_markers:
                meaning = self.footnote_legend.get(m)
                if meaning:
                    meanings[m] = meaning
                    # TC3-AV01: normalize curly/smart quotes before vocab matching.
                    # ARCC (and other filers) store footnote legend text with Unicode
                    # U+201C/U+201D around "Affiliated Person". The CONTROLLED_VOCAB_RULES
                    # patterns use ASCII quotes; without clean_text() the match returns [].
                    flags.extend(classify_footnote_meaning(clean_text(meaning), period_end=self.period_end))
            inv.footnote_meanings = meanings
            inv.footnote_flags = sorted(set(flags))

        # Cross-BDC pattern #2: default-on pledged inference.
        # A legend entry like "Unless otherwise indicated, [securities/portfolio
        # companies] are pledged as collateral for [Credit Facility / 2024 Debt
        # Securitization / etc.]" sets ALL rows as pledged-by-default. The
        # explicit "is not pledged" markers (OBDC fn(9)/(26), HLEND fn(20)/(21),
        # GBDC fn(28), BCRED) carve out the exceptions. Without this rule,
        # HLEND $25.3B, MAIN, OCIC etc. show is_pledged=0 across the entire
        # portfolio even though the 10-K asserts the opposite by default.
        default_on_pledged_re = re.compile(
            r"\bunless\s+(?:otherwise\s+(?:indicated|noted)|expressly\s+noted)\b"
            r".{0,400}\bpledged\b"
            r"|\bpledged\b.{0,300}\bunless\s+(?:otherwise\s+(?:indicated|noted)|expressly\s+noted)\b"
            # Match "each/all/every/any of the [Company's] investments is pledged".
            # Accept ASCII apostrophe, Unicode curly apostrophe (U+2019), or
            # nothing — OSCF fn(3) uses U+2019; OBDC/HLEND use plain ASCII.
            r"|\b(?:each|all|every|any)\s+(?:of\s+)?(?:the\s+)?(?:company[’']?s\s+)?investments?\s+(?:is|are)\s+pledged\b",
            re.I | re.DOTALL,
        )
        default_on = any(
            default_on_pledged_re.search(meaning)
            for meaning in self.footnote_legend.values()
        )
        # Scope: does the default-on rule explicitly limit to debt investments?
        # BCRED fn(1)/(5) reads "Each of the Company's debt investments is
        # pledged as collateral..." — the convention does NOT cover equity / JV
        # holdings. When this debt-scoping language appears, skip equity rows.
        debt_scoped_re = re.compile(
            r"\bdebt\s+investments?\b.{0,80}\bpledged\b"
            r"|\bpledged\b.{0,80}\bdebt\s+investments?\b"
            r"|\bloans?\s+are\s+pledged\b"
            # Negating-form diagnostic. BCRED fn(5): "These debt investments
            # are not pledged as collateral" — the explicit "debt" modifier
            # tells us the umbrella default-on convention only covers debt.
            r"|\bdebt\s+investments?\s+are\s+not\s+pledged\b"
            r"|\bother\s+debt\s+investments?\b.{0,80}\bpledged\b",
            re.I | re.DOTALL,
        )
        debt_scoped = any(
            debt_scoped_re.search(meaning)
            for meaning in self.footnote_legend.values()
        )
        # Markers whose legend entry carries the is_not_pledged flag — these
        # opt rows OUT of the default-on pledged stamp.
        not_pledged_markers: set[str] = set()
        for marker, meaning in self.footnote_legend.items():
            mflags = classify_footnote_meaning(
                clean_text(meaning), period_end=self.period_end
            )
            if "is_not_pledged" in mflags:
                not_pledged_markers.add(marker)
        # If the filing carves out explicit not-pledged exceptions via legend
        # markers but the "unless otherwise indicated" phrasing didn't appear
        # in the legend text (HLEND case — the default-on convention is
        # declared in the MD&A paragraph BEFORE the legend, which the
        # extractor doesn't capture), the existence of not-pledged markers
        # is itself diagnostic of a default-on convention. Treat them
        # symmetrically.
        implicit_default_on = bool(not_pledged_markers) and not default_on
        # Cross-BDC pattern #5: default-on is_level_3 inference.
        # GBDC fn(4)/MAIN fn(18)/ARCC fn(16) declare 'unobservable inputs ...
        # unless otherwise noted' — every non-cash investment is Level 3
        # except those marked by a negating footnote. Mirror the pledged
        # default-on machinery for Level-3.
        # "unless otherwise indicated/noted" OR "unless noted otherwise" OR
        # "unless expressly noted" — three valid orderings in the wild.
        _unless_re = (
            r"\bunless\s+(?:otherwise\s+(?:indicated|noted)|noted\s+otherwise|"
            r"expressly\s+noted)\b"
        )
        default_on_l3_re = re.compile(
            r"(?:valued|determined|fair\s+value(?:s)?)\s+(?:were\s+|was\s+|is\s+|are\s+)?"
            r"(?:using\s+)?(?:significant\s+)?unobservable\s+inputs?\b"
            rf".{{0,80}}{_unless_re}"
            rf"|{_unless_re}"
            r".{0,80}(?:valued|determined|fair\s+value)\s+(?:were\s+|was\s+|is\s+|are\s+)?"
            r"(?:using\s+)?(?:significant\s+)?unobservable\s+inputs?\b"
            # ARCC fn(16) inverse phrasing: "Other than the investments noted
            # by this footnote, the fair value ... is determined using
            # unobservable inputs..." — same semantic (default-on).
            r"|\bother\s+than\s+the\s+investments\s+noted\s+by\s+this\s+footnote\b"
            r".{0,200}\bunobservable\s+inputs?\b",
            re.I | re.DOTALL,
        )
        default_on_l3 = any(
            default_on_l3_re.search(meaning)
            for meaning in self.footnote_legend.values()
        )
        # ARCC fn(16) is the negating marker for L3 default-on (the inverse:
        # "Other than the investments noted by THIS FOOTNOTE..." means rows
        # carrying (16) are NOT L3, everything else is). Also FSK fn(aa)
        # 'Security is classified as Level 1 or Level 2 in fair value
        # hierarchy' — same diagnostic, opposite phrasing. Both resolve to
        # the is_not_level_3 controlled-vocab flag.
        not_l3_markers: set[str] = set()
        inverse_l3_re = re.compile(
            r"\bother\s+than\s+the\s+investments\s+noted\s+by\s+this\s+footnote\b",
            re.I,
        )
        for marker, meaning in self.footnote_legend.items():
            if inverse_l3_re.search(meaning):
                not_l3_markers.add(marker)
            mflags = classify_footnote_meaning(
                clean_text(meaning), period_end=self.period_end
            )
            if "is_not_level_3" in mflags:
                not_l3_markers.add(marker)
        # Implicit default-on: presence of is_not_level_3 markers alone is
        # diagnostic of a default-on L3 convention (mirror the pledged
        # implicit_default_on logic — a filing only carves out 'classified as
        # Level 1/2' exceptions when L3 is the default).
        implicit_default_on_l3 = bool(not_l3_markers) and not default_on_l3
        if default_on_l3 or implicit_default_on_l3:
            for inv in self.investments:
                if inv.is_cash_equivalent:
                    continue
                if inv.investment_category == "Unfunded Commitment" and (
                    inv.fair_value is None or inv.fair_value <= 0
                ):
                    continue
                flags = set(inv.footnote_flags or [])
                if not_l3_markers & set(inv.footnote_markers):
                    flags.discard("is_level_3")
                    inv.is_level_3 = False
                elif "is_level_3" not in flags:
                    flags.add("is_level_3")
                    inv.is_level_3 = True
                inv.footnote_flags = sorted(flags)

        if default_on or implicit_default_on or not_pledged_markers:
            for inv in self.investments:
                # Cash equivalents and pure unfunded-commitment lines are
                # not pledged collateral by economic identity — skip.
                if inv.is_cash_equivalent:
                    continue
                if inv.investment_category == "Unfunded Commitment" and (
                    inv.fair_value is None or inv.fair_value <= 0
                ):
                    continue
                is_equity_row = (
                    inv.investment_category == "Equity"
                    or inv.investment_type in ("Equity", "Preferred", "Warrant", "JV")
                )
                flags = set(inv.footnote_flags or [])
                row_negates = bool(
                    not_pledged_markers & set(inv.footnote_markers)
                )
                if row_negates:
                    flags.discard("is_pledged")
                    inv.is_pledged = False
                elif debt_scoped and is_equity_row:
                    # BCRED D-7: fn(1)/(5) legend explicitly says "Each of the
                    # Company's DEBT investments is pledged..." — strip any
                    # is_pledged flag that the per-row marker-to-flag pass
                    # added based on the literal "pledged as collateral"
                    # match. Equity / JV holdings are not covered.
                    flags.discard("is_pledged")
                    inv.is_pledged = False
                elif (default_on or implicit_default_on) and "is_pledged" not in flags:
                    flags.add("is_pledged")
                    inv.is_pledged = True
                inv.footnote_flags = sorted(flags)
