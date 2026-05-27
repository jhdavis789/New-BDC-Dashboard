"""
Reusable structural helpers. Nothing in here is BDC-specific.

Design rule: every function takes HTML/Tag input and returns a typed result.
No function takes or returns a cell index. No function contains a company
name or ticker whitelist. See research/SKILLS/parsing/flexible-structural.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from bs4 import Tag

# ── Text hygiene ────────────────────────────────────────────────────────────


def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\xa0", " ").replace("​", "").replace(" ", " ")
    s = s.replace("–", "-").replace("−", "-")
    # Normalize curly/smart quotes to ASCII so footnote vocab regexes match.
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"') 
    s = s.replace("\u201e", '"').replace("\u201f", '"')
    # Recover known mojibake patterns from broken upstream encoding.
    # SEC EDGAR submissions occasionally embed the UTF-8 byte sequence
    # "\u00ef\u00bf\u00bd" (the U+FFFD REPLACEMENT CHARACTER serialized as Latin-1) in
    # place of legitimate diacritics. The most common case in BDC filings
    # is the Luxembourg corporate-form "S.\u00e0 r.l." (Soci\u00e9t\u00e9 \u00e0 responsabilit\u00e9
    # limit\u00e9e) mis-encoded as "S.\u00ef\u00bf\u00bd r.l." \u2014 surfaced in ADS Q1 2026 for
    # MetaTiedot Midco S.\u00e0 r.l. (4 rows, $42M FV). The encoded form
    # creates phantom cross-period churn if the filer uses the correct
    # spelling in some periods and the broken spelling in others.
    if "\u00ef\u00bf\u00bd" in s or "\ufffd" in s:
        s = s.replace("S.\u00ef\u00bf\u00bd r.l", "S.\u00e0 r.l")
        s = s.replace("S.\ufffd r.l", "S.\u00e0 r.l")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ── Industry alias normalization ────────────────────────────────────────────
#
# Cross-BDC HTML typos / capitalization variants that produce duplicate
# industry buckets in aggregations. Keys are exact post-clean_text strings;
# values are the canonical form. Applied ONLY to industry banner cells and
# per-row industry column cells (not to companies, instruments, numbers).
#
# Keys must be specific enough that they would never legitimately appear as
# a non-industry cell (company name, instrument, amount, etc.) -- typos like
# "Diversifed" (missing "i") guarantee that. The case-variant
# "Specialty retail" virtually only ever appears as an industry banner in
# SOI tables.
#
# Cases captured (2026-05-16):
#   - OTF:  "Diversifed Financial Services" (HTML typo, missing "i")
#   - OCIC: "Specialty retail" (lowercase variant)
#   - GBDC: "Energy, Equipment & Services" (extra comma -- GICS canonical
#           form has no comma between "Energy" and "Equipment")
INDUSTRY_ALIASES: dict = {
    # OTF -- HTML typo, 2 AAM rows / $37.452M, merges into 9.82% combined
    "Diversifed Financial Services": "Diversified Financial Services",
    # OCIC -- case variant; 8 rows lowercase + 2 rows title-case
    "Specialty retail": "Specialty Retail",
    # GBDC -- extra comma vs GICS canonical
    "Energy, Equipment & Services": "Energy Equipment & Services",
}


def canonicalize_industry(s):
    """Return the canonical industry name for `s` if it matches an alias key;
    otherwise return `s` unchanged. Comparison is on the exact post-trim
    string -- no fuzzy matching, no case folding, no whitespace collapse
    beyond what clean_text already did."""
    if not s:
        return s
    return INDUSTRY_ALIASES.get(s, s)


FOOTNOTE_NUMERIC_RE = re.compile(r"\((\d+)\)")
# Country / geographic / corporate qualifiers that are NEVER footnote
# markers even when they appear inside parens in a company-name cell.
# Capital letters and length > 3 disambiguate "(USA)", "(UK)", "(N.A.)",
# "(REITs)", "(Australia)" from real markers "(a)", "(b)", "(ac)" etc.
# Audit caught these across OBDC (RL Datix "(USA)"), FSK ("Australia",
# "REITs"); cross-BDC fix.
# Match BOTH lowercase ("(a)", "(ac)") and capital-letter ("(F)", "(B)")
# alpha footnote markers used by some filers (e.g. BDEBT).
FOOTNOTE_ALPHA_RE = re.compile(r"\(([A-Za-z]+)\)")

# A real BDC footnote alpha marker is short (≤3 chars) AND all lowercase
# (a, b, c, …, aa, ab, ac, ad, ae). Anything else inside parens — "USA",
# "UK", "REITs", "Australia", "N.A." — is a geographic / corporate
# qualifier and must not be treated as a footnote marker.
def _looks_like_alpha_footnote(token: str) -> bool:
    return bool(token) and len(token) <= 3 and token.islower()


# RC-D: cells whose ENTIRE text is a "large" parenthesized number — has
# a comma, a decimal, or 4+ digits — are accounting negatives, NOT
# footnote markers. Single small `(N)` like `(2)`, `(15)` is a marker.
# `(1,234)` or `(123.5)` or `(9999)` is a negative number.
_PAREN_NEG_NUMBER_ONLY_RE = re.compile(
    r"^\s*\(\s*-?\d[\d.,]{2,}\s*\)\s*$"   # 4+ chars after `(` → number
)

# R6-AV09: single-uppercase-letter rate-reset frequency codes (Q/M/S/D/A/B)
# that appear inline in rate cells as "SOFR (Q)" must not be treated as
# footnote markers.  Module-level constants avoid recompilation per call.
_RATE_RESET_CODES: frozenset = frozenset("QMSABD")
# TC3-ARCC-RESIDUAL: company-name legal-qualifier pattern "(<LETTER>),"  e.g.
# "BCPE Co-Invest (A), LP".  A single uppercase letter immediately followed by
# a comma is a corporate-name parenthetical, never a footnote marker.
_COMPANY_NAME_QUALIFIER_RE = re.compile(r"\(([A-Z])\),")
_RATE_TOKEN_IN_CELL_RE = re.compile(
    r"\b(?:SOFR|SONIA|EURIBOR|LIBOR|PRIME|CORRA|BBSY|BKBM|STIBOR|TIBOR|"
    r"SARON|NIBOR|JIBAR|WIBOR|PRIBOR|BUBOR|HIBOR|AONIA|CDI|TONA|CDOR|"
    # R8-D3: NMFC uses FIXED(Q)* in the reference-rate column for fixed-rate
    # investments that reset quarterly. FIXED is not a benchmark rate but is a
    # rate-cell token whose presence confirms (Q) is a reset-frequency code.
    r"FIXED|"
    r"Base\s+Rate|S\s*\+|SF\s*\+|E\s*\+|P\s*\+"
    # R8-D3: NMFC also uses bare 'P' (Prime abbreviation) followed by a
    # reset-frequency paren like P(Q). Match P only when immediately followed
    # by '(' so we don't over-match in other contexts.
    r"|P(?=\s*\())\b",
    re.I,
)


def extract_footnote_markers(
    s: str,
    is_money_column: bool = False,
    footnote_legend: Optional[dict] = None,
) -> list[str]:
    """Return every footnote marker string like '1', '14', 'a', 'ac' present in s.

    Markers come in two forms in BDC SOIs: numeric '(N)' and lowercase-letter
    '(abc)'. The alpha form can be multi-letter ('(ac)', '(ae)'). Order of
    appearance is preserved. Duplicates removed.

    RC-D: if the cell sits in a money column AND its entire text is a single
    parenthesized number (accounting notation for a negative dollar value,
    e.g. "(79)" means -$79), return [] — it's not a footnote marker.

    For non-money columns, single small `(N)` (1-3 digits) is treated as
    a footnote marker. Multi-digit-with-comma/decimal `(1,234)` always
    looks like a number regardless of column.

    R6-AV09: rate-reset frequency codes (Q/M/S/D) appear inline in rate
    cells as "(Q)" / "(M)" / "(S)" immediately after a benchmark name
    ("SOFR (Q) 5.75%"). These are NOT footnote markers. They are filtered
    when they appear as a lone single uppercase letter AND (a) the cell
    contains a known reference-rate token, OR (b) a legend is provided
    and the letter is not a key in that legend.
    """
    if not s:
        return []
    # Always reject "(NN.NN)" or "(N,NNN)" — those are clearly numbers
    if _PAREN_NEG_NUMBER_ONLY_RE.match(s):
        return []
    # In a money column, ANY standalone "(N)" is a negative dollar amount.
    # Tolerate internal whitespace ("( 8 )") and decimal/comma forms.
    if is_money_column:
        if re.match(r"^\s*\(\s*-?\d[\d.,]*\s*\)\s*$", s):
            return []
    out: list[str] = []
    for m in FOOTNOTE_NUMERIC_RE.findall(s):
        out.append(m)
    for m in FOOTNOTE_ALPHA_RE.findall(s):
        # Skip uppercase or long alpha tokens — they're geographic /
        # corporate qualifiers ("USA", "UK", "REITs", "Australia"), not
        # footnote markers.
        if not _looks_like_alpha_footnote(m):
            continue
        out.append(m)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for m in out:
        if m in seen:
            continue
        seen.add(m)
        uniq.append(m)
    # TC3-ARCC-RESIDUAL: build set of single-uppercase-letter tokens that are
    # corporate-name legal qualifiers "(X)," e.g. "BCPE Co-Invest (A), LP".
    # These appear in the company-name cell and must never be treated as
    # footnote markers regardless of whether the cell contains a rate token.
    company_name_qualifiers: set[str] = {
        m.group(1) for m in _COMPANY_NAME_QUALIFIER_RE.finditer(s)
    }
    # R6-AV09: drop single uppercase letters Q/M/S/D/A/B when they are
    # rate-reset-frequency codes rather than genuine footnote markers.
    # Module-level _RATE_RESET_CODES and _RATE_TOKEN_IN_CELL_RE avoid
    # recompilation on every call.
    cell_has_rate = bool(_RATE_TOKEN_IN_CELL_RE.search(s))
    filtered = []
    for m in uniq:
        # Drop company-name legal qualifier "(X)," — e.g. "(A)" in "Co-Invest (A), LP"
        if len(m) == 1 and m.upper() in company_name_qualifiers:
            continue
        if (
            len(m) == 1
            and m.upper() in _RATE_RESET_CODES
            and cell_has_rate
            and (footnote_legend is None or m not in footnote_legend)
        ):
            continue  # rate-reset code, not a footnote marker
        filtered.append(m)
    return filtered


def strip_footnotes(s: str) -> str:
    if not s:
        return ""
    s = FOOTNOTE_NUMERIC_RE.sub("", s)
    # Strip parenthesized alpha markers — only short, all-lowercase
    # tokens (footnote markers like (a), (ac), (cd)). Uppercase or longer
    # tokens like (USA), (UK), (REITs), (Australia) are geographic /
    # corporate qualifiers and must remain in the company name.
    s = re.sub(r"\(([a-z]{1,3})\)", "", s)
    # BDEBT / GAIN pattern: single-uppercase-letter footnote markers as
    # trailing column-header suffixes. Strips one or more consecutive
    # tail markers like "Issuer(F)", "Ref(B)", "Company and Investment
    # (A)(B)(D)(E)". Only the trailing run is stripped — guards against
    # mid-name (A) in things like "Company A (USA)" or product names
    # with letter qualifiers. Two-or-more-letter uppercase parentheticals
    # (USA, UK, REITs) stay because the regex requires exactly one letter.
    while True:
        new = re.sub(r"\(([A-Z])\)\s*$", "", s).strip()
        if new == s:
            break
        s = new
    # Strip curly-brace footnote markers {N} or {alpha} — used in early ARCC filings
    # (pre-2008 10-Qs) as "Company {1}", "Interest {13}", etc.
    s = re.sub(r"\{[A-Za-z0-9]{1,4}\}", "", s)
    # Strip bare superscript footnote suffixes — comma-separated integers that
    # follow a letter character without any space (CSWC style: "Portfolio Company1,5,6,7").
    # The pattern requires: word char immediately before the digit sequence, and
    # the entire tail is only digits and commas (no letters).
    s = re.sub(r"(?<=[A-Za-z])(\d+(?:,\d+)*)$", "", s)
    # Strip trailing space-separated integer footnote sequences with TWO OR
    # MORE trailing groups (SPCC style: "Portfolio Company 1 2 3 4"). A
    # SINGLE trailing " N" is left in place because it can be a legitimate
    # legal-entity suffix on a company name (ADS Q1 2026: "IOTA HOLDINGS 3",
    # "Actium Midco 3"). Headers with single-digit footnote suffixes (SPCC
    # "Cost 7", "Reference Rate and Spread 5") get the additional strip
    # applied in `normalize_label` where the context is unambiguously a
    # column header.
    s = re.sub(r"(\s+\d+){2,}$", "", s)
    return re.sub(r"\s+", " ", s).strip().rstrip(",").strip()


# ── Number parsing ──────────────────────────────────────────────────────────

_PAREN_NEG = re.compile(r"^\(\s*([\d,][\d,.]*)\s*\)$")
_NUMBER_RE = re.compile(r"^-?\d[\d,]*(\.\d+)?$")


def parse_number(text: str) -> Optional[float]:
    if text is None:
        return None
    t = clean_text(str(text))
    if not t:
        return None
    # Dash-as-zero (filings render '—' for "nil")
    if t in ("-", "—", "–"):
        return 0.0
    # Strip leading currency sigils ($, €, £, ¥) and 3-letter ISO currency
    # codes (EUR, SEK, NOK, DKK, GBP, CAD, AUD, CHF, JPY, NZD) so par values
    # like "EUR 160,000" or "SEK 50,000,000" parse correctly.
    # R6-D13: also strip multi-char currency symbols used by ADS and OCIC:
    #   A$  (AUD),  C$  (CAD),  Skr (SEK),  Nkr (NOK),  ₩  (KRW),  ₣  (CHF)
    # These appear as a leading prefix (possibly standalone in the cell, or
    # combined as "A$ 50,000").  Strip longest first to avoid partial matches.
    t = re.sub(r"^[\$€£¥₩₣₹₺₿]\s*", "", t).strip()
    t = re.sub(r"^(A\$|C\$)\s*", "", t).strip()
    t = re.sub(r"^(Skr|Nkr|Dkr)\s*", "", t, flags=re.I).strip()
    t = re.sub(
        r"^(EUR|GBP|CAD|AUD|CHF|JPY|NZD|SEK|NOK|DKK|MXN|BRL|HKD|SGD|INR|CNY|KRW)\s+",
        "",
        t,
        flags=re.I,
    ).strip()
    t = t.rstrip("%").strip()
    # Strip trailing unit-of-quantity suffixes like "Units", "Shares", "shares"
    # so "13,750,397 Units" → 13750397.
    t = re.sub(r"\s+(units?|shares?|sh\.?)\s*$", "", t, flags=re.I).strip()
    m = _PAREN_NEG.match(t)
    if m:
        inner = m.group(1).replace(",", "")
        try:
            return -float(inner)
        except ValueError:
            return None
    t2 = t.replace(",", "")
    try:
        return float(t2)
    except ValueError:
        return None


def parse_money_number(text: str) -> Optional[float]:
    """Money-cell variant of parse_number: strip trailing footnote markers
    (numeric `(N)` and alpha `(a)`/`(ac)`) from the END of the cell ONLY,
    then parse what remains. Preserves leading parens that are accounting
    negatives (e.g. "( 5 ) (a)" → -5.0; "(11,234)(2)" → -11234.0;
    "(254)" → -254.0).

    For non-money cells, prefer plain parse_number; this helper is for
    amortized_cost / fair_value / par / shares cells where bare `(N)` is
    a negative dollar amount, not a footnote.
    """
    if text is None:
        return None
    t = clean_text(str(text))
    if not t:
        return None
    # R6-D12: paren-negative protection.
    # If the cell IS itself a paren-negative number (e.g. "(254)"), parse
    # directly without the trailing-footnote strip — otherwise the strip
    # would interpret it as a numeric footnote marker `(N)` and consume
    # the entire value. _PAREN_NEG matches `^\(\s*<num>\s*\)$`.
    if _PAREN_NEG.match(t):
        return parse_number(t)
    # OSCF (2026-03-31): the FIRST data row of an SOI continuation page
    # carries the leading "$" sigil inline with the value cell (e.g.
    # "$ (58)" / "$ (113)") instead of in a separate sub-column. Without
    # this branch, the trailing-footnote strip below sees "(58)" and
    # consumes it as a footnote marker, leaving just "$" and ultimately
    # parsing as None. Strip a leading currency sigil — only when the
    # remainder is a clean paren-negative — and parse via parse_number.
    # Scoped to `$ (N)` / `$(N)` shapes; mid-cell parens (e.g. "( 5 ) (a)"
    # with a leading "$") still flow through the trailing-strip path
    # because they have additional content after the closing paren.
    _t_no_sigil = re.sub(r"^[\$€£¥₩₣₹]\s*", "", t).strip()
    if _t_no_sigil != t and _PAREN_NEG.match(_t_no_sigil):
        return parse_number(_t_no_sigil)
    # Strip trailing footnote markers iteratively. Numeric `(\d+)` and alpha
    # `(\w{1,4})`. We strip from the RIGHT only.
    while True:
        new_t = re.sub(r"\s*\([A-Za-z]{1,4}\)\s*$", "", t)
        new_t = re.sub(r"\s*\(\d+\)\s*$", "", new_t).strip()
        if new_t == t:
            break
        t = new_t
        # Safety: if we stripped everything, bail.
        if not t:
            return None
        # After each strip, re-check whether what remains is a paren-negative
        # (e.g. "(254)(3)" → "(254)" after stripping the trailing footnote).
        if _PAREN_NEG.match(t):
            return parse_number(t)
    return parse_number(t)


def parse_percent(text: str) -> Optional[float]:
    """Return the percentage as a decimal percent (e.g. '8.74%' -> 8.74)."""
    n = parse_number(text)
    return n


def parse_basis_points(spread_text: str) -> Optional[float]:
    """'4.75%' or '475 bps' -> 475.0."""
    t = clean_text(spread_text or "")
    if not t:
        return None
    low = t.lower()
    if "bps" in low or "bp" in low:
        m = re.search(r"(-?\d[\d,.]*)", low)
        if m:
            return parse_number(m.group(1))
    # percent form
    pct = parse_percent(t)
    if pct is None:
        return None
    return round(pct * 100.0, 4)


_MONTH_YEAR_RE = re.compile(r"^(\d{1,2})/(\d{4})$")
_MONTH_YEAR_SHORT_RE = re.compile(r"^(\d{1,2})/(\d{2})$")  # MM/YY
_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_US_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_US_SHORT_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{2})$")  # MM/DD/YY (MFIC)


def _expand_2digit_year(yr: str) -> str:
    """Expand 2-digit year to 4-digit via the BDC convention: 00-49 → 2000-2049,
    50-99 → 1950-1999. SOI dates rarely go before 1990, so this is safe."""
    n = int(yr)
    return f"20{yr.zfill(2)}" if n < 50 else f"19{yr.zfill(2)}"


def parse_date(text: str) -> Optional[str]:
    """Normalize filing date strings to ISO (YYYY-MM-DD). Month/Year → YYYY-MM-01.

    Supported formats:
      ISO:           YYYY-MM-DD
      US long:       MM/DD/YYYY (also M/D/YYYY)
      US short:      MM/DD/YY (MFIC) — 2-digit year via 2050 cutoff
      Month-only:    MM/YYYY → YYYY-MM-01
      Month-short:   MM/YY → YYYY-MM-01
    """
    t = clean_text(text or "")
    if not t or t in ("-", "—", "–", "N/A", "n/a"):
        return None
    if (m := _ISO_RE.match(t)):
        return t
    if (m := _US_RE.match(t)):
        mo, da, yr = m.groups()
        return f"{yr}-{int(mo):02d}-{int(da):02d}"
    if (m := _US_SHORT_RE.match(t)):
        mo, da, yr = m.groups()
        return f"{_expand_2digit_year(yr)}-{int(mo):02d}-{int(da):02d}"
    if (m := _MONTH_YEAR_RE.match(t)):
        mo, yr = m.groups()
        return f"{yr}-{int(mo):02d}-01"
    if (m := _MONTH_YEAR_SHORT_RE.match(t)):
        mo, yr = m.groups()
        return f"{_expand_2digit_year(yr)}-{int(mo):02d}-01"
    return None


# ── Reference-rate tokenization ─────────────────────────────────────────────

# Full-name tokens, checked LONGEST FIRST so EURIBOR matches before "OR",
# and STIBOR matches before TIBOR. Add new tokens to this tuple — they
# are sorted at module load.
REFERENCE_RATE_TOKENS_RAW = (
    "SOFR", "SONIA", "EURIBOR", "CDOR", "CORRA", "BBSY", "BBSW", "TONA", "NIBOR",
    "BKBM", "PRIME", "LIBOR", "TIBOR", "STIBOR", "SARON", "CDI", "KORIBOR",
    "JIBAR", "WIBOR", "PRIBOR", "BUBOR", "MOSPRIME", "PRIBORF",
    "HIBOR", "AONIA",
    # R6-D3: "Base Rate" is the colloquial term for the alternate base rate
    # (Federal Funds Rate or Prime Rate).  Normalize to PRIME.
    "BASE RATE",
)
REFERENCE_RATE_TOKENS = tuple(sorted(REFERENCE_RATE_TOKENS_RAW, key=len, reverse=True))

# RC-B: abbreviated forms used by filers in rate cells. These MUST be
# followed by a word boundary + `+` or space+spread to be recognized as a
# reference-rate token (to avoid false-positive on "S" alone). Order:
# longer tokens first to prevent "S" stealing "SOFR" match.
ABBREVIATION_MAP = {
    # BXSL / MFIC / OBDC / FSK / GBDC / ADS style abbreviations
    "SOFR": "SOFR",
    "SONIA": "SONIA",
    "EURIBOR": "EURIBOR",
    "STIBOR": "STIBOR",
    "CDOR": "CDOR",
    "CORRA": "CORRA",
    "LIBOR": "LIBOR",
    "PRIME": "PRIME",
    "TIBOR": "TIBOR",
    "NIBOR": "NIBOR",
    "BBSY": "BBSY",
    "BBSW": "BBSW",
    "BKBM": "BKBM",
    "KORIBOR": "KORIBOR",
    "TONAR": "TONA",   # ADS legend uses "TONAR" interchangeably with "TONA"
    "SARON": "SARON",
    "CDI": "CDI",
    "TONA": "TONA",
    "JIBAR": "JIBAR",
    "WIBOR": "WIBOR",
    "PRIBOR": "PRIBOR",
    "BUBOR": "BUBOR",
    "HIBOR": "HIBOR",
    "AONIA": "AONIA",
    # Abbreviations (3-letter)
    "SOF": "SOFR",
    "SON": "SONIA",        # MFIC GBP: "SON+500, 0.00% Floor"
    # Abbreviations (2-letter)
    "SF": "SOFR",          # ARCC/GBDC/FSK use "SF +" for SOFR
    "SO": "SOFR",
    "SN": "SONIA",         # sometimes "SN +"
    "SR": "STIBOR",        # FSK SEK rows: filing legend explicitly says SR = STIBOR. Never SONIA.
    "SA": "SONIA",         # FSK/OBDC GBP: filing legend "SONIA or 'SA'"
    "BB": "BBSY",          # OBDC AUD: "BBSY or 'BB'"
    "EU": "EURIBOR",
    # Abbreviations (1-letter; checked LAST due to length sort)
    "B": "BBSY",           # FSK AUD rows
    "C": "CORRA",          # FSK CAD rows: filing legend "CORRA"
    "E": "EURIBOR",        # "E +" appears in BCRED/BXSL
    "K": "KORIBOR",        # ADS legend: 3 months KORIBOR ("K") — KRW tranches
    "N": "NIBOR",          # ADS legend: Norwegian Overnight ("NIBOR" or "N") — NOK tranches
    "P": "PRIME",          # "P +"
    "PR": "PRIME",
    "L": "LIBOR",          # "L +" legacy
    "S": "SOFR",           # "S +" for SOFR
    "T": "TONA",           # ADS legend: Japan Tokyo Overnight Average ("TONAR" or "T") — JPY tranches
    # R6-D3: "Base Rate" — colloquial alias for alternate base rate / Prime
    "BASE RATE": "PRIME",
}

# Regex that matches a reference-rate abbreviation at the start of a
# cell or adjacent to "+": `SOFR +`, `S+`, `E + 4.00%`, etc. The token
# must be uppercase and followed by a space/+/EOL to avoid grabbing
# substring of a word (e.g., "Signal" starting with S).
_REF_RATE_INLINE_RE = re.compile(
    r"(?:^|[^A-Za-z])("
    + "|".join(sorted(ABBREVIATION_MAP.keys(), key=len, reverse=True))
    + r")(?:\s*\+|\s+\+|\s*\(|\s+based|\s*$)",
    re.I,
)

RATE_CURRENCY_MAP = {
    "SOFR": "USD",
    "LIBOR": "USD",
    "PRIME": "USD",
    "SONIA": "GBP",
    "EURIBOR": "EUR",
    "CDOR": "CAD",
    "CORRA": "CAD",
    "BBSY": "AUD",
    "BBSW": "AUD",
    "KORIBOR": "KRW",
    "TONA": "JPY",
    "NIBOR": "NOK",
    "BKBM": "NZD",
    "SARON": "CHF",
    "CDI": "BRL",
    "TIBOR": "JPY",
    "STIBOR": "SEK",
    "JIBAR": "ZAR",
    "WIBOR": "PLN",
    "PRIBOR": "CZK",
    "BUBOR": "HUF",
    "HIBOR": "HKD",
    "AONIA": "AUD",
}


def tokenize_reference_rate(text: str) -> Optional[str]:
    """Return the canonical rate token ('SOFR', 'EURIBOR', ...) or None.

    RC-B: recognizes abbreviated forms like 'SF +', 'E +', 'P +', 'S +'
    that filers use as space-saving reference-rate prefixes.
    """
    t = clean_text(text or "")
    if not t:
        return None
    # Full-name substring match first (case-insensitive).
    u = t.upper()
    for tok in REFERENCE_RATE_TOKENS:
        if tok in u:
            # R6-D3: "Base Rate" is a colloquial term for Prime; normalize.
            if tok == "BASE RATE":
                return "PRIME"
            return tok
    # Abbreviation match — only if adjacent to `+` or whitespace+`+`
    m = _REF_RATE_INLINE_RE.search(t)
    if m:
        abbr = m.group(1).upper()
        return ABBREVIATION_MAP.get(abbr)
    return None


def currency_of_reference_rate(rate_token: str | None) -> str:
    if not rate_token:
        return "USD"
    return RATE_CURRENCY_MAP.get(rate_token.upper(), "USD")


# ── RC-C: Rate-cell parser (cash + PIK split) ───────────────────────────────


_PIK_INLINE_RE = re.compile(
    r"\([^)]*?([\d.]+)\s*%\s*(?:PIK|p\.?i\.?k\.?)[^)]*?\)",
    re.I,
)                                             # "(3.50% PIK)" or "(incl. 2.75% PIK)" — inner PIK value
_PIK_PLAIN_RE = re.compile(
    r"([\d.]+)\s*%\s*(?:PIK|p\.?i\.?k\.?)\b",
    re.I,
)                                             # "3.50% PIK" — plain
_PIK_SLASH_RE = re.compile(
    r"([\d.]+)\s*%\s*/\s*(?:PIK|p\.?i\.?k\.?)\b",
    re.I,
)                                             # "3.50%/PIK" — NMFC style


def parse_rate_cell(text: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """Split a rate cell into (cash_rate_pct, pik_rate_pct).

    Supported formats:
      "8.74%"                        -> (8.74, None)
      "—" or "-"                     -> (None, None)   (RC-G: rate em-dash = null)
      "8.00% PIK"                    -> (0.0, 8.00)
      "8.00%/PIK"                    -> (0.0, 8.00)    (NMFC)
      "10.22% (3.50% PIK)"           -> (10.22, 3.50)  (ARCC: total includes PIK)
      "6.00% + 2.50% PIK"            -> (6.00, 2.50)   (ARCC/BXSL alternate)
      "A% / B% PIK"                  -> (A, B)
      "S + 4.75%"                    -> (None, None)   (this is ref+spread, not coupon)
      "SOFR (Q) 4.75%"               -> (4.75, None)   (reference rate + rate)

    Returns (None, None) for unparseable / empty.
    """
    if text is None:
        return (None, None)
    t = clean_text(str(text))
    if not t:
        return (None, None)
    # RC-G: bare em-dash / hyphen → null (not 0.0) for rate fields
    if t in ("-", "—", "–", "N/A", "n/a", ""):
        return (None, None)
    # If cell is ONLY a ref-rate prefix + spread (e.g. "S + 4.75%") with
    # no cash-coupon number, it's a ref+spread cell, not a coupon.
    # Heuristic: starts with letters, has only one `%` right after a `+`,
    # and no post-plus number without the `+` context → treat as ref+spread.
    if re.match(r"^\s*[A-Z]{1,6}\s*\+", t):
        # "S + 4.75%" — only percent is the spread
        return (None, None)

    pik: Optional[float] = None
    cash: Optional[float] = None

    # 1. Inner PIK in parens: "10.22% (3.50% PIK)" or "9.07% (incl. 2.75% PIK)"
    m = _PIK_INLINE_RE.search(t)
    if m:
        try:
            pik = float(m.group(1))
        except ValueError:
            pik = None
        # Detect "incl." or "including" before the PIK number — when
        # present, the OUTER number is the TOTAL (cash + pik), so cash =
        # outer - pik. Without the inclusive marker, ARCC-style: outer IS
        # the reported coupon (cash), and PIK is on top.
        inclusive = bool(re.search(r"\bincl(?:\.|uding)?\b", t, re.I))
        # Remove the PIK clause to parse the outer cash portion
        t_outer = _PIK_INLINE_RE.sub("", t).strip()
        outer = re.search(r"([\d.]+)\s*%", t_outer)
        if outer:
            try:
                outer_val = float(outer.group(1))
                if inclusive and pik is not None:
                    cash = round(outer_val - pik, 4)
                else:
                    cash = outer_val
            except ValueError:
                cash = None
        return (cash, pik)

    # 2. Slash form: "3.50%/PIK"
    m = _PIK_SLASH_RE.search(t)
    if m:
        try:
            pik = float(m.group(1))
        except ValueError:
            pik = None
        # Anything before the "/PIK" that's a percent is cash portion
        before = t[: m.start()]
        cash_match = re.search(r"([\d.]+)\s*%\s*$", before)
        if cash_match:
            try:
                cash = float(cash_match.group(1))
            except ValueError:
                cash = None
        else:
            cash = 0.0
        return (cash, pik)

    # 3. Plain PIK form: "3.50% PIK" (no parens, no slash)
    m = _PIK_PLAIN_RE.search(t)
    if m:
        try:
            pik = float(m.group(1))
        except ValueError:
            pik = None
        # Look for cash portion before the PIK. GBDC and similar SOIs use:
        #   "A% cash / B% PIK"  or  "A% cash B% PIK"   (cash-keyword form)
        # ARCC/BXSL use:
        #   "A% + B% PIK"                              (plus-keyword form)
        before = t[: m.start()]
        # First try "X% cash" — matches GBDC's split-coupon disclosure.
        cash_match = re.search(
            r"([\d.]+)\s*%\s*cash\s*[/,\s]*$", before, re.I
        )
        if not cash_match:
            # Then "A% +" — ARCC/BXSL style.
            cash_match = re.search(r"([\d.]+)\s*%\s*\+\s*$", before)
        if cash_match:
            try:
                cash = float(cash_match.group(1))
            except ValueError:
                cash = None
        else:
            # Probably "N% PIK" alone → cash = 0
            cash = 0.0
        return (cash, pik)

    # 4. Plain percent, no PIK: "8.74%"
    pct_match = re.search(r"([\d.]+)\s*%", t)
    if pct_match:
        try:
            cash = float(pct_match.group(1))
        except ValueError:
            cash = None
        return (cash, None)

    return (None, None)


def split_ref_and_spread(text: Optional[str]) -> tuple[Optional[str], Optional[float]]:
    """Parse a combined "Reference Rate and Spread" cell into (ref, spread_bps).

    Examples:
      "SOFR + 5.75%"                 -> ("SOFR", 575.0)
      "S + 4.75%"                    -> ("SOFR", 475.0)    (ARCC/FSK/GBDC abbr)
      "E + 4.00%"                    -> ("EURIBOR", 400.0)
      "SOFR (Q) + 4.75%"             -> ("SOFR", 475.0)
      "SOFR"                         -> ("SOFR", None)
      "S+450, 0.50% Floor"           -> ("SOFR", 450.0)    (ADS: spread in bps, no %)
      "S+238 Cash plus 2.88% PIK"    -> ("SOFR", 238.0)    (ADS with PIK annotation)
      "Fixed 8.00%"                  -> ("Fixed", None)
      ""                             -> (None, None)
    """
    if text is None:
        return (None, None)
    t = clean_text(str(text))
    if not t:
        return (None, None)
    # Tokenize reference rate
    ref = tokenize_reference_rate(t)
    spread_bps: Optional[float] = None
    # Try percent form first: "+ 4.75%" or "+4.75%"
    spread_match = re.search(r"\+\s*([\d.]+)\s*%", t)
    if spread_match:
        try:
            spread_bps = float(spread_match.group(1)) * 100.0
        except ValueError:
            spread_bps = None
    else:
        # Try bps form: "S+450" or "S + 450" (no % after the number).
        # Take the first number after `+` that's followed by space/comma/EOL/letter (not %).
        bps_match = re.search(r"\+\s*([\d.]+)(?:\s|,|$|[A-Za-z])", t + " ")
        if bps_match:
            try:
                val = float(bps_match.group(1))
                # Heuristic: if value < 25, it's a percent form missing the `%`;
                # otherwise it's already in basis points.
                if val < 25:
                    spread_bps = val * 100.0
                else:
                    spread_bps = val
            except ValueError:
                spread_bps = None
    return (ref, spread_bps)


# ── Structural column-header alignment ──────────────────────────────────────


@dataclass
class ColumnSchema:
    """Mapping from logical column index (after colspan expansion) to label.

    `spans` is a list of (start, end_exclusive, label) tuples built from the
    header row's colspans. `field_for(start)` returns the canonical field
    name associated with that logical column start, using `LABEL_TO_FIELD`
    matching, or None.
    """

    spans: list[tuple[int, int, str]] = field(default_factory=list)
    total_logical: int = 0

    def field_for_logical(self, logical: int) -> Optional[str]:
        for start, end, label in self.spans:
            if start <= logical < end:
                n = normalize_label(label)
                result = LABEL_TO_FIELD.get(n)
                if result is not None:
                    return result
                # Caption-bleed fallback: some filers (OBDC 10-K) collapse a
                # unit caption "(\$ in thousands)" with the column label into a
                # single cell, producing "($ in thousands) company".  After the
                # direct lookup fails, strip any leading parenthesized caption
                # (one set of parens + trailing space) and try again.
                stripped = re.sub(r"^\([^)]*\)\s*", "", n).strip()
                if stripped and stripped != n:
                    return LABEL_TO_FIELD.get(stripped)
                return None
        return None

    def label_for_logical(self, logical: int) -> Optional[str]:
        for start, end, label in self.spans:
            if start <= logical < end:
                return label
        return None


def normalize_label(s: str) -> str:
    """Strip footnote markers + trailing punctuation, normalize slashes,
    drop periods, and lowercase. Collapses minor filer-to-filer whitespace
    variations (e.g. 'Par / Units' vs 'Par/Units').

    R6 fix: filers like OBDC 2025-12-31 prefix the FIRST column header with
    a unit caption like "($ in thousands) Company". Strip these caption
    prefixes so "($ in thousands) Company" → "company".
    """
    s = strip_footnotes(s).lower().strip().rstrip(":").strip()
    # Column-header context: strip even a SINGLE trailing " N" suffix —
    # SPCC files "Cost 7" / "Reference Rate and Spread 5" with the
    # footnote-marker number bare instead of parenthesized. Safe to do
    # here because labels are never legal-entity names.
    s = re.sub(r"\s+\d{1,2}$", "", s).strip()
    # Strip asterisks (some filers use ** for footnotes)
    s = s.replace("*", "").strip()
    # Normalize em-dash and en-dash to ASCII hyphen so "investments—non-controlled/..."
    # (U+2014) matches the same LABEL_TO_FIELD key as "investments-non-controlled/..."
    # (CDIF and any other filer using Unicode dashes in column headers).
    s = s.replace("—", "-").replace("–", "-")
    # R6: strip caption prefixes — parenthesized phrases at the START of
    # the label that aren't part of the column name proper. Examples:
    # "($ in thousands) Company" → "Company"
    # "(in thousands) Cost" → "Cost"
    # "(Amounts in thousands, except share amounts) Company" → "Company"
    # The pattern matches any leading parenthesized phrase that contains
    # a unit-of-measure keyword (thousands/millions/$).
    s = re.sub(
        r"^\(\s*[^)]*?(?:thousands?|millions?|\$\s+in|in\s+\$|amounts?|except\s+share)[^)]*?\)\s*",
        "",
        s,
        flags=re.I,
    ).strip()
    # Normalize slashes with or without surrounding spaces
    s = re.sub(r"\s*/\s*", "/", s)
    # Drop periods and trailing punctuation
    s = s.replace(".", "")
    s = s.rstrip(":,;").strip()
    return re.sub(r"\s+", " ", s)


# Canonical field names per header label. Extend as new BDCs surface new
# labels — never add a company-name or ticker condition here.
LABEL_TO_FIELD: dict[str, str] = {
    # Company / issuer
    "company": "company_name",
    "issuer": "company_name",
    "portfolio company": "company_name",
    "portfolio investments": "company_name",  # AUDX — "Portfolio Investments" header
    "investments": "company_name",  # BXSL / Blackstone style
    "investment": "investment_type_raw",  # ARCC (when first row says "Investment")
    "name of portfolio company": "company_name",
    "industry/company": "company_name",          # ADS, MFIC (hierarchical)
    "country/security/industry/company": "company_name",  # CCAP (4-level hierarchical)
    "portfolio company, location and industry": "company_name",  # NMFC
    "portfolio company and type of investment": "company_name",  # NMFC SLP sub
    "description": "company_name",               # SLRC
    "portfolio company/type of investment": "company_name",
    "investments-non-controlled/non-affiliated": "company_name",  # OCREDIT
    "investments-non-controlled/affiliated": "company_name",
    "investments-controlled/affiliated": "company_name",
    "issuer(f)": "company_name",                 # BDEBT — alpha footnote not stripped
    "issuer": "company_name",
    "issuer name": "company_name",              # PFLT, PNNT — "Issuer Name" header
    "company and investment": "company_name",   # GAIN, GLAD — inline combined header
    "company/investment": "company_name",       # OXSQ — "COMPANY/INVESTMENT" header
    # Business description
    "business description": "business_description",
    "description": "business_description",
    # Investment type / footnotes
    "footnotes": "footnotes_text",
    "footnote": "footnotes_text",
    "type of investment": "investment_type_raw",
    "type of warrant": "investment_type_raw",       # TPVG warrant/equity table
    "type of equity": "investment_type_raw",        # TPVG equity table
    "investment type": "investment_type_raw",
    "investmenttype": "investment_type_raw",        # GCRED collapsed
    "type ofinvestment": "investment_type_raw",     # NMFC collapsed
    "instrument": "investment_type_raw",            # BDEBT — "Instrument" header
    "asset": "investment_type_raw",                 # KKRDL — "Asset" header
    "asset type": "investment_type_raw",            # SAR — "Asset Type" header
    "security": "investment_type_raw",              # SCM — "Security" header carries "First Lien" / "Equity"
    "investment description": "business_description",
    "industry": "industry",
    # Rates
    "coupon": "cash_rate_pct",
    "interest rate": "cash_rate_pct",
    # HTGC: "Interest Rate and Floor (1)" normalizes to "interest rate and floor"
    # (footnote marker stripped). Map to rate_text so split_ref_and_spread runs.
    "interest rate and floor": "cash_rate_pct",
    "current coupon": "cash_rate_pct",
    "cash": "cash_rate_pct",
    "cash rate": "cash_rate_pct",
    "cash interest rate": "cash_rate_pct",    # OSCF "Cash Interest Rate (4)(5)"
    "cash coupon": "cash_rate_pct",
    "rate": "cash_rate_pct",                  # FSK
    "total coupon": "cash_rate_pct",
    "totalcoupon": "cash_rate_pct",           # BDEBT collapsed
    "total rate": "cash_rate_pct",            # MAIN
    "interestrate": "cash_rate_pct",          # GCRED/OCREDIT collapsed
    "pik": "pik_rate_pct",
    "pik rate": "pik_rate_pct",
    "pik component": "pik_rate_pct",
    "floor": "floor_text",                    # FSK floor column
    "interest rate floor": "floor_text",
    # Reference rate / spread — BXSL uses a combined column
    "reference": "reference_rate_text",
    "reference rate": "reference_rate_text",
    "ref rate": "reference_rate_text",
    "reference rate and spread": "reference_rate_and_spread_text",
    "reference rate & spread": "reference_rate_and_spread_text",  # KKRDL — "Reference Rate & Spread"
    # BCSF uses a dedicated "Index (1)" column for the benchmark rate,
    # separate from the spread column.  Map both the bare form and the
    # footnote-annotated form (normalize_label strips the "(1)" footnote
    # marker before look-up, so "index (1)" → "index" after stripping, but
    # register both for safety since normalize_label calls strip_footnotes
    # which removes the numeric marker producing just "index").
    "index": "reference_rate_text",
    "index (1)": "reference_rate_text",
    "spread": "spread_text",
    "base rate spread": "spread_text",
    "basis spread": "spread_text",
    "spread above index": "spread_text",               # GBDC
    "basis point spread above index": "spread_text",   # PFLT, PNNT
    "basis pointspread aboveindex": "spread_text",     # PFLT collapsed
    "spreadaboveindex": "spread_text",                  # GCRED collapsed
    "spread above base rate": "spread_text",
    "spread above reference rate": "spread_text",
    "reference rateand spread": "reference_rate_and_spread_text",  # OCREDIT collapsed
    "ref(b)": "reference_rate_text",                    # BDEBT
    "ref": "reference_rate_text",
    # OBDC pre-2025-Q1 10-K / 10-Q used a single "Interest" header
    # spanning the (Reference Rate, Spread, optional PIK) sub-columns.
    # Treat that as the combined ref+spread block so the parser
    # routes the cell into the same combined-rate splitter that
    # handles "SF + 6.0% (3.5% PIK)" style strings.
    "interest": "reference_rate_and_spread_text",
    # Dates
    "acquisition date": "acquisition_date",
    "initial acquisition date": "acquisition_date",
    "investment date": "acquisition_date",                 # MAIN
    "investmentdate": "acquisition_date",                  # MAIN <br/> joined
    "date": "acquisition_date",                            # generic short form
    "maturity date": "maturity_date",
    "maturitydate": "maturity_date",                       # MAIN/SLRC <br/> joined
    "maturity/expirationdate": "maturity_date",  # NMFC collapsed
    "maturity/expiration date": "maturity_date",
    "maturity": "maturity_date",
    # Money — collapsed forms
    "amortizedcost": "amortized_cost_text",                # GCRED/OCREDIT
    "fairvalue": "fair_value_text",                        # GCRED/OCREDIT
    # Par / shares / units
    "shares/units": "par_or_shares_text",
    "shares": "shares_text",
    "units": "shares_text",
    "number of shares": "shares_text",     # FSK equity SOI
    "number of units": "shares_text",
    "principal": "par_text",
    "par": "par_text",
    "par amount": "par_text",
    "par amount/units": "par_or_shares_text",
    "par/shares": "par_or_shares_text",
    "par/shares/units": "par_or_shares_text",    # ADS
    "par/principal amount": "par_or_shares_text",  # CDIF "Par/ Principal Amount *"
    "shares/principal amount": "par_or_shares_text",  # CCLFX/CELFX "Shares/ Principal Amount"
    "par/units": "par_or_shares_text",
    "par or units": "par_or_shares_text",
    "principal/shares": "par_or_shares_text",
    "principal/units": "par_or_shares_text",
    "principal ($)/shares": "par_or_shares_text",       # GBDC
    "principal ($) / shares": "par_or_shares_text",
    "principal ($)/ shares": "par_or_shares_text",
    "principal ($)/shares(3)": "par_or_shares_text",
    "principal ($) /shares": "par_or_shares_text",      # GCRED collapsed
    "paramount/units": "par_or_shares_text",            # OCREDIT collapsed
    "principal amount": "par_text",
    "principal amount/shares": "par_or_shares_text",
    "principalamount,par valueor shares": "par_or_shares_text",  # NMFC collapsed
    "principal amount, par value or shares": "par_or_shares_text",  # NMFC un-collapsed
    "principal amount,par value or shares": "par_or_shares_text",
    "principal amount par value or shares": "par_or_shares_text",
    "amount": "par_text",
    # Money
    "amortized cost": "amortized_cost_text",
    "cost": "amortized_cost_text",
    "cost ($)": "amortized_cost_text",           # RWAY — currency-annotated column headers
    "fair value": "fair_value_text",
    "fair value ($)": "fair_value_text",          # RWAY — currency-annotated column headers
    "market value": "fair_value_text",          # BCSF
    "value": "fair_value_text",                  # HTGC ("Value" col)
    # % of net assets
    "% of net assets": "pct_net_assets_text",
    "percentage of net assets": "pct_net_assets_text",
    "percent of net assets": "pct_net_assets_text",
    "percent ofnetassets": "pct_net_assets_text",  # NMFC collapsed
    # Total coupon (filer-specific; NMFC uses this instead of 'Coupon')
    "total coupon": "cash_rate_pct",
}


def parse_header_row(row: Tag) -> ColumnSchema:
    """Build a ColumnSchema from an SOI header row by walking colspans.

    Uses plain `get_text()` to concatenate child text nodes. Filers that
    stack multi-word labels using <br/> or <div> blocks (GBDC is the
    notable case) should normalize those at the adapter level via a
    `_postprocess_schema` hook — not here, because injecting spaces
    globally breaks other filers whose data cells use <span> wrappers
    that should stay concatenated (e.g. ADS/MFIC address fields).
    """
    spans: list[tuple[int, int, str]] = []
    running = 0
    for cell in row.find_all(["td", "th"]):
        cs = int(cell.get("colspan", 1) or 1)
        label = clean_text(cell.get_text())
        spans.append((running, running + cs, label))
        running += cs
    return ColumnSchema(spans=spans, total_logical=running)


def row_cell_spans(row: Tag) -> list[tuple[int, int, str]]:
    """For each cell in the row, return (logical_start, logical_end, text).
    Plain `get_text()` — see parse_header_row for the rationale.

    Industry-alias canonicalization (cross-BDC fix 2026-05-16): if the cleaned
    cell text exactly matches a key in INDUSTRY_ALIASES, replace it with the
    canonical form. Keys are HTML typos ("Diversifed Financial Services") or
    capitalization quirks ("Specialty retail") that would never legitimately
    appear as a company-name or instrument cell — so this is safe to apply
    at the row-span layer, which catches both section-banner industry rows
    (read directly from spans in base.py) and per-row industry columns.
    """
    out: list[tuple[int, int, str]] = []
    running = 0
    for cell in row.find_all(["td", "th"]):
        cs = int(cell.get("colspan", 1) or 1)
        text = clean_text(cell.get_text())
        if text in INDUSTRY_ALIASES:
            text = INDUSTRY_ALIASES[text]
        out.append((running, running + cs, text))
        running += cs
    return out


def is_header_row(row: Tag) -> bool:
    """Structural header detection: row whose first non-empty expanded span
    contains a recognized header label identifying the company OR
    investment-type column (some filers leave the company column blank so
    the first non-empty header cell names the investment-type column).
    """
    spans = row_cell_spans(row)
    non_empty = [s for s in spans if s[2]]
    if not non_empty:
        return False
    first_label = normalize_label(non_empty[0][2])
    if first_label in {
        "company",
        "issuer",
        "issuer name",                                      # PFLT, PNNT
        "portfolio company",
        "investments",
        "name of portfolio company",
        "industry/company",                                 # ADS, MFIC
        "country/security/industry/company",                # CCAP
        "portfolio company, location and industry",         # NMFC
        "portfolio company and type of investment",         # NMFC sub-schedule
        "investment type",                                  # GBDC (blank company col)
        "type of investment",
        "description",                                      # SLRC
        "portfolio company/type of investment",
        "company/investment",
        "company and investment",          # GAIN, GLAD — "Company and Investment(A)(B)..."
        "reference asset",                 # GBDC CLO sub-portfolio 2011-2014
    }:
        # Guard against single-cell banner rows that carry the generic
        # "Investments" label as a section header rather than a column
        # header (CCAP's "Investments (1)(2)(3)" hierarchy banner). A real
        # column-header row contains multiple labels — the first one names
        # the company/issuer column and the rest name Investment Type / Rate
        # / Cost / FV / etc. A single-cell row whose label happens to be
        # "investments" is a banner, not a header — it must NOT replace
        # the schema. Apply this only to the generic "investments" /
        # "investment type" labels that have a high false-positive rate;
        # specific labels like "industry/company" (ADS, MFIC) or
        # "country/security/industry/company" (CCAP) are safe because no
        # filer uses them as a banner.
        if first_label in {"investments", "investment type", "type of investment"}:
            if len(non_empty) < 2:
                return False
        return True
    # OCREDIT-style: first cell is "Investments-non-controlled/non-affiliated"
    # (company column header includes affiliation). Recognize by prefix.
    if first_label.startswith("investments-") or first_label.startswith("investments —"):
        # OCREDIT also uses the same prefix as a single-cell SECTION BANNER
        # ("Investments— non-controlled/non-affiliated") immediately after
        # the real column-header row. A real column header has multiple
        # populated cells naming Footnotes / Reference Rate / Maturity /
        # Cost / Fair Value etc.; a banner has just one. Require >=2
        # non-empty cells to avoid the banner row replacing the schema with
        # a single-label one and silently dropping the table's data rows.
        if len(non_empty) < 2:
            return False
        return True
    if first_label.startswith("portfolio company"):
        return True
    # Caption-prefixed company column (OBDC 2023-12-31, others): the cell
    # text collapses a caption like "($ in thousands)" with the column label
    # "Company" → "($ in thousands)company". Accept when "company" / "issuer"
    # / "investments" appear as a SUFFIX after a caption-style prefix.
    for token in ("company", "issuer", "investments"):
        # A caption like "($ in thousands)" or similar followed directly by
        # the column label. Captions usually contain `$`/`%` or are wrapped
        # in parens.
        if first_label.endswith(token) and (
            "(" in first_label or "$" in first_label or "%" in first_label
        ):
            return True
    # Last resort: count recognized column labels in the row. A header row
    # for SOI typically has FV + Cost/Amortized Cost + at least 2 of
    # (Industry, Type of Investment, Interest Rate, Maturity, Par/Principal).
    rec_labels = 0
    label_blob = " ".join(
        normalize_label(t) for _, _, t in non_empty
    )
    for label_term in (
        "fair value", "amortized cost", " cost ", "industry", "type of investment",
        "interest rate", "maturity", "principal", "par", "shares",
    ):
        if label_term.strip() in label_blob:
            rec_labels += 1
    if rec_labels >= 5:
        return True
    return False


# ── Row classification primitives (structural only) ─────────────────────────


_TOTAL_RE = re.compile(
    # Accept hyphenated "Sub-Total" (EPTV: "Sub-Total: Artificial
    # Intelligence (23.5%)*"), bare "Total", "Subtotal", "Sub Total",
    # and the net-assets summary banner ("NET ASSETS - 100.0%" in
    # CION's SOI). The net-assets row sits after the grand-total as
    # a portfolio-vs-net-assets reconciliation line and is never an
    # investment.
    r"^\s*(total|subtotal|sub\s+total|sub[-–—]total|net\s+assets\b)\b",
    re.I,
)


def is_total_like_text(s: str) -> bool:
    return bool(_TOTAL_RE.match(s or ""))


def is_total_row_structurally(row_or_spans, schema: "ColumnSchema | None" = None) -> bool:
    """R5-D3: distinguish a real "Total <X>" subtotal row from an issuer
    name that begins with "Total " (e.g. MFIC's "Total Power Limited").

    A real subtotal row has NO investment-type, NO rate, NO maturity
    cell populated. An issuer-name-starting-with-"Total" data row HAS
    those columns populated.

    Returns True only when:
      (a) first non-empty text matches `^Total\\b` AND
      (b) no investment_type / cash_rate / maturity_date / par cells
          are populated under the schema (the row is monetary-only).
    """
    if isinstance(row_or_spans, Tag):
        spans = row_cell_spans(row_or_spans)
    else:
        spans = row_or_spans
    non_empty = [s for s in spans if s[2]]
    if not non_empty:
        return False
    if not is_total_like_text(non_empty[0][2]):
        return False
    if schema is None:
        return True
    DATA_FIELDS = (
        "investment_type_raw",
        "reference_rate_text",
        "reference_rate_and_spread_text",
        "cash_rate_pct",
        "spread_text",
        "pik_rate_pct",
        "maturity_date",
        "acquisition_date",
        "par_text",
        "par_or_shares_text",
        "shares_text",
    )
    # Count UNIQUE data-field names populated (not raw cell hits). When a
    # logical column spans multiple physical cells — e.g. EPTV's
    # "Par Amount (4)" span carries one cell with "$" and an adjacent cell
    # with the numeric amount, both mapping to par_text — the old per-cell
    # tally inflated the count and falsely classified per-issuer subtotals
    # ("Total Beam Technologies, Inc." with par+cost+FV but no
    # type/rate/maturity) as data rows.
    populated_fields = set()
    for start, _end, text in non_empty:
        if not text:
            continue
        fld = schema.field_for_logical(start)
        if fld in DATA_FIELDS:
            populated_fields.add(fld)
    # If 2+ DISTINCT per-position data fields are populated (type, rate,
    # maturity, par, etc.), this is a real data row whose company name
    # happens to start with "Total".
    return len(populated_fields) < 2


def classify_row(row: Tag, schema: ColumnSchema | None) -> str:
    """Return one of: header, total, subtotal, industry, data, spacer, skip.

    Structural rules only — no company/industry whitelist:
      * header: first non-empty text matches a known header label.
      * spacer: no non-empty cells, or a single empty cell.
      * total:  first non-empty text begins with 'Total' or 'Subtotal'.
      * industry: exactly one non-empty cell positioned in the 'company'
                  column of the schema, whose text does not parse as a
                  number and is not a total. Industry labels have no entity
                  suffix and aren't currency symbols.
      * subtotal: all non-empty cells are numeric and fall in the monetary
                  columns (amortized_cost, fair_value, %-of-net-assets) —
                  i.e. no company-column content.
      * data: anything else with ≥2 non-empty cells spread across multiple
              logical columns.
    """
    spans = row_cell_spans(row)
    non_empty = [s for s in spans if s[2]]
    if not non_empty:
        return "spacer"
    if is_header_row(row):
        return "header"

    first_text = non_empty[0][2]
    if is_total_like_text(first_text):
        # R5-D3 guard: real subtotal rows have NO investment-type / rate /
        # maturity / par columns populated. Issuer-name rows that happen
        # to start with "Total " (e.g. MFIC's "Total Power Limited") have
        # those columns populated and must NOT be classified as total.
        if is_total_row_structurally(spans, schema):
            return "total"

    # If there's a schema, use it to decide industry vs subtotal.
    if schema is not None:
        # Which logical column does the first non-empty cell occupy?
        first_start = non_empty[0][0]
        first_field = schema.field_for_logical(first_start)

        if len(non_empty) == 1 and first_field == "company_name":
            text = first_text
            # Industry headers: no entity suffix, not a number, not currency
            if parse_number(text) is None and not re.search(
                r"\b(LLC|L\.L\.C\.|Inc\.|Corp\.?|L\.P\.|Ltd\.|PLC|Holdings?|Company|Corporation|Partners|Partnership|S\.A\.|GmbH|AG|N\.V\.)\b",
                text,
            ) and text not in {"$", "€", "£"}:
                return "industry"

        # Subtotal: every non-empty cell is numeric and every one falls in a
        # monetary column (not the company / investment columns).
        monetary_fields = {
            "amortized_cost_text",
            "fair_value_text",
            "pct_net_assets_text",
            "par_text",
            "par_or_shares_text",
            "shares_text",
        }
        all_monetary = True
        # Tokens that are allowed as standalone cells in a subtotal row
        # without being numeric — currency sigils and percent signs.
        allowed_nontext = {"$", "€", "£", "¥", "%", "—", "-"}
        for start, end, text in non_empty:
            fld = schema.field_for_logical(start)
            if fld not in monetary_fields:
                all_monetary = False
                break
            if parse_number(text) is None and text not in allowed_nontext:
                all_monetary = False
                break
        if all_monetary and len(non_empty) >= 1:
            # RC-J: reject if the FIRST non-empty cell falls in (or overlaps)
            # the company_name column — that's a share-count on an equity
            # row, not a subtotal. Find the company-column boundary from
            # the schema.
            company_col_end = None
            for s_start, s_end, s_label in schema.spans:
                if LABEL_TO_FIELD.get(normalize_label(s_label)) == "company_name":
                    company_col_end = s_end
                    break
            first_start = non_empty[0][0]
            if company_col_end is not None and first_start < company_col_end:
                # First cell is inside the company column — equity share
                # count, not a subtotal.
                pass
            elif first_start > schema.spans[0][1] - 1:
                return "subtotal"

    # Default: treat as data if it has reasonable spread
    return "data" if len(non_empty) >= 2 else "skip"


def get_cell_text_for_field(
    row_spans: list[tuple[int, int, str]], schema: ColumnSchema, field_name: str
) -> Optional[str]:
    """Return the text of the cell whose logical column range falls under
    `field_name` in the schema.

    If multiple cells map to the same field, they are joined with a single
    space (this happens when a filer splits '$' and the numeric value across
    two sub-columns, and the header labels them both under 'Amortized Cost').
    """
    parts: list[str] = []
    for start, end, text in row_spans:
        if not text:
            continue
        fld = schema.field_for_logical(start)
        if fld == field_name:
            parts.append(text)
    if not parts:
        return None
    joined = " ".join(parts).strip()
    # Industry-alias canonicalization (cross-BDC fix 2026-05-16): even though
    # row_cell_spans already canonicalizes per-cell text, the joined form
    # could still match an alias key (e.g. if a filer splits the label across
    # two sub-columns). Apply once more on the assembled industry value.
    if field_name == "industry":
        joined = INDUSTRY_ALIASES.get(joined, joined)
    return joined


# ── Footnote legend parsing ─────────────────────────────────────────────────


def parse_footnote_legend(text_region: str) -> dict[str, str]:
    """Find entries like '(N) …' or '(ac) …' in a block of text and return
    a map from marker -> meaning (first sentence / up to next marker).

    Used on the tail of the SOI section where the legend is printed.
    """
    out: dict[str, str] = {}
    # Split on markers, retaining them
    pattern = re.compile(r"\(([0-9a-z]{1,3})\)")
    # Collect all markers with their spans
    matches = list(pattern.finditer(text_region or ""))
    for i, m in enumerate(matches):
        marker = m.group(1)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text_region)
        body = clean_text(text_region[body_start:body_end])
        # Trim at the first double-space-or-period-then-capital (sentence end)
        # but keep meaningful length
        if not body:
            continue
        # Only register the first occurrence (the legend is typically the
        # first long-form description; earlier in-row occurrences are short).
        if marker in out:
            # prefer longer meaning
            if len(body) > len(out[marker]) + 20:
                out[marker] = body
            continue
        out[marker] = body
    return out


# ── Controlled-vocab footnote classifier ────────────────────────────────────


# Controlled-vocab rules — applied to LEGEND MEANINGS, not row text.
# These must be PRECISE: each regex should match the specific phrase that a
# legend uses to define the flag, never a casual mention. For example, we
# do NOT derive 'is_subordinated' from a legend that casually mentions
# "senior secured and subordinated loans" (disjunctive); the loan's
# seniority is a row-level fact from investment_type_raw.
CONTROLLED_VOCAB_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnon[-\s]?accrual\s+status", re.I), "is_non_accrual"),
    (re.compile(r"\bloan\s+was\s+on\s+non[-\s]?accrual", re.I), "is_non_accrual"),
    # Variants captured from the coverage-floor audit:
    #   BBDC fn(26) — "Non-accrual investment."
    #   ICMB fn(7) — "Classified as non-accrual asset."
    #   MAIN fn(14) — "Non-accrual and non-income producing debt investment."
    # Pattern matches "non-accrual" before investment/loan/asset/note,
    # allowing up to 40 chars of qualifier/conjunction text between
    # ("and non-income producing debt"); plus the "classified as"
    # lead-in and the "<noun> was on non-accrual" tail-in variants.
    (re.compile(r"\bnon[-\s]?accrual\b.{0,40}\b(?:investment|loan|asset|note|position)\b", re.I), "is_non_accrual"),
    (re.compile(r"\bclassified\s+as\s+non[-\s]?accrual", re.I), "is_non_accrual"),
    (re.compile(r"\binvestment\s+was\s+on\s+non[-\s]?accrual", re.I), "is_non_accrual"),
    (re.compile(r"\bnot\s+a\s+qualifying\s+asset", re.I), "is_non_qualifying_asset"),
    # HTGC fn10: "deems not 'qualifying assets'" or 'not "qualifying assets"' (any quote style)
    (re.compile(r'\bdeems?\s+not\W{0,3}qualifying\s+assets?', re.I), "is_non_qualifying_asset"),
    # GBDC fn(8): "treated as a non-qualifying asset under Section 55(a)..."
    # Generic catch for any "non-qualifying asset" phrasing without
    # requiring "not a" lead-in. Section-55(a) reference is the
    # unambiguous BDC 1940-Act marker, so we also accept that on its own.
    (re.compile(r"\bnon[-\s]?qualifying\s+asset", re.I), "is_non_qualifying_asset"),
    (re.compile(r"\bSection\s+55\s*\(a\)\s+of\s+the\s+(?:Investment\s+Company\s+Act|1940\s+Act)", re.I), "is_non_qualifying_asset"),
    (re.compile(r"\bis\s+a\s+qualifying\s+asset\b", re.I), "is_qualifying_asset"),
    (re.compile(r"\blevel\s*3\s+(fair\s+value|input|measurement|investment|asset|securit)", re.I), "is_level_3"),
    (re.compile(r"\bvalued\s+using\s+(?:significant\s+)?unobservable\s+inputs\b", re.I), "is_level_3"),
    (re.compile(r"\bsignificant\s+unobservable\s+inputs\b", re.I), "is_level_3"),
    (re.compile(r"\bconsidered\s+level\s*3\s+(investment|asset|securit)", re.I), "is_level_3"),
    (re.compile(r"\blevel\s*2\s+(fair\s+value|input|measurement|investment|asset|securit)", re.I), "is_level_2"),
    (re.compile(r"\blevel\s*1\s+(fair\s+value|input|measurement|investment|asset|securit)", re.I), "is_level_1"),
    # Inverse-marker pattern. FSK fn(aa) "Security is classified as Level 1
    # or Level 2 in the Company's fair value hierarchy" — rows carrying (aa)
    # are NOT Level 3; everything else IS L3 by default (cross-BDC pattern #5
    # inverse form, mirrors HLEND fn(20)/(21) pledged-exception convention).
    (re.compile(r"\b(?:securit(?:y|ies)|investments?)\s+(?:is|are)\s+classified\s+as\s+level\s*(?:1|2|one|two)\b", re.I), "is_not_level_3"),
    (re.compile(r"\blevel\s*(?:1|2|one|two)\s+(?:or\s+level\s*(?:1|2|one|two)\s+)?in\s+(?:the|our|its|company[’']?s?)\s*\bfair\s+value\s+hierarch", re.I), "is_not_level_3"),
    (re.compile(r"\bpayment[-\s]?in[-\s]?kind\b|\b\(?pik\)?\s+interest", re.I), "has_pik"),
    (re.compile(r"\bvariable\s+rate\s+loans?", re.I), "is_variable_rate"),
    (re.compile(r"\bfixed[-\s]?rate\s+(loan|note|security)", re.I), "is_fixed_rate"),
    (re.compile(r"\binterest\s+rate\s+floor", re.I), "has_floor"),
    # Control vs non-control affiliation. 1940 Act legend language:
    #   non-controlled: "Affiliated Person because it owns 5% or more..."
    #   controlled:     "both an Affiliated Person and Control this portfolio..."
    # Match only the unambiguous phrasings. "both ... and 'Control'" is the
    # diagnostic phrase for the controlled-affiliate legend entry.
    (re.compile(r'\bboth\s+an?\s+["""]?affiliated\s+person["""]?\s+and\s+["""]?control["""]?\s+this\s+portfolio', re.I), "is_controlled_affiliate"),
    (re.compile(r'\bboth\s+an?\s+["""]?affiliated\s+person["""]?\s+(?:of\s+)?and\s+["""]?control["""]?\s+this\s+portfolio', re.I), "is_controlled_affiliate"),
    (re.compile(r'\bboth\s+an?\s+["""]?affiliated\s+person["""]?\s+and\s+["""]?control', re.I), "is_controlled_affiliate"),
    (re.compile(r'\b(deemed\s+to\s+)?["""]?control["""]?\s+a\s+portfolio\s+company\s+if\s+it\s+owns?\s+more\s+than\s+25\s*%', re.I), "is_controlled_affiliate"),
    # MFIC (5) phrasing: "deemed to exercise a controlling influence over the management or policies"
    (re.compile(r'\b(deemed\s+to\s+)?exercise\s+a\s+controlling\s+influence\b', re.I), "is_controlled_affiliate"),
    (re.compile(r'\bcontrolling\s+influence\s+over\s+the\s+(management|policies)\b', re.I), "is_controlled_affiliate"),
    # HTGC fn7: "Control investment as defined under the 1940 Act in which [firm] owns at least 25%"
    (re.compile(r'\bcontrol\s+investment.{0,80}owns\s+at\s+least\s+25\s*%', re.I), "is_controlled_affiliate"),
    (re.compile(r'\b(deemed\s+to\s+be\s+)?an?\s+["""]?affiliated\s+person["""]?\s+because\s+it\s+owns?\s+5\s*%', re.I), "is_non_controlled_affiliate"),
    (re.compile(r'\b(deemed\s+to\s+be\s+)?an?\s+["""]?affiliated\s+person["""]?\s+(of\s+a\s+portfolio\s+company\s+)?if\s+it\s+owns?\s+5\s*%', re.I), "is_non_controlled_affiliate"),
    # GBDC fn(28): "affiliated person of the portfolio company as the
    # Company owns five percent or more". Accept "five percent" or "5%"
    # with flexible "as the Company owns" connector.
    (re.compile(r'\b(deemed\s+to\s+be\s+)?an?\s+["""]?affiliated\s+person["""]?\b[^.]{0,80}\bowns?\s+(?:5\s*%|five\s*percent)', re.I), "is_non_controlled_affiliate"),
    # BCSF fn10 phrasing: "deemed to be an 'affiliated person' of the Company
    # as the Company owns 5% or more..." — uses "as the Company owns" instead
    # of "because it owns" / "if it owns".
    (re.compile(r'\baffiliated\s+person.{0,40}as\s+\w+\s+(?:company|fund|trust)\s+owns?\s+5\s*%', re.I), "is_non_controlled_affiliate"),
    # GBDC fn28 phrasing: "...affiliated person of the portfolio company
    # as the Company owns five percent or more..." — longer window, and
    # uses "five percent" as the spelled-out variant. Same 1940-Act
    # non-controlled-affiliate marker.
    (re.compile(r'\baffiliated\s+person.{0,80}\bas\s+\w+\s+(?:company|fund|trust)\s+owns?\s+(?:5\s*%|five\s+percent)', re.I), "is_non_controlled_affiliate"),
    # GBDC fn29 phrasing: "...affiliated person of and 'control' this
    # portfolio company as the Company owns more than 25% of the
    # portfolio company's outstanding voting..." — controlled-affiliate
    # marker with "more than 25%" / "twenty-five percent" variants.
    (re.compile(r'\baffiliated\s+person.{0,40}(?:and\s+)?["""]?control["""]?.{0,80}(?:more\s+than\s+25\s*%|twenty[-\s]five\s+percent)', re.I), "is_controlled_affiliate"),
    # MFIC (4) phrasing: 'Denotes investments in which we are an "Affiliated Person"' (no 5%/25% language).
    # Match the bare "Affiliated Person" + 1940 Act marker as the non-controlled case
    # provided it does NOT also contain controlling-influence language (handled
    # by classify_footnote_meaning's disambiguator).
    (re.compile(r'\bdenotes\s+investments?\s+in\s+which\s+we\s+are\s+an?\s+["""]?affiliated\s+person', re.I), "is_non_controlled_affiliate"),
    (re.compile(r'\b(?:are|is)\s+an?\s+["""]?affiliated\s+person["""]?[,.]?\s+as\s+defined\s+in\s+the\s+investment\s+company\s+act', re.I), "is_non_controlled_affiliate"),
    (re.compile(r'\bpledged\s+as\s+collateral', re.I), "is_pledged"),
    # Negating-marker patterns: a legend entry that explicitly says "not pledged"
    # marks the rows carrying THIS marker as NOT pledged. Used together with the
    # default-on pledged logic in base._apply_footnotes (when a legend entry has
    # is_pledged AND wording "unless otherwise indicated", every other row is
    # treated as pledged by default; rows with the is_not_pledged marker override).
    # OBDC fn(9)/(26), HLEND fn(20)/(21), GBDC fn(28), BCRED fn(varies) all use
    # this convention.
    (re.compile(r'\bis\s+not\s+pledged\s+as\s+collateral', re.I), "is_not_pledged"),
    (re.compile(r'\bnot\s+pledged\s+as\s+collateral', re.I), "is_not_pledged"),
    (re.compile(r'\binvestment\s+is\s+not\s+pledged', re.I), "is_not_pledged"),
    (re.compile(r'\bsecurity\s+is\s+not\s+pledged', re.I), "is_not_pledged"),
    (re.compile(r'\binvestments?\s+are\s+not\s+pledged', re.I), "is_not_pledged"),
    (re.compile(r'\bthese\s+investments?\s+are\s+not\s+pledged', re.I), "is_not_pledged"),
    # MAIN fn(16): "Investment is not encumbered as security for the Credit
    # Facilities". HTGC fn(13)-style "not subject to credit-facility security
    # interest". Treat "encumbered" as a synonym for "pledged" — both refer
    # to the same secured-debt collateral pool.
    (re.compile(r'\b(?:is\s+)?not\s+encumbered\b', re.I), "is_not_pledged"),
    (re.compile(r'\bnot\s+subject\s+to\s+(?:a\s+)?(?:security\s+interest|pledge)', re.I), "is_not_pledged"),
    # Unfunded: only match when the legend entry uses per-row singular language
    # that refers to THIS SPECIFIC investment's funded status.
    #
    # R8-D2 fix: removed the two "schedule-header" patterns that fired on
    # DEFINITIONAL footnotes (ASIF fn11, ARCC fn13, NMFC fn15) which describe
    # an ENTIRE SCHEDULE, not a per-row status:
    #   - r'\bcommitments?\s+to\s+fund\b'        (fires on "had the following
    #     commitments to fund various revolving and delayed draw loans")
    #   - r'\bhad\s+the\s+following\s+commitments' (same ASIF/ARCC fn pattern)
    #   - r'\bdrawn\s+or\s+undrawn\b...(revolver)' (fires on NMFC fn15:
    #     "Par value amounts represent the drawn or undrawn portion of revolvers")
    # These were causing 297 false positives in ASIF, 36 in NMFC, 399 in ARCC.
    # Per-row unfunded markers use singular subject language (see below).
    (re.compile(r'\bno\s+amounts?\s+(were\s+)?funded', re.I), "is_unfunded_commitment"),
    # HTGC fn17: "unfunded contractual commitment available at the request of this portfolio company"
    (re.compile(r'\bunfunded\s+contractual\s+commitment\b', re.I), "is_unfunded_commitment"),
    # MAIN fn25: "The position is unfunded and no interest income is being earned"
    (re.compile(r'\bthe\s+position\s+is\s+unfunded\b', re.I), "is_unfunded_commitment"),
    (re.compile(r'\bunfunded\s+and\s+no\s+interest', re.I), "is_unfunded_commitment"),
    # OBDC fn(24): "the Company is deemed to 'control'... 1940 Act"
    (re.compile(r'\bdeemed\s+to\s+["""]?control["""]?.{0,40}1940\s+Act', re.I), "is_controlled_affiliate"),
    (re.compile(r'\b1940\s+Act.{0,40}\bdeemed\s+to\s+["""]?control', re.I), "is_controlled_affiliate"),
    # BCSF fn11 phrasing: "As defined in the 1940 Act, the Company is deemed to
    # “control” this portfolio company..." — curly-quote wrapping of
    # "control" is U+201C/U+201D, not matched by the ASCII ["""]? char class
    # above.  Simpler non-quote-dependent form covers BCSF and any other filer
    # that uses Unicode smart-quotes around the control verb.
    (re.compile(r'\b1940\s+Act.{0,60}\bdeemed\s+to\b.{0,10}\bcontrol\b', re.I), "is_controlled_affiliate"),
    (re.compile(r'\bdeemed\s+to\b.{0,10}\bcontrol\b.{0,60}\b1940\s+Act', re.I), "is_controlled_affiliate"),
    # NMFC fn(37): "Company 'Controls'... power to vote more than 25%"
    (re.compile(r'\bpower\s+to\s+vote\s+more\s+than\s+25', re.I), "is_controlled_affiliate"),
    (re.compile(r'\bdenotes\s+investments?\s+in\s+which\s+.{0,60}["""]?controls?["""]?\b', re.I), "is_controlled_affiliate"),
    (re.compile(r'\bletters?\s+of\s+credit\s+issued', re.I), "has_letter_of_credit"),
    (re.compile(r'\bforeign\s+currency|\bdenominated\s+in\s+[A-Z]{3}', re.I), "is_foreign_currency"),
    (re.compile(r'\btax\s+basis\s+(differs|varies|of)', re.I), "tax_basis_differs_from_gaap"),
    (re.compile(r'\bco[-\s]?invest(ed|ment)\s+in\s+the\s+SDLP', re.I), "is_sdlp_co_investment"),
]


_NOT_CONTROL_DISAMBIG = re.compile(
    r'(?:is\s+)?not\s+deemed\s+to\s+["""]?control',
    re.I,
)
_AND_CONTROL_DISAMBIG = re.compile(
    r'(?:deemed\s+to\s+be\s+an?\s+["""]?affiliated\s+person["""]?\s+)?'
    r'and\s+deemed\s+to\s+["""]?control',
    re.I,
)
# R6-AV05: "not pledged as collateral" — fired when a legend entry says the
# investment is EXCLUDED from pledged set (e.g. BXSL fn5).  The base regex
# fires is_pledged because it sees "pledged as collateral"; this pattern
# strips the flag when the surrounding text is explicitly negative.
# Policy-text disambiguator for is_non_accrual. The vocab rules match
# "non-accrual status" and "non-accrual <noun>" so they fire on policy
# text like "Loans are generally placed on non-accrual status when…"
# too. Drop is_non_accrual when the legend body is clearly a POLICY
# description rather than a per-row assertion.
_NA_POLICY_DISAMBIG = re.compile(
    r"\bgenerally\s+placed\s+on\s+non[-\s]?accrual"
    r"|\bare\s+placed\s+on\s+non[-\s]?accrual\s+status\s+when\b"
    r"|\binvestments?\s+that\s+are\s+expected\s+to\s+pay.+non[-\s]?accrual",
    re.I,
)


# Prior-period date in a non-accrual footnote. ARCC pattern:
#   (8)  "Loan was on non-accrual status as of December 31, 2025."   ← current
#   (10) "Loan was on non-accrual status as of December 31, 2024."   ← prior
# When the footnote names a year that is NOT the filing's period_end
# year, drop is_non_accrual — the row was on NA last period but not this
# period.
_NA_AS_OF_DATE_RE = re.compile(
    r"\bas\s+of\s+(?:january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\s+\d{1,2}\s*,\s*(\d{4})",
    re.I,
)


_NOT_PLEDGED_RE = re.compile(
    r'\b(?:is\s+not|are\s+not|not)\s+pledged(?:\s+as\s+collateral)?|'
    r'\bexclude(?:d)?\s+from\s+pledged|'
    r'\bnot\s+available\s+to\s+satisfy\s+the\s+creditors',
    re.I,
)


def classify_footnote_meaning(meaning: str, period_end: Optional[str] = None) -> list[str]:
    """Return the list of controlled-vocab flags triggered by this meaning
    string. Multiple flags may trigger at once (e.g. 'Non-accrual. PIK.').

    Disambiguator: some FSK-style legend entries quote BOTH the 5% rule and
    the 25% rule in their definition before stating which CATEGORY this
    footnote denotes (e.g. "...is deemed to be an "affiliated person" but
    is NOT deemed to "control"."). Both flag-rules will fire on the verbose
    quote; we drop the wrong one based on the trailing disambiguator phrase.

    R6-AV05: a legend entry that says "these investments are NOT pledged as
    collateral" fires the is_pledged rule via substring match.  Post-process:
    if the text also matches the NOT-pledged pattern, drop the flag.
    """
    flags: list[str] = []
    for rx, flag in CONTROLLED_VOCAB_RULES:
        if rx.search(meaning or ""):
            flags.append(flag)
    if (
        "is_controlled_affiliate" in flags
        and "is_non_controlled_affiliate" in flags
        and meaning
    ):
        if _NOT_CONTROL_DISAMBIG.search(meaning):
            flags = [f for f in flags if f != "is_controlled_affiliate"]
        elif _AND_CONTROL_DISAMBIG.search(meaning):
            flags = [f for f in flags if f != "is_non_controlled_affiliate"]
    # R6-AV05: drop is_pledged when the legend text explicitly says NOT pledged.
    if "is_pledged" in flags and meaning and _NOT_PLEDGED_RE.search(meaning):
        flags = [f for f in flags if f != "is_pledged"]
    # Policy-text disambiguator: drop is_non_accrual when the legend
    # text is describing the filer's NA POLICY rather than asserting
    # this row is on non-accrual. The SLRC marker (m) is the canonical
    # example: "Investments that are expected to pay regularly scheduled
    # interest in cash are generally placed on non-accrual status when…"
    if "is_non_accrual" in flags and meaning and _NA_POLICY_DISAMBIG.search(meaning):
        flags = [f for f in flags if f != "is_non_accrual"]
    # Prior-period NA disambiguator: footnote names a year != period_end's
    # year (e.g. ARCC fn(10) "as of December 31, 2024" in a 2025 filing).
    if "is_non_accrual" in flags and meaning and period_end:
        m = _NA_AS_OF_DATE_RE.search(meaning)
        if m:
            footnote_year = m.group(1)
            period_year = period_end[:4]
            if footnote_year != period_year:
                flags = [f for f in flags if f != "is_non_accrual"]
    return flags


# ── Currency sigil detection ────────────────────────────────────────────────


CURRENCY_SIGILS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

# R6-D13: multi-char currency prefixes that must be checked before the
# single-char sigils so "A$" doesn't fall through to "$" → USD.
_MULTICHAR_CURRENCY_MAP: list[tuple[str, str]] = [
    ("A$",  "AUD"),   # Australian dollar (ADS, OCIC, OTF)
    ("C$",  "CAD"),   # Canadian dollar (ADS, OCIC)
    ("Skr", "SEK"),   # Swedish krona
    ("Nkr", "NOK"),   # Norwegian krone
    ("Dkr", "DKK"),   # Danish krone
    ("₩",   "KRW"),   # Korean won
    ("₣",   "CHF"),   # Swiss franc (legacy Unicode)
    ("₹",   "INR"),   # Indian rupee
    ("₺",   "TRY"),   # Turkish lira
]


_ISO_CURRENCY_RE = re.compile(
    r"(?:^|\s)(GBP|EUR|AUD|CAD|SEK|NOK|DKK|CHF|JPY|NZD|HKD|SGD|MXN|BRL|INR|"
    r"ZAR|KRW|TRY|CNY|RUB|PLN|CZK|HUF|ILS|AED|SAR)(?:$|\s|\)|,|\.|\d)",
    re.I,
)


def infer_currency_from_cell(text: str) -> Optional[str]:
    if not text:
        return None
    # R6-D13: check multi-char prefixes first (longest-first within group)
    # so "A$" is not matched as plain "$" → USD.
    t = text.strip()
    for prefix, cur in _MULTICHAR_CURRENCY_MAP:
        # Case-insensitive prefix check for alpha prefixes, exact for symbols
        if prefix.isalpha():
            if t.lower().startswith(prefix.lower()):
                return cur
        else:
            if t.startswith(prefix):
                return cur
    for sig, cur in CURRENCY_SIGILS.items():
        if sig in text:
            return cur
    # ISO-3 code in the cell text (BXSL emits "GBP", "EUR", "AUD" as
    # standalone cell content adjacent to the par-amount column; BCRED's
    # Riser/Excelitas GBP+EUR tranches use the same pattern). Bound by
    # word boundaries to avoid matching company-name fragments
    # ("EURasia Group" must not match "EUR").
    m = _ISO_CURRENCY_RE.search(text)
    if m:
        return m.group(1).upper()
    return None


# ──────────────────────────────────────────────────────────────────────
# Banner-propagation helper (reusable across adapters)
# ──────────────────────────────────────────────────────────────────────
#
# Many BDC SOIs print "section banners" as standalone short rows that
# describe the seniority (or affiliation, or industry) of the data rows
# that follow. The base extract pipeline routes these rows to either
# classify_row()=='industry' (single non-empty cell in company column)
# or simply ingests them as data rows that get dropped because they have
# no monetary cells.
#
# Either way, the seniority label never makes it onto the actual data
# rows — so downstream consumers see investment_type_raw=None on every
# investment unless the filer also prints the seniority inline.
#
# This helper inverts the problem: AFTER the base pipeline has run and
# `parser.investments` is populated, walk every table in document order
# and look for banner rows. For each matched banner, propagate the
# corresponding seniority label to every investment whose source row
# came after the banner (and before the next one).
#
# Each adapter wires its own banner-regex → seniority-label dict so
# they can encode their own dialect. The default set below covers the
# canonical phrases used by Blue Owl Technology Finance / FSK / CION /
# HLEND / GBDC-summary-style / OCREDIT — i.e. the 70%+ of BDCs whose
# section banners are phrased plainly.

_DEFAULT_BANNER_PATTERNS: list[tuple["re.Pattern[str]", str]] = [
    # Sub-Categories matter when they're more specific. Order is
    # CRITICAL: first match wins, so revolver / DDTL / second-lien
    # patterns must run before the generic "First Lien" / "Senior
    # Secured" catchalls.
    (re.compile(r"\b(?:revolv|revolver)\b", re.I), "Revolver"),
    (re.compile(r"\bdelayed[-\s]?draw\b|\bDDTLs?\b", re.I), "Delayed Draw Term Loan"),
    (re.compile(r"\bsecond[-\s]?lien\b|^2L\b", re.I), "Second Lien Senior Secured Loan"),
    (re.compile(r"\bunitranche\b|\bone[\s\-]?stop\b", re.I), "Unitranche"),
    (re.compile(r"\bfirst[\s\-]?lien\b|^1L\b", re.I), "First Lien Senior Secured Loan"),
    (re.compile(r"\bsenior\s+secured\s+(?:loans?|notes?|debt|first)\b", re.I), "Senior Secured Loan"),
    (re.compile(r"^\s*senior\s+secured\b", re.I), "Senior Secured Loan"),
    # ANTARES uses bare "Secured Debt" / "Secured Loans" as the
    # first-lien section banner. Match it as a SENIOR-SECURED bucket
    # rather than the more specific First-Lien label, since the
    # banner doesn't actually disambiguate first vs second lien.
    (re.compile(r"^\s*secured\s+(?:debt|loans?|notes?)\b", re.I), "Senior Secured Loan"),
    (re.compile(r"\bsubordinated\s+(?:loans?|debt|notes?)\b", re.I), "Subordinated Debt"),
    (re.compile(r"\bmezzanine\b", re.I), "Mezzanine"),
    (re.compile(r"\b(?:unsecured\s+debt|senior\s+unsecured)\b", re.I), "Unsecured Debt"),
    (re.compile(r"\b(?:asset[\s\-]?based\s+finance|ABF)\b", re.I), "Asset Based Finance"),
    (re.compile(r"\b(?:CLO|collateralized\s+loan\s+obligation)s?\b", re.I), "CLO"),
    (re.compile(r"\bjoint\s+venture\b", re.I), "Joint Venture"),
    (re.compile(r"\bpreferred\s+(?:stock|equity|interests?|units?)\b", re.I), "Preferred Stock"),
    (re.compile(r"\bcommon\s+(?:stock|equity|interests?|units?)\b", re.I), "Common Stock"),
    (re.compile(r"^\s*warrants?\b", re.I), "Warrant"),
    (re.compile(r"^\s*equity\b(?!\s+real\s+estate)", re.I), "Equity Investment"),
    (re.compile(r"^\s*equity\s*/\s*other\b", re.I), "Equity Investment"),
]

# Banner-row shape: at most this many non-empty cells (sub-totals and
# data rows generally have many more). Tuned to allow "X — pct %"
# style banners which can render as 2-3 cells under the parser.
_BANNER_MAX_NONEMPTY_CELLS = 4
# Banner labels can't legitimately exceed this length — anything longer
# is probably a real company name or business description.
_BANNER_MAX_LABEL_CHARS = 120


def _row_first_text(row: "Tag") -> str:
    """Return the first non-empty cell text in a row, normalized."""
    try:
        spans = row_cell_spans(row)
    except Exception:
        return ""
    for _, _, t in spans:
        if t and t.strip():
            return t.strip()
    return ""


def _row_nonempty_count(row: "Tag") -> int:
    try:
        spans = row_cell_spans(row)
    except Exception:
        return 0
    return sum(1 for _, _, t in spans if t and t.strip())


def detect_banner_label(
    text: str,
    patterns: list[tuple["re.Pattern[str]", str]] = _DEFAULT_BANNER_PATTERNS,
) -> Optional[str]:
    """Return the canonical seniority label this banner-row text
    advertises, or None if the text doesn't look like a section banner.

    A banner shape is a short label followed by an optional percent
    marker (e.g. "Senior Secured Loans—First Lien— 128.6 %"). We
    accept the bare label too, since some filers (HLEND, ANTARES,
    CION older filings) print "First Lien Debt" alone.
    """
    if not text or len(text) > _BANNER_MAX_LABEL_CHARS:
        return None
    # Strip trailing pct markers / footnote markers so the regex below
    # doesn't have to anchor on the entire string.
    stripped = re.sub(r"\s*[—\-–]\s*\(?\s*[\d.,]+\s*%[^A-Za-z]*$", "", text)
    stripped = re.sub(r"\s*\([a-z0-9]\)\s*$", "", stripped, flags=re.I)
    for rx, label in patterns:
        if rx.search(stripped):
            return label
    return None


def propagate_section_banners(
    parser,
    patterns: Optional[list[tuple["re.Pattern[str]", str]]] = None,
    overwrite: bool = False,
) -> int:
    """Walk every parsed table in document order, identify section
    banners, and stamp investment_type_raw on each investment based on
    the most recent banner before it.

    Returns the number of investments whose investment_type_raw was set
    by this call (useful for tests / debug logs).

    Parameters
    ----------
    parser : an extracted V2SOIParser instance — must expose
        `tables` (list[Tag]) and `investments` (list[Investment] each
        with `_source_table_index` and `_source_row_index`).
    patterns : optional override of (regex, label) tuples. The default
        set covers the canonical phrases ("First Lien", "Second Lien",
        "Subordinated Debt", "Asset Based Finance", "Joint Venture",
        "Preferred Stock", "Common Stock", "Warrant", etc.).
    overwrite : if True, replace investment_type_raw even when the
        adapter already extracted one. Default False — only fill
        blanks, never clobber adapter work.
    """
    if not getattr(parser, "tables", None) or not getattr(parser, "investments", None):
        return 0
    use_patterns = patterns or _DEFAULT_BANNER_PATTERNS

    # Restrict banner discovery to SOI tables. Cover/narrative/issuance
    # tables (SCM, PSEC) routinely contain phrases like "Common Stock,
    # par value..." or "Senior Secured" in plain prose which would
    # otherwise stamp every position with the wrong seniority.
    soi_tables = getattr(parser, "_main_soi_tables", None) or parser.tables
    soi_table_indices = {parser.tables.index(t) for t in soi_tables}

    # Build a banner-by-position map.
    banner_at: dict[tuple[int, int], str] = {}
    for ti, table in enumerate(parser.tables):
        if ti not in soi_table_indices:
            continue
        for ri, row in enumerate(table.find_all("tr")):
            if _row_nonempty_count(row) > _BANNER_MAX_NONEMPTY_CELLS:
                continue
            text = _row_first_text(row)
            label = detect_banner_label(text, use_patterns)
            if label:
                banner_at[(ti, ri)] = label

    if not banner_at:
        return 0

    sorted_positions = sorted(banner_at.items())  # [((ti, ri), label), ...]

    # Walk investments in document order and stamp.
    inv_sorted = sorted(
        parser.investments,
        key=lambda inv: (inv._source_table_index, inv._source_row_index),
    )
    bi = 0
    current = None
    n_stamped = 0
    for inv in inv_sorted:
        pos = (inv._source_table_index, inv._source_row_index)
        while bi < len(sorted_positions) and sorted_positions[bi][0] <= pos:
            current = sorted_positions[bi][1]
            bi += 1
        if current is None:
            continue
        if not inv.investment_type_raw or overwrite:
            inv.investment_type_raw = current
            n_stamped += 1
    return n_stamped


# ──────────────────────────────────────────────────────────────────────
# Affiliation-section propagation
# ──────────────────────────────────────────────────────────────────────
#
# Most BDC SOIs partition their holdings across affiliation buckets using
# section-header rows like:
#   "Non-controlled/non-affiliated company investments"
#   "Non-Controlled/Affiliated Investments"
#   "Controlled/Affiliated Investments"
#   "Controlled affiliate company equity investments"
#
# These headers are single-cell rows that sit above their child positions.
# The base parser doesn't propagate them generically, so adapters that
# don't have custom logic (GBDC, OCIC, OTF, ASIF, ADS, MAIN, …) end up
# with every row tagged "Non-controlled/non-affiliated" regardless of
# which section it sits in. Confirmed defect in GBDC: $305M of affiliate
# / controlled investments mis-categorized.
#
# Per-adapter opt-in via `USE_AFFILIATION_PROPAGATION = True` keeps this
# safe — adapters that already have custom affiliation logic (HLEND,
# ANTARES, FSK) are unaffected.

_AFFILIATION_PATTERNS: list[tuple["re.Pattern[str]", str]] = [
    # Order matters: most specific first. Both "affiliate" (bare) and
    # "affiliated" (with -d) accepted because GBDC uses both forms.
    # Negative lookbehind prevents "non-controlled" matching the bare
    # "controlled" pattern.
    (re.compile(
        r"\bnon[-\s]?controlled\s*/\s*non[-\s]?affiliated?\b"
        r"|\bnon[-\s]?controlled,?\s*non[-\s]?affiliated?\b",
        re.I,
    ), "Non-controlled/non-affiliated"),
    (re.compile(
        r"\bnon[-\s]?controlled[-\s]*affiliated?\b"
        r"|\bnon[-\s]?controlled\s*/\s*affiliated?\b"
        r"|\bnon[-\s]?controlled,?\s*affiliated?\b",
        re.I,
    ), "Non-controlled affiliate"),
    (re.compile(
        r"(?<!non-)(?<!non )\bcontrolled[-\s]*affiliated?\b"
        r"|(?<!non-)(?<!non )\bcontrolled\s*/\s*affiliated?\b"
        r"|(?<!non-)(?<!non )\bcontrolled,?\s*affiliated?\b",
        re.I,
    ), "Controlled affiliate"),
]

# Tighter shape constraint than the seniority banner walker: affiliation
# section headers are very short (1-3 cells, label only).
_AFFILIATION_MAX_NONEMPTY_CELLS = 3


def detect_affiliation_banner(text: str) -> Optional[str]:
    """Return the canonical category this section-header text declares,
    or None if it's not an affiliation banner. Categories are exactly
    one of: 'Non-controlled/non-affiliated', 'Non-controlled affiliate',
    'Controlled affiliate'."""
    if not text or len(text) > 200:
        return None
    for rx, category in _AFFILIATION_PATTERNS:
        if rx.search(text):
            return category
    return None


def propagate_affiliation_banners(parser, overwrite: bool = False) -> int:
    """Walk every SOI table in document order, identify affiliation
    section banners, and stamp `category` + `is_controlled_affiliate` /
    `is_non_controlled_affiliate` on each investment based on the most
    recent banner before it.

    Returns the number of investments updated.

    Adapter opt-in: set `USE_AFFILIATION_PROPAGATION = True` on the
    parser class. The base `extract()` checks for this flag and invokes
    this helper at the end of extract.
    """
    if not getattr(parser, "tables", None) or not getattr(parser, "investments", None):
        return 0

    soi_tables = getattr(parser, "_main_soi_tables", None) or parser.tables
    soi_table_indices = {parser.tables.index(t) for t in soi_tables}

    banner_at: dict[tuple[int, int], str] = {}
    for ti, table in enumerate(parser.tables):
        if ti not in soi_table_indices:
            continue
        for ri, row in enumerate(table.find_all("tr")):
            if _row_nonempty_count(row) > _AFFILIATION_MAX_NONEMPTY_CELLS:
                continue
            text = _row_first_text(row)
            category = detect_affiliation_banner(text)
            if category:
                banner_at[(ti, ri)] = category

    if not banner_at:
        return 0

    sorted_positions = sorted(banner_at.items())

    inv_sorted = sorted(
        parser.investments,
        key=lambda inv: (inv._source_table_index, inv._source_row_index),
    )
    bi = 0
    current_cat: Optional[str] = None
    n_stamped = 0
    for inv in inv_sorted:
        pos = (inv._source_table_index, inv._source_row_index)
        while bi < len(sorted_positions) and sorted_positions[bi][0] <= pos:
            current_cat = sorted_positions[bi][1]
            bi += 1
        if current_cat is None:
            continue
        # Only stamp when the field is empty/default or overwrite=True.
        # Adapter-level explicit values (HLEND, FSK) never get clobbered.
        should_stamp = overwrite or not inv.category or inv.category == "Non-controlled/non-affiliated"
        if should_stamp and inv.category != current_cat:
            inv.category = current_cat
            n_stamped += 1
            # Sync footnote_flags + boolean projections.
            flags = set(inv.footnote_flags or [])
            flags.discard("is_controlled_affiliate")
            flags.discard("is_non_controlled_affiliate")
            inv.is_controlled_affiliate = False
            inv.is_non_controlled_affiliate = False
            if current_cat == "Controlled affiliate":
                flags.add("is_controlled_affiliate")
                inv.is_controlled_affiliate = True
            elif current_cat == "Non-controlled affiliate":
                flags.add("is_non_controlled_affiliate")
                inv.is_non_controlled_affiliate = True
            inv.footnote_flags = sorted(flags)
    return n_stamped
