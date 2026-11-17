"""Deterministic page-integrity pass for assembled pages: brand, links, images.

The LLM fills content slots; it cannot know what the assembled page will contain
(which sections exist, which image hosts the rendering browser can reach, what the
other blocks called the brand — in split mode every block is filled by an
independent call). So three properties that LOOK like copy are enforced here, in
code, after the plan is resolved and before any block renders:

* **One brand.** A single canonical brand name (the caller's ``brand.name``, else
  the first brand the plan itself wrote) is stamped into every ``brand`` slot, and
  any other brand spelling the plan used is rewritten to it in every text slot.
  Live run 01a156a6 shipped "Hearth & Crumb" in the header and "The Daily Crumb"
  in the footer.
* **No dead links.** Every href slot resolves to a section id that exists on the
  page, a ``mailto:``/``tel:`` taken from the brief, or an absolute http(s) URL the
  plan wrote. ``#``, empty, dangling ``#x`` and site-relative paths (which point at
  nothing inside a single self-contained artifact) are rewritten to the best
  matching real section; a nav/footer link with no plausible target is removed.
* **No unverified image hotlinks.** An image slot is kept only when it is an
  image this service produced for the page (exact value — the inlined hero) or
  on a ``CATALOG_IMAGE_HOST_ALLOWLIST`` host. Anything else — an Unsplash hotlink
  the browser blocks with ``net::ERR_BLOCKED_BY_ORB``, an LLM ``data:`` URI, a
  MinIO URL, a relative path — is dropped, and the block renders its fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from src.services.catalog.registry import BlockManifest

# Stable, human-readable section ids per block key (the anchors links resolve to).
SECTION_IDS: dict[str, str] = {
    "navbar": "top",
    "hero": "home",
    "logos": "partners",
    "features": "features",
    "stats": "highlights",
    "testimonials": "testimonials",
    "pricing": "pricing",
    "faq": "faq",
    "hours": "hours",
    "cta": "get-started",
    "footer": "contact",
}

# Words in a link label / dangling target that point at a given block key.
# Ordered: the first block whose synonyms match (and which exists) wins.
_LINK_SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pricing", ("pricing", "price", "prices", "plan", "plans", "cost", "rates")),
    ("faq", ("faq", "faqs", "question", "questions", "help")),
    ("testimonials", ("review", "reviews", "testimonial", "testimonials", "customers")),
    ("logos", ("partners", "partner", "clients", "press")),
    ("stats", ("stats", "numbers", "results", "impact", "highlights")),
    (
        "features",
        (
            "features",
            "feature",
            "menu",
            "products",
            "product",
            "services",
            "service",
            "offer",
            "offerings",
            "about",
            "story",
            "why",
            "how",
            "work",
            "shop",
            "collection",
        ),
    ),
    (
        "cta",
        (
            "order",
            "start",
            "started",
            "book",
            "reserve",
            "signup",
            "sign",
            "join",
            "buy",
            "subscribe",
            "try",
            "demo",
            "get",
            "preorder",
        ),
    ),
    # Before ``footer``: a "View Opening Hours" / "Visit us" CTA lands on the hours
    # block when the page has one (it used to land on the footer, which has none).
    (
        "hours",
        (
            "hours",
            "opening",
            "open",
            "visit",
            "location",
            "locations",
            "find",
            "address",
            "directions",
            "when",
        ),
    ),
    (
        "footer",
        ("contact", "location", "locations", "visit", "hours", "find", "address"),
    ),
    ("hero", ("home", "top")),
)

_CONTACT_WORDS = frozenset({"contact", "email", "mail", "call", "phone", "reach", "write"})
_CALL_WORDS = frozenset({"call", "phone"})

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# A phone number: optional +, then 7-15 digits allowing space/dot/dash/paren separators.
_PHONE_RE = re.compile(r"(?<![\w])\+?\d[\d\s().-]{6,18}\d(?![\w])")

# Brand slot default in the catalog — never treated as a real brand.
_PLACEHOLDER_BRANDS = frozenset({"", "brand"})


@dataclass
class IntegrityReport:
    """What the integrity pass changed (surfaced for logs/tests, not tenant-facing)."""

    brand: str = ""
    brand_aliases: list[str] = field(default_factory=list)
    links_rewritten: int = 0
    links_removed: int = 0
    images_dropped: int = 0
    hours_dropped: int = 0
    testimonials_dropped: int = 0
    testimonial_blocks_removed: int = 0


def extract_contact(text: str) -> dict[str, str]:
    """Pull a contact email / phone out of the brief text (empty dict if none)."""
    found: dict[str, str] = {}
    if not text:
        return found
    email = _EMAIL_RE.search(text)
    if email:
        found["email"] = email.group(0)
    phone = _PHONE_RE.search(_EMAIL_RE.sub(" ", text))
    if phone:
        digits = re.sub(r"[^\d+]", "", phone.group(0))
        if 7 <= len(digits.lstrip("+")) <= 15:
            found["phone"] = digits
    return found


def assign_section_ids(keys: list[str]) -> list[str]:
    """One unique id per rendered block, in order (``features``, ``features-2`` ...)."""
    used: dict[str, int] = {}
    ids: list[str] = []
    for key in keys:
        base = SECTION_IDS.get(key, key)
        n = used.get(base, 0) + 1
        used[base] = n
        ids.append(base if n == 1 else f"{base}-{n}")
    return ids


# ---------------------------------------------------------------------------
# Brand
# ---------------------------------------------------------------------------


def _clean_brand(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _PLACEHOLDER_BRANDS else text


def canonical_brand(resolved: list[tuple[BlockManifest, dict]], brand: dict | None) -> str:
    """The one brand name the page uses: the caller's, else the plan's first."""
    given = _clean_brand((brand or {}).get("name"))
    if given:
        return given
    for manifest, content in resolved:
        if "brand" in manifest.slots:
            found = _clean_brand(content.get("brand"))
            if found:
                return found
    return ""


def _replace_aliases(value: object, aliases: list[str], canonical: str) -> object:
    if isinstance(value, str):
        for alias in aliases:
            value = re.sub(re.escape(alias), canonical, value, flags=re.IGNORECASE)
        return value
    if isinstance(value, list):
        return [_replace_aliases(v, aliases, canonical) for v in value]
    if isinstance(value, dict):
        return {k: _replace_aliases(v, aliases, canonical) for k, v in value.items()}
    return value


def enforce_brand(
    resolved: list[tuple[BlockManifest, dict]],
    brand: dict | None,
    report: IntegrityReport,
) -> str:
    """Stamp one brand into every brand slot and rewrite other spellings in copy."""
    canonical = canonical_brand(resolved, brand)
    report.brand = canonical
    if not canonical:
        return ""
    aliases: list[str] = []
    for manifest, content in resolved:
        if "brand" in manifest.slots:
            found = _clean_brand(content.get("brand"))
            if found and found.casefold() != canonical.casefold() and found not in aliases:
                aliases.append(found)
            content["brand"] = canonical
    # Longest first so "The Daily Crumb Co" is replaced before "The Daily Crumb".
    aliases.sort(key=len, reverse=True)
    report.brand_aliases = aliases
    if aliases:
        for manifest, content in resolved:
            for name, spec in manifest.slots.items():
                if spec.type in ("text", "list") and name != "brand":
                    content[name] = _replace_aliases(content.get(name), aliases, canonical)
    return canonical


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def enforce_images(
    resolved: list[tuple[BlockManifest, dict]],
    trusted: TrustedImages,
    report: IntegrityReport,
) -> None:
    """Blank every image slot whose URL is not on a source this service controls."""
    for manifest, content in resolved:
        for name, spec in manifest.slots.items():
            if spec.type == "image":
                if content.get(name) and not trusted.allows(content[name]):
                    content[name] = ""
                    report.images_dropped += 1
            elif spec.type == "list":
                image_fields = [k for k, t in spec.item_slots.items() if t == "image"]
                for item in content.get(name) or []:
                    for k in image_fields:
                        if item.get(k) and not trusted.allows(item[k]):
                            item[k] = ""
                            report.images_dropped += 1


@dataclass(frozen=True)
class TrustedImages:
    """Where an assembled page may load ``<img>`` from.

    ``own`` holds the exact image values THIS service produced for this page — the
    generated hero as a ``data:image/jpeg;base64,...`` URI. A value is trusted only
    if it is byte-for-byte one of them: an LLM-written ``data:`` URI (or any other
    value) is never trusted by shape. ``hosts`` is the operator allowlist
    (``CATALOG_IMAGE_HOST_ALLOWLIST``), matched by exact host or host:port on the
    parsed URL — never a substring test.

    No MinIO host is trusted: a ``minio.test`` URL is plain http, tailnet-only
    and (presigned) expiring, so it breaks under the served page's
    ``img-src https: data:`` CSP and outside the cluster. Our own images are
    inlined instead, which keeps the page self-contained.
    """

    hosts: frozenset[str] = frozenset()
    own: frozenset[str] = frozenset()

    def allows(self, url: object) -> bool:
        text = str(url or "").strip()
        if not text:
            return False
        if text in self.own:
            return True
        parts = urlsplit(text)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return False
        host = parts.hostname.lower()
        try:
            port = parts.port
        except ValueError:
            return False
        hostport = f"{host}:{port}" if port else host
        return host in self.hosts or hostport in self.hosts


def trusted_image_hosts(allowlist: list[str]) -> frozenset[str]:
    """Hostnames (and host:port) of the operator image allowlist."""
    hosts: set[str] = set()
    for entry in allowlist:
        text = str(entry or "").strip().lower()
        if not text:
            continue
        if "://" in text:
            parts = urlsplit(text)
            if parts.hostname:
                hosts.add(parts.hostname)
                if parts.port:
                    hosts.add(f"{parts.hostname}:{parts.port}")
            continue
        hosts.add(text)
        hosts.add(text.split(":", 1)[0])
    return frozenset(hosts)


# ---------------------------------------------------------------------------
# Opening hours
# ---------------------------------------------------------------------------
#
# Hours are compared as PARSED, NORMALISED ranges — ``(start_min, end_min)`` minutes
# after midnight, per day — never as bare numbers. "9am-5pm" and "09:00-17:00" are
# the same range (540, 1020); "9:30-17:00" is not "9:00-17:30" even though the two
# use the same digits; and a bare "9-5" means 9:00-17:00 (the end is smaller than
# the start, so it is read as pm).

_DAYS: tuple[tuple[str, ...], ...] = (
    ("monday", "mon", "mo"),
    ("tuesday", "tue", "tues", "tu"),
    ("wednesday", "wed", "we"),
    ("thursday", "thu", "thur", "thurs", "th"),
    ("friday", "fri", "fr"),
    ("saturday", "sat", "sa"),
    ("sunday", "sun", "su"),
)
_DAY_INDEX: dict[str, int] = {alias: i for i, names in enumerate(_DAYS) for alias in names}
_DAY_GROUPS: dict[str, frozenset[int]] = {
    "daily": frozenset(range(7)),
    "everyday": frozenset(range(7)),
    "weekdays": frozenset(range(5)),
    "weekday": frozenset(range(5)),
    "weekends": frozenset({5, 6}),
    "weekend": frozenset({5, 6}),
}
_SEP = r"\s*(?:-|\u2013|\u2014|to|until|till|through|thru)\s*"
_DAY_WORD = r"[A-Za-z]{2,9}\.?"
_DAY_RANGE_RE = re.compile(rf"\b({_DAY_WORD}){_SEP}({_DAY_WORD})(?![A-Za-z])", re.IGNORECASE)
_DAY_TOKEN_RE = re.compile(r"\b[A-Za-z]{2,9}\b")
_T = r"(\d{1,2})(?:\s*[:.h]\s*(\d{2}))?\s*(a\.?\s?m\.?|p\.?\s?m\.?)?"
_RANGE_RE = re.compile(rf"(?<![\w:.$£€¥]){_T}{_SEP}{_T}(?![\w:])", re.IGNORECASE)
# A lone clock time: needs minutes or am/pm — a bare "7" ("open 7 days") is not a time.
_POINT_RE = re.compile(rf"(?<![\w:.$£€¥]){_T}(?![\w:])", re.IGNORECASE)
_CLOSED_RE = re.compile(r"\bclosed\b", re.IGNORECASE)
_HOURS_WORDS = frozenset({"hours", "open", "opening", "opens", "close", "closes", "closing"})
_ANY_DAY = -1

Range = tuple[int, int]


def _meridiem(raw: str | None) -> str | None:
    if not raw:
        return None
    return "pm" if raw.lower().startswith("p") else "am"


def _to_minutes(hour: int, minute: int, meridiem: str | None) -> int:
    if meridiem == "am" and hour == 12:
        hour = 0
    elif meridiem == "pm" and hour < 12:
        hour += 12
    return hour * 60 + minute


def _range_from_match(m: re.Match[str]) -> Range | None:
    h1, m1, ap1, h2, m2, ap2 = m.groups()
    a, b = int(h1), int(h2)
    mi1, mi2 = int(m1 or 0), int(m2 or 0)
    if a > 24 or b > 24 or mi1 > 59 or mi2 > 59:
        return None
    mer1, mer2 = _meridiem(ap1), _meridiem(ap2)
    if mer1 is None and mer2 is not None:
        # "9-5pm" -> 9am-5pm; "1-5pm" -> 1pm-5pm: inherit the end's meridiem
        # unless that would put the start after the end ("11-2pm" = 11am-2pm).
        mer1 = mer2 if _to_minutes(a, mi1, mer2) < _to_minutes(b, mi2, mer2) else "am"
    if mer2 is None and mer1 is not None:
        mer2 = mer1
    start, end = _to_minutes(a, mi1, mer1), _to_minutes(b, mi2, mer2)
    if end <= start and mer2 is None and b < 12:
        end += 12 * 60  # bare "9-5" -> 9:00-17:00
    return start, end


def _days_in(clause: str) -> set[int]:
    days: set[int] = set()
    for m in _DAY_RANGE_RE.finditer(clause):
        lo = _DAY_INDEX.get(m.group(1).lower().rstrip("."))
        hi = _DAY_INDEX.get(m.group(2).lower().rstrip("."))
        if lo is not None and hi is not None:
            span = range(lo, hi + 1) if lo <= hi else [*range(lo, 7), *range(hi + 1)]
            days.update(span)
    for tok in _DAY_TOKEN_RE.findall(clause):
        low = tok.lower()
        # Two-letter aliases ("we", "su", "th") are English words; only honour them
        # inside an explicit day range above.
        if low in _DAY_INDEX and len(low) >= 3:
            days.add(_DAY_INDEX[low])
        elif low in _DAY_GROUPS:
            days.update(_DAY_GROUPS[low])
    return days


@dataclass
class _Hours:
    """Ranges (and lone times) a text states, keyed by day (``_ANY_DAY`` = unspecified)."""

    ranges: dict[int, set[Range]] = field(default_factory=dict)
    points: set[int] = field(default_factory=set)

    def all_ranges(self) -> set[Range]:
        out: set[Range] = set()
        for rs in self.ranges.values():
            out |= rs
        return out

    def is_empty(self) -> bool:
        return not self.ranges and not self.points


def parse_hours(text: object, *, days: set[int] | None = None) -> _Hours:
    """Parse every time range (and lone clock time) in ``text``, per day.

    Day words scope the ranges next to them. A text that names its days BEFORE the
    first range ("Tue-Sat 7am-3pm, Sun 8:30-1pm") gives each range the days
    written between it and the previous range; one that puts them AFTER
    ("9-5 weekdays, 10-2 Saturday") gives each range the days up to the next one.
    ``days`` overrides both (an hours row carries its day in its own field). A
    range with no day is stored under ``_ANY_DAY``.
    """
    raw = str(text or "")
    out = _Hours()
    matches = [m for m in _RANGE_RE.finditer(raw) if _range_from_match(m) is not None]
    days_before = bool(matches) and bool(_days_in(raw[: matches[0].start()]))
    prev_end = 0
    for i, m in enumerate(matches):
        if days is not None:
            scope = days
        elif days_before:
            scope = _days_in(raw[prev_end : m.start()])
        else:
            nxt = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            scope = _days_in(raw[m.end() : nxt])
        rng = _range_from_match(m)
        if rng is not None:
            for d in sorted(scope) or [_ANY_DAY]:
                out.ranges.setdefault(d, set()).add(rng)
        prev_end = m.end()
    rest = _RANGE_RE.sub(" ", raw)
    for m in _POINT_RE.finditer(rest):
        hour, minute, ap = m.groups()
        if minute is None and ap is None:
            continue
        if int(hour) <= 24 and int(minute or 0) <= 59:
            out.points.add(_to_minutes(int(hour), int(minute or 0), _meridiem(ap)))
    return out


def _stated(claim: _Hours, brief: _Hours) -> bool:
    """Every range/time ``claim`` states is one the brief states (for that day)."""
    if claim.is_empty():
        return False
    every = brief.all_ranges()
    for day, ranges in claim.ranges.items():
        allowed = every if day == _ANY_DAY else brief.ranges.get(day, set())
        allowed = allowed | brief.ranges.get(_ANY_DAY, set())
        if not ranges <= allowed:
            return False
    endpoints = {t for rng in every for t in rng} | brief.points
    return claim.points <= endpoints


def _hours_row_grounded(item: dict, brief: str, brief_hours: _Hours) -> bool:
    day_text = str(item.get("day") or "").strip()
    time = str(item.get("time") or "").strip()
    if not time or not day_text:
        return False
    row_days = _days_in(day_text)
    claim = parse_hours(time, days=row_days or None)
    if not claim.is_empty():
        return _stated(claim, brief_hours)
    if _CLOSED_RE.search(time):
        return bool(_CLOSED_RE.search(brief))
    # Free text with no clock time ("By appointment"): only if the brief says it.
    return time.casefold() in brief.casefold()


def _faq_answer_grounded(item: dict, brief_hours: _Hours) -> bool:
    if not (_words(item.get("question"), item.get("answer")) & _HOURS_WORDS):
        return True  # not about opening hours
    claim = parse_hours(item.get("answer"))
    return claim.is_empty() or _stated(claim, brief_hours)


def enforce_hours(
    resolved: list[tuple[BlockManifest, dict]], brief: str, report: IntegrityReport
) -> None:
    """Opening hours come ONLY from the brief — never from the model's imagination.

    * ``hours`` block: a row is kept only if every time range it states equals a
      range the brief states for that day (normalised minutes, am/pm resolved).
      With no hours in the brief every row goes, and the block renders "Opening
      hours on request". An address using any word the brief lacks is blanked.
    * ``faq`` block: a Q&A about opening/closing whose answer states a range or
      time the brief does not is an invented fact (the live run answered "When
      are you open?" with made-up hours) — the whole pair is dropped.
    """
    brief = brief or ""
    brief_hours = parse_hours(brief)
    for manifest, content in resolved:
        if manifest.key == "hours":
            items = content.get("items") or []
            kept = [i for i in items if _hours_row_grounded(i, brief, brief_hours)]
            report.hours_dropped += len(items) - len(kept)
            content["items"] = kept
            address = str(content.get("address") or "").strip()
            if address and not _words(address) <= _words(brief):
                content["address"] = ""
        elif manifest.key == "faq":
            items = content.get("items") or []
            kept = [i for i in items if _faq_answer_grounded(i, brief_hours)]
            report.hours_dropped += len(items) - len(kept)
            content["items"] = kept


# ---------------------------------------------------------------------------
# Testimonials
# ---------------------------------------------------------------------------

# Every quotation mark / apostrophe form a model or a brief may use. Removed (not
# mapped) before comparison, so `“Best bread”`, `"Best bread"` and `Best bread` match.
_QUOTE_CHARS_RE = re.compile(
    "[\"'\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f\u00ab\u00bb\u2039\u203a`\u00b4]"
)
_DASHES_RE = re.compile("[\u2010-\u2015\u2212]")
_EDGE_PUNCT = " .,;:!?-\u2026"
#: A shorter "quote" ("Great!", "Love it") is not evidence the brief quoted anyone:
#: three words of ordinary brief copy would match almost any page.
_MIN_QUOTE_WORDS = 3


def _norm_quote(text: object) -> str:
    """Casefold, drop quotation marks, unify dashes, collapse whitespace, trim edges."""
    value = _QUOTE_CHARS_RE.sub("", str(text or ""))
    value = _DASHES_RE.sub("-", value).replace("\u2026", "...")
    return " ".join(value.casefold().split()).strip(_EDGE_PUNCT)


def _testimonial_grounded(item: dict, norm_brief: str) -> bool:
    quote = _norm_quote(item.get("quote"))
    return len(quote.split()) >= _MIN_QUOTE_WORDS and quote in norm_brief


def enforce_testimonials(
    resolved: list[tuple[BlockManifest, dict]], brief: str, report: IntegrityReport
) -> None:
    """A testimonial is a quote from a REAL customer — it comes ONLY from the brief.

    Same rule as opening hours: the live run (01a15968, "Bakkerij De Molen") shipped
    "Mark T." and "Elena R." praising a bakery whose brief quoted nobody. A quote is
    kept only when its text (quotation marks, dashes, case and whitespace normalised)
    appears verbatim in the brief; an author/role using any word the brief lacks is
    blanked rather than attributed to an invented person. A testimonials block left
    with no quote is REMOVED from the page (mutates ``resolved`` in place) — never a
    headline over an empty grid, never placeholders. Run before
    :func:`assign_section_ids`, so ``#testimonials`` links are then dropped/rewritten
    by :func:`enforce_links` like any link to an absent section.
    """
    norm_brief = _norm_quote(brief)
    brief_words = _words(brief)
    kept_blocks: list[tuple[BlockManifest, dict]] = []
    for manifest, content in resolved:
        if manifest.key != "testimonials":
            kept_blocks.append((manifest, content))
            continue
        items = [i for i in content.get("items") or [] if isinstance(i, dict)]
        kept = [i for i in items if _testimonial_grounded(i, norm_brief)]
        report.testimonials_dropped += len(items) - len(kept)
        for item in kept:
            for key in ("author", "role"):
                value = str(item.get(key) or "").strip()
                if value and not _words(value) <= brief_words:
                    item[key] = ""
        if not kept:
            report.testimonial_blocks_removed += 1
            continue
        content["items"] = kept
        kept_blocks.append((manifest, content))
    resolved[:] = kept_blocks


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


def _words(*texts: object) -> set[str]:
    words: set[str] = set()
    for text in texts:
        for w in re.split(r"[^a-z0-9]+", str(text or "").lower()):
            if w:
                words.add(w)
    return words


@dataclass
class _LinkContext:
    ids_by_key: dict[str, str]
    all_ids: frozenset[str]
    contact: dict[str, str]


def _is_external(href: str) -> bool:
    return href.lower().startswith(("http://", "https://", "mailto:", "tel:"))


def _match_section(words: set[str], ctx: _LinkContext, own_key: str) -> str | None:
    for key, synonyms in _LINK_SYNONYMS:
        if key == own_key or key not in ctx.ids_by_key:
            continue
        if words & set(synonyms):
            return f"#{ctx.ids_by_key[key]}"
    return None


def _match_contact(words: set[str], contact: dict[str, str]) -> str | None:
    """A "Contact"/"Call"/"Email" link becomes the brief's real mailto:/tel:."""
    if not words & _CONTACT_WORDS:
        return None
    if words & _CALL_WORDS and contact.get("phone"):
        return f"tel:{contact['phone']}"
    if contact.get("email"):
        return f"mailto:{contact['email']}"
    if contact.get("phone"):
        return f"tel:{contact['phone']}"
    return None


def _button_fallback(ctx: _LinkContext, own_key: str) -> str | None:
    """A button must land somewhere real: the closing CTA band, else contact, else top."""
    for key in ("cta", "footer", "navbar", "hero"):
        if key != own_key and key in ctx.ids_by_key:
            return f"#{ctx.ids_by_key[key]}"
    if ctx.contact.get("email"):
        return f"mailto:{ctx.contact['email']}"
    return None


def _resolve(
    href: object, label: object, ctx: _LinkContext, own_key: str, *, button: bool
) -> str | None:
    """Return a live href for this link, or None when no real target exists."""
    raw = str(href or "").strip()
    if _is_external(raw) or (raw.startswith("#") and raw[1:] in ctx.all_ids):
        return raw
    target = raw.lstrip("#/.").rsplit(".", 1)[0] if raw else ""
    if target in ctx.all_ids:
        return f"#{target}"
    words = _words(label, target)
    found = _match_contact(words, ctx.contact) or _match_section(words, ctx, own_key)
    if found or not button:
        return found
    return _button_fallback(ctx, own_key)


# (href slot, label slot) pairs per block that are rendered as buttons.
_BUTTON_SLOTS: dict[str, tuple[tuple[str, str], ...]] = {
    "navbar": (("cta_href", "cta_label"),),
    "hero": (("primary_href", "primary_label"), ("secondary_href", "secondary_label")),
    "cta": (("primary_href", "primary_label"),),
}


def _enforce_buttons(
    manifest: BlockManifest, content: dict, ctx: _LinkContext, report: IntegrityReport
) -> None:
    for href_slot, label_slot in _BUTTON_SLOTS.get(manifest.key, ()):
        if href_slot not in manifest.slots:
            continue
        before = str(content.get(href_slot) or "")
        label = content.get(label_slot)
        if href_slot == "secondary_href" and not label:
            continue  # the secondary button is not rendered without a label
        resolved_href = _resolve(before, label, ctx, manifest.key, button=True)
        if resolved_href is None:
            if href_slot == "secondary_href":
                content[label_slot] = ""  # hide the button rather than ship it dead
                report.links_removed += 1
            continue
        if resolved_href != before:
            report.links_rewritten += 1
        content[href_slot] = resolved_href


def _enforce_list_links(
    manifest: BlockManifest, content: dict, ctx: _LinkContext, report: IntegrityReport
) -> None:
    for name, spec in manifest.slots.items():
        link_keys = [k for k in ("href", "cta_href") if k in spec.item_slots]
        if spec.type != "list" or not link_keys:
            continue
        href_key = link_keys[0]
        is_button = href_key == "cta_href"  # pricing tier CTA: must land somewhere
        label_key = "cta_label" if is_button else "label"
        kept: list[dict] = []
        for item in content.get(name) or []:
            before = str(item.get(href_key) or "")
            label = item.get(label_key) or (item.get("name") if is_button else None)
            resolved_href = _resolve(before, label, ctx, manifest.key, button=is_button)
            if resolved_href is None:
                report.links_removed += 1  # a nav/footer link to nothing is removed
                continue
            if resolved_href != before:
                report.links_rewritten += 1
            kept.append({**item, href_key: resolved_href})
        content[name] = kept


def enforce_links(
    resolved: list[tuple[BlockManifest, dict]],
    section_ids: list[str],
    contact: dict[str, str],
    report: IntegrityReport,
) -> None:
    """Rewrite every href slot to a real target; drop list links that have none."""
    ids_by_key: dict[str, str] = {}
    for (manifest, _), sid in zip(resolved, section_ids, strict=True):
        ids_by_key.setdefault(manifest.key, sid)
    ctx = _LinkContext(ids_by_key=ids_by_key, all_ids=frozenset(section_ids), contact=contact)
    for manifest, content in resolved:
        _enforce_buttons(manifest, content, ctx, report)
        _enforce_list_links(manifest, content, ctx, report)
