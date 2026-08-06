"""
Output guard — streaming blocklist filter for the RAG answer stream.

Applies a multilingual blocked-term list (Shutterstock LDNOOBW) to the answer in transit,
using a sliding-window buffer so a banned term split across streamed chunks is still caught
and never reaches the chatui client.

Mechanism (continuously growth buffer):
  1. Hold-back: keep the last ``max_len - 1`` chars un-flushed each chunk, where
     ``max_len`` is the longest blocklist term. Guarantees no term can complete
     inside already-flushed text without the next chunk arriving. Protects
     multi-word phrases (which contain internal separators).
  2. Word-boundary cut: within the flushable region, release only up to the last
     whitespace/punctuation boundary, so a single token is never split
     mid-word.
  3. On hit: stop, drop the buffer, emit a fixed notice once (latched).
  4. On stream end: final full-buffer scan, then flush the remainder.

Arabic: scanning is done on a *normalized copy* of the buffer (diacritics,
tatweel and zero-width marks stripped; alef/ya/teh-marbuta variants folded) while
the **original** text is what gets flushed. Because a hit drops the *entire*
buffer (we never partially redact), match offsets in the normalized copy never
need to be mapped back to the original — so no index map is required.
"""


import re
import logging
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from components.utils import load_instance_yaml

logger = logging.getLogger(__name__)


# --- separators used for the word-boundary flush cut -------------------------
# Includes ASCII + Arabic + common CJK/full-width punctuation so non-Latin answers
# still flush at clause boundaries rather than buffering a whole sentence.
_PUNCT = set(".,!?;:()[]{}\"'«»…،؛؟-—–/\\|@#*~`"
             "。、！？，；：「」『』（）【】〜・　")

# Matching strategy is chosen PER TERM by script content (not by the `[lang]`
# header, which is only organizational — lists occasionally mix scripts, e.g.
# ASCII entries like "sm"/"xx" inside the Japanese list). A term containing any
# character from a non-space-delimited script (CJK ideographs, kana, Thai) is
# matched as a SUBSTRING; every other term (Latin, Cyrillic, Arabic, Devanagari,
# Hangul, …) is matched WHOLE-WORD. This prevents short ASCII tokens in a CJK
# list from substring-matching ordinary English text.
def _has_substring_script(text: str) -> bool:
    """True if `text` contains a character from a script without word delimiters
    (→ substring matching). Hangul is excluded (Korean uses spaces)."""
    for ch in text:
        o = ord(ch)
        if (0x3040 <= o <= 0x30FF or    # Hiragana + Katakana
                0xFF66 <= o <= 0xFF9D or  # halfwidth Katakana
                0x4E00 <= o <= 0x9FFF or  # CJK Unified Ideographs
                0x3400 <= o <= 0x4DBF or  # CJK Extension A
                0xF900 <= o <= 0xFAFF or  # CJK Compatibility Ideographs
                0x0E00 <= o <= 0x0E7F):   # Thai
            return True
    return False


def _is_sep(ch: str) -> bool:
    """True if `ch` is a word separator (whitespace or punctuation)."""
    return ch.isspace() or ch in _PUNCT


# Hard ceiling on an unbroken (no-separator) run before _flush_safe force-cuts
# it anyway, well above any real blocklist term's max_len. Bounds self.buf (and
# therefore normalize_arabic() cost) even for adversarial separator-free input
# (e.g. a huge URL/base64 blob) that would otherwise never hit a word boundary.
_MAX_UNBROKEN_RUN = 512


# --- Arabic normalization ----------------------------------------------------
# Harakat/tanwin (U+064B–U+0652), superscript alef (U+0670), tatweel (U+0640).
_AR_DIACRITICS = set(chr(c) for c in range(0x064B, 0x0653)) | {"ٰ", "ـ"}
# Zero-width and bidi marks an attacker could splice into a word.
_ZERO_WIDTH = {"​", "‌", "‍", "‎", "‏", "﻿"}
# Letter-form folding so colloquial spelling matches the canonical blocklist term.
_AR_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
})


def normalize_arabic(s: str) -> str:
    """
    Normalize Arabic text for blocklist matching (NOT for display).

    Strips diacritics/tatweel/zero-width marks, folds alef/ya/waw/teh-marbuta
    variants, and lowercases (harmless for Arabic, helps mixed Latin). The result
    is only ever used for *scanning*; the original text is what we flush.
    """
    s = unicodedata.normalize("NFC", s)
    s = "".join(ch for ch in s if ch not in _AR_DIACRITICS and ch not in _ZERO_WIDTH)
    return s.translate(_AR_FOLD).lower()


@dataclass
class CompiledBlocklist:
    """Pre-compiled blocklist regexes plus the hold-back size."""
    boundary_re: Optional[re.Pattern]   # whole-word match on the Arabic-normalized buffer
    substring_re: Optional[re.Pattern]  # substring match on the raw buffer (CJK/Thai)
    max_len: int  # longest term length → drives the (max_len - 1) hold-back


def _boundary_pattern(terms: List[str]) -> Optional[re.Pattern]:
    """
    Compile an alternation of whole-word matches for `terms`.

    Uses explicit non-word lookarounds (`(?<!\\w) … (?!\\w)`) rather than `\\b`:
    Arabic/Cyrillic/Devanagari letters are word chars, so the lookarounds give
    clean separator-based boundaries while `\\b` would fire on script transitions.
    Case-insensitive, Unicode-aware.
    """
    cleaned = [re.escape(t) for t in terms if t]
    if not cleaned:
        return None
    body = "|".join(cleaned)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE | re.UNICODE)


def _substring_pattern(terms: List[str]) -> Optional[re.Pattern]:
    """
    Compile an alternation of plain substring matches (no word boundaries).

    For scripts that don't delimit words with spaces (Chinese, Japanese, Thai),
    where whole-word boundaries would never fire.
    """
    cleaned = [re.escape(t) for t in terms if t]
    if not cleaned:
        return None
    return re.compile("|".join(cleaned), re.IGNORECASE | re.UNICODE)


def load_blocklist(path: str) -> Dict[str, List[str]]:
    """
    Parse a sectioned blocklist file into {lang: [terms]}.

    Format: `[<lang>]` section headers, one term per line, `#` comments and blank
    lines ignored. Terms before any header default to the `en` section.
    """
    sections: Dict[str, List[str]] = {"en": []}
    current = "en"
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1].strip().lower()
                sections.setdefault(name, [])
                current = name
                continue
            sections[current].append(line)
    return sections


def compile_blocklist(terms_by_lang: Dict[str, List[str]]) -> CompiledBlocklist:
    """
    Compile loaded terms into a CompiledBlocklist, bucketing each term by
    matching strategy by script content (see `_has_substring_script`).

    Two compiled regexes:
      - `boundary_re`: whole-word match, evaluated against the Arabic-normalized
        buffer. Terms are normalized too, so Arabic-script terms fold to their
        canonical form; normalization is harmless for Latin/Cyrillic/etc. (it just
        lowercases, which the IGNORECASE flag already handles).
      - `substring_re`: plain substring match on the raw buffer, for non-delimited
        scripts (zh/ja/th).

    `max_len` (longest term, measured on the normalized form for boundary terms)
    sizes the streaming hold-back.
    """
    boundary_terms: List[str] = []
    substring_terms: List[str] = []
    for terms in terms_by_lang.values():
        for t in terms:
            if not t:
                continue
            if _has_substring_script(t):
                substring_terms.append(t)
            else:
                # whole-word match on the normalized buffer; normalize folds
                # Arabic-script terms and is harmless (lowercasing) for the rest.
                boundary_terms.append(normalize_arabic(t))

    boundary_terms = sorted({t for t in boundary_terms if t})
    substring_terms = sorted({t for t in substring_terms if t})

    max_len = max((len(t) for t in boundary_terms + substring_terms), default=1)

    return CompiledBlocklist(
        boundary_re=_boundary_pattern(boundary_terms),
        substring_re=_substring_pattern(substring_terms),
        max_len=max_len,
    )


def load_compiled_blocklist(path: str) -> CompiledBlocklist:
    """
    Convenience: load the base list from `path`, merge in
    INSTANCE_CONFIG_DIR/instance.yaml's `blocklist` key (additions only, per language —
    layered on top of the shipped list, never removes from it), then compile. Used at
    startup.
    """
    terms_by_lang = load_blocklist(path)
    instance_additions = load_instance_yaml().get("blocklist", {})
    for lang, terms in instance_additions.items():
        if not isinstance(terms, list):
            raise ValueError(
                f"instance.yaml 'blocklist.{lang}' must be a list of terms, got {type(terms).__name__}"
            )
        terms_by_lang[lang] = terms_by_lang.get(lang, []) + terms

    compiled = compile_blocklist(terms_by_lang)
    logger.info(
        f"Output guard blocklist compiled from {path} ({len(terms_by_lang)} languages"
        f"{', with instance additions' if instance_additions else ''}, max_len={compiled.max_len})"
    )
    return compiled


class StreamingBlocklistFilter:
    """
    Stateful per-request streaming blocklist filter.

    Construct one per request from a shared `CompiledBlocklist`. Drive it with
    `feed(chunk)` for each streamed text chunk, then `flush_final()` once at the
    end. Both return `(text_to_emit, hit)`: emit `text_to_emit` to the client; if
    `hit` is True, stop streaming the answer (the buffer is dropped and `notice`
    is returned exactly once).
    """

    def __init__(self, compiled: CompiledBlocklist, notice: str):
        self.c = compiled
        self.notice = notice
        self.buf = ""
        self.blocked = False

    @staticmethod
    def _match(pattern: Optional[re.Pattern], text: str, at_end: bool) -> bool:
        """
        True if `pattern` matches a whole word in `text`.

        Mid-stream (`at_end=False`) a match whose right edge is the end of the
        buffer is treated as TENTATIVE and ignored: the next chunk may extend the
        word (e.g. "anal" → "analysis", "cock" → "cocktail"), and the regex's
        `(?!\\w)` lookahead is otherwise satisfied by end-of-string and would
        false-positive. Such a match sits inside the hold-back tail, so it is
        never flushed; it is reconfirmed once a following char arrives, or at
        stream end (`at_end=True`) where end-of-buffer is a real word boundary.
        """
        if pattern is None:
            return False
        n = len(text)
        for m in pattern.finditer(text):
            if at_end or m.end() < n:
                return True
        return False

    def _scan(self, text: str, at_end: bool) -> bool:
        """
        True if any blocked term occurs in `text`.

        Boundary terms are matched against the normalized buffer with the tentative
        end-of-buffer rule. Substring terms are matched on the raw buffer and are
        always definitive (no right-boundary requirement → no tentative case)."""
        if self.c.boundary_re is not None and self._match(self.c.boundary_re, normalize_arabic(text), at_end):
            return True
        if self.c.substring_re is not None and self.c.substring_re.search(text):
            return True
        return False

    def _flush_safe(self) -> str:
        """
        Release the safe prefix: up to the last word boundary that lies before
        the (max_len - 1) hold-back point. Never cuts mid-token — except when a
        single unbroken run (no separators) exceeds _MAX_UNBROKEN_RUN, where we
        force-cut at the hold-back point anyway to bound buffer growth on
        adversarial separator-free input. Correctness (no missed cross-chunk
        term) only depends on the cut point being <= limit; the word-boundary
        search is a cosmetic refinement on top of that, not a requirement.
        """
        hold = self.c.max_len - 1
        limit = len(self.buf) - hold
        if limit <= 0:
            return ""
        cut = -1
        for i in range(limit - 1, -1, -1):
            if _is_sep(self.buf[i]):
                cut = i
                break
        if cut == -1:
            if limit > _MAX_UNBROKEN_RUN:
                # Force-cut retains max_len chars (one more than the normal `hold`),
                # not just `hold`. A max_len-length term whose match lands exactly at
                # the current buffer end is "tentative" (not yet confirmed — see
                # _match) and needs to survive one more call to be reconfirmed; `hold`
                # alone is one char short of that guarantee, so the force-cut path
                # needs the extra margin that the separator-based cut gets for free
                # (it never flushes past a boundary at all when cut == -1).
                p = len(self.buf) - self.c.max_len
                if p > 0:
                    emit, self.buf = self.buf[:p], self.buf[p:]
                    return emit
            return ""  # no boundary in the flushable region yet — keep buffering
        p = cut + 1
        emit, self.buf = self.buf[:p], self.buf[p:]
        return emit

    def feed(self, chunk: str) -> Tuple[str, bool]:
        """
        Append `chunk`, scan the whole buffer, and either block or flush the
        safe prefix.
        """
        if self.blocked:
            return "", True
        self.buf += chunk
        if self._scan(self.buf, at_end=False):
            self.blocked = True
            self.buf = ""
            return self.notice, True
        return self._flush_safe(), False

    def flush_final(self) -> Tuple[str, bool]:
        """End of stream: final full scan, then flush whatever remains."""
        if self.blocked:
            return "", True
        if self._scan(self.buf, at_end=True):
            self.blocked = True
            self.buf = ""
            return self.notice, True
        emit, self.buf = self.buf, ""
        return emit, False
