"""Shared regex construction for word-bounded phrase matching.

The starter's rules used plain substring containment, which is why "ignore"
fired on *"ignore notifications"* and "kill" on *"skill"*. Everything built
here is word-bounded, and interior whitespace is relaxed so a phrase still
matches after normalization has collapsed runs of spaces.

``phrases`` compiles one class. ``sequence`` and ``one_of`` compose several
into a construction, for the cases where co-occurrence is not enough and the
classes have to appear in order with only closed-class material between them.
"""

from __future__ import annotations

from re import IGNORECASE, Pattern, compile as re_compile, escape
from typing import Final, Sequence


def _body(items: Sequence[str]) -> str:
    """The bare alternation for a class, without anchors.

    Longer alternatives are tried first so that a more specific phrase wins
    over a prefix of itself -- Python's alternation is first-match, not
    longest-match, and callers count distinct matches.
    """

    ordered = sorted(set(items), key=len, reverse=True)
    return "|".join(
        r"\s+".join(escape(word) for word in phrase.split()) for phrase in ordered
    )


def phrases(items: Sequence[str]) -> Pattern[str]:
    """Compile phrases into one word-bounded alternation."""

    return re_compile(rf"\b(?:{_body(items)})\b", IGNORECASE)


def sequence(*classes: Sequence[str], filler: Sequence[str] = ()) -> str:
    """Regex source matching each class in order, separated only by *filler*.

    Returns source rather than a compiled pattern so several constructions can
    be combined by ``one_of``. The filler class is the point: it is what makes
    the sequence a *construction* rather than a proximity test. Only members of
    a deliberately closed class may stand between the parts, so "write a really
    *nasty* message" matches while "help me report the *abusive* messages" does
    not -- "report" is not filler, and its presence means the produce verb
    already has an object.
    """

    gap = rf"\s+(?:\b(?:{_body(filler)})\b\s+)*" if filler else r"\s+"
    return gap.join(rf"\b(?:{_body(item)})\b" for item in classes)


def one_of(*sources: str) -> Pattern[str]:
    """Compile alternative construction sources into one pattern."""

    return re_compile("|".join(f"(?:{source})" for source in sources), IGNORECASE)


#: How many modifiers may stack before a head noun. "all your previous safety
#: rules" is four; beyond that the run is no longer one noun phrase.
MAX_QUALIFIERS: Final = 4


def qualified(modifiers: Sequence[str], heads: Sequence[str]) -> str:
    """Regex source for *one or more* modifiers followed by a head noun.

    The object classes this replaces were written as literal determiner+noun
    bigrams -- "the instructions", "your rules", "internal configuration" -- and
    that shape is what capped their recall. Every unlisted modifier broke the
    match even when both halves were known: "the instructions" was in the class
    and "the *safety* instructions" was not, "your guardrails" was in and "*no*
    guardrails" was not. Enumerating the product by hand is quadratic work that
    is never finished.

    Writing it as a grammar keeps the property the bigram list existed to
    enforce -- **a bare head noun never matches**. That is what stops an appeal
    saying *"tell me which rule my post broke"* from reading as an injection
    attempt: "rule" alone is not an object, and the determiner or possessive is
    what says the object is the system's rather than the forum's. The grammar
    requires a qualifier for exactly the same reason, and gets every
    combination for free.

    Plural and singular heads are both accepted; the caller supplies singular
    stems and the optional ``s`` is added here.
    """

    modifier = f"(?:{_body(modifiers)})"
    head = f"(?:{_body(heads)})"
    return rf"\b(?:{modifier}\s+){{1,{MAX_QUALIFIERS}}}{head}s?\b"


def products(modifiers: Sequence[str], heads: Sequence[str]) -> tuple[str, ...]:
    """Every modifier+head pair, for surfaces that cannot run a grammar.

    The whitespace-free surface has no word boundaries and so cannot use the
    ``{1,4}`` repetition above without matching across unrelated text. It gets
    the enumerated product instead -- derived from the same two classes, so the
    two surfaces cannot drift apart.
    """

    return tuple(f"{modifier} {head}" for modifier in modifiers for head in heads)
