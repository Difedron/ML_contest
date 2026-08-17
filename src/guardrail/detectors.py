"""Ordered keyword signals for the intentionally weak starter policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from re import escape, search
from typing import Protocol, Sequence, TypeAlias, runtime_checkable

from common import Action, ReasonCode
from guardrail.concepts import Observation, Origin
from guardrail.normalization import normalize_text


@dataclass(frozen=True, slots=True)
class Signal:
    """A detector's proposed action and reason."""

    action: Action
    reason_code: ReasonCode
    origin: Origin = field(default=Origin.MESSAGE, compare=False)
    confidence: float = field(default=1.0, compare=False)


Finding: TypeAlias = Signal | Observation


@runtime_checkable
class Detector(Protocol):
    """Structural interface implemented by starter signal detectors."""

    def detect(self, text: str) -> Signal | None:
        """Return one signal for a flattened input, or no signal."""


@dataclass(frozen=True, slots=True)
class KeywordRule:
    """One ordered policy concept and its literal trigger terms."""

    action: Action
    reason_code: ReasonCode
    keywords: tuple[str, ...]


DEFAULT_KEYWORD_RULES = (
    KeywordRule(
        Action.BLOCK,
        ReasonCode.PROMPT_OVERRIDE,
        (
            "ignore all prior",
            "ignore prior",
            "ignore previous",
            "ignore policy",
            "ignore the system",
            "disregard",
            "override instructions",
            "override the instructions",
            "system prompt",
            "developer message",
            "hidden instructions",
            "secret instructions",
            "reveal the prompt",
            "show the prompt",
            "show system prompt",
            "leak the prompt",
        ),
    ),
    KeywordRule(
        Action.BLOCK,
        ReasonCode.MODERATION_EVASION,
        (
            "bypass moderation",
            "bypass checks",
            "bypass the checks",
            "bypass the filter",
            "bypass safety",
            "evade moderation",
            "evade the safety",
            "evade safety checks",
            "avoid the filter",
            "avoid moderation",
            "get around moderation",
            "get past moderation",
            "slip past moderation",
            "banned material",
        ),
    ),
    KeywordRule(
        Action.BLOCK,
        ReasonCode.GENERATE_ABUSE,
        (
            "threaten",
            "intimidate",
            "kill",
            "bully",
            "insult them",
            "insult him",
            "insult her",
            "insult my",
            "write an insult",
            "write insults",
            "draft insults",
            "humiliate",
            "abusive message",
            "hostile message",
            "make them afraid",
            "scare them",
            "force them",
            "pressure them",
        ),
    ),
    KeywordRule(
        Action.BLOCK,
        ReasonCode.PRIVATE_DATA_REQUEST,
        (
            "reporter identity",
            "reporter details",
            "who reported",
            "who filed",
            "private data",
            "private information",
            "home address",
            "personal address",
            "phone number",
            "email address",
            "contact details",
            "contact information",
            "real name",
            "ip address",
            "dox",
            "doxx",
        ),
    ),
)


class OrderedKeywordDetector:
    """Return the first matching concept according to configured rule order."""

    def __init__(
        self, rules: Sequence[KeywordRule] | None = None
    ) -> None:
        configured = tuple(rules) if rules is not None else DEFAULT_KEYWORD_RULES
        self._rules = tuple(
            KeywordRule(
                rule.action,
                rule.reason_code,
                tuple(
                    normalize_text(keyword).control_stripped
                    for keyword in rule.keywords
                ),
            )
            for rule in configured
        )

    def detect(self, text: str) -> Signal | None:
        flattened = normalize_text(text).control_stripped
        for rule in self._rules:
            if any(
                _contains_keyword(flattened, keyword)
                for keyword in rule.keywords
            ):
                return Signal(rule.action, rule.reason_code)
        return None


def _contains_keyword(text: str, keyword: str) -> bool:
    return search(rf"(?<!\w){escape(keyword)}(?!\w)", text) is not None
