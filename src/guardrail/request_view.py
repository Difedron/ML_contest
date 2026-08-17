"""Per-field normalized views of a request.

The starter engine joins the message and every evidence item into one string
before any detector runs. That is cheap, but it discards the only fact several
taxonomy families depend on: whether the requester *authored* a piece of text
or merely *supplied* it. ``RequestView`` keeps the fields apart and normalizes
each exactly once, so detectors can report provenance alongside their verdict.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from operator import attrgetter
from typing import Final, Protocol, TypeAlias, runtime_checkable

from common import Evidence, EvidenceKind, GuardrailRequest, TrustedContext
from guardrail.concepts import Origin
from guardrail.detectors import Detector, Finding, Signal
from guardrail.normalization import normalize_text


# Per-field ceiling on the text handed to similarity scoring.
#
# The contract allows a 4096-character message plus three 8192-character
# evidence items, but NFKC normalization is not length preserving: U+FDFA
# expands to eighteen characters, so a fully legal request can normalize to
# roughly 446k characters. Cosine scoring over char 3-5-grams then costs about
# 450ms, and a 500-case suite would exhaust the tester's 60s run budget and
# score zero -- while every individual case still finishes inside the 2s
# per-request timeout, so no single-case test can observe the problem.
#
# Cosine scoring over char 3-5-grams then costs about 450ms.
MAX_VECTOR_TEXT_LENGTH: Final = 8_192

# The same ceiling, for the surface the pattern layer reads.
#
# The budget above was once justified by "substring matching stays linear and
# costs roughly a millisecond even at that size". Measured, the pattern layer
# costs **833 ms** on such a request: ~20 wide alternations over 440k
# characters, and Python's ``re`` builds no DFA and does not release the GIL,
# so the tester's concurrency does not recover it either. A suite of padded
# requests would exhaust the 60s run budget and score zero, while every single
# request still finishes inside the 2s per-request timeout -- so no
# single-case test can observe it. That is the same failure mode the vector
# budget exists to prevent, on the layer that was assumed to be cheap.
#
# The cap costs no recall, and that is a property of the contract rather than
# a hope. ``MessageText`` is capped at 4096 characters and ``EvidenceText`` at
# 8192, so this ceiling is **twice the largest field the contract permits**.
# NFKC on ordinary prose is near length-preserving, so any field of genuine
# text arrives whole with the margin untouched. Only a deliberate expansion
# can cross it -- U+FDFA alone expands eighteen-fold -- and what lies past the
# cut is that expansion: deterministic ligature text carrying no attack the
# retained half does not already carry.
MAX_MATCH_TEXT_LENGTH: Final = 16_384

#: Chooses which view of a field a detector reads.
TextSelector: TypeAlias = Callable[["FieldView"], str]

#: The full control-stripped field. The default for substring-style detectors.
MATCH_TEXT: Final[TextSelector] = attrgetter("control_stripped")
#: The length-budgeted field, for detectors whose cost grows with input size.
VECTOR_TEXT: Final[TextSelector] = attrgetter("vector_text")

#: Detects on one already-normalized string. See ``FieldScoped``.
FieldDetect: TypeAlias = Callable[[str], "Signal | None"]


@dataclass(frozen=True, slots=True)
class FieldView:
    """One request field, normalized once and reused by every detector."""

    origin: Origin
    kind: EvidenceKind | None
    normalized: str
    control_stripped: str
    #: All whitespace removed. Recovers "b y p a s s", which control-stripping
    #: cannot: the separators there are real spaces, not invisible characters.
    #: Matched by long stems only -- word boundaries do not exist here.
    dense: str
    has_suspicious_controls: bool
    vector_text: str
    truncated: bool

    @classmethod
    def build(
        cls,
        text: str,
        origin: Origin,
        kind: EvidenceKind | None = None,
        *,
        vector_budget: int = MAX_VECTOR_TEXT_LENGTH,
    ) -> FieldView:
        view = normalize_text(text)
        stripped = view.control_stripped[:MAX_MATCH_TEXT_LENGTH]
        truncated = len(stripped) > vector_budget
        return cls(
            origin=origin,
            kind=kind,
            normalized=view.normalized,
            control_stripped=stripped,
            dense="".join(stripped.split()),
            has_suspicious_controls=view.has_suspicious_controls,
            vector_text=stripped[:vector_budget] if truncated else stripped,
            truncated=truncated,
        )


@dataclass(frozen=True, slots=True)
class RequestView:
    """A request whose message and evidence are normalized and kept apart."""

    message: FieldView
    evidence: tuple[FieldView, ...]
    context: TrustedContext

    @classmethod
    def build(
        cls,
        request: GuardrailRequest,
        *,
        vector_budget: int = MAX_VECTOR_TEXT_LENGTH,
    ) -> RequestView:
        def evidence_view(item: Evidence) -> FieldView:
            return FieldView.build(
                item.text,
                Origin.EVIDENCE,
                item.kind,
                vector_budget=vector_budget,
            )

        return cls(
            message=FieldView.build(
                request.message, Origin.MESSAGE, vector_budget=vector_budget
            ),
            evidence=tuple(evidence_view(item) for item in request.evidence),
            context=request.context,
        )

    @property
    def fields(self) -> tuple[FieldView, ...]:
        """The message followed by every evidence item, in request order."""

        return (self.message, *self.evidence)


@runtime_checkable
class RequestDetector(Protocol):
    """A detector that reads the whole request rather than one flat string."""

    def inspect(self, view: RequestView) -> Sequence[Finding]:
        """Return everything this detector finds, tagged with its origin."""


@dataclass(frozen=True, slots=True)
class FieldScoped:
    """Run a per-string detection function against each field separately.

    Wrapping a callable rather than a ``Detector`` lets the same adapter serve
    two purposes: it adapts any object-shaped detector via ``obj.detect``, and
    it lets built-in detectors expose a variant that skips the re-normalization
    ``detect`` must perform on untrusted input. ``RequestView`` has already
    normalized every field, so repeating that work per detector is pure cost.
    """

    detect: FieldDetect
    select: TextSelector = MATCH_TEXT

    def inspect(self, view: RequestView) -> tuple[Signal, ...]:
        signals = []
        for field_view in view.fields:
            signal = self.detect(self.select(field_view))
            if signal is not None:
                signals.append(replace(signal, origin=field_view.origin))
        return tuple(signals)


def as_request_detector(detector: Detector | RequestDetector) -> RequestDetector:
    """Return ``detector`` unchanged if request-aware, otherwise adapt it.

    Injected detectors predating ``RequestDetector`` keep working untouched;
    they simply see each field in turn instead of one flattened string.
    """

    if isinstance(detector, RequestDetector):
        return detector
    return FieldScoped(detector.detect)
