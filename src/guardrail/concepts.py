"""The internal vocabulary the pipeline reasons in.

The taxonomy partitions *situations*, not content, and those situations factor
along three axes: what the writer is doing (``Act``), what it is about
(``Concept``), and which field it arrived in (``Origin``). Families that differ
in action always differ on at least one axis, so a detector never has to name a
family -- it reports one axis and lets ``resolver`` do the naming.

Nothing here is ever serialized. ``ReasonCode`` belongs to the wire; ``Concept``
and ``Act`` belong to the reasoning, and keeping them apart is what stops a
detector from quietly deciding an action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Origin(StrEnum):
    """Which part of a request supports an observation.

    Provenance is structural rather than statistical: an observation either
    rests on text the requester authored, on text they merely supplied as
    evidence, or on a trusted context field. Fusion may therefore order origins
    with more confidence than it orders scores.
    """

    MESSAGE = "message"
    EVIDENCE = "evidence"
    CONTEXT = "context"


class Concept(StrEnum):
    """What a field is about, independent of who wrote it.

    Topic alone can never decide an action: ``HOSTILITY`` in a message is a
    demand to produce abuse, while the same topic in evidence is a specimen
    someone forwarded for review. That is why these are concepts rather than
    reason codes -- the pair ``(concept, origin)`` is the smallest thing that
    determines a family.
    """

    INJECTION = "injection"
    EVASION = "evasion"
    HOSTILITY = "hostility"
    PRIVACY = "privacy"
    FRAUD = "fraud"
    PERIL = "peril"

    #: An adverse topic matched but the subtype did not separate. Never means
    #: "evidence exists" -- it still requires a positive match. The floor keeps
    #: a subtype failure from costing an *action*, spending only reason
    #: precision to protect it.
    ADVERSE_UNKNOWN = "adverse_unknown"

    #: An operation outside the caller's grant. Not a topic at all: it is a
    #: fact about the request, established by comparing two vouched-for enums.
    GRANT_VIOLATION = "grant_violation"


class Act(StrEnum):
    """What the writer is doing, where that changes how topics are read.

    Only three acts are worth detecting. A topic in the message is an authored
    attack *by default*, and an act only ever suppresses that default, so
    everything not listed here needs no detector.
    """

    ASK_ANALYZE = "ask_analyze"
    ASK_ABOUT = "ask_about"
    REPORT = "report"


@dataclass(frozen=True, slots=True)
class Observation:
    """One concept found in one field.

    Confidence is a gate and a tiebreak, never a summand: it is compared only
    against observations in the same tier, and never added, averaged, or
    weighed against an observation resting on a better kind of evidence. It is
    excluded from equality for the same reason -- two observations of the same
    concept in the same field are the same finding, however strongly each was
    supported.
    """

    concept: Concept
    origin: Origin = Origin.MESSAGE
    confidence: float = field(default=1.0, compare=False)
