"""The verdict table: the one place a ``ReasonCode`` is chosen.

Quotedness is not detected; it is the ``origin`` column. ``INJECTION`` in a
message is ``PROMPT_OVERRIDE``, and the same text in evidence is
``QUOTED_INJECTION`` -- so detectors run identically on every field and this
table decides what a match *means*.

**An absent cell emits nothing.** No separate reachability rule is needed: a
``(concept, origin)`` pair with no entry is a hole, and a hole falls through to
ALLOW. That is how ``FRAUD`` authored in a message resolves -- no BLOCK reason
code covers soliciting fraud, so it cannot be a test case, so firing on it
would be pure false-positive risk against the utility families.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping

from common import Action, ReasonCode, Route
from guardrail.concepts import Concept, Origin


#: ``(concept, origin)`` -> the family's action and reason code.
#:
#: Read the two columns side by side: the same concept is an attack when the
#: requester authored it and inert material when they merely supplied it.
VERDICT: Final[
    Mapping[tuple[Concept, Origin], tuple[Action, ReasonCode]]
] = MappingProxyType(
    {
        # Authored in the message -- the requester is asking for it.
        (Concept.INJECTION, Origin.MESSAGE): (
            Action.BLOCK,
            ReasonCode.PROMPT_OVERRIDE,
        ),
        (Concept.EVASION, Origin.MESSAGE): (
            Action.BLOCK,
            ReasonCode.MODERATION_EVASION,
        ),
        (Concept.HOSTILITY, Origin.MESSAGE): (
            Action.BLOCK,
            ReasonCode.GENERATE_ABUSE,
        ),
        (Concept.PRIVACY, Origin.MESSAGE): (
            Action.BLOCK,
            ReasonCode.PRIVATE_DATA_REQUEST,
        ),
        # Peril reaches this table only through the REPORT act, which requires
        # immediacy as well; bare peril is a past-tense report, not an
        # emergency, and never becomes an observation at all.
        (Concept.PERIL, Origin.MESSAGE): (
            Action.ESCALATE,
            ReasonCode.IMMINENT_SAFETY_RISK,
        ),
        # Supplied as evidence -- forwarded material, to be handled as content
        # rather than as an instruction.
        (Concept.INJECTION, Origin.EVIDENCE): (
            Action.ALLOW_AS_DATA,
            ReasonCode.QUOTED_INJECTION,
        ),
        # Evasion joins injection: both steer the system rather than harm the
        # person quoted at.
        (Concept.EVASION, Origin.EVIDENCE): (
            Action.ALLOW_AS_DATA,
            ReasonCode.QUOTED_INJECTION,
        ),
        (Concept.HOSTILITY, Origin.EVIDENCE): (
            Action.ALLOW_AS_DATA,
            ReasonCode.QUOTED_ABUSE,
        ),
        # Soliciting credentials or money from a person is what a scam
        # specimen looks like, so privacy and fraud share a quoted family.
        (Concept.PRIVACY, Origin.EVIDENCE): (
            Action.ALLOW_AS_DATA,
            ReasonCode.QUOTED_SCAM,
        ),
        (Concept.FRAUD, Origin.EVIDENCE): (
            Action.ALLOW_AS_DATA,
            ReasonCode.QUOTED_SCAM,
        ),
        (Concept.ADVERSE_UNKNOWN, Origin.EVIDENCE): (
            Action.ALLOW_AS_DATA,
            ReasonCode.QUOTED_ABUSE,
        ),
        # A fact about the request, not a claim about its text.
        (Concept.GRANT_VIOLATION, Origin.CONTEXT): (
            Action.BLOCK,
            ReasonCode.UNAUTHORIZED_ACTION,
        ),
    }
)

# Holes, stated so a reader does not have to diff the table against the enum:
#
#   (FRAUD, MESSAGE)            no BLOCK family covers soliciting fraud
#   (ADVERSE_UNKNOWN, MESSAGE)  the floor is an evidence-side device only
#   (PERIL, EVIDENCE)           a forwarded report, not an emergency in progress
#
# Each falls through to ALLOW, which is the intended reading.


#: How an attack reads once it turns out to be quoted rather than authored.
#:
#: Only the legacy ``Signal`` path needs this -- an ``Observation`` already
#: resolves to the right family via ``VERDICT``, because origin was part of the
#: lookup. Identity entries let the mapping be applied unconditionally.
QUOTED_OF: Final[Mapping[ReasonCode, ReasonCode]] = MappingProxyType(
    {
        ReasonCode.GENERATE_ABUSE: ReasonCode.QUOTED_ABUSE,
        ReasonCode.PROMPT_OVERRIDE: ReasonCode.QUOTED_INJECTION,
        ReasonCode.MODERATION_EVASION: ReasonCode.QUOTED_INJECTION,
        ReasonCode.PRIVATE_DATA_REQUEST: ReasonCode.QUOTED_SCAM,
        ReasonCode.QUOTED_ABUSE: ReasonCode.QUOTED_ABUSE,
        ReasonCode.QUOTED_SCAM: ReasonCode.QUOTED_SCAM,
        ReasonCode.QUOTED_INJECTION: ReasonCode.QUOTED_INJECTION,
    }
)


#: The four ALLOW families are textually near-indistinguishable, so the route
#: -- which the caller vouched for -- carries the reason. This touches only
#: reason precision and can never change an action.
ROUTE_ALLOW_REASONS: Final[Mapping[Route, ReasonCode]] = MappingProxyType(
    {
        Route.APPEAL: ReasonCode.APPEAL_DECISION,
        Route.ACCOUNT_SAFETY: ReasonCode.ACCOUNT_SAFETY,
        Route.REPORT: ReasonCode.POLICY_QUESTION,
        Route.GENERAL: ReasonCode.ORDINARY_SUPPORT,
    }
)


def resolve(
    concept: Concept, origin: Origin
) -> tuple[Action, ReasonCode] | None:
    """Return the family this observation names, or ``None`` for a hole."""

    return VERDICT.get((concept, origin))


def allow_reason(route: Route, asked_about: bool = False) -> ReasonCode:
    """Return the ALLOW reason for a request that produced no verdict.

    The route is the default and stays the default: it is the field the caller
    vouched for, and the four ALLOW families are textually near-identical
    without it.

    The act refines it only where it actually fired. A request on the general
    route that opens by asking what the rules are is a policy question, not
    ordinary support -- the act knows something the route does not, because
    ``general`` is where everything unrouted lands. On the other three routes
    the route is already the better answer, so the act is ignored.

    This can only ever move a reason code: the caller reached T4, so the action
    is ALLOW either way.
    """

    if asked_about and Route(route) is Route.GENERAL:
        return ReasonCode.POLICY_QUESTION
    return ROUTE_ALLOW_REASONS[Route(route)]
