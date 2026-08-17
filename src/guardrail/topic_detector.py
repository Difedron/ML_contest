"""The topic layer, gated by the act frame.

Detection and suppression are kept in one detector because they are one
decision: whether a topic found in the message is an authored attack or
somebody else's material under discussion. Splitting them would leave a window
in which a suppressed topic exists as an observation.
"""

from __future__ import annotations

from typing import Final

from guardrail.acts import SUPPRESSING, detect_acts
from guardrail.concepts import Concept, Observation, Origin
from guardrail.request_view import FieldView, RequestView
from guardrail.topics import adverse_cue_hits, labelled_subtype, observe


#: How much a suspicious control character adds to a topic that already
#: matched. Never enough to matter on its own, because it is never applied on
#: its own: obfuscation corroborates a reading, it does not establish one.
CONTROL_BOOST: Final = 0.1

#: Confidence for the floor. Below every real subtype match, so a field that
#: does separate always outranks one that does not.
FLOOR_CONFIDENCE: Final = 0.5


class TopicDetector:
    """Report topics per field, suppressing message topics under an act."""

    def inspect(self, view: RequestView) -> tuple[Observation, ...]:
        observations: list[Observation] = []

        if not (detect_acts(view) & SUPPRESSING):
            observations.extend(self._field(view.message))

        for field_view in view.evidence:
            found = self._field(field_view)
            observations.extend(found or self._floor(field_view, view))

        return tuple(observations)

    def _field(self, field_view: FieldView) -> tuple[Observation, ...]:
        return observe(
            field_view.control_stripped,
            field_view.dense,
            field_view.origin,
            CONTROL_BOOST if field_view.has_suspicious_controls else 0.0,
        )

    def _floor(
        self, field_view: FieldView, view: RequestView
    ) -> tuple[Observation, ...]:
        """Catch evidence that is plainly adverse but does not name a subtype.

        Reached only when no topic matched the field. Without it, evidence that
        was obviously hostile or fraudulent -- but not confidently *which* --
        emitted nothing and fell through to ALLOW, spending whole utility
        families to protect a fraction of a reason code. The floor keeps the
        action right and lets the subtype be wrong.

        It still demands a positive cue. "Evidence exists" is not one, because
        the case this has to survive is an ALLOW-family request that happens to
        carry a benign attachment.

        When the requester named a category, that names the subtype. The tier
        has already fixed the action, so their label can only move the reason
        code -- and a moderator forwarding material almost always says what
        they think it is, which generalises better than guessing from the
        specimen.
        """

        if not adverse_cue_hits(field_view.control_stripped):
            return ()
        concept = (
            labelled_subtype(view.message.control_stripped)
            or Concept.ADVERSE_UNKNOWN
        )
        return (Observation(concept, Origin.EVIDENCE, FLOOR_CONFIDENCE),)
