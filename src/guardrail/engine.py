"""Orchestration for the intentionally weak starter guardrail."""

from __future__ import annotations

from collections.abc import Sequence

from common import GuardrailDecision, GuardrailRequest
from guardrail.classifier import classify
from guardrail.detectors import Detector, OrderedKeywordDetector, Signal
from guardrail.normalization import normalize_text
from guardrail.policy import StarterPolicy
from guardrail.vector_detector import create_starter_prototype_detector


class StarterGuardrail:
    """Normalize, flatten, detect, and fuse a request."""

    def __init__(
        self,
        detectors: Sequence[Detector] | None = None,
        policy: StarterPolicy | None = None,
    ) -> None:
        self._use_contextual_classifier = detectors is None and policy is None
        self._detectors = (
            tuple(detectors)
            if detectors is not None
            else (
                OrderedKeywordDetector(),
                create_starter_prototype_detector(),
            )
        )
        self._policy = policy or StarterPolicy()

    def check(self, request: GuardrailRequest) -> GuardrailDecision:
        if self._use_contextual_classifier:
            return classify(request)

        views = [normalize_text(request.message)]
        views.extend(normalize_text(evidence.text) for evidence in request.evidence)
        flattened = " ".join(view.control_stripped for view in views)

        signals: list[Signal] = []
        for detector in self._detectors:
            signal = detector.detect(flattened)
            if signal is not None:
                signals.append(signal)

        # Deliberately preserve the starter authorization gap by not comparing
        # requested_operation with allowed_operations.
        return self._policy.decide(signals, request.context.route)
