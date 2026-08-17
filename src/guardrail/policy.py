"""Deterministic signal fusion for the starter guardrail."""

from __future__ import annotations

from collections.abc import Iterable

from common import Action, GuardrailDecision, ReasonCode, Route
from guardrail.detectors import Signal


POLICY_VERSION = "starter-v1"

ROUTE_ALLOW_REASONS = {
    Route.APPEAL: ReasonCode.APPEAL_DECISION,
    Route.ACCOUNT_SAFETY: ReasonCode.ACCOUNT_SAFETY,
    Route.REPORT: ReasonCode.POLICY_QUESTION,
    Route.GENERAL: ReasonCode.ORDINARY_SUPPORT,
}


class StarterPolicy:
    """Choose the highest-priority signal, otherwise allow by route."""

    def __init__(self, policy_version: str = POLICY_VERSION) -> None:
        self.policy_version = policy_version

    def decide(
        self, signals: Iterable[Signal], route: Route
    ) -> GuardrailDecision:
        ordered_signals = tuple(signals)
        for action in (Action.ESCALATE, Action.BLOCK, Action.ALLOW_AS_DATA):
            signal = _first_with_action(ordered_signals, action)
            if signal is not None:
                return GuardrailDecision(
                    action=signal.action,
                    reason_code=signal.reason_code,
                    policy_version=self.policy_version,
                )

        signal = _first_with_action(ordered_signals, Action.ALLOW)
        if signal is not None:
            return GuardrailDecision(
                action=signal.action,
                reason_code=signal.reason_code,
                policy_version=self.policy_version,
            )

        return GuardrailDecision(
            action=Action.ALLOW,
            reason_code=ROUTE_ALLOW_REASONS[Route(route)],
            policy_version=self.policy_version,
        )


def _first_with_action(
    signals: Iterable[Signal],
    action: Action,
) -> Signal | None:
    return next((signal for signal in signals if signal.action is action), None)


def fuse_signals(
    signals: Iterable[Signal],
    route: Route,
    *,
    policy_version: str = POLICY_VERSION,
) -> GuardrailDecision:
    """Functional entry point for deterministic starter-policy fusion."""

    return StarterPolicy(policy_version).decide(signals, route)
