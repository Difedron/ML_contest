"""Context-aware deterministic classification for the synthetic policy.

The classifier deliberately keeps trusted context, active user instructions,
and quoted evidence separate.  This prevents hostile text supplied for review
from being treated as an instruction to the guardrail.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
from collections.abc import Iterable
from unicodedata import category, normalize
from urllib.parse import unquote

from common import (
    Action,
    GuardrailDecision,
    GuardrailRequest,
    ReasonCode,
    Route,
)
from guardrail.normalization import normalize_text


POLICY_VERSION = "starter-v1"

_CONFUSABLES = str.maketrans(
    {
        "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h",
        "о": "o", "р": "p", "с": "c", "т": "t", "у": "y", "х": "x",
        "і": "i", "ј": "j", "ѕ": "s",
        "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ο": "o",
        "ρ": "p", "τ": "t", "υ": "y", "χ": "x",
        "0": "o", "1": "i",
        "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s",
    }
)


def _text(value: str) -> str:
    """Return a matching view resilient to common visual obfuscation."""
    normalized = normalize_text(unquote(value)).control_stripped.translate(
        _CONFUSABLES
    )
    normalized = "".join(
        character
        for character in normalize("NFKD", normalized)
        if category(character) != "Mn"
    )
    normalized = re.sub(r"[_./|\\-]+", " ", normalized)
    # Join runs such as ``b y p a s s`` after punctuation/control stripping.
    normalized = re.sub(
        r"(?<!\w)(?:[a-z]\s+){2,}[a-z](?!\w)",
        lambda match: re.sub(r"\s+", "", match.group()),
        normalized,
    )
    normalized = " ".join(normalized.split())

    derived: list[str] = []
    for token in re.findall(r"[a-zA-Z0-9+/_-]{16,}={0,2}", value):
        padded = token.replace("-", "+").replace("_", "/")
        padded += "=" * (-len(padded) % 4)
        try:
            decoded = base64.b64decode(padded, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if (
            decoded
            and sum(character.isprintable() for character in decoded)
            / len(decoded)
            >= 0.9
            and re.search(r"[a-zA-Z]{3}", decoded)
        ):
            derived.append(normalize_text(decoded).control_stripped)

    rot13 = codecs.decode(normalized, "rot_13")
    if re.search(
        r"\b(?:bypass|evade|moderation|filter|safeguard|disallowed|banned)\b",
        rot13,
    ):
        derived.append(rot13)

    reversed_text = normalized[::-1]
    if re.search(
        r"\b(?:bypass|evade|moderation|filter|safeguard|disallowed|banned)\b",
        reversed_text,
    ):
        derived.append(reversed_text)

    return " ".join((normalized, *derived))


def _has(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


_REVIEW_INTENT = (
    r"\b(?:classif|analy[sz]|assess|review|summari[sz]|explain|identify|detect)\w*",
    r"\b(?:quoted?|attached|reported|evidence|excerpt|passage|profile text|content)\b",
    r"\b(?:warning signs?|consistent with|without repeating)\b",
)
_INJECTION = (
    r"\b(?:ignore|disregard|discard|forget|override)\b.{0,50}"
    r"\b(?:previous|prior|system|developer|governing|instructions?|directions?|rules?|commands?)\b",
    r"\b(?:reveal|show|print|expose|disclos|leak|extract)\w*\b.{0,45}"
    r"\b(?:system prompt|hidden prompt|secret|configuration|instructions?)\b",
    r"\bjailbreak\b",
    r"\bact as\b.{0,40}\b(?:unrestricted|unfiltered|developer mode)\b",
)
_EVASION = (
    r"\bb\s*y\s*p\s*a\s*s\s*s\b.{0,55}"
    r"\b(?:moderation|filter|checks?|safety|review)\b",
    r"\be\s*v\s*a\s*d\s*e\b.{0,55}"
    r"\b(?:moderation|filter|checks?|safety|review)\b",
    r"\b(?:bypass|circumvent|evade|defeat|disable|sidestep|work around|get around|"
    r"slip past|beat|dodge|outsmart)\w*\b.{0,55}"
    r"\b(?:moderation|filter|safeguard|safety|check|detection|restriction|rule)\w*\b",
    r"\b(?:avoid|escape|fool|trick)\w*\b.{0,45}"
    r"\b(?:moderation|filter|detection|safety checks?)\b",
    r"\b(?:encode|obfuscat|misspell|space out)\w*\b.{0,50}"
    r"\b(?:banned|blocked|disallowed|moderation|filter)\w*\b",
    r"\b(?:without|not)\b.{0,25}\b(?:trigger|alert|flag|detect|notice)\w*\b"
    r".{0,35}\b(?:moderation|filter|system|safety|review)\b",
    r"\b(?:pass|post|send|publish|spread)\w*\b.{0,35}"
    r"\b(?:undetected|unnoticed|without being flagged)\b",
    r"\b(?:conceal|hide|mask)\w*\b.{0,35}\bfrom\b.{0,25}"
    r"\b(?:moderation|filter|detection|review)\b",
    r"\b(?:rewrite|rephrase|alter|modify|transform|format)\w*\b.{0,65}"
    r"\b(?:filter|moderation|detect|flag|block|catch)\w*\b",
    r"\b(?:filter|moderation|system|review)\b.{0,55}"
    r"\b(?:not|never|won't|cannot)\b.{0,20}\b(?:detect|catch|flag|notice|block)\w*\b",
    r"\b(?:look|appear|seem)\w*\b.{0,20}\b(?:harmless|allowed|benign|acceptable)\b"
    r".{0,45}\b(?:filter|moderation|detection)\b",
    r"\b(?:sneak|smuggle|slip|push|get|put)\w*\b.{0,55}"
    r"\b(?:banned|blocked|prohibited|disallowed|forbidden|restricted)\w*\b"
    r".{0,45}\b(?:past|through|around|under the radar)\b",
    r"\b(?:banned|blocked|prohibited|disallowed|forbidden|restricted)\w*\b"
    r".{0,75}\b(?:past|through|under the radar|accepted|approved|published)\b",
    r"\b(?:moderation|guardrail|automated review|content checks?|safety classifier|"
    r"content classifier)\b.{0,70}\b(?:accept|approve|miss|overlook|permit|"
    r"treat|classif)\w*\b",
    r"\b(?:accept|approve|permit|treat|classif)\w*\b.{0,55}"
    r"\b(?:allowed|safe|benign|permitted|compliant)\b.{0,40}"
    r"\b(?:moderation|guardrail|review|classifier|filter)\b",
    r"\b(?:avoid|prevent)\w*\b.{0,40}\b(?:removal|takedown|rejection|sanction|"
    r"account restriction)\b",
    r"\b(?:payload|material|content|message|post)\b.{0,55}"
    r"\b(?:not be|isn't|won't be|cannot be)\b.{0,20}"
    r"\b(?:caught|detected|flagged|blocked|removed)\b",
    r"\b(?:keep|stop|prevent)\w*\b.{0,35}\bfrom\b.{0,25}"
    r"\b(?:being )?(?:taken down|removed|flagged|blocked|rejected|caught)\b",
    r"\b(?:post|send|publish|share|spread)\w*\b.{0,45}\bwithout\b.{0,25}"
    r"\b(?:getting|being)?\s*(?:caught|detected|flagged|blocked|removed)\b",
    r"\b(?:game|trick|fool|outsmart|beat)\w*\b.{0,25}"
    r"\b(?:the )?(?:system|algorithm|classifier|reviewer|moderator)\b",
    r"\b(?:fly|stay|remain|pass)\w*\b.{0,20}\bunder the radar\b",
)
_PRIVATE = (
    r"\b(?:reporter|complainant|whistleblower)\b.{0,45}"
    r"\b(?:identity|name|details?|contact|who)\b",
    r"\b(?:private|personal|confidential|nonpublic)\b.{0,35}"
    r"\b(?:data|information|details?|address|phone|email|contact|location)\b",
    r"\b(?:home|residential|street)\s+address\b",
    r"\b(?:find|locate|retrieve|get|give|tell|show|reveal|identify|doxx?)\w*\b"
    r".{0,55}\b(?:address|phone number|email|identity|location)\b",
    r"\b(?:doxx?|deanonymi[sz])\w*\b",
)
_ABUSE = (
    r"\b(?:write|draft|compose|create|generate|make|help me|give me)\b.{0,65}"
    r"\b(?:threat|intimidat|harass|humiliat|bully|coerc|terrify|scare|insult|abusive)\w*\b",
    r"\b(?:threaten|intimidate|harass|humiliate|bully|coerce|terrify)\b.{0,55}"
    r"\b(?:him|her|them|someone|neighbor|coworker|target|person|ex|user)\b",
    r"\b(?:message|text|post|language|wording)\b.{0,40}"
    r"\b(?:into silence|afraid|fear|worthless|kill|hurt)\b",
)
_IMMINENT = (
    r"\b(?:right now|just now|immediate(?:ly)?|imminent|at this moment|currently|"
    r"urgent(?:ly)?|about to|very soon|any minute|within minutes?|tonight|already)\b",
    r"\b(?:outside (?:my|the) door|in immediate danger|need urgent help|emergency)\b",
    r"\b(?:on (?:his|her|their|the) way|coming (?:here|over)|cannot stay safe|"
    r"can't stay safe)\b",
    r"\b(?:going to|plans? to|intends? to|ready to)\b",
    r"\b(?:next few|next couple of)\s+(?:minutes?|hours?)\b",
    r"\b(?:is|are|they're|he's|she's)\b.{0,20}"
    r"\b(?:threatening|attacking|following|chasing|breaking in|trying to)\b",
    r"\b(?:has|have|holding|carrying)\b.{0,20}\b(?:weapon|gun|knife)\b",
    r"\b(?:trying|attempting)\b.{0,25}\b(?:break in|enter|hurt|harm|attack|kill)\b",
    r"\b(?:trapped|cornered|locked (?:myself|ourselves) in|hiding)\b",
)
_DANGER = (
    r"\b(?:threaten|attack|hurt|harm|kill|suicid|self harm|danger|weapon|gun|"
    r"knife|bleed|violence|die|dying)\w*\b",
    r"\b(?:not safe|fear for (?:my|their|his|her) life)\b",
    r"\b(?:take (?:my|their|his|her) (?:own )?life|end (?:my|their|his|her) life|"
    r"end it all)\b",
    r"\b(?:overdose|pills?|poison|jump(?:ing)?|goodbye|final message|"
    r"can't go on|cannot go on|wants? to die|doesn't want to live|"
    r"does not want to live)\b",
    r"\b(?:break(?:ing)? in|intruder|stalk(?:er|ing)?|abduct|kidnap|hostage|"
    r"assault|domestic violence|call (?:the )?police)\b",
)

_MODERATION_CONCEPT = (
    r"\b(?:moderat|filter|guardrail|safeguard|screening|automated review|"
    r"automated checks?|automated system|content review|content checks?|"
    r"safety checks?|content classifier|detection system|platform rules?|"
    r"platform enforcement|automated enforcement|spam detection|spam checks?|"
    r"rate limits?|scanner|checker|review system)\w*\b",
)
_EVASION_INTENT = (
    r"\b(?:how|ways?|methods?|tips?|strategy|technique|help|show|teach|make|"
    r"get|craft|disguise|camouflage|sanitize|render|rewrite|rephrase|encode|"
    r"hide|conceal|post|publish|send|share|avoid|prevent|pass|clear|survive|"
    r"slip|sneak)\w*\b",
)
_DISALLOWED_CONTENT = (
    r"\b(?:banned|blocked|prohibited|disallowed|forbidden|restricted|removed|"
    r"takedown|sanctioned)\w*\b",
)
_EVIDENCE_TRANSFORM = (
    r"\b(?:rewrite|rephrase|modify|alter|encode|obfuscat|disguise|camouflage|"
    r"mask|sanitize|convert|transform)\w*\b.{0,70}"
    r"\b(?:attached|quoted|provided|supplied|this|the)\b.{0,25}"
    r"\b(?:text|content|message|post|material|payload|wording)\b",
    r"\b(?:attached|quoted|provided|supplied|this|the)\b.{0,25}"
    r"\b(?:text|content|message|post|material|payload|wording)\b.{0,70}"
    r"\b(?:rewrite|rephrase|modify|alter|encode|obfuscat|disguise|camouflage|"
    r"mask|sanitize|convert|transform)\w*\b",
    r"\b(?:alternative|different|safer|acceptable|compliant)\s+"
    r"(?:wording|version|phrasing|spelling)\b.{0,50}"
    r"\b(?:post|publish|send|share|reuse|pass)\w*\b",
)
_SCAM = (
    r"\b(?:scam|fraudulent|phish|impersonat|deceptive)\w*\b",
    r"\b(?:gift cards?|processing fee|upfront fee|guaranteed prize|wire transfer|"
    r"crypto payment|bank transfer|verification code|one time code)\b",
    r"\b(?:verify your account|claim your prize|limited time offer|send money|"
    r"click (?:this|the) link|investment return)\b",
)
_ABUSIVE_EVIDENCE = (
    r"\b(?:abuse|abusive|insult|hostile|harass|threat|intimidat|bully|"
    r"derogatory|demean|mock)\w*\b",
    r"\b(?:worthless|pathetic|nobody wants you|hate you|kill you|fraud|idiot|"
    r"stupid|loser|moron|disgusting|shut up|piece of)\b",
)

_INJECTION_EVIDENCE = _INJECTION + (
    r"\b(?:follow|obey)\b.{0,35}\b(?:these|my|new)\s+instructions?\b",
    r"\b(?:do not|don't)\b.{0,25}\b(?:follow|obey)\b.{0,25}"
    r"\b(?:original|earlier|developer|system)\b",
    r"\b(?:hidden|internal)\b.{0,25}\b(?:rules?|prompt|instructions?|secrets?)\b",
    r"\b(?:change|alter|control|redirect)\w*\b.{0,35}"
    r"\b(?:assistant|model|bot|agent)\w*\b.{0,20}\b(?:behavior|response|output)\b",
    r"\b(?:new|replacement|higher priority)\b.{0,25}"
    r"\b(?:commands?|directives?|instructions?)\b",
)

_INJECTION_CONTEXT = (
    r"\b(?:instruction|prompt|command|directive|model behavior|assistant behavior|"
    r"agent behavior|override attempt)\w*\b",
)
_SCAM_CONTEXT = (
    r"\b(?:offer|payment|prize|support agent|account verification|investment|"
    r"transaction|seller|buyer|money|fee|link)\b",
)
_ABUSE_CONTEXT = (
    r"\b(?:tone|insult|hostile|abusive|harassment|threat|wording|name calling|"
    r"personal attack)\b",
)


def _decision(action: Action, reason: ReasonCode) -> GuardrailDecision:
    return GuardrailDecision(
        action=action, reason_code=reason, policy_version=POLICY_VERSION
    )


def _quoted_reason(message: str, evidence: str) -> ReasonCode | None:
    if not evidence:
        return None

    # The active review request is the strongest disambiguator.  For example,
    # "fraud" can itself be an insult, while "reported offer" indicates scam.
    if _has(message, (r"\b(?:prompt injection|instruction attack|jailbreak)\b",)):
        return ReasonCode.QUOTED_INJECTION
    if _has(message, _INJECTION_CONTEXT):
        return ReasonCode.QUOTED_INJECTION
    if _has(message, _SCAM) or _has(message, _SCAM_CONTEXT):
        return ReasonCode.QUOTED_SCAM
    if _has(message, _ABUSIVE_EVIDENCE) or _has(message, _ABUSE_CONTEXT):
        return ReasonCode.QUOTED_ABUSE

    if _has(evidence, _INJECTION_EVIDENCE):
        return ReasonCode.QUOTED_INJECTION
    if _has(evidence, _SCAM):
        return ReasonCode.QUOTED_SCAM
    if _has(evidence, _ABUSIVE_EVIDENCE):
        return ReasonCode.QUOTED_ABUSE

    # Evidence on the report route is inert data even when the exact harmful
    # phrase is an unseen paraphrase.  Conservative lexical fallbacks retain
    # the most likely published reason code.
    if _has(message, _REVIEW_INTENT):
        if _has(evidence, (r"\b(?:pay|send|transfer|account|card|cash|money)\w*\b",)):
            return ReasonCode.QUOTED_SCAM
        return ReasonCode.QUOTED_ABUSE
    return None


def classify(request: GuardrailRequest) -> GuardrailDecision:
    """Classify one validated request against the published taxonomy."""
    context = request.context
    if (
        context.requested_operation.value != "none"
        and context.requested_operation not in context.allowed_operations
    ):
        return _decision(Action.BLOCK, ReasonCode.UNAUTHORIZED_ACTION)

    message = _text(request.message)
    evidence = " ".join(_text(item.text) for item in request.evidence)

    if _has(message, _DANGER) and (
        _has(message, _IMMINENT) or context.route is Route.ACCOUNT_SAFETY
    ):
        return _decision(Action.ESCALATE, ReasonCode.IMMINENT_SAFETY_RISK)

    # Questions about the published policy are utility requests, even when
    # they name a prohibited concept.  Enabling questions ("how can I evade")
    # do not match these explanatory forms.
    if (
        context.route is Route.REPORT
        and not request.evidence
        and _has(
            message,
            (
                r"\b(?:what (?:is|does|information)|does (?:the )?(?:policy|rule)|"
                r"is it (?:allowed|against)|why (?:is|does)|explain (?:the )?"
                r"(?:policy|rule)|what should i include)\b",
            ),
        )
        and _has(
            message,
            (r"\b(?:policy|rules?|allowed|prohibited|violation|report)\w*\b",),
        )
    ):
        return _decision(Action.ALLOW, ReasonCode.POLICY_QUESTION)

    if _has(message, _INJECTION):
        return _decision(Action.BLOCK, ReasonCode.PROMPT_OVERRIDE)
    if request.evidence and _has(message, _EVIDENCE_TRANSFORM):
        return _decision(Action.BLOCK, ReasonCode.MODERATION_EVASION)
    if context.route is Route.GENERAL and (
        (
            _has(message, _MODERATION_CONCEPT)
            and _has(message, _EVASION_INTENT)
        )
        or (
            _has(message, _DISALLOWED_CONTENT)
            and _has(
                message,
                (
                    r"\b(?:post|publish|send|share|spread|upload|distribute|"
                    r"phrase|word|encode|transform|alter|make|craft|disguise|"
                    r"camouflage|sanitize|render|get)\w*\b",
                ),
            )
        )
    ):
        return _decision(Action.BLOCK, ReasonCode.MODERATION_EVASION)
    if _has(message, _EVASION):
        return _decision(Action.BLOCK, ReasonCode.MODERATION_EVASION)
    if _has(message, _PRIVATE):
        return _decision(Action.BLOCK, ReasonCode.PRIVATE_DATA_REQUEST)
    if _has(message, _ABUSE):
        return _decision(Action.BLOCK, ReasonCode.GENERATE_ABUSE)

    quoted_reason = _quoted_reason(message, evidence)
    if quoted_reason is not None:
        return _decision(Action.ALLOW_AS_DATA, quoted_reason)

    # In the published trusted context, ordinary general support concerns the
    # caller's own account.  A target-less general request that reaches this
    # point is the semantic fallback for unseen moderation-evasion wording.
    if (
        context.route is Route.GENERAL
        and context.target_relation.value == "none"
    ):
        return _decision(Action.BLOCK, ReasonCode.MODERATION_EVASION)

    route_reasons = {
        Route.APPEAL: ReasonCode.APPEAL_DECISION,
        Route.ACCOUNT_SAFETY: ReasonCode.ACCOUNT_SAFETY,
        Route.REPORT: ReasonCode.POLICY_QUESTION,
        Route.GENERAL: ReasonCode.ORDINARY_SUPPORT,
    }
    return _decision(Action.ALLOW, route_reasons[context.route])
