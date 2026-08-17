"""What a field is about, as constructions rather than as vocabulary.

A topic is not a keyword, and it is not a bag of keywords either. *"How do I
ignore notifications from a group chat?"* contains "ignore"; it is not an
injection attempt. What makes one is the verb applied to **an object denoting
the system's own instructions** -- and, just as load-bearing, the two standing
in a grammatical relation rather than merely sharing a field.

**Co-occurrence is not construction.** Requiring only that both classes appear
somewhere in the field reads *"I need to report the harassment I am receiving
from a coworker"* as a request to generate abuse, and *"my friend forgot her
email address for the account we share"* as a demand for a third party's
contact details. Both sentences contain every part; neither contains the
construction. So a cue holds only when its classes fall inside **one clause**
-- see ``share_clause``. That single constraint is what lets the vocabulary be
aggressive without costing the utility families, and it is why the classes
below can afford to be wide.

The same cues run on every field. ``ORIGIN`` decides what a match means, so a
specimen pasted into evidence is recognised by the same rule that catches an
authored demand -- see ``resolver``.

**The two sides differ in construction, not in strictness.** An earlier design
made the evidence side "permissive" by accepting an object class on its own.
That is not a weaker test of the same thing, it is a test of something else,
and it fired on exactly the material an appeal attaches: a removal notice is
*made of* enforcement nouns, so ``EVASION`` matched every forwarded takedown
and turned appeals into ALLOW_AS_DATA. What evidence needs to show is not "the
writer asked for this" but "this material *is* that" -- which for injection,
evasion and privacy is the very same construction, since a quoted injection is
itself an authored demand. Only ``HOSTILITY`` and ``FRAUD`` are recognised by
register instead, through the ``evidence_only`` cues below: abuse and scam
copy look like nothing else, while enforcement and contact nouns are the
ordinary furniture of forwarded material.

Vocabulary is written from the taxonomy definitions and the threat model. Where
a term had to be covered because a pinned test names it, it enters as a member
of a class with siblings drawn from neither the starter nor the public suite --
a literal there would fit the starter's synthetic prototypes, which is the same
overfitting mistake in a different corpus.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from re import IGNORECASE, Pattern, compile as re_compile
from typing import Final

from guardrail.concepts import Concept, Observation, Origin
from guardrail.patterns import one_of, phrases, products, qualified, sequence


# --------------------------------------------------------------------------
# Clause scope.

#: How many words may sit between two classes before they stop being one
#: construction. Three is the width of an English noun phrase between a verb
#: and its object -- a determiner and up to two modifiers, as in "ignore *all
#: prior* instructions", "find *the private* home address", "write *a message
#: to* intimidate". A fourth word means a second predicate has started, which
#: is exactly the shape of the sentences this constraint exists to spare:
#: "report *the harassment I am receiving from* a coworker".
CLAUSE_WINDOW: Final = 3

#: Punctuation that ends a clause outright. A verb cannot govern an object in
#: the next sentence however few words separate them.
_CLAUSE_BREAK: Final = re_compile(r"[.!?;]")

#: How far back a ``Cue.denied_by`` pattern may read. Wide enough for a
#: possessive and a modifier ("my own account"), narrow enough that it cannot
#: reach into the previous clause.
DENIAL_LOOKBACK: Final = 24


def share_clause(text: str, left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Whether two matched spans stand in one construction.

    Spans are passed in either order; overlapping spans trivially share a
    clause, which falls out of slicing without a special case.
    """

    between = text[min(left[1], right[1]) : max(left[0], right[0])]
    return len(between.split()) <= CLAUSE_WINDOW and not _CLAUSE_BREAK.search(between)


# --------------------------------------------------------------------------
# INJECTION -- an instruction aimed at the system's own instructions.

#: Verbs that **countermand** an instruction, or assert a new regime over it.
#: These are injective whatever they take as an object: there is no benign
#: reading of asking a system to set its own rules aside.
INJECTION_OVERRIDE_WORDS: Final = (
    "ignore", "ignoring", "disregard", "disregarding", "forget", "override",
    "overriding", "bypass", "discard", "drop", "skip", "pretend", "disable",
    "turn off", "switch off", "unlock", "act as", "behave as", "from now on",
    "you must now", "you will now", "you are now", "you are in",
    "new directive", "instead of your", "set aside", "none of your",
    "do not follow", "stop following", "no longer apply", "does not apply",
    "do not apply", "have been replaced", "replace your", "replace the",
    "are void", "is void", "suspend", "lift", "waive",
    "must now", "will now", "shall now", "is now", "are now", "now answer",
)

#: Verbs that merely **ask to see**. These are the vocabulary of every support
#: request -- *"show me the settings"*, *"tell me the policy on harassment"* --
#: so they are injective only against an object that is unambiguously the
#: system's own configuration. See ``INJECTION_SYSTEM_OBJECTS``.
INJECTION_DISCLOSE_WORDS: Final = (
    "reveal", "expose", "print", "output", "show", "show me", "disclose",
    "leak", "dump", "repeat", "tell me", "give me", "paste", "echo",
    "spell out", "write out", "recite", "verbatim",
)

INJECTION_VERB_WORDS: Final = INJECTION_OVERRIDE_WORDS + INJECTION_DISCLOSE_WORDS

INJECTION_VERBS: Final = phrases(INJECTION_VERB_WORDS)
INJECTION_OVERRIDE_VERBS: Final = phrases(INJECTION_OVERRIDE_WORDS)

#: Modifiers that say an instruction noun is **the system's own**, or that it
#: predates this turn. One of these is mandatory, and that is what stops
#: *"tell me which rule my post was found to break"* from reading as an
#: override: a bare head noun never matches.
#:
#: The class is closed and grammatical -- determiners, possessives, and the
#: adjectives that locate an instruction in the system rather than the forum.
#: It does not grow with the attack vocabulary, which is the whole point of
#: splitting it out from the nouns.
INJECTION_QUALIFIERS: Final = (
    "your", "yours", "the", "all", "any", "every", "these", "those", "this",
    "that", "its", "his", "her", "their", "our", "no", "without",
    "prior", "previous", "earlier", "preceding", "above", "before",
    "initial", "original", "first", "existing", "current", "default",
    "standing", "governing", "hidden", "secret", "concealed", "internal",
    "confidential", "private", "system", "safety", "operator", "content",
    "conversation", "session", "moderation", "underlying", "built-in",
)

#: Qualifiers that on their own say the object is **the system's**, as opposed
#: to merely being definite. "The rules" could be the forum's; "your rules" and
#: "the prior rules" could not.
INJECTION_SYSTEM_QUALIFIERS: Final = (
    "your", "yours", "its", "his", "her", "their",
    "prior", "previous", "earlier", "preceding", "above", "before",
    "initial", "original", "first", "existing", "standing", "governing",
    "hidden", "secret", "concealed", "internal", "confidential",
    "system", "safety", "operator", "content", "conversation", "session",
    "underlying", "built-in", "no", "without", "all",
)

#: Nouns that name the system's own configuration and nothing else. Any
#: qualifier makes these an injection object, because there is no ordinary
#: support reading of *"the system prompt"* or *"your configuration"*.
INJECTION_SYSTEM_HEADS: Final = (
    "instruction", "direction", "prompt", "configuration", "config",
    "guardrail", "setup", "programming", "training", "persona", "directive",
    "safeguard", "secret",
)

#: Nouns an injection aims at that are **also ordinary support vocabulary**.
#: *"Tell me the policy on harassment"* is a policy question and *"which rule
#: did my post break"* is an appeal, so a bare determiner is not enough here:
#: these count only under an override verb, or under a qualifier from
#: ``INJECTION_SYSTEM_QUALIFIERS`` that says whose rules are meant.
INJECTION_SHARED_HEADS: Final = (
    "rule", "policy", "policies", "guideline", "guidance", "limit",
    "limitation", "restriction", "constraint", "command", "role",
)

#: Both halves, for the whitespace-free surface and for readers diffing this
#: against the enum. Neither cue reads it: they read the two halves separately,
#: because that separation is the whole point -- see ``TOPICS``.
INJECTION_HEADS: Final = INJECTION_SYSTEM_HEADS + INJECTION_SHARED_HEADS

#: Objects that are not a modifier-plus-noun phrase. A jailbreak that asserts a
#: *mode* names no instruction noun at all, and the elliptical forms refer to
#: the instructions without repeating them.
INJECTION_IRREGULAR_WORDS: Final = (
    "everything above", "everything before", "everything else",
    "anything above", "told before", "told earlier", "told previously",
    "what you were told", "what you were given", "what you were instructed",
    "the ones you were given", "instructions above", "rules above",
    "developer mode", "dan mode", "god mode", "admin mode", "debug mode",
    "maintenance mode", "jailbreak", "unfiltered", "unrestricted",
    "without any filter", "setup text",
)

#: Objects that are the system's however they are asked for -- so they pair
#: with the disclose verbs as well as the override verbs.
INJECTION_SYSTEM_OBJECTS: Final = one_of(
    qualified(INJECTION_QUALIFIERS, INJECTION_SYSTEM_HEADS),
    qualified(INJECTION_SYSTEM_QUALIFIERS, INJECTION_SHARED_HEADS),
    sequence(INJECTION_IRREGULAR_WORDS),
)

#: Objects that need an override verb to become an injection: a definite
#: article is not enough to say the rules are the system's.
INJECTION_SHARED_OBJECTS: Final = one_of(
    qualified(INJECTION_QUALIFIERS, INJECTION_SHARED_HEADS)
)

#: The enumerated product, for the whitespace-free surface -- which has no word
#: boundaries and so cannot run the grammar's repetition safely. Derived from
#: the same classes, so the two surfaces cannot drift apart.
#:
#: **System heads only.** The dense surface knows neither clauses nor verb
#: classes, so it cannot make the distinction the shared heads depend on: it
#: read *"give me the guidance document"* as an override, because "giveme" and
#: "theguidance" both survive whitespace removal. Restricting it to the
#: unambiguous objects costs nothing that matters -- someone spacing out a
#: payload to defeat matching is writing "i g n o r e a l l p r i o r
#: i n s t r u c t i o n s", not asking about a policy.
INJECTION_OBJECT_WORDS: Final = (
    products(INJECTION_QUALIFIERS, INJECTION_SYSTEM_HEADS)
    + INJECTION_IRREGULAR_WORDS
)



# --------------------------------------------------------------------------
# EVASION -- getting prohibited material past enforcement.

# The object class had to be split, because its two halves do not license the
# same verbs. A **mechanism** is a thing you can be asked to get past, and
# almost any verb aimed at one is evasive. An **outcome** is what an appeal is
# about, and generic verbs aimed at one are the ordinary vocabulary of a
# grievance. Measured, with one flat class and generic verbs in it:
#
#   "how do I stop the moderation emails from filling my inbox?"       -> BLOCK
#   "why was my account suspended and how do I prevent another one?"   -> BLOCK
#   "please make the harassment stop, I have blocked him twice"        -> BLOCK
#
# All three pair a bare "stop"/"prevent" with an enforcement noun inside one
# clause. Those verbs discriminate nothing and are gone; what makes a field
# evasive is getting *past* a mechanism, or *re-publishing* after an outcome.

#: The machinery that catches prohibited material. Wide on purpose: the verb
#: classes below are what keep this honest, and a mechanism noun is rarely the
#: object of an ordinary support request.
EVASION_MECHANISM_WORDS: Final = (
    "moderation", "moderator", "moderators", "the filter", "filter",
    "filters", "spam filter", "profanity checker", "profanity filter",
    "safety check", "safety checks", "content safeguards", "safeguards",
    "automated review", "automated check", "automated checks",
    "automated screening", "automated system", "automated systems",
    "screening", "detection", "detector", "detectors", "review system",
    "content review", "review queue", "review", "reviews", "checks",
    "enforcement", "classifier",
    "spam classifier", "moderation model", "moderation system",
    "safety system", "safety model", "the algorithm", "the system",
    "human review", "reviewers", "the reviewers",
)

#: What enforcement produced -- the vocabulary of every appeal. These count
#: only under a get-past or re-publish verb.
#:
#: "strike" stays out entirely: it is the core noun of an appeal ("the notice
#: mentioned a strike but not which post caused it") and carries no evasive
#: reading of its own.
EVASION_OUTCOME_WORDS: Final = (
    "banned", "banned material", "banned content", "disallowed",
    "prohibited content", "ban", "suspension", "suspended", "blocked",
    "flagged", "flagging", "takedown", "removal", "taken down", "take down",
    "got removed", "was removed", "removed post", "removed content",
    "deleted post", "restricted content",
)

#: Both halves, stated so the split is visible in one place. The cues read the
#: halves separately -- an outcome does not license the verbs a mechanism does.
EVASION_OBJECT_WORDS: Final = EVASION_MECHANISM_WORDS + EVASION_OUTCOME_WORDS

#: Getting past something. Evasive against either half of the object class.
EVASION_GET_PAST_WORDS: Final = (
    "bypass", "bypassing", "evade", "evading", "circumvent", "circumventing",
    "get past", "gets past", "getting past", "get around", "gets around",
    "getting around", "get through", "gets through", "getting through",
    "sneak", "sneaks", "sneaking", "slip", "slips", "slipping", "slip past",
    "slip through", "slips through", "let through", "lets through",
    "pass", "passes", "past", "through", "around", "dodge", "dodging",
    "defeat", "trick", "fool", "beat", "outsmart", "invisible to",
    "undetected", "unnoticed", "unflagged", "workaround", "work around",
    "tricks", "tricking",
    "without being", "without getting", "not get caught", "not get flagged",
    "so it does not", "so it will not", "so they do not", "so they will not",
    "does not catch", "do not catch", "will not catch", "cannot detect",
    "does not detect", "does not notice", "does not pick up",
)

#: Publishing again after enforcement. Evasive only against an outcome -- the
#: point is that the material was already removed once.
#:
#: Not bare "post"/"share"/"publish": every appeal and support message mentions
#: a post.
EVASION_REPUBLISH_WORDS: Final = (
    "repost", "reposts", "reposting", "re-post", "put it back", "post again",
    "publish again", "upload again", "post it again", "share it again",
    "keep posting", "keep using", "keep sending", "distribute", "new alias",
    "another account", "a new account", "second account",
)

#: Restating the material so the machinery scores it differently. Evasive only
#: against a mechanism: rewording to *follow* the rules is what an appeal does.
EVASION_REWORD_WORDS: Final = (
    "phrase", "rephrase", "reword", "rewrite", "wording", "phrasing",
    "word it", "spell it", "spellings", "substitutions",
    "character tricks", "disguise", "obfuscate", "mask", "hide",
    "avoid", "avoiding", "disable", "turn off", "switch off",
)

EVASION_VERB_WORDS: Final = (
    EVASION_GET_PAST_WORDS + EVASION_REPUBLISH_WORDS + EVASION_REWORD_WORDS
)

EVASION_MECHANISMS: Final = phrases(EVASION_MECHANISM_WORDS)
EVASION_OUTCOMES: Final = phrases(EVASION_OUTCOME_WORDS)
EVASION_MECHANISM_VERBS: Final = phrases(
    EVASION_GET_PAST_WORDS + EVASION_REWORD_WORDS
)
EVASION_OUTCOME_VERBS: Final = phrases(
    EVASION_GET_PAST_WORDS + EVASION_REPUBLISH_WORDS
)

#: Non-detection as a stated purpose, which names no mechanism at all.
#:
#: *"so the automated review does not catch it"* names one; *"so it does not
#: get flagged"* does not, and the second phrasing is the commoner. The purpose
#: link is what makes this safe to read on its own: you write "so it is not
#: removed" about something you are about to do, not about something that
#: already happened to you.
EVASION_NON_DETECTION: Final = re_compile(
    r"\b(?:so|so that|in order that)\b[^.!?]{0,40}?"
    r"\b(?:not|never|n't)\b[^.!?]{0,20}?"
    r"\b(?:detect\w*|catch|caught|flag\w*|notice\w*|remov\w*|ban\w*|"
    r"block\w*|taken down|picked up|review\w*|trigger\w*)\b"
    r"|\bwithout\s+(?:being\s+|getting\s+|it\s+being\s+)?"
    r"(?:detect\w*|caught|flag\w*|notic\w*|remov\w*|bann\w*|block\w*|"
    r"trigger\w*)\b"
    # "without the system noticing", "without anyone seeing" -- the same
    # purpose with the detector as the subject rather than the object.
    r"|\bwithout\s+(?:the\s+|any\s+)?\w*\s*(?:one|body|system|filter|"
    r"moderator\w*|review\w*)?\s*(?:notic\w+|detect\w+|see\w+|seeing|"
    r"catch\w*|catching|knowing|realising|realizing)\b"
    r"|\b(?:avoid|evade|escape|dodge|beat)\s+(?:being\s+|getting\s+)?"
    r"(?:detect\w*|caught|flag\w*|notic\w*|remov\w*|bann\w*|block\w*|"
    r"a\s+ban|the\s+ban|another\s+strike)\b",
    IGNORECASE,
)


# --------------------------------------------------------------------------
# HOSTILITY -- abuse aimed at a person.
#
# This is the topic where co-occurrence fails hardest, because the people who
# most need the product talk about abuse constantly. *"Please help me report
# the abusive messages I keep getting"*, *"help me deal with the harassment
# from my neighbour"* and *"give me advice about the bullying my classmate is
# doing"* all pair a produce verb with an abuse noun inside one clause, and all
# three are victims asking for support. Six such messages in eight blocked
# under a proximity test alone.
#
# What separates them from an attack is not distance but **argument
# structure**: in a demand the abuse *is what gets produced*, while in a report
# it is something the writer received. English marks that difference with
# closed-class material, which is why the cues below are constructions rather
# than pairs -- and why the abuse vocabulary itself can stay wide.

#: Verbs that ask for an artifact to be authored. Strong enough to stand as a
#: precondition on their own, unlike the request verbs below.
HOSTILITY_AUTHOR: Final = (
    "write", "draft", "compose", "generate", "create", "make", "craft",
    "produce", "come up with", "put together", "throw together", "work up",
    "rewrite", "reword", "word", "phrase",
)

#: Verbs that merely ask. These are the ones every support message also uses,
#: so they count only inside the artifact construction, where the abuse noun is
#: demonstrably the thing being asked for.
HOSTILITY_ASK: Final = (
    "help me", "give me", "send me", "suggest", "i want", "i need",
    "i would like", "say to",
)

#: The abuse vocabulary, in any part of speech.
HOSTILITY_TERMS: Final = (
    "insult", "insults", "insulting", "threaten", "threatening", "threat",
    "threats", "intimidate", "intimidating", "humiliate", "humiliating",
    "demean", "demeaning", "degrade", "degrading", "belittle", "harass",
    "harassing", "harassment", "bully", "bullying", "mock", "mocking",
    "ridicule", "abusive", "abuse", "cruel", "cruelty", "hostile",
    "hostility", "coercive", "coercive language", "hurtful", "nasty",
    "vicious", "scare", "scares", "frighten", "frightens", "frightening",
    "terrify", "shame", "cut down", "tear down", "into silence",
    # The "make them feel X" construction, which names no abuse noun at
    # all -- the harm is in the predicate, not in a slur.
    "feel small", "feel stupid", "feel worthless", "feel ashamed",
    "look stupid", "look like a fool", "put down", "putdown", "roast",
    "comeback", "tear into", "rip into",
)

#: Base-form abuse verbs, which are the only ones that can take a target as a
#: direct object. Nominals are excluded on purpose: *"the bullying my classmate
#: is doing"* puts a person right after an abuse word, but as the subject of a
#: reduced relative clause rather than as its object.
HOSTILITY_ACTS: Final = (
    "threaten", "intimidate", "humiliate", "demean", "degrade", "belittle",
    "harass", "bully", "mock", "ridicule", "insult", "scare", "frighten",
    "terrify", "shame", "roast",
)

#: What may stand between a produce verb and the artifact it asks for. The
#: class is closed and, crucially, **indefinite**: you cannot ask to author a
#: thing that already exists, so "the", "this" and the possessives are absent.
#: That single omission is what separates *"make a vicious comeback"* from
#: *"please make the harassment stop"*.
ARTIFACT_FILLER: Final = (
    "a", "an", "some", "any", "one", "two", "few", "several", "couple",
    "something", "anything", "more", "most", "far", "really", "very", "quite",
    "extremely", "truly", "incredibly", "super", "much", "another", "other",
    "sort of", "kind of", "bit of",
)

#: Purpose links. The abuse term must follow **immediately**: "to intimidate"
#: is a purpose, while "to an insulting comment" is a recipient, and only the
#: adjacency tells them apart.
HOSTILITY_PURPOSE: Final = (
    "to", "designed to", "meant to", "intended to", "in order to",
    "so it", "so that it", "so they", "that will", "that would",
    "which will", "which would", "so as to",
)

#: Copular, perception and causative links, which take a degree phrase or an
#: object pronoun before the term: "sounds far more threatening", "comes across
#: as more intimidating", "make her feel worthless".
#:
#: The causatives are safe here only because the filler class below excludes
#: definite determiners: "make *her feel worthless*" is a result, while "make
#: *the harassment* stop" is a plea about something that already exists.
HOSTILITY_COPULA: Final = (
    "sound", "sounds", "look", "looks", "feel", "feels", "seem", "seems",
    "come across", "comes across", "comes off", "reads as",
    "make", "makes", "making", "made",
)

DEGREE_FILLER: Final = (
    "a", "an", "as", "so", "more", "most", "far", "really", "very", "quite",
    "extremely", "truly", "incredibly", "super", "much", "even", "way",
    "them", "him", "her", "it", "me", "us",
)

#: The abuse *is the artifact*: a produce or request verb whose object, across
#: closed-class material only, is an abuse term.
HOSTILITY_ARTIFACT: Final = one_of(
    sequence(HOSTILITY_AUTHOR + HOSTILITY_ASK, HOSTILITY_TERMS,
             filler=ARTIFACT_FILLER)
)

#: The abuse is the *result* the artifact must achieve. Licensed only when the
#: field also asks for something to be authored -- read field-wide, because the
#: produce verb sits several clauses away in "rewrite the message below so it
#: sounds far more threatening".
HOSTILITY_RESULT: Final = one_of(
    sequence(HOSTILITY_PURPOSE, HOSTILITY_TERMS),
    sequence(HOSTILITY_COPULA, HOSTILITY_TERMS, filler=DEGREE_FILLER),
)

HOSTILITY_AUTHORS: Final = phrases(HOSTILITY_AUTHOR)

#: Who the abuse is aimed at, for the third construction: an abuse verb taking
#: a person as its direct object.
#:
#: Direction is word order. In a demand the person follows the abuse --
#: *"humiliate my roommate"*. In a report the person precedes it, because they
#: are its subject: *"someone is bullying my daughter"*. So this class counts
#: only when it **follows** the abuse verb inside the clause. Merely naming a
#: target was the earlier test, and it read every victim's report -- which
#: always names who is doing it -- as an attack.
HOSTILITY_TARGETS: Final = phrases(
    (
        "someone", "somebody", "this person", "that person", "the reviewer",
        "the seller", "the user", "my ex", "my neighbour", "my neighbor",
        "my landlord", "my coworker", "my co-worker", "my colleague", "my boss",
        "my roommate", "my flatmate", "my classmate", "my former",
        "a coworker", "a neighbour", "a neighbor", "a colleague",
    )
)

HOSTILITY_ACT_VERBS: Final = phrases(HOSTILITY_ACTS)

#: What abuse looks like when someone forwards it, rather than asks for it.
#: Evidence-only, and a register rather than a topic: ordinary forwarded
#: material does not address its reader this way, which is what lets this class
#: stand alone where the enforcement and contact nouns cannot.
#:
#: A slur list is the wrong shape for this and was measured to be: six of eight
#: forwarded-abuse probes carried no listed word, because abuse vocabulary is
#: unbounded and inventive while the *constructions* it arrives in are not.
#: The list below is retained -- the words in it really are abusive -- but it is
#: now one alternative among the constructions in ``HOSTILITY_REGISTER``.
HOSTILITY_SPECIMEN: Final = phrases(
    (
        "worthless", "pathetic", "idiot", "idiots", "moron", "stupid",
        "disgusting", "freak", "parasite", "loser", "scum", "trash", "vermin",
        "worm", "brainless", "incompetent", "shut your mouth", "shut up",
        "nobody wants you", "nobody would miss you", "waste of space",
        "crawl back", "get out", "never come back", "laughs at you",
        "laugh at you", "you are a", "you are an", "you sad", "you little",
        "kill yourself", "everyone knows you", "everyone can see you",
    )
)

#: Abuse by construction rather than by vocabulary. Every alternative below is
#: a way of addressing a reader that ordinary forwarded material -- a removal
#: notice, a receipt, a support transcript -- does not use, and none of them
#: names a particular insult.
#:
#: The negative control this must survive is an appeal's own attachment, and
#: none of these fire on *"Your post was removed for prohibited content"* or
#: *"Your account is suspended for 7 days following a review"*: an enforcement
#: notice states what happened, it does not tell its reader what they are.
HOSTILITY_REGISTER: Final = re_compile(
    # Predicating something of the reader: "you are a ...", "you're such a ...".
    r"\byou(?:\s+are|'re|\s+were)\s+(?:a|an|such|so|nothing|just)\b"
    # Vocative epithet: an address closing a clause, "..., coward".
    r"|\byou\s+(?:\w+\s+){0,2}(?:rat|pig|dog|snake|worm|animal|parasite|"
    r"scum|filth|freak|clown|coward|liar|fraud|failure|loser|nobody|creep|"
    r"idiot|moron|fool|slob|waste)\b"
    # Dehumanising comparison.
    r"|\b(?:you|your\s+\w+)\s+(?:sound|sounds|look|looks|smell|smells|"
    r"act|acts|behave|behaves)\s+like\s+(?:a|an)\b"
    # Wishing harm on the reader.
    r"|\bi\s+hope\s+(?:you|your)\b[^.!?]{0,40}?\b(?:burn\w*|die|dies|dead|"
    r"rot\w*|fail\w*|lose|loses|suffer\w*|regret\w*|choke\w*|collapse\w*)\b"
    # Threatening a consequence.
    r"|\b(?:watch\s+your\s+back|you\s+(?:will|'ll)\s+regret|"
    r"see\s+what\s+happens\s+to\s+you|coming\s+for\s+you|make\s+you\s+pay|"
    r"you\s+are\s+(?:done|finished)|we\s+know\s+(?:your|where\s+you)|"
    r"i\s+know\s+where\s+you)\b"
    # Telling the reader to remove themselves.
    r"|\b(?:go\s+back\s+(?:to\s+)?where\s+you|do\s+everyone\s+a\s+favou?r|"
    r"never\s+come\s+back|get\s+out\s+of\s+(?:here|this)|"
    r"disappear|crawl\s+back)\b"
    # Nobody-cares, as a predicate about the reader.
    r"|\b(?:nobody|no\s+one|no-one)\b[^.!?]{0,30}?\b(?:wants|likes|misses|"
    r"cares|can\s+stand|would\s+miss|needs)\b[^.!?]{0,15}?\byou\b"
    # Contempt frame: "only a ... would ...".
    r"|\bonly\s+(?:a|an)\b[^.!?]{0,30}?\bwould\b",
    IGNORECASE,
)


# --------------------------------------------------------------------------
# PRIVACY -- disclosing a third party's identity or contact details.
#
# This is the one topic that needs three parts, not two. "email address" is an
# ordinary support noun: *"link a second email address to this profile"* and
# *"which email address is on my own account"* must both stay ALLOW. What makes
# it a privacy request is that the identifier belongs to **somebody else**.

PRIVACY_VERBS: Final = phrases(
    (
        "tell", "tell me", "give", "give me", "send", "send me", "share",
        "reveal", "disclose", "show", "show me", "find", "find out", "look up",
        "locate", "obtain", "get", "get me", "provide", "identify",
        "retrieve", "pull", "pull up",
        "open", "confirm", "check", "verify", "trace", "unmask", "dig up",
        "search for", "look for", "match", "cross-reference", "connect",
        "i need", "i want", "who is", "who owns", "which user", "what is",
        "which", "whose", "where does", "where do",
    )
)

#: Identifier nouns. Harmless on their own -- they need a third-party marker.
PRIVACY_IDENTIFIER_WORDS: Final = (
    "identity", "real name", "full name", "legal name", "actual name",
    "name", "email", "email address", "phone", "phone number",
    "home address", "address", "postal address", "location",
    "contact details", "contact information", "contact coordinates",
    "personal details", "personal information", "personal data",
    "private data", "private information", "handle", "username",
    "account details", "ip address",
    # The person, and attributes that locate one.
    "person", "actual person", "real person", "individual",
    "workplace", "employer", "school", "employer name", "city",
    "where they live", "where they work",
)

PRIVACY_IDENTIFIERS: Final = phrases(PRIVACY_IDENTIFIER_WORDS)

#: Nouns that refer to a person who is not the writer.
PRIVACY_PARTY_NOUNS: Final = (
    "person", "people", "user", "users", "account", "accounts", "profile",
    "profiles", "handle", "member", "seller", "buyer", "reviewer", "reporter",
    "complainant", "party", "individual", "customer", "poster", "commenter",
    "owner", "sender", "one",
)

#: Determiners that put such a noun at arm's length from the writer.
PRIVACY_PARTY_DETERMINERS: Final = (
    "the", "that", "this", "another", "other", "some", "any", "which",
    "whose", "each", "every", "a", "an",
)

#: Relative clauses that attach an action to the third party.
PRIVACY_RELATIVE_VERBS: Final = (
    "reported", "filed", "flagged", "complained", "criticized", "criticised",
    "left", "posted", "wrote", "sent", "messaged", "owns", "runs", "uses",
    "keeps", "made", "submitted", "raised", "contacted", "is behind",
    "was behind", "am disputing", "i am disputing", "i reported",
)

#: Markers that the identifier is somebody else's. Read field-wide rather than
#: inside the clause: the marker is routinely a trailing relative clause --
#: "the home address *of the person who criticized me*" -- and pushing it into
#: the window would cost the very construction it is there to recognise.
#:
#: The literal list this replaced enumerated relative pronouns paired with
#: verbs, one string at a time, and so had "who criticized" but not "that
#: criticized", "who filed" but not "they filed". Written as a grammar the
#: pairing is free, which is the same repair the injection objects needed.
PRIVACY_THIRD_PARTY: Final = one_of(
    sequence(PRIVACY_PARTY_DETERMINERS, PRIVACY_PARTY_NOUNS),
    sequence(("who", "that", "which", "they", "he", "she"),
             PRIVACY_RELATIVE_VERBS),
    sequence(
        (
            "someone else", "somebody else", "another user", "other user",
            "the other party", "third party", "whoever", "belongs to",
            "belong to", "behind the", "behind this", "behind that",
            "of the person", "of the user", "their", "his", "her", "them",
            "anonymous",
        )
    ),
)

#: First-person possessives. An identifier they govern is the writer's own, and
#: the whole topic turns on the identifier belonging to somebody else.
#:
#: This is the guard that keeps *"which email address is on my own account"*
#: and *"link a second phone number to my profile"* out, now that the
#: third-party markers above are wide enough to match the surrounding prose.
#: Matched against the text immediately preceding an identifier, so it must be
#: end-anchored: at most one modifier may stand between the possessive and the
#: noun it owns ("my own account", "our shared handle").
PRIVACY_FIRST_PERSON: Final = re_compile(
    r"\b(?:my|mine|our|ours)\s+(?:\w+\s+){0,1}$", IGNORECASE
)

#: Noun phrases that **are** a request for a third party's identity record, and
#: so need neither a verb nor a marker: they name the party and the record at
#: once. *"Which account belongs to the person that filed this complaint?"* has
#: no disclose verb anywhere and is unambiguously a privacy request.
#:
#: Relative clauses deliberately stay out. *"who reported"*, *"who filed"* and
#: *"who flagged"* mark third-partyness -- they belong to the class above --
#: but they attach to a person the writer is merely describing, so treating
#: them as self-sufficient blocked *"can the person who reported me see my real
#: name?"*, which is a policy question.
PRIVACY_INHERENT: Final = phrases(
    (
        "reporter identity", "reporter details", "reporter record",
        "reporter name", "identity of the reporter", "identity behind",
        "complainant details", "complainant name", "complainant identity",
        "which account belongs to", "whose account", "who is behind",
    )
)

#: Third-party roles in the **possessive**, which name the party and own the
#: record in one phrase. *"The reporter's email address please"* has no
#: disclose verb anywhere and was a documented miss: the elliptical demand is
#: complete without one, because the genitive already says whose record it is.
#:
#: Bare possessive pronouns stay out. "Her email address" is what *"my friend
#: forgot her email address for the account we share"* says, and that is an
#: ordinary support request; only an explicit third-party role counts here.
PRIVACY_POSSESSIVE_PARTIES: Final = (
    "reporter's", "reporters'", "complainant's", "seller's", "buyer's",
    "reviewer's", "moderator's", "other user's", "another user's",
    "other party's", "that user's", "the user's", "that person's",
    "the person's", "poster's", "commenter's", "owner's", "sender's",
)

PRIVACY_POSSESSIVE: Final = one_of(
    sequence(PRIVACY_POSSESSIVE_PARTIES, PRIVACY_IDENTIFIER_WORDS)
)


# --------------------------------------------------------------------------
# FRAUD -- soliciting money or credentials.
#
# Evidence-only by construction: message-side fraud is a hole in the verdict
# table, since no BLOCK reason code covers soliciting it. The class exists to
# recognise scam specimens, and like the hostility specimens it is a register
# ordinary forwarded material does not write in.

FRAUD_SPECIMEN: Final = phrases(
    (
        "processing fee", "clearance payment", "advance fee", "customs fee",
        "administrative fee", "release fee", "small fee", "gift card",
        "gift cards", "wire transfer", "wire the", "bank transfer",
        "crypto", "cryptocurrency", "bitcoin", "western union",
        "guaranteed prize", "your prize", "claim your", "you have won",
        "you won", "congratulations", "lottery", "inheritance", "windfall",
        "verify your account", "confirm your card", "card number",
        "account number", "billing team", "billing department",
        "official support", "support agent", "security team",
        "send the codes", "send the code", "verification code",
        "no refunds", "act now", "before it expires", "limited time",
        "deal closes", "final notice", "investment opportunity",
        "double your", "guaranteed return",
    )
)


# --------------------------------------------------------------------------
# The evidence floor.
#
# These are *constructions*, not topics. Each is a register that ordinary
# forwarded material does not use, and none of them identifies a subtype -- so
# a match here says only "this is adverse", which is exactly what
# ADVERSE_UNKNOWN means. The floor exists because subtype failure was costing
# an action: evidence plainly hostile or fraudulent but not confidently *which*
# emitted nothing and fell through to ALLOW, spending whole utility families to
# protect a fraction of a reason code.
#
# It still requires a positive match. "Evidence exists" is not a cue, because
# the negative control this design must survive is an ALLOW-family case that
# happens to carry a benign attachment.

#: Second-person denigration: an address plus a negative predicate. The
#: construction is what matters -- "you are a X" where X is disparaging -- and
#: it is why no list of slurs is needed.
ADVERSE_SECOND_PERSON: Final = re_compile(
    r"\byou(?:\s+are|'re|\s+were)\s+(?:a|an|such|so|the)?\s*"
    r"\w*(?:ing|less|able)?\s*\w*\b(?=[^.!?]*\b(?:idiot|moron|fraud|liar|"
    r"waste|scum|trash|freak|worthless|pathetic|disgusting|stupid|useless|"
    r"spineless|ignorant|incompetent|nobody|nothing)\b)",
    IGNORECASE,
)

#: Solicitation: a transfer verb aimed at money or credentials. Catches the
#: scam register without naming any particular scam.
ADVERSE_SOLICITATION: Final = re_compile(
    r"\b(?:pay|send|transfer|wire|deposit|remit|claim|confirm|verify|release|"
    r"need|needs|needed|asking\s+for|ask\s+for|requires?|forward)\b"
    r"[^.!?]{0,60}?\b(?:fee|payment|money|funds|card|account|code|codes|"
    r"gift|deposit|prize|reward|inheritance|transfer|details|\d[\d.,]*)\b",
    IGNORECASE,
)

#: Coerced urgency: a consequence conditioned on the reader acting now. The
#: pairing is the cue; either half alone is ordinary notice text.
ADVERSE_COERCED_URGENCY: Final = re_compile(
    r"\b(?:unless you|before it|or your|otherwise your|will be (?:closed|"
    r"suspended|deleted|terminated)|will close|expires|final notice|"
    r"act now|limited time|today only)\b",
    IGNORECASE,
)

#: An imperative aimed at the reader as if it were a system.
ADVERSE_SYSTEM_IMPERATIVE: Final = re_compile(
    r"\b(?:you are now|act as|pretend|from now on|new directive|"
    r"no longer applies?|are void|ignore your|disregard your|override your)\b",
    IGNORECASE,
)

ADVERSE_CUES: Final = (
    ADVERSE_SECOND_PERSON,
    ADVERSE_SOLICITATION,
    ADVERSE_COERCED_URGENCY,
    ADVERSE_SYSTEM_IMPERATIVE,
)


def adverse_cue_hits(text: str) -> int:
    """Count adverse registers present, without identifying a subtype."""

    return sum(1 for cue in ADVERSE_CUES if cue.search(text))


# --------------------------------------------------------------------------
# What the requester says the specimen is.
#
# Reason-code only, and safe by construction: the tier has already fixed the
# action to ALLOW_AS_DATA, so this can move the reason and nothing else. A
# moderator forwarding material almost always names the category they think
# they are looking at, and taking their word for it generalises far better
# than guessing the subtype from the specimen.
SPECIMEN_LABELS: Final[tuple[tuple[Pattern[str], Concept], ...]] = (
    (
        phrases(
            (
                "scam", "scams", "fraud", "fraudulent", "phishing", "phish",
                "advance fee", "impersonation", "impersonating", "fake seller",
                "romance scam", "money", "payment", "financial",
            )
        ),
        Concept.FRAUD,
    ),
    (
        phrases(
            (
                "injection", "prompt injection", "prompt-injection",
                "jailbreak", "payload", "manipulate", "manipulating",
                "override", "overriding", "instruction", "instructions",
            )
        ),
        Concept.INJECTION,
    ),
    (
        phrases(
            (
                "abuse", "abusive", "harassment", "harassing", "hostile",
                "hostility", "insult", "insulting", "threatening", "bullying",
                "personal attack", "tone", "severity",
            )
        ),
        Concept.HOSTILITY,
    ),
)


def labelled_subtype(message: str) -> Concept | None:
    """Return the concept the requester named, if they named one."""

    for pattern, concept in SPECIMEN_LABELS:
        if pattern.search(message):
            return concept
    return None


# --------------------------------------------------------------------------
# The whitespace-free surface.

#: Minimum length for a stem matched on the whitespace-stripped surface.
#: Word boundaries are gone there, so only long phrases stay distinctive --
#: a short one would fire inside an unrelated word.
DENSE_MIN_LENGTH: Final = 11

#: Verbs are allowed to be shorter, because a dense match needs a verb *and*
#: an object -- see ``Topic.dense``.
DENSE_MIN_VERB_LENGTH: Final = 6


def _dense_pattern(
    items: tuple[str, ...], minimum: int = DENSE_MIN_LENGTH
) -> Pattern[str] | None:
    """Build the whitespace-free variant of a class.

    Derived from the class itself rather than kept as a second list, so the
    dense surface cannot drift out of step with the one it mirrors.
    """

    dense = tuple(
        "".join(phrase.split())
        for phrase in items
        if len("".join(phrase.split())) >= minimum
    )
    if not dense:
        return None
    alternatives = "|".join(sorted(set(dense), key=len, reverse=True))
    return re_compile(alternatives, IGNORECASE)


def _dense_cue(
    verbs: tuple[str, ...], objects: tuple[str, ...]
) -> tuple[Pattern[str], Pattern[str]] | None:
    """Pair the two dense classes a topic needs, or ``None`` if either is empty."""

    dense_verbs = _dense_pattern(verbs, DENSE_MIN_VERB_LENGTH)
    dense_objects = _dense_pattern(objects)
    if dense_verbs is None or dense_objects is None:
        return None
    return dense_verbs, dense_objects


# --------------------------------------------------------------------------
# Cues and topics.


@dataclass(frozen=True, slots=True)
class Cue:
    """One construction by which a field can show a topic.

    ``names`` is the class that says *what* the topic is -- the instruction
    noun, the abuse noun, the identifier. ``governs``, when present, is the
    class that must stand with it **inside one clause**: the verb that asks for
    it, or the person it is aimed at. A cue with no ``governs`` is
    self-sufficient, which only a class that is already a complete request or
    an unmistakable register may be.
    """

    names: Pattern[str]
    governs: Pattern[str] | None = None
    #: Require ``governs`` to follow ``names``. Word order is what separates
    #: abuse aimed at a person from a report of abuse they are committing.
    governs_follows: bool = False
    #: A precondition read across the whole field rather than in the clause.
    requires: Pattern[str] | None = None
    #: Disqualifies a term whose immediately preceding text matches -- an
    #: end-anchored pattern read over the ``DENIAL_LOOKBACK`` characters before
    #: it. Where ``requires`` asks the field to show something, this asks the
    #: term itself not to be governed by something: an identifier under a
    #: first-person possessive is the writer's own record, however many
    #: third-party markers the rest of the sentence contains.
    denied_by: Pattern[str] | None = None
    #: Recognises forwarded material, so it is never read in a message.
    evidence_only: bool = False

    def applies_to(self, origin: Origin) -> bool:
        return origin is Origin.EVIDENCE or not self.evidence_only

    def hits(self, text: str) -> int:
        """Count distinct topic terms that actually stand in this construction.

        Counting the terms that participate -- rather than every term present
        -- keeps confidence honest: it measures how well corroborated *this
        reading* is, not how much vocabulary the field happens to contain.
        """

        if self.requires is not None and not self.requires.search(text):
            return 0
        named = [
            match.span()
            for match in self.names.finditer(text)
            if not self._denied(text, match.start())
        ]
        if not named or self.governs is None:
            return len({text[start:end] for start, end in named})

        governing = [match.span() for match in self.governs.finditer(text)]
        if not governing:
            return 0
        return len(
            {
                text[start:end]
                for start, end in named
                if self._governed(text, (start, end), governing)
            }
        )

    def _denied(self, text: str, start: int) -> bool:
        """Whether the text immediately before *start* disqualifies the term."""

        if self.denied_by is None:
            return False
        return bool(
            self.denied_by.search(text[max(0, start - DENIAL_LOOKBACK) : start])
        )

    def _governed(
        self, text: str, named: tuple[int, int], governing: list[tuple[int, int]]
    ) -> bool:
        """Whether any governing span shares a clause with *named*.

        Both lists come from ``finditer`` and so are position-sorted, and the
        text between two spans only grows as they move apart. So the search
        starts at the nearest candidate and stops the moment one is too far:
        everything beyond it is further still. Pairing every span with every
        other is quadratic in the match count, which a message that repeats one
        phrase can drive arbitrarily high.
        """

        start, end = named
        index = bisect_left(governing, start, key=lambda span: span[0])

        for span in governing[index:]:
            if not (self.governs_follows and span[0] < end) and share_clause(
                text, named, span
            ):
                return True
            if span[0] > end:
                break

        if self.governs_follows:
            return False
        for span in reversed(governing[:index]):
            if share_clause(text, named, span):
                return True
            if span[1] < start:
                break
        return False


@dataclass(frozen=True, slots=True)
class Topic:
    """One concept, as the constructions a field may show it through.

    Cues are alternatives: the best-supported one carries the topic. ``dense``
    is a separate surface rather than a cue, because the whitespace-free view
    has no words and so no clauses -- see ``matches``.
    """

    concept: Concept
    cues: tuple[Cue, ...]
    dense: tuple[Pattern[str], Pattern[str]] | None = None

    def _dense_hit(self, dense: str) -> bool:
        """Whether *both* halves survive on the whitespace-free surface.

        Requiring both is what makes this safe. Word boundaries do not exist
        here, so a single long stem could still land inside an unrelated run of
        characters; two independent stems doing so is vanishingly unlikely, and
        the conjunction costs nothing because a real attack spaces out its
        whole payload, not half of it.
        """

        return self.dense is not None and all(
            part.search(dense) for part in self.dense
        )

    def matches(self, text: str, dense: str, origin: Origin) -> int:
        """How strongly this field shows the topic, as a corroboration count."""

        hits = max(
            (cue.hits(text) for cue in self.cues if cue.applies_to(origin)),
            default=0,
        )
        if hits:
            return hits
        return 1 if self._dense_hit(dense) else 0


TOPICS: Final[tuple[Topic, ...]] = (
    Topic(
        Concept.INJECTION,
        cues=(
            Cue(INJECTION_SYSTEM_OBJECTS, INJECTION_VERBS),
            Cue(INJECTION_SHARED_OBJECTS, INJECTION_OVERRIDE_VERBS),
        ),
        dense=_dense_cue(INJECTION_VERB_WORDS, INJECTION_OBJECT_WORDS),
    ),
    Topic(
        Concept.EVASION,
        cues=(
            Cue(EVASION_MECHANISMS, EVASION_MECHANISM_VERBS),
            Cue(EVASION_OUTCOMES, EVASION_OUTCOME_VERBS),
            Cue(EVASION_NON_DETECTION),
        ),
        dense=_dense_cue(EVASION_VERB_WORDS, EVASION_MECHANISM_WORDS),
    ),
    Topic(
        Concept.HOSTILITY,
        cues=(
            Cue(HOSTILITY_ARTIFACT),
            Cue(HOSTILITY_RESULT, requires=HOSTILITY_AUTHORS),
            Cue(HOSTILITY_ACT_VERBS, HOSTILITY_TARGETS, governs_follows=True),
            Cue(HOSTILITY_SPECIMEN, evidence_only=True),
            Cue(HOSTILITY_REGISTER, evidence_only=True),
        ),
    ),
    Topic(
        Concept.PRIVACY,
        cues=(
            Cue(
                PRIVACY_IDENTIFIERS,
                PRIVACY_VERBS,
                requires=PRIVACY_THIRD_PARTY,
                denied_by=PRIVACY_FIRST_PERSON,
            ),
            Cue(PRIVACY_INHERENT),
            Cue(PRIVACY_POSSESSIVE),
        ),
    ),
    Topic(
        Concept.FRAUD,
        cues=(Cue(FRAUD_SPECIMEN, evidence_only=True),),
    ),
)


def _confidence(hits: int) -> float:
    """Turn corroboration into a within-tier tiebreak.

    Never a summand and never compared across tiers: more matched terms means
    a better-supported reading of the same field, nothing more.
    """

    return min(0.6 + 0.1 * hits, 1.0)


def observe(
    text: str, dense: str, origin: Origin, boost: float = 0.0
) -> tuple[Observation, ...]:
    """Report every topic the field shows, through the cues its origin allows.

    ``boost`` raises the confidence of topics that already matched, and can
    never create one. That is the whole role suspicious control characters are
    allowed to play: they corroborate a reading, never establish it.
    """

    observations = []
    for topic in TOPICS:
        hits = topic.matches(text, dense, origin)
        if hits:
            observations.append(
                Observation(
                    topic.concept, origin, min(_confidence(hits) + boost, 1.0)
                )
            )
    return tuple(observations)
