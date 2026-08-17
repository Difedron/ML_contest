"""Detection of imminent risk to a person's physical safety.

IMMINENT_SAFETY_RISK is the only family whose action is ESCALATE, and it is
the one family where a wrong answer is not symmetric: blocking a person in
danger is far worse than routing an over-eager report to a human. The
detector is still built to be strict, because escalation that fires on
ordinary urgency stops being a signal at all.

Strictness comes from requiring a conjunction rather than from a high
threshold. Peril alone describes a report ("someone threatened me last year").
Immediacy alone describes ordinary impatience ("I need this fixed right now").
Only together do they describe an emergency.
"""

from __future__ import annotations

from re import Pattern
from typing import Final

from guardrail.concepts import Concept, Observation, Origin
from guardrail.patterns import phrases as _phrases
from guardrail.request_view import RequestView
from guardrail.topics import Cue


# Physical harm to the person writing, or to someone they are reporting on.
#
# Deliberately written from the victim's side rather than the addressee's:
# "hurt me" and "hurt themselves" are cries for help, while "hurt you" is a
# threat, which is GENERATE_ABUSE and belongs in a BLOCK family. That keeps
# quoted hostility from reading as an emergency.
#
# Violence, as a verb paired with whom it lands on. The pairing used to be an
# enumerated product of adjacent phrases -- every verb concatenated with every
# victim -- and adjacency is not how English puts a victim after a verb.
# Measured, all of these missed:
#
#   "my neighbour is beating his wife right now"
#   "a stranger grabbed her arm and she is running away"
#   "my ex just broke in and is coming up the stairs"
#
# The direction constraint is the part worth keeping, and ``Cue`` already
# expresses it: ``governs_follows`` requires the victim to come *after* the
# verb, which is what separates "hurt me" from "intimidate my former coworker"
# -- in a report the victim follows the verb, in a demand the target does too
# but the verb is a request to author, not a description of violence.
HARM_VERBS: Final = (
    "hurt", "hurting", "harm", "harming", "kill", "killing", "attack",
    "attacking", "attacked", "assault", "assaulting", "assaulted", "beat",
    "beating", "beat up", "stab", "stabbing", "stabbed", "shoot", "shooting",
    "strangle", "strangling", "choke", "choking", "hit", "hitting", "grab",
    "grabbed", "grabbing", "drag", "dragged", "dragging", "threaten",
    "threatening", "threatened", "follow", "following", "followed", "stalk",
    "stalking", "chase", "chasing", "corner", "cornered", "coming at",
    "coming after", "coming for", "coming up", "coming toward",
    "coming towards", "going to hurt", "going to kill", "going to attack",
    "closing in on", "trapped", "locked in with", "throwing things at",
    "screaming at", "punch", "punched", "punching", "burn", "burning",
    "waiting for", "watching", "found where", "knows where",
)

#: Whom the harm lands on. Written from the victim's side: first person, the
#: writer's dependants, and the third parties a bystander report names.
#:
#: Deliberately excludes the coworker/neighbour/roommate vocabulary that
#: ``topics.HOSTILITY_TARGETS`` uses. Those are the people an *attacker* names
#: when demanding an abusive message, and letting them count here would put
#: GENERATE_ABUSE requests on the escalation tier, which outranks BLOCK.
VICTIMS: Final = (
    "me", "i", "myself", "us", "him", "her", "them", "himself", "herself",
    "themselves", "my kid", "my kids", "my child", "my children",
    "my family", "my mother", "my mum", "my mom", "my father", "my dad",
    "my sister", "my brother", "my daughter", "my son", "my wife",
    "my husband", "my partner", "my friend", "my baby", "my grandmother",
    "his wife", "her husband", "his girlfriend", "her boyfriend",
    "the child", "the kid", "a child", "the woman", "the man", "the victim",
    "someone", "somebody", "a stranger", "the neighbour", "the neighbor",
)

#: Reflexives, which need no verb pairing because the verb is implied by the
#: object: "hurt themselves" is peril whoever wrote it.
#: Self-harm already acted on: the act is in the past tense or the means are
#: in hand, so the emergency is under way whatever else the message says.
UNFOLDING_SELF_HARM: Final = (
    "the pills in my hand", "pills in my hand", "took the pills",
    "swallowed the pills", "swallowed a bottle", "swallowed a whole",
    "overdosed",
)

#: Self-harm as intent or ideation. Grave, but not by itself an event in
#: progress -- these still need an immediacy marker.
STANDING_SELF_HARM: Final = (
    "hurt myself", "hurt himself", "hurt herself", "hurt themselves",
    "harm myself", "harm himself", "harm herself", "harm themselves",
    "kill myself", "kill himself", "kill herself", "kill themselves",
    "end my life", "end his life", "end her life", "end their life",
    "take my own life", "take his own life", "take her own life",
    "take their own life", "suicide", "suicidal", "self harm", "selfharm",
    "overdose", "have the pills", "bottle of pills",
    "whole bottle of", "took a bottle", "going to jump", "about to jump",
    "on the bridge",
    "on the ledge", "off the roof", "do not want to live",
    "does not want to live", "cannot go on", "can not go on", "end it all",
    # Ideation stated without naming the act. People writing these do not
    # reach for the clinical noun, and every one of them is the same report.
    "do not want to be alive", "does not want to be alive",
    "want to die", "wants to die", "wanting to die", "better off dead",
    "no reason to live", "nothing to live for", "not be here anymore",
    "not want to be here", "said goodbye", "saying goodbye",
    "final message", "last message to anyone", "cannot do this anymore",
    "can not do this anymore", "give up on everything",
)

SELF_HARM: Final = UNFOLDING_SELF_HARM + STANDING_SELF_HARM

#: A weapon is peril only once someone is holding or using it -- bare "gun"
#: is a policy question about weapon imagery, and bare "armed" is the armed
#: forces.
#:
#: Wielded, meaning the weapon is in play right now. These carry their own
#: immediacy; a blade already drawn does not become more urgent for the writer
#: adding "right now".
WIELDED_WEAPONS: Final = (
    "with a knife", "with a gun", "with a weapon", "with a blade",
    "holding a knife", "holding a gun", "pulled a knife", "pulled a gun",
    "pointing a gun", "waving a knife", "waving a gun", "drew a knife",
    "took out a knife", "got a knife out", "knife in his hand",
    "knife in her hand", "gun in his hand", "gun in her hand",
    "one of them has a knife", "one of them has a gun",
)

#: Possessed, meaning capability rather than an event. Peril, but it still
#: takes an immediacy marker to become an emergency: *"my neighbour has a gun"*
#: is a report, *"my neighbour has a gun and is at my door"* is not.
POSSESSED_WEAPONS: Final = (
    "has a knife", "has a gun", "has a weapon", "is armed", "armed and",
)

WEAPONS: Final = WIELDED_WEAPONS + POSSESSED_WEAPONS

#: States that are an emergency without needing a verb or an actor, and that
#: describe an event already under way. Present-continuous by construction.
UNFOLDING_STATES: Final = (
    "in immediate danger", "bleeding badly", "unconscious", "not breathing",
    "breaking into my", "banging on my door", "kicking the door",
    "kicking my door", "kicking in the door", "forcing the door",
    "breaking the door", "smashing the door", "broke into the house",
    "broke into my house", "just broke in", "coming up the stairs",
    "hiding in the bathroom", "hiding in the closet", "hiding under",
    "barricaded", "screaming through the wall", "hear her screaming",
    "hear him screaming", "screaming for help",
)

#: The same, for states that describe a situation rather than an event in
#: progress. These need an immediacy marker of their own.
STANDING_STATES: Final = (
    "in danger", "in serious danger", "my life is in danger", "not safe here",
    "not safe at home", "broke the door", "broke into my", "broken into my",
    "outside my door", "outside my house", "at my door",
)

EMERGENCY_STATES: Final = UNFOLDING_STATES + STANDING_STATES

#: Peril that needs no verb-victim pairing: the phrase is the emergency.
#:
#: The directed harm construction is *not* in here -- it is a ``Cue``, because
#: a verb and its victim stand a clause apart rather than adjacent. This tuple
#: is what remains: phrases that carry peril on their own.
PERIL_PHRASES: Final = SELF_HARM + WEAPONS + EMERGENCY_STATES

# Happening now or imminently -- an adverb class and a tense class.
#
# Bare "now" stays excluded: it is ordinary impatience in nearly every support
# message. The tense markers carry what it used to be asked to carry, and they
# carry it better -- "is closing in on me" is present-continuous, which is the
# grammatical form of an event in progress, while "now" is just a word that
# often accompanies one.
IMMEDIACY_PHRASES: Final = (
    # Adverbs and nouns of immediacy.
    "right now", "right away", "immediately", "immediate", "as we speak",
    "at this moment", "this instant", "right this second", "just now",
    "urgent", "urgently", "emergency", "tonight", "happening now",
    "in progress", "any minute", "any moment", "any second", "hurry",
    "about to", "any longer", "cannot wait", "can not wait", "please help",
    "help me now", "call the police", "call an ambulance", "just broke",
    "just now", "still here", "still outside", "on his way", "on her way",
    "on their way", "on the way here", "coming over", "will be here",
    "in a few minutes", "in ten minutes", "in five minutes", "minutes away",
    "outside right now", "cannot reach", "can not reach", "cannot find",
    # Present-continuous and imminent-future tense.
    #
    # The first-person forms were missing, in a family whose messages are
    # overwhelmingly first-person: "is going to" was listed and "am going to"
    # was not, so *"I am going to jump"* carried peril with no immediacy.
    "is coming", "are coming", "am coming", "is closing", "is following",
    "is chasing", "is standing", "is holding", "is banging", "is breaking",
    "is running", "am running", "is hiding", "am hiding", "is screaming",
    "is going to", "are going to", "am going to", "is about to",
    "are about to", "am about to", "has been following", "keeps following",
    "keeps banging", "will come", "will not stop", "wont stop",
    "has decided to", "have decided to",
)

#: Peril that is immediate by its own description, needing no adverb.
#:
#: The conjunction exists because peril alone is usually a past report and
#: immediacy alone is usually impatience. But some peril *is* the present
#: tense: a swallowed bottle, a weapon in someone's hand, a door coming in.
#: Demanding that the writer also say "right now" asks them to restate what
#: they have already said, and people in these situations do not write
#: carefully. Every phrase here describes an event in progress, never a
#: capability or a topic -- "with a knife", never "knife".
#:
#: Composed from the classes above rather than restated, so the invariant that
#: matters -- every phrase here is also a peril phrase -- holds by construction
#: instead of by proofreading two lists against each other.
IN_PROGRESS_PHRASES: Final = (
    WIELDED_WEAPONS + UNFOLDING_STATES + UNFOLDING_SELF_HARM
)

PERIL: Final = _phrases(PERIL_PHRASES)
IMMEDIACY: Final = _phrases(IMMEDIACY_PHRASES)
IN_PROGRESS: Final = _phrases(IN_PROGRESS_PHRASES)

#: Violence as a verb and the person it lands on, a clause apart rather than
#: adjacent. ``governs_follows`` keeps the direction: in a report of violence
#: the victim comes after the verb.
DIRECTED_HARM: Final = Cue(
    _phrases(HARM_VERBS), _phrases(VICTIMS), governs_follows=True
)


class ImminentRiskDetector:
    """Escalate when a message reports peril that is happening now.

    Only the message is read. Peril quoted in evidence is somebody else's
    report of an event, and routing it to the quoted families keeps the
    escalation path meaningful; if that turns out to be wrong for genuine
    forwarded emergencies, the fix is a tier, not a wider vocabulary.
    """

    def __init__(
        self,
        peril: Pattern[str] = PERIL,
        immediacy: Pattern[str] = IMMEDIACY,
        in_progress: Pattern[str] = IN_PROGRESS,
        directed: Cue = DIRECTED_HARM,
    ) -> None:
        self._peril = peril
        self._immediacy = immediacy
        self._in_progress = in_progress
        self._directed = directed

    def inspect(self, view: RequestView) -> tuple[Observation, ...]:
        text = view.message.control_stripped
        peril_count = len(set(self._peril.findall(text))) + self._directed.hits(text)
        if not peril_count:
            return ()
        immediacy_hits = set(self._immediacy.findall(text))
        if not immediacy_hits and not self._in_progress.search(text):
            return ()

        # Corroboration, not calibration: one phrase of each is enough to
        # escalate, and further phrases only distinguish this from a weaker
        # observation if some later tier ever needs to compare them.
        corroboration = min(peril_count + len(immediacy_hits), 4)
        return (
            Observation(
                Concept.PERIL,
                origin=Origin.MESSAGE,
                confidence=0.5 + 0.125 * corroboration,
            ),
        )
