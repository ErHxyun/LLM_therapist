# src/reflection_validation.py

import json
import re

from src.local_llm.types import LLMTask
from src.utils.llm_client import llm_complete, llm_complete_task
from src.utils.llm_output_contracts import normalize_decision_output, parse_binary_decision
# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.session_event_logger import log_llm_event
logger = get_logger("ReflectionValidation")

# Prompt for the reasoner: checks if follow-up is related to topic or original response
RV_FOLLOW_UP_SYSTEM_REASONER_PROMPT = '''You are an intelligent agent who have strong reasoning capability and psychology and mental health commonsense knowledge. 
You are in a conversation session with a user. You need to evaluate if the user provide a follow-up response that's related to the original respone or the conversation topic. 


You will be provided with:
The conversattion topic, the original response, and the followup response in the format of '{"Topic": XXXX, "Original Response": XXXX, "Follow Up Response": XXXX}'

DECISION = 0 if the follow-up response is related to the "Original Response" or the "Topic". Otherwise, DECISION = 1.



Response format:
DECISION: 0/1

Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response. Don't output any open-ended questions or invitation for follow-up to the user.


Example 1:
{"Topic": Managing mood, "Original Response": I am sad recently."Follow Up Response": I am sad because I am homesick. I haven't been back home for a few years due to Covid-19. }
DECISION: 0

Example 2:
{"Topic": Family support, "Original Response": I don't feel my family is supportive. "Follow Up Response": I live away from my parents and family. We are in two different countries. We don't usually talk a lot. You know, they don't know what happened in my life and I don't know what's happening to them as well.}
DECISION: 0

Example 3:
{"Topic": Taking medication as prescribed, "Original Response": I don't want to follow the prescription., "Follow Up Response": " I have been trying to exercise more and eat healthier. I want to try and handle my symptoms naturally before resorting to medication."}
DECISION: 0

Example 4:
{"Topic": Participating primary and mental health care, "Original Response": I haven't gone to my prescriber for a long time. "Follow Up Response": I've been trying to pick up running as a hobby. I find it helps clear my mind and relieve stress. Plus, it's a great way to stay fit and healthy.}
DECISION: 1

Example 5:
{"Topic": Organizing personal possessions and doing housework, "Original Response": I never mop the floor. "Follow Up Response": Recently, I started learning how to cook. I'm trying to make dishes from different cuisines. Yesterday, I made pasta for the first time and it turned out really good.}
DECISION: 1
'''

RV_GUIDE_REDIRECT_TEMPLATE = (
    'Guide: Thank you for sharing that. I want to return to what you mentioned earlier: '
    '"{original_response}"{ending} Could you tell me more about that?'
)

RV_GUIDE_TOPIC_REDIRECT_TEMPLATE = (
    "Guide: Thank you for sharing that. I want to return to the current topic, "
    "{topic}. Could you tell me more about that?"
)

RV_GUIDE_PROFESSIONAL_HELP = (
    "Guide: I am concerned about your safety. This may need support from a qualified "
    "professional. If you are in immediate danger, please call emergency services now. "
    "If you are in the U.S., you can call or text 988 for crisis support."
)

_SERIOUS_RISK_PATTERN = re.compile(
    r"("
    r"\bkill myself\b|"
    r"\bend my life\b|"
    r"\bwant to die\b|"
    r"\bsuicid(?:e|al)\b|"
    r"\bself[-\s]?harm\b|"
    r"\bhurt myself\b|"
    r"\bharm myself\b|"
    r"\boverdose\b|"
    r"\bkill someone\b|"
    r"\bhurt someone\b|"
    r"\bharm others\b|"
    r"\bhurt others\b|"
    r"\bcan(?:not|'t) breathe\b|"
    r"\bchest pain\b|"
    r"\bseizure\b|"
    r"\bstroke\b"
    r")",
    re.IGNORECASE,
)

# Prompt for validation: provides empathic validation and support
RV_FOLLOW_UP_VALIDATION_SYSTEM_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are in the conversation with a client. You need to provide empathic validation and support to the client.

You will be provided with:
1. The conversattion topic.
2. The original response from the client.
3. The follow-up response from the client to the question 'Can you tell me more about it?'.
These infromation will be provided in the format of '{"Topic": XXXX, "Original Response": XXXX, "Follow-up Response": XXXX}'

Goal:
You need to provide empathic validation and support to the client based on the conversation topic, origianl response, and followup response.
You need to first express the understanding to the client's follow-up response, and then try to lead the client to the right direction.
Don't read into the clients' mind and make too much assumptions. Try to use the phrases used by the client in your response, instead of rephrasing too much.
Don't output any open-ended questions or invitation for follow-up to the user.

Formatting constraints (very important):
- Use ASCII characters only. Do not output smart quotes or special dashes.
- Replace any of these with ASCII equivalents: ‘ ’ “ ” … • – — ‑ → ' " ... * -
- Use the ASCII hyphen '-' for hyphenated words (e.g., write 'mid-afternoon', 'check-in').
- Avoid non-breaking hyphen (U+2011) and en/em dashes entirely.

Response format:
VALIDATION: xxxx

Example 1:
{"Topic": Maintaining stable weight, "Original Response": My weight increased a lot recently. "Follow-up Response": I am a ISFP. I like to follow my heart.My personality leads me to just eat whenever I want. And I usually don’t control how much I eat.}
VALIDATION: I completely understand that as an ISFP, rules may not be your preference. Nonetheless, it's crucial to be aware of how your eating habits affect your health. If you've gained weight and believe your eating habits are a factor, consider adjusting your diet and lifestyle. Seeking guidance from a healthcare professional or dietitian can assist in creating a tailored plan for a healthier lifestyle.

Example 2:
{"Topic": Maintaining stable weight, "Original Response": My weight increased a lot recently. "Follow-up Response":  I have upcoming deadlines. So I often do stress eating.}
VALIDATION: It sounds like you’re facing pressure from deadlines, leading to stress eating and weight gain. This is a common response to stress, but you can manage it by identifying triggers, opting for healthy snacks, and maintaining regular meal times. Talking to friends, family, or a therapist can also provide support and help you find healthier coping strategies. 

Example 3:
{"Topic": Managing mood, "Original Response": I am sad recently. "Follow-up Response": "My sadness stems from a variety of factors. I've been dealing with a lot of stress at work, and it's been affecting my mood and overall well-being. I'm also finding it hard to connect with my friends due to the pandemic. This isolation has been making me feel quite depressed."}
VALIDATION: It sounds like you're going through a really challenging time. The stress from work, coupled with the isolation brought on by the pandemic, can significantly impact your mental health. It's important to acknowledge these feelings and understand that it's okay to feel overwhelmed by these circumstances. During pandemic, staying connected with friends and family can be challenging, but you can try to schedule regular video calls or join online communities or groups with similar interests. It may also be beneficial to talk about your feelings with someone you trust. Whether it's a friend, family member, or a mental health professional, sharing your experiences can provide relief and offer perspectives that might help you cope better.

'''

RV_VALIDATION_PROFESSIONAL_HELP = (
    "VALIDATION: I am concerned about your safety. This may need support from a qualified "
    "professional. If you are in immediate danger, please call emergency services now. "
    "If you are in the U.S., you can call or text 988 for crisis support."
)

RV_VALIDATION_BOUNDED_SYSTEM_PROMPT = '''You are the CaiTI Reflection-Validation Validator.
You speak to the client after the R-V Reasoner has decided their follow-up response is related to the original response/topic.

Input format:
{"Topic": XXXX, "Original Response": XXXX, "Follow-up Response": XXXX}

Goal:
Provide brief empathic validation and support using Motivational Interviewing style.

Required behavior:
1. Mirror the client's follow-up using their own words.
2. Briefly connect it to the original response or topic.
3. Include affective reflection only for emotion words the client explicitly stated.
4. Include one affirmation or supportive sentence.

Boundaries:
- Do not ask questions.
- Do not diagnose.
- Do not provide clinical, medical, diet, exercise, medication, or treatment advice.
- Do not propose coping strategies or action plans.
- Do not infer unstated mood, personality, causes, or risk.
- Do not say "you feel", "it sounds like", or "it seems like" unless the same emotion word appears in the input.
- Do not use emotion words such as overwhelmed, anxious, depressed, sad, stressed, worried, or afraid unless that word appears in the input.
- If no emotion is explicitly stated, use simple reflection and supportive validation without naming an emotion.
- Keep it concise: 2-3 sentences.
- Output ASCII only.

Response format:
VALIDATION: xxxx
'''

def parse_rv_decision(text: str, default: str = "1") -> str:
    """
    Parse an R-V reasoner decision.
    Adapter raw output is expected to be 0/1; older prompts may return DECISION: 0/1.
    """
    return parse_binary_decision(text, default=default)

def _format_rv_reasoner_input(topic: str, original_response: str, follow_up_response: str) -> str:
    return json.dumps(
        {
            "Topic": topic,
            "Original Response": original_response,
            "Follow Up Response": follow_up_response,
        }
    )

def _format_rv_validation_input(topic: str, original_response: str, follow_up_response: str) -> str:
    return (
        f'{{"Topic": {topic!r}, "Original Response": {original_response!r}, '
        f'"Follow-up Response": {follow_up_response!r}}}'
    )

def _clean_redirect_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().strip('"')

_EMOTION_WORDS_REQUIRING_EXPLICIT_INPUT = (
    "overwhelmed",
    "anxious",
    "depressed",
    "sad",
    "stressed",
    "worried",
    "afraid",
)


def _remove_unstated_emotion_sentences(sentences: list[str], source_texts: tuple[str, ...]) -> list[str]:
    context = " ".join(str(text or "").lower() for text in source_texts)
    if not context:
        return sentences

    kept = []
    for sentence in sentences:
        lower = sentence.lower()
        has_unstated_emotion = any(
            emotion in lower and emotion not in context
            for emotion in _EMOTION_WORDS_REQUIRING_EXPLICIT_INPUT
        )
        if not has_unstated_emotion:
            kept.append(sentence)
    return kept


def clean_rv_validation_text(text: str, source_texts: tuple[str, ...] = ()) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "VALIDATION: I hear what you shared, and I appreciate you explaining more."

    match = re.search(
        r"VALIDATION:\s*(.+?)(?:\n\n|\nVALIDATION:|\nGuide:|$)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        raw = match.group(1).strip()

    hard_stops = [
        "IMPORTANT:",
        "INVALID:",
        "VALID SCORE",
        "SCORE:",
        "Now respond",
        "Correct -",
        "Correct:",
        "Note:",
        "Follow-up RESPONSE",
        "Output is invalid",
    ]
    lower = raw.lower()
    cut = None
    for marker in hard_stops:
        pos = lower.find(marker.lower())
        if pos != -1:
            cut = pos if cut is None else min(cut, pos)
    if cut is not None:
        raw = raw[:cut].strip()

    raw = re.sub(r"^(VALIDATION|validation)\s*:\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip()

    question_pos = raw.find("?")
    if question_pos != -1:
        raw = raw[:question_pos].rstrip(". ") + "."

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", raw) if sentence.strip()]
    sentences = _remove_unstated_emotion_sentences(sentences, source_texts)
    if len(sentences) > 3:
        sentences = sentences[:3]
    raw = " ".join(sentences).strip()

    if raw and raw[-1] not in ".!?":
        raw += "."
    return f"VALIDATION: {raw}" if raw else "VALIDATION: I hear what you shared, and I appreciate you explaining more."

def _has_serious_risk(*texts: str) -> bool:
    return any(_SERIOUS_RISK_PATTERN.search(str(text or "")) for text in texts)

def _build_rv_guide_redirect(topic: str, original_response: str) -> str:
    original = _clean_redirect_text(original_response)
    if original:
        ending = "" if original[-1] in ".!?" else "."
        return RV_GUIDE_REDIRECT_TEMPLATE.format(original_response=original, ending=ending)

    topic_text = _clean_redirect_text(topic) or "the current topic"
    return RV_GUIDE_TOPIC_REDIRECT_TEMPLATE.format(topic=topic_text)

def rv_reasoner(topic: str, original_question: str, original_response: str, follow_up_response: str) -> str:
    """
    Use the reasoner prompt to determine if the follow-up is related to the topic or original response.
    Returns the decision as a string.
    """
    logger.info("Running reflection validation reasoner.")
    payload = _format_rv_reasoner_input(topic, original_response, follow_up_response)
    raw = llm_complete_task(
        LLMTask.TASK3_RV_REASONER,
        RV_FOLLOW_UP_SYSTEM_REASONER_PROMPT,
        payload,
        max_new_tokens=8,
    ).text
    contract = normalize_decision_output(raw, default="1")
    decision = contract.decision
    normalized = contract.normalized_output
    log_llm_event(
        task=LLMTask.TASK3_RV_REASONER,
        dimension=topic,
        score=decision,
        segment_text=follow_up_response,
        question_text=original_question,
        raw_llm_output=raw,
        normalized_output=normalized,
        metadata={"original_response": original_response, "reasoner_payload": payload},
    )
    logger.debug(
        "R-V reasoner normalized decision: %s (raw=%s, topic=%s, original_question=%s)",
        decision,
        raw,
        topic,
        original_question,
    )
    return normalized

def rv_guide(topic: str, original_question: str, original_response: str, follow_up_response: str) -> str:
    """
    Redirect the user back to the original score-2 response after an invalid follow-up.
    This module intentionally avoids clinical advice; severe risk routes to professional help.
    """
    logger.info("Running reflection validation guide.")
    if _has_serious_risk(original_response, follow_up_response):
        guide = RV_GUIDE_PROFESSIONAL_HELP
    else:
        guide = _build_rv_guide_redirect(topic, original_response)

    log_llm_event(
        task="rv_guide",
        dimension=topic,
        segment_text=follow_up_response,
        question_text=original_question,
        raw_llm_output=guide,
        normalized_output=guide,
        metadata={
            "original_response": original_response,
            "mode": "professional_help" if guide == RV_GUIDE_PROFESSIONAL_HELP else "redirect",
        },
    )
    logger.debug("R-V guide output: %s (topic=%s, original_question=%s)", guide, topic, original_question)
    return guide

def rv_validation(topic: str, original_question: str, original_response: str, follow_up_response: str) -> str:
    """
    Generate bounded affective reflection and affirmation after a valid follow-up.
    Severe risk routes to professional help.
    """
    logger.info("Running reflection validation support/validation.")
    if _has_serious_risk(original_response, follow_up_response):
        validation = RV_VALIDATION_PROFESSIONAL_HELP
        raw = validation
        mode = "professional_help"
    else:
        payload = _format_rv_validation_input(topic, original_response, follow_up_response)
        raw = llm_complete(RV_VALIDATION_BOUNDED_SYSTEM_PROMPT, payload)
        validation = clean_rv_validation_text(raw, source_texts=(original_response, follow_up_response))
        mode = "bounded_generation"

    log_llm_event(
        task="rv_validation",
        dimension=topic,
        segment_text=follow_up_response,
        question_text=original_question,
        raw_llm_output=raw,
        normalized_output=validation,
        metadata={
            "original_response": original_response,
            "mode": mode,
        },
    )
    logger.debug(
        "R-V validation output: %s (topic=%s, original_question=%s)",
        validation,
        topic,
        original_question,
    )
    return validation
