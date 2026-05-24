# src/text_generators.py
from src.utils.llm_client import llm_complete

from src.utils.log_util import get_logger
logger = get_logger("TextGenerators")

# The following functions generate prompts and call OpenAI's API to generate various types of text transformations.
# Each function is commented to explain its purpose and logic.

def generate_prompt_synonymous_sentences(user_input):
    """
    Generate a prompt for the model to rephrase a screening question.

    This must preserve question form because the output is spoken directly to
    the user by the voice shell.
    """
    return """Rephrase the screening question while preserving its meaning.

Rules:
- Return one question only.
- Keep it as a question ending in a question mark.
- Do not answer the question.
- Do not add labels, prefixes, examples, arrows, or explanations.

Question: Have you experienced significant weight change recently?
Rephrased question: Have you noticed any significant changes in your weight recently?
Question: How has your mood been? Have you been able to manage your moods?
Rephrased question: How has your mood been lately, and have you been able to manage it?
Question: How's your eating? Are you eating regularly?
Rephrased question: How have your eating habits been, and have you been eating regularly?
Question: {}
Rephrased question:""".format(
        str(user_input).strip()
    )

def _strip_rewrite_prefix(text: str) -> str:
    candidate = str(text or "").strip().strip("\"'")
    prefixes = (
        "answer:",
        "rephrased question:",
        "question:",
        "output:",
        "rewrite:",
        "rewritten question:",
    )
    changed = True
    while changed:
        changed = False
        lowered = candidate.lower().lstrip()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                candidate = candidate.lstrip()[len(prefix):].strip().strip("\"'")
                changed = True
                break
    return candidate

def clean_question_rewrite(raw: str) -> str:
    """
    Extract the first plausible user-facing question from model output.
    """
    text = str(raw or "").strip()
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    candidate = lines[0] if lines else text
    if "=>" in candidate:
        candidate = candidate.split("=>", 1)[0].strip()
    candidate = _strip_rewrite_prefix(candidate)

    if "?" in candidate:
        candidate = candidate[: candidate.find("?") + 1]
    return " ".join(candidate.split()).strip()

def is_valid_question_rewrite(candidate: str) -> bool:
    """
    Reject rewrite artifacts that would turn a screening question into an answer.
    """
    text = " ".join(str(candidate or "").split()).strip()
    if not text or "?" not in text:
        return False
    lowered = text.lower()
    forbidden_fragments = (
        "=>",
        "answer:",
        "user:",
        "rephrased question:",
        "output:",
    )
    if any(fragment in lowered for fragment in forbidden_fragments):
        return False
    answer_starts = (
        "i am ",
        "i'm ",
        "i have ",
        "i haven't ",
        "i do ",
        "i don't ",
        "my ",
        "we ",
    )
    if lowered.startswith(answer_starts):
        return False
    return True

def generate_synonymous_sentences(question_text):
    """
    Generate a safe paraphrase for a question, falling back to the original.
    """
    original_question = " ".join(str(question_text or "").split()).strip()
    
    raw = llm_complete(
        "You rephrase screening questions. Return exactly one question and nothing else.",
        generate_prompt_synonymous_sentences(original_question)
    )
    results = clean_question_rewrite(raw)
    if not is_valid_question_rewrite(results):
        logger.warning(
            "Question rewrite rejected; falling back to original. raw=%r cleaned=%r original=%r",
            raw,
            results,
            original_question,
        )
        results = original_question
    logger.info(f"generate_synonymous_sentences: {results}")
    return results

def generate_prompt_therapist(user_input):
    """
    Generate a prompt for the model to act as a therapist in a conversation.
    The prompt provides several example exchanges and then asks the model to respond to the user's input.
    """
    return """Chat with people as a therapist.

    User: I feel so depressed daily.
    Answer: I am so sorry to hear that. It's OK to feel a little bit depressed but you need to figure out a way to makes you feel better. You can talk to a friend or family member. Or you can reach out to a therapist. And I am always here to support you.
    User: I don't want to talk.
    Answer: I get that you don’t want to have this conversation. But it's important to share your feelings with others and find out ways to make you feel better. 
    User: My partner wants to check my messages everyday.
    Answer: When you having a controlling partner, you might want to know the following items. Understand Controlling Personality Types. Recognize the Part of You That Accepts Another's Control. Take Back Responsibility for Your Life. Decide Whether You Need or Want Controlling Men in Your Life. Know What You Want Out of Life. Learn and Practice Assertiveness. Set Healthy Boundaries.
    User: I don't know what's going on with me.
    Answer: It's fine not to know the reason why you don't feel well now. Doing medication might help you understand yourself better. Or you can reach out to your family members, friends, or therapist to help you out.
    User:{}
    Answer:""".format(
        user_input.capitalize()
    )

def generate_therapist_chat(user_input):
    """
    Use OpenAI API to generate a therapist-like response to the user's input.
    """
    result = llm_complete(
        "Chat with people as a virtual AI therapist.",
        generate_prompt_therapist(user_input)
    )
    logger.info(f"generate_therapist_chat: {result}")
    return result

def generate_prompt_change(user_input):
    """
    Generate a prompt for the model to convert a first-person sentence to a second-person sentence.
    The prompt provides several examples and then asks the model to convert the user's input.
    """
    return """　Change from first-person sentence to second-person.

    User: I feel so depressed daily.
    Answer: You feel so depressed daily.
    User: I am so happy.
    Answer: You are so happy.
    User: I am under a lot of pressure.
    Answer: You are under a lot of pressure.
    User:{}
    Answer:""".format(
        user_input.capitalize()
    )

def generate_change(user_input):
    """
    Use OpenAI API to convert a first-person sentence to a second-person sentence.
    """
    resp = llm_complete(
        "Convert first-person to second-person statements.",
        generate_prompt_change(user_input)
    )
    logger.debug(resp)
    return resp

def generate_prompt_change_positive(user_input):
    """
    Generate a prompt for the model to convert a question to a positive declarative sentence.
    The prompt provides several examples and then asks the model to convert the user's input.
    """
    return """　Change from question to positive declarative sentence.

    User: Do you have coping skills to help you calm down.
    Answer: You have coping skills to help you calm down.
    User: Do you have self-harming behaviours?
    Answer: You have self-harming behaviours.
    User: Are you involved in any legal issues recently?
    Answer: You are involved in some legal issues recently.
    User:{}
    Answer:""".format(
        user_input.capitalize()
    )

def generate_change_positive(user_input):
    """
    Use OpenAI API to convert a question to a positive declarative sentence.
    """
    resp = llm_complete(
        "Turn a question into a positive declarative sentence.",
        generate_prompt_change_positive(user_input)
    )
    logger.debug(resp)
    return resp

def generate_prompt_change_negative(user_input):
    """
    Generate a prompt for the model to convert a question to a negative declarative sentence.
    The prompt provides several examples and then asks the model to convert the user's input.
    """
    return """　Change from question to negative declarative sentence.

    User: Do you have coping skills to help you calm down.
    Answer: You don't have coping skills to help you calm down.
    User: Do you feel productive?
    Answer: You don't feel productive.
    User: Have you done anything creative recently?
    Answer: You haven't done anything creative recently.
    User:{}
    Answer:""".format(
        user_input.capitalize()
    )

def generate_change_negative(user_input):
    """
    Use OpenAI API to convert a question to a negative declarative sentence.
    """
    resp = llm_complete(
        "Turn a question into a negative declarative sentence.",
        generate_prompt_change_negative(user_input)
    )
    logger.debug(resp)
    return resp
