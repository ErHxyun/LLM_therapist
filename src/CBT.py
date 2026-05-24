import re

from src.local_llm.types import LLMTask
from src.session.control import NullSessionControl
from src.utils.llm_client import llm_complete, llm_complete_task
from src.utils.llm_output_contracts import normalize_decision_output, parse_binary_decision

# Set up logger for this module
from src.utils.log_util import get_logger
from src.utils.io_record import get_resp_log, log_question, log_system_message, set_question_prefix
from src.utils.session_event_logger import log_llm_event
logger = get_logger("CBT")


PROMPTER_CBT_STAGE0_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are reviewing the therapy session history and trying to ask the patient to choose a dimension that he/she would like to work on through this CBT process.
Only choose those dimensions that received a score of 2 in the conversation history.
Response format:
QUESTION: xxxx
'''

REASONER_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.



You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The user's answer towards the CBT question "Can you try to identify any unhelpful thoughts you have that contribute to this situation?". This is the step that the patient tries to recognize negative thoughts. These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.

You will be provideed with several examples in the format of STATEMENT: xxxxx; UNHELPFUL_THOUGHTS: xxxxx;


Usually the patient's statement and responses contain situation that is not valid or useful. As an AI assistant, you need to examine the validaity and utility of the patient's response.
There are 13 possible common cognitive distortions that the patient might encounter. And you might want to pay attention to.
1. Filtering: focusing on the negative but ignore the positive
2. Polarized thinking/extreme thinking: seeing everything in all-or-nothing terms.
3. Control fallacies: assumes only self or other takes all the responsibility and is to be blamed. Includes personalization (assuming self is responsible) and blaming (assuming others at fault). 
4. Fallacy of fairness: assumes life should be fair
5. Overgeneralization: assumes a rule from one experience, using one experience for all future experiences. 
6. Emotional reasoning: “if I feel it, it must be true.” Using emotional “terms” for all the situations. 
7. Fallacy of change: expects others to change
8. “shoulds”: using personal rules to judge self and others if the rules broken
9. Catastrophizing: expecting the worst case scenario.
10. Heaven’s reward fallacy: expecting to be rewarded in some way.
11. Always being right: being wrong is unacceptable, needs to be right all the time. 
12. Personalization (like control fallacies): assuming self is responsible.
13. Jumping to conclusions: make assumptions based on little evidence


If any of these cognitive distortion is included in the UNHELPFUL_THOUGHTS, the user may still properly identifies the unhelpful thoughts. But outline the cognitive distortions in analysis.

Your goal is:
Justify if the user is identify the unhelpful thoughts properly in the statement(0: identified properly, 1: not properly identified).
You also need to provide analysis to justify your decision. 


Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.





Example 1:
"STATEMENT: I have concern with your recent spending habits. I spent a lot of money. I spent a lot of money on clothing.; UNHELPFUL_THOUGHTS: I have issue on spending habits because I buy too much clothes.;"
DECISION: 0


Example 2:
"STATEMENT: I haven't done any creative work recently. I just don't know what are the creative things I can do.; UNHELPFUL THOUGHTS: I'm just not a creative person. I don't have any good ideas, and even if I did, they wouldn't be worth pursuing. "
DECISION:0


Example 3:
"STATEMENT: I have concern with my recent spending habits. I spent a lot of money. I spent a lot of money on clothing. RESPONSE: I like to go shopping "
DECISION:1

'''

REASONER_CBT_STAGE2_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
3. The patient's answer to the CBT question "Can you challenge your thought?". This is the step when the patient begin to challenge the UNHELPFUL_THOUGHTS in the STATEMENT after recognizing and analyzing these thoughts. Challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?

You will be provideed with several examples in the format of STATEMENT: xxxxx; UNHELPFUL_THOUGHTS: xxxxx; CHALLENGE: xxxx;

Usually the patient's statement and responses contain situation that is not valid or useful. As an AI assistant, you need to examine the validaity and utility of the patient's response.
There are 13 possible common cognitive distortions that the patient might encounter. And you might want to pay attention to.
1. Filtering: focusing on the negative but ignore the positive
2. Polarized thinking/extreme thinking: seeing everything in all-or-nothing terms.
3. Control fallacies: assumes only self or other takes all the responsibility and is to be blamed. Includes personalization (assuming self is responsible) and blaming (assuming others at fault). 
4. Fallacy of fairness: assumes life should be fair
5. Overgeneralization: assumes a rule from one experience, using one experience for all future experiences. 
6. Emotional reasoning: “if I feel it, it must be true.” Using emotional “terms” for all the situations. 
7. Fallacy of change: expects others to change
8. “shoulds”: using personal rules to judge self and others if the rules broken
9. Catastrophizing: expecting the worst case scenario.
10. Heaven’s reward fallacy: expecting to be rewarded in some way.
11. Always being right: being wrong is unacceptable, needs to be right all the time. 
12. Personalization (like control fallacies): assuming self is responsible.
13. Jumping to conclusions: make assumptions based on little evidence


Your goal is:
Justify if the patient challenges the unhelpful thoughts (UNHELPFUL_THOUGHTS) properly. (0: properly challenge the unhelpful thoughts, 1: not challenge the unhelpful thoughts)
Note that:
1. The patient might identify the unhelpful thoughts in a wrong way (with cognitive distortions). In this case, the patient might challenge the STATEMENT or some unhelpful thoughts that related to this STATEMENT that is not explicitly identified, which is acceptable.
2. It would be acceptable if the patient not fully challenge the validity and usability of the unhelpful thoughts/situation. As long as the CHALLENGE is related to the STATEMENT and UNHELPFUL THOUGHTS, it is acceptable.
Make notes about the distortions in the analysis.
You also need to provide analysis to justify your decision. 


Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.



Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself.; CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?;"
DECISION: 0


Example 2:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself.; CHALLENGE: I don't know how to challenge my thoughts;"
DECISION: 1

Example 3:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself.; CHALLENGE: I am not cool to engage in the social events;"
DECISION: 1

Example 4:
"STATEMENT: I don't smoke cigarettes, but I vape every day. I vape when I am working hard or debugging.; UNHELPFUL_THOUGHTS: I can't work or solve problems effectively without vaping.; CHALLENGE: There are several potential health impacts of vaping. It's bad for my lung.;"
DECISION: 0

'''

REASONER_CBT_STAGE3_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
3. The patient's response to challenge the UNHELPFUL_THOUGHTS after recognizing and analyzing these thoughts. Challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?
4. The patient's answer to the CBT question "What is another way of thinking about this situation?". This is the step that the patient tires to reframe your unhelpful thoughts into more balanced, realistic, and constructive ones. This process is about changing the way the patient thinks about the situation, which can lead to changes in emotions and behaviors.


You will be provideed with several examples in the format of STATEMENT: xxxxx; UNHELPFUL_THOUGHTS: xxxxx; CHALLENGE: xxxxx; REFRAME: xxxxx;

Your goal is:
Justify if the patient reframes the unhelpful thoughts properly. (0: properly reframe the unhelpful thoughts, 1: fail to reframe the unhelpful thoughts).
You also need to provide analysis to justify your decision. 


Response format:
DECISION: 0/1
Provide response with [DECISION] only. Do not put excessive analysis and small talk in the response.




Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'; REFRAME: People may have their own concerns and may not be focused on me all the time and I've had good conversations in the past without embarrassing myself.;"
DECISION: 0

Example 2:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say.; UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'; REFRAME: I'm finding it hard to reframe them. I still believe that: 'People are definitely judging me; I just know it.;"
DECISION: 1
'''

# Rephrase recap for Stage 3 based on user's CHALLENGE
RECAP_CBT_STAGE3_CHALLENGE_PROMPT = '''You are a concise and supportive therapist-assistant.

You will be provided with:
1) The patient's brief STATEMENT of the situation
2) The patient's identified UNHELPFUL_THOUGHTS
3) The patient's CHALLENGE to those thoughts

Your task is to rephrase the patient's CHALLENGE into a short recap that reminds the patient what they already did to challenge their thoughts. Be neutral, supportive, and concise.

Rules:
- 1-2 sentences only
- No extra headers or labels, output the recap directly
- Do not add new ideas beyond the user's content
- Use second-person neutral tone (you/your)
'''

GUIDE_CBT_STAGE1_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to answer the cognitive behavioural therapy (CBT) questions based-on patient's statement provided.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.


Your goal is:
Try to recognize negative thoughts. These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic. You are trying to understand the patient's unhelpful thoughts, , so please answer the questions using the second person.


Response format:
UNHELPFUL_THOUGHTS: xxxx

You will be provideed with several examples with the statement and example unhelpful thoughts in the format of "STATEMENT: xxxxx, UNHELPFUL_THOUGHTS: xxxxxx". 



Example 1:
STATEMENT: I have not taken days off recently. Paper deadline is coming up! I don't even have time to sleep. 
UNHELPFUL_THOUGHTS: Your unhelpful thoughts might be taking days off will hinder your progress on meeting the paper deadline.

Example 2:
STATEMENT: I don't chat a lot with my colleagues. I can talk to them about work, but I can't talk to them about life. I can't seem to find common ground for life conversations with them. My personal life is quite dull and lacks the variety of personal and family activities that they have.
UNHELPFUL_THOUGHTS: Everyone will think you are boring so you don't chat with your colleagues might be your unhelpful thought.
'''

GUIDE_CBT_STAGE2_PROMPT  = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.

Your goal is:
Try to help the patient challenge the unhelpful thoughts (UNHELPFUL_THOUGHTS) properly. After recognizing and analyzing these UNHELPFUL_THOUGHTS, challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?

Response format:
CHALLENGE: xxxx

You will be provideed with several examples with the statement and example unhelpful thoughts in the format of "STATEMENT: xxxxx. UNHELPFUL_THOUGHTS: xxxxxx. CHALLENGE: xxxxxx". 



Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say. UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. 
CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'"


Example 2:
"STATEMENT: My issue is procrastination, especially when it comes to completing assignments for work or school. UNHELPFUL_THOUGHTS: When faced with a task I need to complete, I often have negative thoughts like: 'I'll never finish this on time; I'm so lazy.' 
CHALLENGE: asking myself: I have successfully complete a similar fairly challenging school project before. I might not be fair to label myself as lazy just because I'm struggling with this task."

'''

GUIDE_CBT_STAGE3_PROMPT = '''You are an AI assistant who has rich psychology and mental health commonsense knowledge and strong reasoning abilities.
You are trying to justify if the patient is effectively going through and responding to cognitive behavioural therapy (CBT) questions.

You will be provided with:
1. The statement of the patient towards one day-to-day functioning issue or mental health issue that he/she would like to work on through this CBT process.
2. The patient's response to recognize unhelpful thoughts in his/her statement (UNHELPFUL_THOUGHTS). These thoughts that go through the patient's mind when he/she experience this issue. These thoughts can be self-critical, overly pessimistic, or unrealistic.
3. The patient's response to challenge the UNHELPFUL_THOUGHTS after recognizing and analyzing these thoughts. Challenge means questioning the validity of these thoughts. Are there alternative, more balanced, or rational thoughts that might be more helpful in the situation?
4. The patient's response to reframe the UNHELPFUL_THOUGHTS into more balanced, realistic, and constructive ones. This process is about changing the way the patient thinks about the situation, which can lead to changes in emotions and behaviors.


Your goal is:
Try to reframe the unhelpful thoughts (UNHELPFUL_THOUGHTS) for the patient. This is the step to reframe the patient's unhelpful thoughts into more balanced, realistic, and constructive ones. 


Response format:
REFRAME: xxxx




Example 1:
"STATEMENT: I don't participate in community. I get anxious when there are a lot of people around me. I don't know what to say. UNHELPFUL_THOUGHTS: When I'm in a social situation, I often have negative thoughts like: (1) Everyone is judging me and (2) I'll say something stupid and embarrass myself. CHALLENGE: I can challenge these negative thoughts by asking myself: 'Is there any real evidence that people are constantly judging me? and have there been times when people genuinely seemed interested in talking to me?'."
REFRAME: People may have their own concerns and may not be focused on you all the time. You may had good conversations in the past without embarrassing your self.

Example 2:
"STATEMENT: I often avoid speaking up in meetings at work or in front of others. I’m afraid my ideas aren’t good enough. UNHELPFUL_THOUGHTS: If I speak up, people will think my ideas are silly. Others are much smarter than me, so my opinion doesn’t matter. CHALLENGE: I can challenge these thoughts by asking: ‘Have my colleagues ever reacted negatively when I spoke before?’ and ‘Do people usually respect different opinions, even if they’re not perfect?’."
REFRAME: My ideas have value, and sharing them can contribute to the discussion. Others are likely focused on the topic, not on judging me, and speaking up can help me grow more confident.
'''

BOUNDED_CBT_STAGE1_GUIDE_PROMPT = '''You are the CaiTI CBT Stage 1 Guide.
The Stage 1 Reasoner decided the client's response did not yet identify an unhelpful thought.

Input format:
STATEMENT: xxxx

Task:
Generate a brief scaffold that helps the client identify their own unhelpful thought about the statement.

Boundaries:
- Do not write the unhelpful thought for the client.
- Do not diagnose.
- Do not provide clinical advice or action plans.
- Do not ask what the statement means.
- Focus only on the thought, belief, or self-talk that came up for the client.
- Do not introduce new interpretations, causes, emotions, or examples.
- Ask the client to answer in their own words.
- Prefer asking for "one unhelpful thought" directly.
- 1-2 sentences only.

Response format:
GUIDE: xxxx
'''

BOUNDED_CBT_STAGE2_GUIDE_PROMPT = '''You are the CaiTI CBT Stage 2 Guide.
The Stage 2 Reasoner decided the client's response did not yet challenge the unhelpful thought.

Input format:
STATEMENT: xxxx
UNHELPFUL_THOUGHTS: xxxx

Task:
Generate a brief scaffold that helps the client challenge their own unhelpful thought.

Boundaries:
- Do not write the challenge for the client.
- Do not diagnose.
- Do not provide clinical advice or action plans.
- You may mention checking whether the thought is always true, evidence for/against it, or a more balanced view.
- Keep the scaffold process-focused; do not provide content-specific evidence or a ready-made challenge.
- Do not introduce new interpretations, causes, emotions, examples, or advice.
- Do not ask for examples or past situations; ask for the challenge itself.
- Ask the client to answer in their own words.
- 1-2 sentences only.

Response format:
GUIDE: xxxx
'''

BOUNDED_CBT_STAGE3_GUIDE_PROMPT = '''You are the CaiTI CBT Stage 3 Guide.
The Stage 3 Reasoner decided the client's response did not yet reframe the unhelpful thought.

Input format:
STATEMENT: xxxx
UNHELPFUL_THOUGHTS: xxxx
CHALLENGE: xxxx

Task:
Generate a brief scaffold that helps the client reframe their own unhelpful thought into a more balanced, realistic, constructive thought.

Boundaries:
- Do not write the reframe for the client.
- Do not diagnose.
- Do not provide clinical advice or action plans.
- Do not provide a possible balanced thought, conclusion, reassurance, or new interpretation.
- Refer to the user's challenge only as material they can use; do not complete the reframe for them.
- Do not introduce new interpretations, causes, emotions, examples, or advice.
- Do not ask for examples or other situations; ask for one balanced sentence in the client's own words.
- Do not mention worth, value, competence, identity, or self-esteem unless the client already used those words.
- Ask the client to answer in their own words.
- 1-2 sentences only.

Response format:
GUIDE: xxxx
'''

ADAPTER_CBT_STAGE1_PROMPT = REASONER_CBT_STAGE1_PROMPT
ADAPTER_CBT_STAGE2_PROMPT = REASONER_CBT_STAGE2_PROMPT
ADAPTER_CBT_STAGE3_PROMPT = REASONER_CBT_STAGE3_PROMPT

CBT_PROFESSIONAL_HELP_MESSAGE = (
    "This part seems difficult right now. We will end CBT for now, and it may be "
    "helpful to seek support from a mental health professional for a more effective "
    "CBT experience."
)


def parse_cbt_decision(raw_text: str, default: str = "1") -> str:
    return parse_binary_decision(raw_text, default=default)


_NUMBER_WORDS = {
    "one": 1,
    "first": 1,
    "two": 2,
    "second": 2,
    "three": 3,
    "third": 3,
    "four": 4,
    "fourth": 4,
    "five": 5,
    "fifth": 5,
    "six": 6,
    "sixth": 6,
    "seven": 7,
    "seventh": 7,
    "eight": 8,
    "eighth": 8,
    "nine": 9,
    "ninth": 9,
    "ten": 10,
    "tenth": 10,
    "eleven": 11,
    "twelfth": 12,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS_NUMBER_WORDS = {"twenty": 20, "thirty": 30}


def extract_cbt_choice_number(answer: str) -> int | None:
    answer = str(answer or "").strip().lower()
    digit_match = re.search(r"\d+", answer)
    if digit_match:
        return int(digit_match.group(0))

    words = re.findall(r"[a-z]+", answer.replace("-", " "))
    for index, word in enumerate(words[:-1]):
        if word in _TENS_NUMBER_WORDS and words[index + 1] in _NUMBER_WORDS:
            return _TENS_NUMBER_WORDS[word] + _NUMBER_WORDS[words[index + 1]]
    for word in words:
        if word in _NUMBER_WORDS:
            return _NUMBER_WORDS[word]
    for word in words:
        if word in _TENS_NUMBER_WORDS:
            return _TENS_NUMBER_WORDS[word]
    return None


def _format_stage1_input(statement: str, unhelpful_thoughts: str) -> str:
    return f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts};"'


def _format_stage2_input(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    return (
        f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; '
        f'CHALLENGE: {challenge};"'
    )


def _format_stage3_input(
    statement: str,
    unhelpful_thoughts: str,
    challenge: str,
    reframe: str,
) -> str:
    return (
        f'"STATEMENT: {statement}; UNHELPFUL_THOUGHTS: {unhelpful_thoughts}; '
        f'CHALLENGE: {challenge}; REFRAME: {reframe};"'
    )


def _chat_complete(system_content: str, user_content: str):
    return llm_complete(system_content, user_content)

def clean_cbt_guide_text(text: str, fallback: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return fallback

    match = re.search(r"GUIDE:\s*(.+?)(?:\n\n|\nGUIDE:|$)", raw, flags=re.IGNORECASE | re.DOTALL)
    if match:
        raw = match.group(1).strip()
    raw = re.sub(r"^(GUIDE|Guide|guide)\s*:\s*", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", raw) if sentence.strip()]
    if len(sentences) > 2:
        raw = " ".join(sentences[:2]).strip()
    if raw and raw[-1] not in ".!?":
        raw += "."
    return f"GUIDE: {raw}" if raw else fallback


def _stage3_guide_oversteps(guide: str) -> bool:
    lower = str(guide or "").lower()
    overstep_markers = (
        "your worth",
        "your value",
        "your competence",
        "your identity",
        "self-esteem",
        "doesn't define",
        "does not define",
        "you are not",
        "you're not",
    )
    return any(marker in lower for marker in overstep_markers)

def stage0_prompter(history: str) -> str:
    payload = f"HISTORY: {history}"
    return _chat_complete(PROMPTER_CBT_STAGE0_PROMPT, payload)

def stage1_reasoner(statement: str, unhelpful_thoughts: str, dimension: str | None = None) -> str:
    payload = _format_stage1_input(statement, unhelpful_thoughts)
    raw = llm_complete_task(
        LLMTask.TASK4_CBT_STAGE1,
        ADAPTER_CBT_STAGE1_PROMPT,
        payload,
        max_new_tokens=8,
    ).text
    contract = normalize_decision_output(raw)
    decision = contract.decision
    normalized = contract.normalized_output
    log_llm_event(
        task=LLMTask.TASK4_CBT_STAGE1,
        dimension=dimension,
        score=decision,
        segment_text=unhelpful_thoughts,
        question_text=statement,
        raw_llm_output=raw,
        normalized_output=normalized,
        metadata={"cbt_stage": 1, "statement": statement, "reasoner_payload": payload},
    )
    return normalized

def stage2_reasoner(
    statement: str,
    unhelpful_thoughts: str,
    challenge: str,
    dimension: str | None = None,
) -> str:
    payload = _format_stage2_input(statement, unhelpful_thoughts, challenge)
    raw = llm_complete_task(
        LLMTask.TASK4_CBT_STAGE2,
        ADAPTER_CBT_STAGE2_PROMPT,
        payload,
        max_new_tokens=8,
    ).text
    contract = normalize_decision_output(raw)
    decision = contract.decision
    normalized = contract.normalized_output
    log_llm_event(
        task=LLMTask.TASK4_CBT_STAGE2,
        dimension=dimension,
        score=decision,
        segment_text=challenge,
        question_text=statement,
        raw_llm_output=raw,
        normalized_output=normalized,
        metadata={
            "cbt_stage": 2,
            "statement": statement,
            "unhelpful_thoughts": unhelpful_thoughts,
            "reasoner_payload": payload,
        },
    )
    return normalized

def stage3_reasoner(
    statement: str,
    unhelpful_thoughts: str,
    challenge: str,
    reframe: str,
    dimension: str | None = None,
) -> str:
    payload = _format_stage3_input(statement, unhelpful_thoughts, challenge, reframe)
    raw = llm_complete_task(
        LLMTask.TASK4_CBT_STAGE3,
        ADAPTER_CBT_STAGE3_PROMPT,
        payload,
        max_new_tokens=8,
    ).text
    contract = normalize_decision_output(raw)
    decision = contract.decision
    normalized = contract.normalized_output
    log_llm_event(
        task=LLMTask.TASK4_CBT_STAGE3,
        dimension=dimension,
        score=decision,
        segment_text=reframe,
        question_text=statement,
        raw_llm_output=raw,
        normalized_output=normalized,
        metadata={
            "cbt_stage": 3,
            "statement": statement,
            "unhelpful_thoughts": unhelpful_thoughts,
            "challenge": challenge,
            "reasoner_payload": payload,
        },
    )
    return normalized

def stage1_guide(statement: str) -> str:
    payload = f"STATEMENT: {statement}"
    raw = _chat_complete(BOUNDED_CBT_STAGE1_GUIDE_PROMPT, payload)
    guide = clean_cbt_guide_text(
        raw,
        "GUIDE: Please identify one unhelpful thought in your own words.",
    )
    log_llm_event(
        task="cbt_stage1_guide",
        segment_text=statement,
        raw_llm_output=raw,
        normalized_output=guide,
        metadata={"mode": "bounded_generation", "cbt_stage": 1, "payload": payload},
    )
    return guide

def stage2_guide(statement: str, unhelpful_thoughts: str) -> str:
    payload = f"STATEMENT: {statement}\nUNHELPFUL_THOUGHTS: {unhelpful_thoughts}"
    raw = _chat_complete(BOUNDED_CBT_STAGE2_GUIDE_PROMPT, payload)
    guide = clean_cbt_guide_text(
        raw,
        "GUIDE: Please challenge that thought in your own words by checking whether it is always true.",
    )
    log_llm_event(
        task="cbt_stage2_guide",
        segment_text=unhelpful_thoughts,
        question_text=statement,
        raw_llm_output=raw,
        normalized_output=guide,
        metadata={"mode": "bounded_generation", "cbt_stage": 2, "payload": payload},
    )
    return guide

def stage3_guide(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    payload = (
        f"STATEMENT: {statement}\n"
        f"UNHELPFUL_THOUGHTS: {unhelpful_thoughts}\n"
        f"CHALLENGE: {challenge}"
    )
    raw = _chat_complete(BOUNDED_CBT_STAGE3_GUIDE_PROMPT, payload)
    fallback = "GUIDE: Please use your challenge to write one balanced sentence in your own words."
    guide = clean_cbt_guide_text(raw, fallback)
    if _stage3_guide_oversteps(guide):
        guide = fallback
    log_llm_event(
        task="cbt_stage3_guide",
        segment_text=challenge,
        question_text=statement,
        raw_llm_output=raw,
        normalized_output=guide,
        metadata={
            "mode": "bounded_generation",
            "cbt_stage": 3,
            "unhelpful_thoughts": unhelpful_thoughts,
            "payload": payload,
        },
    )
    return guide

def recap_stage3_challenge(statement: str, unhelpful_thoughts: str, challenge: str) -> str:
    challenge_text = " ".join(str(challenge or "").split())
    recap = (
        f'You challenged the thought by saying: "{challenge_text}"'
        if challenge_text
        else "You have already challenged the unhelpful thought."
    )
    log_llm_event(
        task="cbt_stage3_recap",
        segment_text=challenge,
        question_text=statement,
        raw_llm_output=recap,
        normalized_output=recap,
        metadata={
            "mode": "challenge_recap",
            "cbt_stage": 3,
            "unhelpful_thoughts": unhelpful_thoughts,
        },
    )
    return recap

__all__ = [
    "stage0_prompter",
    "stage1_reasoner",
    "stage2_reasoner",
    "stage3_reasoner",
    "parse_cbt_decision",
    "stage1_guide",
    "stage2_guide",
    "stage3_guide",
]

def run_cbt(question_lib, session_control=None):
    """
    Run CBT stages 0-3 after screening is finished or user said stop.
    Stage 0: ask user to choose a dimension with score=2 to work on.
    Stages 1-3: unhelpful thoughts -> challenge -> reframe, with reasoning and guidance.
    """
    logger.info("Starting CBT flow (stages 0-3).")
    session_control = session_control or NullSessionControl()
    session_control.mark_cbt()
    session_control.checkpoint("cbt")
    # 0) Collect dimensions with score=2
    # candidates: list of (idx_shown, i, j, label_internal, name_human)
    candidates = []
    idx = 1
    for i in range(1, len(question_lib) + 1):
        for j in range(1, len(question_lib[str(i)]) + 1):
            entry = question_lib[str(i)][str(j)]
            if any((s == 2) for s in entry.get("score", [])):
                candidates.append((
                    idx,
                    i,
                    j,
                    entry["label"],
                    entry.get("name", entry["label"]),
                ))
                idx += 1

    if not candidates:
        logger.info("No dimensions with score=2. Skipping CBT.")
        log_system_message("We do not have a dimension at score 2 to work on today. We will conclude here.")
        return

    # Stage 0: directly ask the user to choose a dimension by the shown index
    lines = [
        "Thank you for answering the questions.",
        "According to your previous responses, you have issue in:",
    ]
    for k, _, _, _, name0 in candidates:
        lines.append(f"{k}) {name0}")
    lines.append(
        "Which dimension would you like to work on today? "
        "Tell me the dimension number. For example: 1"
    )
    q0_clean = " \n".join(lines)
    session_control.checkpoint("cbt")
    log_question(q0_clean)
    resp = get_resp_log()
    session_control.checkpoint("cbt")
    if isinstance(resp, str) and resp.strip().lower().find("stop") != -1:
        logger.info("User requested stop at CBT stage 0.")
        return

    def _pick_candidate(answer: str):
        ans = str(answer).strip().lower()
        # Prefer selecting by the shown index (e.g., "1")
        n = extract_cbt_choice_number(ans)
        if n is not None:
            for (k0, i0, j0, lbl0, name0) in candidates:
                if k0 == n:
                    return (i0, j0, lbl0, name0)
        # Fallback: try matching by human name or internal label keyword
        for (_, i0, j0, lbl0, name0) in candidates:
            if name0.lower() in ans or lbl0.lower() in ans:
                return (i0, j0, lbl0, name0)
        return None

    chosen = _pick_candidate(resp)
    if chosen is None:
        # one retry to clarify
        opts = "; ".join([f"{k}) {name0}" for (k, _, _, _, name0) in candidates])
        log_question(
            f"Please reply with a single number between 1 and {len(candidates)}. "
            f"Example: 1. Options: {opts}"
        )
        resp = get_resp_log()
        session_control.checkpoint("cbt")
        if isinstance(resp, str) and resp.strip().lower().find("stop") != -1:
            logger.info("User requested stop at CBT stage 0 retry.")
            return
        chosen = _pick_candidate(resp)
        if chosen is None:
            logger.info("Failed to parse user choice for CBT stage 0. Exit CBT.")
            log_system_message("I could not determine your choice. We will stop CBT for now.")
            return

    i_sel, j_sel, label_sel, name_sel = chosen
    logger.info(f"CBT dimension chosen: [{label_sel}] ({name_sel}) at ({i_sel},{j_sel}).")
    session_control.checkpoint("cbt")

    # Stage 1: derive statement from RV notes of the chosen dimension
    # Prefer the latest RV follow-up response (followup_resp_1),
    # then fallback to followup_resp, then original_resp.
    statement = ""
    notes_list = question_lib[str(i_sel)][str(j_sel)].get("notes", [])
    for note_entry in reversed(notes_list):
        if not isinstance(note_entry, list):
            continue
        # Only consider RV note entries by checking the presence of rv fields
        has_rv_field = any((isinstance(x, str) and ("rv_decision:" in x or "rv_validation:" in x)) for x in note_entry)
        if not has_rv_field:
            continue
        # Try to extract in priority order
        for s in note_entry:
            if isinstance(s, str) and s.startswith("followup_resp_1: "):
                statement = s.split(": ", 1)[1]
                break
        if statement:
            break
        for s in note_entry:
            if isinstance(s, str) and s.startswith("followup_resp: "):
                statement = s.split(": ", 1)[1]
                break
        if statement:
            break
        for s in note_entry:
            if isinstance(s, str) and s.startswith("original_resp: "):
                statement = s.split(": ", 1)[1]
                break
        if statement:
            break

    # Add recap prefix (similar to RV), then ask to identify unhelpful thoughts
    recap = (
        f"Let us work on dimension '{name_sel}'. "
        f"From our record, you mentioned that: {statement}"
    )
    set_question_prefix(recap)
    log_question("Can you try to identify any unhelpful thoughts you have that contribute to this situation?")
    unhelpful = get_resp_log()
    if isinstance(unhelpful, str) and unhelpful.strip().lower().find("stop") != -1:
        logger.info("User requested stop at CBT stage 1.")
        return
    session_control.checkpoint("cbt")

    # Reason and guide up to two retries
    dec1_raw = stage1_reasoner(statement, unhelpful, dimension=label_sel)
    dec1 = parse_cbt_decision(dec1_raw)
    retry = 0
    while dec1 == "1" and retry < 2:
        guide1 = stage1_guide(statement)
        log_question(guide1)
        log_question("Please provide your UNHELPFUL_THOUGHTS again, in one sentence.")
        unhelpful = get_resp_log()
        if isinstance(unhelpful, str) and unhelpful.strip().lower().find("stop") != -1:
            logger.info("User requested stop during CBT stage 1 retry.")
            return
        session_control.checkpoint("cbt")
        dec1_raw = stage1_reasoner(statement, unhelpful, dimension=label_sel)
        dec1 = parse_cbt_decision(dec1_raw)
        retry += 1
    if dec1 == "1":
        log_system_message(CBT_PROFESSIONAL_HELP_MESSAGE)
        # record brief CBT notes
        question_lib[str(i_sel)][str(j_sel)]["notes"].append([
            f"CBT_dimension: {label_sel}",
            f"CBT_statement: {statement}",
            f"CBT_unhelpful_thoughts: {unhelpful}",
            "CBT_stage: 1_failed"
        ])
        return

    # Stage 2: challenge the unhelpful thoughts
    session_control.checkpoint("cbt")
    log_question("Now, how could you challenge those unhelpful thoughts? Please write a brief challenge.")
    challenge = get_resp_log()
    if isinstance(challenge, str) and challenge.strip().lower().find("stop") != -1:
        logger.info("User requested stop at CBT stage 2.")
        return
    session_control.checkpoint("cbt")

    dec2_raw = stage2_reasoner(statement, unhelpful, challenge, dimension=label_sel)
    dec2 = parse_cbt_decision(dec2_raw)
    retry = 0
    while dec2 == "1" and retry < 2:
        guide2 = stage2_guide(statement, unhelpful)
        log_question(guide2)
        log_question("Please try to CHALLENGE the unhelpful thoughts again, in one sentence.")
        challenge = get_resp_log()
        if isinstance(challenge, str) and challenge.strip().lower().find("stop") != -1:
            logger.info("User requested stop during CBT stage 2 retry.")
            return
        session_control.checkpoint("cbt")
        dec2_raw = stage2_reasoner(statement, unhelpful, challenge, dimension=label_sel)
        dec2 = parse_cbt_decision(dec2_raw)
        retry += 1
    if dec2 == "1":
        log_system_message(CBT_PROFESSIONAL_HELP_MESSAGE)
        question_lib[str(i_sel)][str(j_sel)]["notes"].append([
            f"CBT_dimension: {label_sel}",
            f"CBT_statement: {statement}",
            f"CBT_unhelpful_thoughts: {unhelpful}",
            f"CBT_challenge: {challenge}",
            "CBT_stage: 2_failed"
        ])
        return

    # Stage 3: reframe the thought (prepend an LLM-rephrased recap of user's CHALLENGE)
    session_control.checkpoint("cbt")
    recap3 = recap_stage3_challenge(statement, unhelpful, challenge)
    set_question_prefix(recap3.strip())
    log_question("Finally, can you reframe the unhelpful thought into a more balanced, constructive one?")
    reframe = get_resp_log()
    if isinstance(reframe, str) and reframe.strip().lower().find("stop") != -1:
        logger.info("User requested stop at CBT stage 3.")
        return
    session_control.checkpoint("cbt")

    dec3_raw = stage3_reasoner(statement, unhelpful, challenge, reframe, dimension=label_sel)
    dec3 = parse_cbt_decision(dec3_raw)
    retry = 0
    while dec3 == "1" and retry < 2:
        guide3 = stage3_guide(statement, unhelpful, challenge)
        log_question(guide3)
        log_question("Please REFRAME again in one or two sentences.")
        reframe = get_resp_log()
        if isinstance(reframe, str) and reframe.strip().lower().find("stop") != -1:
            logger.info("User requested stop during CBT stage 3 retry.")
            return
        session_control.checkpoint("cbt")
        dec3_raw = stage3_reasoner(statement, unhelpful, challenge, reframe, dimension=label_sel)
        dec3 = parse_cbt_decision(dec3_raw)
        retry += 1
    if dec3 == "1":
        log_system_message(CBT_PROFESSIONAL_HELP_MESSAGE)
        question_lib[str(i_sel)][str(j_sel)]["notes"].append([
            f"CBT_dimension: {label_sel}",
            f"CBT_statement: {statement}",
            f"CBT_unhelpful_thoughts: {unhelpful}",
            f"CBT_challenge: {challenge}",
            f"CBT_reframe: {reframe}",
            "CBT_stage: 3_failed"
        ])
        return

    # Success
    question_lib[str(i_sel)][str(j_sel)]["notes"].append([
        f"CBT_dimension: {label_sel}",
        f"CBT_statement: {statement}",
        f"CBT_unhelpful_thoughts: {unhelpful}",
        f"CBT_challenge: {challenge}",
        f"CBT_reframe: {reframe}",
        "CBT_stage: success"
    ])
    log_system_message("Great work today. We completed the CBT steps for this topic. Thank you for your effort.")
