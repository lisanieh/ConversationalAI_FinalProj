"""Every prompt the assistant agent sends to its LLM, word for word.

The paper must report all prompts verbatim; ``run_agent.py`` copies this file's
contents into each run's log folder. Bump PROMPT_VERSION whenever any text here
changes. Questions and repair requests are templates (cheap, deterministic,
and the robot customer reads the label, not the wording).
"""

PROMPT_VERSION = "a1"

EXTRACT_PROMPT = """\
TASK: EXTRACT
You are the dialogue state tracker of a hotel and restaurant booking assistant \
on a phone line. The customer's words come from a speech recognizer and may \
contain recognition errors.
Allowed slots: {slots}
Meaning: area (north, south, east, west, centre), pricerange (cheap, moderate, \
expensive), type (hotel or guesthouse), food (cuisine), day (day of the week), \
people (number of people), stay (number of nights), time (booking time), name \
(the customer's full name), email, phone.
Booking domain known so far: {domain}
The assistant just asked about: {asked}

Return ONLY a JSON object of the form
{{"domain": "hotel" | "restaurant" | null, "slots": {{"<slot>": "<value>"}}}}
with the slots the customer mentions in their LAST message.
Copy each value exactly as it appears in the transcript. Do NOT correct \
spelling, do NOT guess, and do NOT repeat values from earlier turns.
If the customer says they have no preference for the slot that was asked, use \
"dontcare". If nothing new is mentioned, return {{"domain": null, "slots": {{}}}}."""

MISHEAR_PROMPT = """\
TASK: MISHEAR
A speech recognizer transcribed a customer on the phone. Transcript: "{transcript}"
From it you extracted {slot} = "{value}".
How likely is it (0-100) that this value is NOT what the customer actually said?
Answer with JSON only: {{"p_wrong": <number>}}"""

CHOOSE_PROMPT = """\
TASK: CHOOSE
A speech recognizer transcribed a customer on the phone. Transcript: "{transcript}"
From it you extracted {slot} = "{value}".
Choose how to make sure this value is right:
ACCEPT (use it as is), CONFIRM (read it back to the customer), SPELL (ask the \
customer to spell it aloud), TYPE (ask the customer to type it). Typing is the \
most reliable but costs the customer the most effort, and some customers cannot type.
Answer with JSON only: {{"action": "<ACCEPT|CONFIRM|SPELL|TYPE>"}}"""

ALL_PROMPTS = {"EXTRACT_PROMPT": EXTRACT_PROMPT, "MISHEAR_PROMPT": MISHEAR_PROMPT,
               "CHOOSE_PROMPT": CHOOSE_PROMPT}
