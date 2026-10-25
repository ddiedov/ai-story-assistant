
from pathlib import Path
from ollama import chat

BASE_DIR = Path(__file__).parent
SYSTEM_PROMPT_PATH = BASE_DIR / "prompts" / "story_system.txt"

system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

task = """
Analyze the following task:

"Implement concurrent changes handling for the Promotion object, following the approach used for other objects."

Return exactly these sections:

1. CONFIRMED FACTS
List only facts explicitly stated in the task.

2. DRAFT STORY
AS A [role, or NOT SPECIFIED]
I WANT TO [capability supported by the task]
SO THAT [outcome, or NOT SPECIFIED]

3. MISSING INFORMATION
List the questions that must be answered before the Story can be considered ready for development.

Rules:
- Never invent a role, business outcome, requirement, or implementation detail.
- Do not treat a plausible assumption as a confirmed fact.
- Do not explain your reasoning.
- Keep the response concise.
"""

response = chat(
    model="qwen2.5:3b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ],
    think=False,
    options={"num_predict": 500},
)

print("CONTENT:", repr(response.message.content))
print("THINKING:", repr(getattr(response.message, "thinking", None)))
print("DONE:", response.done_reason)