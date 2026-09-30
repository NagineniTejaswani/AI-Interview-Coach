import json
from pathlib import Path
import random

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROGRESS_FILE = DATA_DIR / "progress.json"


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"weak_areas": [], "scores": []}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")

QUESTION_BANK = {
    "python": [
        "Explain the difference between a list and a tuple in Python.",
        "What are decorators and when would you use one?",
        "How does Python manage memory?",
    ],
    "sql": [
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "How would you find duplicate rows in a table?",
        "What is an index and when can it hurt performance?",
    ],
    "system design": [
        "How would you design a URL shortener?",
        "How would you scale an API that gets 10x more traffic?",
    ],
    "behavioral": [
        "Tell me about a time you handled a disagreement in a team.",
        "Describe a project that failed and what you learned.",
    ],
}


def save_weak_area(topic, note="No note provided"):
    progress = load_progress()
    progress["weak_areas"].append({"topic": topic, "note": note})
    save_progress(progress)
    return f"Saved weak area: {topic}"


def get_practice_question(topic):
    topic_lower = topic.lower()
    for key, questions in QUESTION_BANK.items():
        if key in topic_lower or topic_lower in key:
            return random.choice(questions)
    return "No bank question for that topic. Make up a relevant question yourself."


def log_score(question, score):
    try:
        score = max(1, min(10, int(float(str(score).split("/")[0].strip()))))
    except (ValueError, TypeError):
        score = 5  # fallback to neutral score on bad input
    progress = load_progress()
    progress["scores"].append({"question": question, "score": score})
    save_progress(progress)
    return f"Logged score {score}/10"


save_weak_area_json = {
    "name": "save_weak_area",
    "description": "Save a topic the candidate struggled with, so it can be reviewed later. Use it when an answer shows a clear weakness.",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The topic or skill the candidate is weak in"},
            "note": {"type": "string", "description": "Short explanation of what went wrong in the answer"},
        },
        "required": ["topic"],
        "additionalProperties": False,
    },
}

get_practice_question_json = {
    "name": "get_practice_question",
    "description": "Get a practice interview question on a given topic. Use it when the candidate asks for more practice.",
    "parameters": {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The topic to practice, such as python, sql or system design"},
        },
        "required": ["topic"],
        "additionalProperties": False,
    },
}

log_score_json = {
    "name": "log_score",
    "description": "Log a score from 1 to 10 for the candidate's answer. Use it after giving feedback on every answer.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The interview question that was asked"},
            "score": {"type": "integer", "description": "Score from 1 (poor) to 10 (excellent)"},
        },
        "required": ["question", "score"],
        "additionalProperties": False,
    },
}


tools = [
    {"type": "function", "function": save_weak_area_json},
    {"type": "function", "function": get_practice_question_json},
    {"type": "function", "function": log_score_json},
]

tool_map = {
    "save_weak_area": save_weak_area,
    "get_practice_question": get_practice_question,
    "log_score": log_score,
}

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        tool = tool_map.get(tool_name)
        try:
            result = tool(**arguments) if tool else f"Unknown tool: {tool_name}"
        except Exception as error:
            result = f"Tool failed: {error}"

        results.append(
            {"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id}
        )
    return results