from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import os

from coach.context import read_resume, build_system_prompt
from coach.tools import tools, handle_tool_calls

load_dotenv(override=True)

MODEL_NAME = "openai/gpt-oss-120b"
MAX_TOOL_ROUNDS = 5
MAX_TURNS = 10
MAX_HISTORY_MESSAGES = 12  # Sliding window: keep up to 6 turns (12 messages) in context

CSS_PATH = Path(__file__).resolve().parent / "style.css"
CUSTOM_CSS = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""

# Prevent blank/whitespace-only submissions before Gradio adds the user bubble
JS = """
() => {
  const guard = () => {
    // Guard submit button
    document.querySelectorAll(
      'button[aria-label="Submit"], button.submit-button, button.submit'
    ).forEach(btn => {
      if (btn.dataset.blankGuard) return;
      btn.dataset.blankGuard = '1';
      btn.addEventListener('click', e => {
        const ta = btn.closest('form')?.querySelector('textarea')
                   ?? document.querySelector('textarea');
        if (ta && !ta.value.trim()) {
          e.preventDefault();
          e.stopImmediatePropagation();
        }
      }, true);
    });

    // Guard Enter key in every textarea
    document.querySelectorAll('textarea').forEach(ta => {
      if (ta.dataset.blankGuard) return;
      ta.dataset.blankGuard = '1';
      ta.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey && !ta.value.trim()) {
          e.preventDefault();
          e.stopImmediatePropagation();
        }
      }, true);
    });

    // Smooth autoscroll for chat container
    const scrollContainers = document.querySelectorAll(
      '[data-testid="chatbot"] .message-wrap, [data-testid="chatbot"] .bubble-wrap, [data-testid="chatbot"] > div > div, .chatbot-container .message-wrap'
    );
    scrollContainers.forEach(el => {
      if (el.style.scrollBehavior !== 'smooth') {
        el.style.scrollBehavior = 'smooth';
      }
    });
  };

  setTimeout(guard, 500);
  new MutationObserver(guard).observe(document.body, { childList: true, subtree: true });
}
"""

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
)


def _setup_error(msg):
    """Return a consistent error tuple for the setup screen."""
    return "", msg, gr.update(), gr.update()


def clean_history(history):
    """Flatten Gradio history to plain role/content dicts, applying the sliding window first."""
    history = history[-MAX_HISTORY_MESSAGES:]  # apply window before iterating
    cleaned = []
    for item in history:
        content = item["content"]
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        cleaned.append({"role": item["role"], "content": content})
    return cleaned


def start_session(resume_path, jd_text):
    if not resume_path:
        return _setup_error("Please upload your resume first.")

    if not jd_text or not jd_text.strip():
        return _setup_error("Please paste a job description.")

    try:
        resume = read_resume(resume_path)
    except Exception as error:
        return _setup_error(f"Could not read your resume: {error}")

    prompt = build_system_prompt(resume, jd_text.strip())
    return prompt, "Ready.", gr.update(visible=False), gr.update(visible=True)


def reset_session():
    return "", "", gr.update(visible=True), gr.update(visible=False)


def chat(message, history, system_prompt):
    message_text = message.strip() if isinstance(message, str) else ""
    if not message_text:
        raise gr.Error("Please type a message before sending.", duration=2)
    if not system_prompt:
        raise gr.Error("Session not started. Please upload your resume and job description first.", duration=3)

    cleaned = clean_history(history)

    turns_used = sum(1 for m in cleaned if m["role"] == "user")
    if turns_used >= MAX_TURNS:
        return "**Session complete.** Click 'Setup New Interview' to prepare for another role."

    system = [{"role": "system", "content": system_prompt}]
    messages = system + cleaned + [{"role": "user", "content": message_text}]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools
        )
    except Exception as error:
        return f"Sorry, I couldn't reach the model: {error}"

    rounds = 0
    while response.choices[0].finish_reason == "tool_calls" and rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        assistant_message = response.choices[0].message
        results = handle_tool_calls(assistant_message.tool_calls)

        messages.append(assistant_message)
        messages.extend(results)

        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, tools=tools
        )

    reply = response.choices[0].message.content
    if not reply:
        return "I got stuck while processing that. Please send your answer again."
    return reply


EXAMPLES = [
    ["Start the interview."],
    ["Ask me a technical question based on my resume and this role."],
    ["Ask me a scenario or behavioral question for this role."],
    ["Give me a practice question to test my core skills."],
]

THEME = gr.themes.Soft(primary_hue="emerald", neutral_hue="slate")

with gr.Blocks(title="AI Interview Coach", fill_height=True) as demo:
    session_prompt = gr.State("")

    # Setup Screen Area
    with gr.Column(visible=True, elem_classes="setup-screen") as setup_area:
        gr.HTML(
            """
            <div class="setup-header">
                <span class="setup-badge">Mock Interview AI</span>
                <h1 class="setup-title">AI Interview Coach</h1>
                <p class="setup-subtitle">Upload your resume and target job description to start personalized mock interview practice.</p>
            </div>
            """
        )

        with gr.Row(equal_height=True, elem_classes="setup-row"):
            resume_file = gr.File(
                label="Candidate Resume (PDF or TXT)",
                file_types=[".pdf", ".txt"],
                type="filepath",
                scale=1,
            )
            jd_box = gr.Textbox(
                label="Target Job Description",
                placeholder="Paste the target job description, requirements, or role details here...",
                scale=1,
                elem_classes="jd-box",
            )

        start_btn = gr.Button("Start Interview", variant="primary", size="lg")
        status = gr.Markdown()

    # Chat / Coach Screen Area
    with gr.Column(visible=False, elem_classes="chat-wrapper") as coach_area:
        with gr.Row(elem_classes="chat-top-bar"):
            gr.Markdown("### 💬 Mock Interview Session", scale=9)
            reset_btn = gr.Button("Change Role", size="sm", elem_classes="compact-reset-btn", scale=0, min_width=110)

        greeting = [
            {
                "role": "assistant",
                "content": "Welcome! I've reviewed your resume and the target job description. Type **'start'** or pick a prompt below to begin your first question.",
            }
        ]
        gr.ChatInterface(
            chat,
            chatbot=gr.Chatbot(value=greeting, scale=1, autoscroll=True),
            textbox=gr.Textbox(
                placeholder="Type your answer or ask for feedback...",
                container=False,
                scale=7,
                submit_btn="➤",
            ),
            additional_inputs=[session_prompt],
            examples=EXAMPLES,
        )

    start_btn.click(
        start_session,
        inputs=[resume_file, jd_box],
        outputs=[session_prompt, status, setup_area, coach_area],
    )

    reset_btn.click(
        reset_session,
        inputs=[],
        outputs=[session_prompt, status, setup_area, coach_area],
    )

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CUSTOM_CSS, js=JS)