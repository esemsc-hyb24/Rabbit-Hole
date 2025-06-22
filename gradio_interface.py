import gradio as gr
import subprocess
from pathlib import Path
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

# === Load analysis context ===
CONTEXT_FILE = "analysis_summary.txt"
def load_analysis_context():
    return Path(CONTEXT_FILE).read_text() if Path(CONTEXT_FILE).exists() else "No analysis yet. Please run analysis."

analysis_context = load_analysis_context()

# === LLM Setup (Ollama) ===
llm = Ollama(model="gemma3:4b")
chat_template = PromptTemplate.from_template(
    "You're a helpful reflection agent that nudges users toward a balanced information diet and healthy behavior.\n\n"
    "Here is the user's current media and search cluster analysis:\n"
    "{context}\n\n"
    "User says: {user_input}\n\n"
    "Respond with guidance that reflects their focus. If they’re focused on doom, suggest hopeful content. "
    "If too much work, suggest rest. If too much rest, suggest a focus boost. Be concise and warm.\n\n"
    "Response:"
)

# === Action for "Analyze" Button ===
def run_analysis():
    try:
        subprocess.run(["python", "rabbit_hole_analysis.py"], check=True)
        global analysis_context
        analysis_context = load_analysis_context()
        return (
            "✅ Analysis complete!",
            "clusters.png",
            "topics_pie.png",
            "sentiment_pie.png"
        )
    except subprocess.CalledProcessError as e:
        return ("❌ Analysis failed. See terminal for details.", None, None, None)

# === Chatbot Handler ===
def chat(user_input, history):
    # Build conversation string
    conversation = ""
    for i, (user_msg, bot_reply) in enumerate(history):
        conversation += f"User: {user_msg}\nAssistant: {bot_reply}\n"
    conversation += f"User: {user_input}\nAssistant:"

    # Final prompt with system + analysis context
    full_prompt = (
        "You're a helpful reflection agent that nudges users toward a balanced information diet and healthy behavior.\n\n"
        f"Here is the user's media/search analysis context:\n{analysis_context}\n\n"
        "Here is the current conversation:\n"
        f"{conversation}"
    )

    response = llm.invoke(full_prompt).strip()
    history.append((user_input, response))
    return history, history

with gr.Blocks(css="""
.primary-button {
    font-size: 18px;
    padding: 12px 24px;
    background: linear-gradient(to right, #3a7bd5, #00d2ff);
    color: white;
    border-radius: 10px;
    border: none;
    margin-top: 20px;
}

.gradio-container {
    max-width: 1100px;
    margin: auto;
    font-family: 'Segoe UI', sans-serif;
    background-color: #111;   /* dark background */
    color: white;             /* global text color */
}

h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}
""") as demo:
    
    gr.Markdown("# 🕳️ Rabbit Hole")
    gr.Markdown("""
    **Rabbit Hole** is your media reflection tool. It analyzes your recent searches and reading habits to detect:
    
    - Topic concentration
    - Emotional tone
    - Possible "echo chambers"

    After analysis, chat with our agent to get balance tips or fresh perspectives.
    """)

    # Analysis button
    analyze_btn = gr.Button("🐇 Run Analysis", elem_classes="primary-button")
    status = gr.Textbox(label="Status", interactive=False)

    # === Plot section ===

    # Cluster map (single full-width)
    gr.Markdown("### 🧭 Cluster Map")
    img1 = gr.Image(value="placeholder.png", label="Cluster Map", show_download_button=True)

    # Topic + Sentiment pie charts side-by-side
    gr.Markdown("### 📊 Topic & Sentiment Breakdown")
    with gr.Row():
        img2 = gr.Image(value="placeholder.png", label="Topic Distribution", show_download_button=True)
        img3 = gr.Image(value="placeholder.png", label="Sentiment Breakdown", show_download_button=True)

    analyze_btn.click(fn=run_analysis, outputs=[status, img1, img2, img3])

    # Chat section
    gr.Markdown("### 🤖 Reflection Agent")
    gr.Markdown("_Talk about how you’ve been consuming media, or just ask for feedback..._")

    chatbot = gr.Chatbot(label="Your Info Balance Assistant", height=350)
    msg = gr.Textbox(
        label="Enter your message...",
        lines=2,
        placeholder="e.g. I've been focused on AI collapse stories... should I worry?"
    )
    submit_btn = gr.Button("Submit")

    # Enable both Enter and button
    msg.submit(fn=chat, inputs=[msg, chatbot], outputs=[chatbot, chatbot])
    submit_btn.click(fn=chat, inputs=[msg, chatbot], outputs=[chatbot, chatbot])

    gr.Markdown("— built with ❤️ and 🐇 by your hackathon team")

demo.launch()