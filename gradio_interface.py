import gradio as gr
import subprocess
from pathlib import Path
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate

# === Load context from analysis summary ===
CONTEXT_FILE = "analysis_summary.txt"
def load_analysis_context():
    return Path(CONTEXT_FILE).read_text() if Path(CONTEXT_FILE).exists() else "No analysis found yet."

analysis_context = load_analysis_context()

# === LangChain Ollama LLM setup ===
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

# === Action for Analyze Button ===
def run_analysis():
    try:
        subprocess.run(["python", "rabbit_hole_analysis.py"], check=True)
        # Reload context after analysis
        global analysis_context
        analysis_context = load_analysis_context()
        return (
            "✅ Analysis complete!",
            "clusters.png",
            "topics_pie.png",
            "sentiment_pie.png"
        )
    except subprocess.CalledProcessError:
        return ("❌ Analysis failed. Check the script output.", None, None, None)

# === LLM Chat Handler ===
def chat(user_input, history):
    prompt = chat_template.format(context=analysis_context, user_input=user_input)
    response = llm.invoke(prompt)
    history.append((user_input, response.strip()))
    return history, history

# === Gradio Interface ===
with gr.Blocks() as demo:
    gr.Markdown("# 🕳️ Rabbit Hole")
    gr.Markdown("""
    **Rabbit Hole** analyzes your search and content consumption to reveal topic focus, sentiment balance,
    and potential biases. Press **Analyze** to begin, then chat with the reflection agent below.
    """)

    with gr.Row():
        analyze_btn = gr.Button("🔍 Analyze")
        status = gr.Textbox(label="Status")

    with gr.Row():
        img1 = gr.Image(label="Cluster Map")
        img2 = gr.Image(label="Topic Distribution")
        img3 = gr.Image(label="Sentiment Distribution")

    analyze_btn.click(fn=run_analysis, outputs=[status, img1, img2, img3])

    gr.Markdown("### 🤖 Reflection Agent")
    chatbot = gr.Chatbot(label="Your Info Balance Assistant")
    msg = gr.Textbox(label="Ask or reflect...", placeholder="e.g., I feel overwhelmed by AI news lately.")
    msg.submit(chat, [msg, chatbot], [chatbot, chatbot])

demo.launch()