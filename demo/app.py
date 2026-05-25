# ── Imports ───────────────────────────────────────────────────────────────────
import torch
import gradio as gr
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID = "Feyerade/german-support-llama-1b-distilled"

# Exact system prompt used during training and inference (nb02 / nb04)
SYSTEM_PROMPT = (
    "Du bist ein professioneller Kundenservice-Mitarbeiter. "
    "Antworte auf Deutsch und halte dich strikt an folgendes Format:\n"
    "Beginne mit einer kurzen Empathie- oder Begruessungsformel (z.B. 'Das tut mir leid', 'Vielen Dank fuer Ihre Anfrage').\n"
    "Gib deine Loesungsschritte als nummerierte Aufzaehlung — nicht als Fliestext.\n"
    "Schliesse mit einem Hilfsangebot ab (z.B. 'Bei weiteren Fragen stehe ich Ihnen gerne zur Verfuegung').\n"
    "Verwende eine professionelle, hoefliche Sprache (Sie-Form).\n"
    "Maximal 150 Woerter."
)

EXAMPLE_QUERIES = [
    "Meine Bestellung ist noch nicht angekommen. Wo ist mein Paket?",
    "Ich möchte einen Artikel zurückschicken. Wie läuft das ab?",
    "Ich wurde doppelt abgerechnet. Was kann ich tun?",
    "Ich kann mich nicht mehr in mein Konto einloggen.",
    "Die App stürzt beim Öffnen immer wieder ab.",
]

# ── Model loading (runs once at startup) ──────────────────────────────────────
print(f"Loading {MODEL_ID} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

print("Model ready.")

# ── Streaming chat function ───────────────────────────────────────────────────
def predict(message, history):
    '''
        Generator function for the Gradio ChatInterface.
        Builds the full conversation, runs model.generate() in a background thread,
        and yields the accumulated response string token by token for streaming output.

        Args:
            message (str): the current user input.
            history (list): list of [user_msg, assistant_msg] pairs from previous turns,
                            supplied automatically by Gradio.

        Returns:
            str: accumulated response so far, updated on each new token.
    '''

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # ChatML format, consistent with nb02 / nb04
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,           # suppress prompt echo
        skip_special_tokens=True,   # strip EOS/pad tokens
    )

    # generate() is blocking; thread allows concurrent iteration over the streamer
    thread = Thread(
        target=model.generate,
        kwargs=dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        ),
    )
    thread.start()

    # Gradio expects cumulative text, not individual tokens
    partial = ""
    for token in streamer:
        partial += token
        yield partial

demo = gr.ChatInterface(
    fn=predict,
    title="German Customer Support · Student Model Demo",
    description=(
        "Chat with a 1B German customer-support model distilled from a "
        "QLoRA fine-tuned from a 7B teacher model. Pick an example query or type your own."
    ),
    examples=EXAMPLE_QUERIES,
    cache_examples=False,
    textbox=gr.Textbox(
        placeholder="Schreiben Sie Ihre Anfrage auf Deutsch ...",
        container=False,
        scale=7,
    ),
)

if __name__ == "__main__":
    demo.launch()
