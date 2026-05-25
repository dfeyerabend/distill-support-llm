# ── Imports ───────────────────────────────────────────────────────────────────
import torch
import gradio as gr
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_ID = "Feyerade/german-support-student-1.5b-distilled"

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
    # Build the full conversation: system prompt + past turns + new message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user",      "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    # Tokenize using ChatML format — same as nb02 / nb04
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    # Streamer acts as a queue: generate() writes decoded tokens in,
    # we read them out one by one in the loop below
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,         # don't echo the input tokens back
        skip_special_tokens=True, # strip <|im_end|> and similar
    )

    # model.generate() is blocking — it runs until all tokens are done.
    # We put it in a background thread so the main thread is free to
    # iterate over the streamer at the same time.
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

    # Yield the accumulated response so far on every new token.
    # Gradio re-renders the chat bubble on each yield.
    partial = ""
    for token in streamer:
        partial += token
        yield partial

