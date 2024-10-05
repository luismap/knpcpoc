from typing import List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

# from huggingface_hub import login as lg
from threading import Thread


model_id = "google/gemma-2-9b-it"
model_id_sm = "openai-community/gpt2"

tok = AutoTokenizer.from_pretrained(model_id_sm)
model = AutoModelForCausalLM.from_pretrained(model_id_sm)
streamer = TextIteratorStreamer(tok)


def generate_inputs(text: List[str] = ["tell me a joke"]):
    inputs = tok(text, return_tensors="pt")
    return inputs


def get_stream_output(tokenized_input, max_new_tokens: int = 30):
    generation_kwargs = dict(
        tokenized_input, streamer=streamer, max_new_tokens=max_new_tokens
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for e in streamer:
        yield e
