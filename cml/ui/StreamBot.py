from typing import List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
import os
from huggingface_hub import login

login(os.getenv("hf_token"))

from huggingface_hub import login as lg
from threading import Thread

bnb_4bit_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


model_id = "google/gemma-2-9b-it"
model_id_sm = "openai-community/gpt2"
current_model = model_id_sm

tokenizer = AutoTokenizer.from_pretrained(current_model)
decoder_kwargs = {"skip_special_tokens": True}
model = AutoModelForCausalLM.from_pretrained(
    current_model,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_4bit_config,
)

streamer = TextIteratorStreamer(tokenizer, decoder_kwargs=decoder_kwargs)


def generate_inputs(chat: List[dict[str, str]] = []):
    inputs = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )  # .to("cuda")
    return inputs


def get_stream_output(tokenized_input, max_new_tokens: int = 30):
    generation_kwargs = dict(
        inputs=tokenized_input, streamer=streamer, max_new_tokens=max_new_tokens
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    for e in streamer:
        yield e
