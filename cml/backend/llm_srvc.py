from huggingface_hub import login
import os
import torch
from transformers import pipeline, BitsAndBytesConfig
from core.utils.ChatHistory import ChatHistory

login(os.getenv("hf_token"))

bnb_4bit_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

user_bank = ChatHistory()

model_id = "google/gemma-2-9b-it"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
        "quantization_config": bnb_4bit_config,
    },
    max_new_tokens=10000,
    # device=device,  # replace with "mps" to run on a Mac device
)


def ai_chat(args):
    """will received a payload in the form
    ex.
    {"user_id":"default", "question": "hello"}
    and return and asnwer to the question
    """
    user_id = args["user_id"]
    question = args["question"]
    messages = user_bank.get_history(user_id=user_id)

    messages.append({"role": "user", "content": question})

    rsp = pipe(messages)
    ai_rsp = rsp[0]["generated_text"][-1]
    user_bank.add_history(user_id=user_id, question=question, ai_response=ai_rsp)
    return {"answer": ai_rsp["content"].strip()}
