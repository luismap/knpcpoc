from huggingface_hub import login
import os
import torch
from transformers import pipeline, BitsAndBytesConfig

login(os.getenv("hf_token"))

bnb_4bit_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


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


def query_expansion(args):
    """will received a payload in the form
    ex.
    {"question": "what color is blue", "n_gen": 2}
    and return and n_gen related questions
    {"answer": "ans1 ... n_gen"}

    """
    question = args["question"]
    try:
        n_gen = args["n_gen"]
    except KeyError:
        n_gen = 4

    messages = [
        {
            "role": "user",
            "content": f"""You are a helpful expert assistant on the energy sector. Your users are asking questions differents reports.
            Suggest up to {n_gen} additional related questions to help them find the information they need, for the provided question.
            Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
            Make sure they are complete questions, and that they are related to the original question.
            Output one question per line. Do not number the questions.
            ##Question:
            {question}
            """,
        }
    ]

    rsp = pipe(messages)
    ai_rsp = rsp[0]["generated_text"][-1]
    return {"answer": ai_rsp["content"].strip()}
