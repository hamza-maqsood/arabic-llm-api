import logging

from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# --###########################################################################

accelerator = Accelerator()

model_path = "brainiac-origin/jais-chat-30b-8bit"

device = None
model = None
tokenizer = None


def get_response(text, _tokenizer=tokenizer, _model=model):
    input_ids = _tokenizer(text, return_tensors="pt").input_ids
    inputs = input_ids.to(device)
    input_len = inputs.shape[-1]
    generate_ids = _model.generate(
        inputs,
        top_p=0.9,
        temperature=0.3,
        max_length=2048 - input_len,
        min_length=input_len + 4,
        repetition_penalty=1.2,
        do_sample=True,
    )
    response = _tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    response = response.split("### Response: [|AI|]")
    return response


# --###########################################################################

app = FastAPI()

origins = ["*"]

# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LLMRequest(BaseModel):
    prompt: str


class LLMResponse(BaseModel):
    response: str


@app.post("/generate-text", response_model=LLMResponse)
async def generate_text(graph: LLMRequest):
    global tokenizer, model
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not ready to serve yet")
    return get_response(graph.prompt, tokenizer, model)


@app.on_event("startup")
async def app_startup():
    global device, model, tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logging.info("downloading model")


# --###########################################################################

if __name__ == "__main__":
    logging.info("-------------Starting Program!-------------")
    uvicorn.run(app, host="0.0.0.0", port=8888)
