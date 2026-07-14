from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re

# Initialize FastAPI Application

app = FastAPI(
    title="AI Text Summarizer",
    description="Text Summarization using Fine-Tuned T5 Transformer",
    version="1.0.0"
)

# Static Files & Templates

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load Model & Tokenizer

MODEL_PATH = "./saved_summary_model"
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)

# Device Configuration

if torch.backends.mps.is_available() :
    device = torch.device("mps")
elif torch.cuda.is_available() :
    device = torch.device("cuda")
else :
    device = torch.device("cpu")

model.to(device)
model.eval()

print(f"Running on : {device}")

# Request Schema

class DialogueInput(BaseModel) :
    dialogue: str


# Text Cleaning

def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip().lower()


# Summarization Function

def summarize_dialogue(dialogue: str) -> str :
    dialogue = "summarize: " + clean_data(dialogue)
    inputs = tokenizer(
        dialogue,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad() :
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )
    return summary


# Routes

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput) :
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary" : summary}