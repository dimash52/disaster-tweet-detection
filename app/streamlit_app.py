import streamlit as st
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

st.set_page_config(
    page_title="Disaster Tweet Detection",
    page_icon="./icon.ico"
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "distilbert"


@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

st.title("Disaster Tweet Detection")

st.write("Enter a tweet and the model will classify it")

tweet = st.text_area("Tweet", height=150)

if st.button("Predict"):

    inputs = tokenizer(
        tweet,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)

        preds = probs.argmax().item()

        confidence = probs.max().item()

        if preds == 1:
            st.error(f'Disaster error ({confidence: .2%})')
        else:
            st.success(f'Non-Disaster Tweet Confidence {confidence: .2%}')
