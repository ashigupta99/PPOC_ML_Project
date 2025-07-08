import streamlit as st
import torch
import pickle
import os
import requests
import base64
from dotenv import load_dotenv
from transformers import BertForSequenceClassification, BertTokenizerFast
from huggingface_hub import hf_hub_download

# Set base path (important for deployment!)
BASE_PATH = os.path.dirname(__file__)

# Load CSS and background
def load_css_with_bg(css_path, bg_path):
    with open(bg_path, "rb") as f:
        bg_data = base64.b64encode(f.read()).decode()
    with open(css_path, "r") as f:
        css = f.read()
    css = css.replace("{{bg_image}}", bg_data)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Use full paths
css_path = os.path.join(BASE_PATH, "assets", "styles.css")
bg_path = os.path.join(BASE_PATH, "backgrounds", "news_bg1.jpg")
load_css_with_bg(css_path, bg_path)

# Load environment variables
load_dotenv()

# Load model, tokenizer and temperature
@st.cache_resource
def load_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained("aashi219/fake-news-bert")
    tokenizer = BertTokenizerFast.from_pretrained("aashi219/fake-news-bert")
    temp_path = hf_hub_download(
        repo_id="aashi219/fake-news-bert",
        filename="temperature.pkl",
        repo_type="model"
    )
    with open(temp_path, "rb") as f:
        optimal_temp = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, optimal_temp, device

model, tokenizer, optimal_temp, device = load_model_and_tokenizer()

# Load Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.warning("🔑 Google Gemini API key not found. Please set GEMINI_API_KEY as an environment variable.")

GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# Predict with temperature scaling
def predict_with_temperature(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        logits = logits / optimal_temp
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
        confidence = probs[0, pred.item()].item()
    label = "Real" if pred.item() == 1 else "Fake"
    return label, confidence

# Ask Gemini for explanation
def gemini_llm_explanation(headline):
    prompt = (
        "You are a news fact-checking assistant. "
        "Given a news headline, explain in 2-3 sentences whether it is likely real or fake, "
        "and provide reasoning. If the claim is plausible, speculative, or known to be false, mention that.\n\n"
        f"Headline: \"{headline}\"\nExplanation:"
    )
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"], None
        else:
            return None, f"Error: {response.text}"
    except Exception as e:
        return None, str(e)

# Adjust confidence if LLM is uncertain
def adjust_confidence_with_llm(confidence, explanation):
    uncertain_keywords = [
        "partially", "uncertain", "incorrect", "not confirmed",
        "plausible", "speculative", "rumor", "unverified", "not entirely accurate",
        "may be", "possibly", "could be", "mixed", "disputed", "debated"
    ]
    if explanation:
        explanation_lower = explanation.lower()
        if any(word in explanation_lower for word in uncertain_keywords):
            return max(confidence * 0.6, 0.5)
    return confidence

# App UI
st.markdown('<div class="news-main-card">', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="news-title"><span style="font-size:1.7rem;">📰</span> Fake News Detector!!</div>', unsafe_allow_html=True)
    st.markdown('<div class="news-subtitle">BERT + Google Gemini LLM (for explanation)</div>', unsafe_allow_html=True)

    user_input = st.text_area("Paste a news headline or article:")

    if st.button("Check News"):
        if user_input.strip():
            label, conf = predict_with_temperature(user_input)
            with st.spinner("Getting LLM explanation..."):
                explanation, error = gemini_llm_explanation(user_input)
            adj_conf = adjust_confidence_with_llm(conf, explanation)

            # Override label to 'Uncertain' if confidence is low
            if adj_conf < 0.65:
                label = "Uncertain"

            st.markdown(
                f'<div class="confidence-info"><b>Prediction:</b> {label} '
                f'<b>(Confidence:</b> {adj_conf*100:.1f}%)</div>',
                unsafe_allow_html=True
            )

            if label == "Uncertain":
                st.markdown(
                    '<div class="uncertain">⚠️ Model is unsure about this news. '
                    'Please verify it from trusted sources.</div>',
                    unsafe_allow_html=True
                )

            if explanation:
                st.markdown(f'<div class="llm-explanation"><b>LLM says:</b> {explanation}</div>', unsafe_allow_html=True)
                if adj_conf < conf and label != "Uncertain":
                    st.info("Confidence reduced due to detected uncertainty or partial correctness in LLM explanation.")
            else:
                st.error("LLM explanation unavailable.")
                st.code(error)
        else:
            st.warning("Please enter some text.")

st.markdown('</div>', unsafe_allow_html=True)
