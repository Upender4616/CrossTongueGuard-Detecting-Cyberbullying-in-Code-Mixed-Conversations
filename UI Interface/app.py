import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Cyberbullying Moderation Chat",
    page_icon="💬",
    layout="centered"
)

st.title("💬 CrossTongueGuard:Detecting Cyberbullying in Code-Mixed Conversations")
st.caption("Messages are checked before being displayed")

# -------------------------------
# Session State
# -------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=2
    ).to(device)
    model.eval()
    return tokenizer, model

muril_tok, muril_model = load_model("google/muril-base-cased")
mdeberta_tok, mdeberta_model = load_model("microsoft/mdeberta-v3-base")
xlmr_tok, xlmr_model = load_model("xlm-roberta-base")

# -------------------------------
# Prediction
# -------------------------------
def predict_prob(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    return float(probs[1])

# -------------------------------
# Classification Thresholds
# -------------------------------
LOW, HIGH = 0.45, 0.60

def classify(p):
    if p >= HIGH:
        return "Cyberbullying"
    elif p <= LOW:
        return "Not Cyberbullying"
    return "Uncertain"

# -------------------------------
# Lexical Abuse Detector
# -------------------------------
ABUSIVE_WORDS = [
    "fuck", "fucking", "bitch", "lanja", "makalode",
    "bakwas", "pichi", "ass", "boobs", "sex",
    "idiot", "stupid", "mental", "slut",
    #English abuse
    "idiot", "stupid", "dumb", "moron", "loser",
    "bitch", "bastard", "asshole",
    "fuck", "fucking", "shit", "bullshit",
    "slut", "whore", "pervert", "creep",

    # Hindi / Hinglish abuse
    "bakwas", "chutiya", "madarchod", "bhenchod",
    "gandu", "harami", "kamina", "pagal",
    "saala", "kutta", "nalayak",

    # Telugu / Telugu-English slang
    "arey", "pichi", "mental", "lanja",
    "lanjakodaka", "bewarse", "chetta",
    "dongana", "vedhava", "bakwasgadu",
    "makalode",

    # General harassment terms
    "ugly", "disgusting", "hate you",
    "go die", "kill yourself", "nonsense"
]

def has_abuse(text):
    text = text.lower()
    return any(w in text for w in ABUSIVE_WORDS)

# -------------------------------
# Chat Display
# -------------------------------
st.subheader("📨 Chat")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Input Box
# -------------------------------
# -------------------------------
# Chat Input (FIXED)
# -------------------------------
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

user_text = st.chat_input("Type a message...")

if user_text:
    st.session_state.pending_input = user_text

# -------------------------------
# Process Message Immediately
# -------------------------------
if st.session_state.pending_input:

    text = st.session_state.pending_input
    st.session_state.pending_input = None  # clear buffer

    # ---- Model predictions (hidden) ----
    p1 = predict_prob(text, muril_tok, muril_model)
    p2 = predict_prob(text, mdeberta_tok, mdeberta_model)
    p3 = predict_prob(text, xlmr_tok, xlmr_model)

    labels = [
        classify(p1),
        classify(p2),
        classify(p3)
    ]

    is_cyberbullying = (
        has_abuse(text)
        or labels.count("Cyberbullying") >= 2
    )

    # ---- Chat behavior ----
    if is_cyberbullying:
        st.session_state.chat.append({
            "role": "assistant",
            "content": "🚨 **Message blocked** due to cyberbullying."
        })
    else:
        st.session_state.chat.append({
            "role": "user",
            "content": text
        })
        st.session_state.chat.append({
            "role": "assistant",
            "content": "✅ Message delivered."
        })

    st.rerun()
