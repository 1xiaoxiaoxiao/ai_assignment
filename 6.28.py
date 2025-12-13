# ===============================================
# Hotel FAQ Chatbot (Lightweight, SVM + spaCy-sm)
# ===============================================

import streamlit as st
import pandas as pd
import re
import spacy
import joblib
import random
import time
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------
# 1. Configuration
# -------------------------
CONFIDENCE_THRESHOLD = 0.75

# -------------------------
# 2. Load spaCy small model (tokenization + lemmatization)
# -------------------------
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# -------------------------
# 3. Text preprocessing
# -------------------------
def preprocess_text(text):
    """
    Lowercase, remove punctuation, lemmatize, remove stopwords
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# -------------------------
# 4. Load model & vectorizer
# -------------------------
@st.cache_resource
def load_model_resources():
    try:
        svm_model = joblib.load("intent_model_spacy.joblib")
        vectorizer = joblib.load("tfidf_vectorizer_spacy.joblib")
        return svm_model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}")
        return None, None

svm_model, vectorizer = load_model_resources()

# -------------------------
# 5. Responses
# -------------------------
RESPONSES = {
    "ask_room_price": "Our deluxe room costs RM180 per night.",
    "ask_booking": "I can help you book a room. Please provide your date and number of guests.",
    "ask_checkin_time": "Check-in time starts from 2:00 PM.",
    "ask_checkout_time": "Check-out time is before 12:00 PM.",
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Thank you for visiting. Have a nice day!"
}

# -------------------------
# 6. Prediction function
# -------------------------
def predict_intent(user_input):
    if svm_model is None or vectorizer is None:
        return "setup_error", "Model not loaded.", "N/A", 0.0

    start_time = time.time()
    cleaned = preprocess_text(user_input)
    vec = vectorizer.transform([cleaned])
    predicted_class = svm_model.predict(vec)[0]

    # pseudo-confidence via decision_function
    if hasattr(svm_model, "decision_function"):
        scores = svm_model.decision_function(vec)
        confidence_score = max(scores[0]) if len(scores.shape) > 1 else abs(scores[0])
    else:
        confidence_score = 1.0  # fallback

    confidence_display = f"{confidence_score*100:.2f}%"
    response_time = time.time() - start_time
    response_text = RESPONSES.get(predicted_class, "Sorry, I do not understand your request.")

    return predicted_class, response_text, confidence_display, response_time

# -------------------------
# 7. Streamlit Chatbot
# -------------------------
def main():
    st.set_page_config(page_title="Hotel FAQ Chatbot", layout="centered")
    st.title("🏨 Astra Imperium Hotel Chatbot")
    st.caption(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    st.markdown("Ask me about room rates, booking, check-in/out times, and more!")

    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": RESPONSES.get("greeting", "Hello!")})

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "intent" in msg:
                st.caption(f"Intent: **{msg['intent']}**, Confidence: **{msg['confidence']}**, Time: **{msg['time']:.4f}s**")

    # User input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Analyzing..."):
            intent, response, confidence, resp_time = predict_intent(user_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "intent": intent,
                "confidence": confidence,
                "time": resp_time
            })
            st.rerun()

if __name__ == "__main__":
    main()
