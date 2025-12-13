# =====================================================
# Hotel Customer Support Chatbot (SVM + spaCy) for Streamlit
# =====================================================

import streamlit as st
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import load
import random
import time

# -------------------------
# 1. Configuration
# -------------------------
CONFIDENCE_THRESHOLD = 0.75  # For SVM, this can be used if we use decision_function

# -------------------------
# 2. Load spaCy Model
# -------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------
# 3. Text Preprocessing
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# -------------------------
# 4. Load Model and Vectorizer
# -------------------------
@st.cache_resource
def load_resources():
    try:
        svm_model = load("intent_model_spacy.joblib")
        vectorizer = load("tfidf_vectorizer_spacy.joblib")
        return svm_model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Missing model/vectorizer file: {e.filename}")
        return None, None

svm_model, vectorizer = load_resources()

# -------------------------
# 5. Responses
# -------------------------
responses = {
    "ask_room_price": "Our deluxe room costs RM180 per night.",
    "ask_booking": "I can help you book a room. Please provide your date and number of guests.",
    "ask_checkin_time": "Check-in time starts from 2:00 PM.",
    "ask_checkout_time": "Check-out time is before 12:00 PM.",
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Thank you for visiting. Have a nice day!"
}

# -------------------------
# 6. Suggested Questions
# -------------------------
PROMPT_MAPPING = {
    "ask_room_price": "What is the room price?",
    "ask_booking": "I want to book a room.",
    "ask_checkin_time": "What is the check-in time?",
    "ask_checkout_time": "What is the check-out time?"
}

SUGGESTED_INTENTS = list(PROMPT_MAPPING.keys())

# -------------------------
# 7. Predict Intent Function
# -------------------------
def predict_intent(user_input):
    if svm_model is None or vectorizer is None:
        return "setup_error", "Model not loaded.", "N/A", 0.0

    start_time = time.time()
    cleaned = preprocess_text(user_input)
    vec = vectorizer.transform([cleaned])

    # For SVM, we can use decision_function as pseudo-confidence
    decision_scores = svm_model.decision_function(vec)
    predicted_index = decision_scores.argmax() if len(decision_scores.shape) > 1 else 0
    if len(decision_scores.shape) > 1:
        confidence_score = max(decision_scores[0])
    else:
        confidence_score = abs(decision_scores[0])  # fallback for single-class

    intent_name = svm_model.classes_[predicted_index]
    response = responses.get(intent_name, "Sorry, I do not understand your request.")
    confidence_display = f"{confidence_score*100:.2f}%"

    end_time = time.time()
    response_time = end_time - start_time

    return intent_name, response, confidence_display, response_time

# -------------------------
# 8. Streamlit Chatbot
# -------------------------
def main():
    st.set_page_config(page_title="Hotel AI Assistant (SVM + spaCy)", layout="centered")
    st.title("🏨 Astra Imperium Hotel Chatbot (SVM + spaCy)")
    st.caption(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")

    # --- Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        greeting = responses.get("greeting", "Hello! How can I assist you?")
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "intent" in message:
                st.caption(f"Intent: **{message['intent']}** | Confidence: **{message['confidence']}** | Time: **{message['time']:.4f}s**")
            st.markdown(message["content"])

    # --- Suggested Questions Buttons ---
    if "random_intents" not in st.session_state:
        st.session_state.random_intents = random.sample(SUGGESTED_INTENTS, min(4, len(SUGGESTED_INTENTS)))

    current_intents = st.session_state.random_intents
    if current_intents:
        st.markdown("**Suggested Questions:**")
        cols = st.columns(len(current_intents))
        for i, intent_key in enumerate(current_intents):
            prompt_instruction = PROMPT_MAPPING[intent_key]
            with cols[i]:
                if st.button(prompt_instruction, key=f"btn_{intent_key}", use_container_width=True):
                    st.session_state.pending_input = prompt_instruction
                    del st.session_state.random_intents
                    st.rerun()

    # --- Handle User Input ---
    user_input = None
    if st.session_state.pending_input:
        user_input = st.session_state.pending_input
        st.session_state.pending_input = None
    else:
        user_input = st.chat_input("How can I help you?")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        current_user_input = st.session_state.messages[-1]["content"]
        with st.spinner("Analyzing query..."):
            intent_name, response, confidence_display, response_time = predict_intent(current_user_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "intent": intent_name,
                "confidence": confidence_display,
                "time": response_time
            })
            st.rerun()

if __name__ == "__main__":
    main()
