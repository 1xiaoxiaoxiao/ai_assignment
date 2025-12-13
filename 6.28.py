import streamlit as st
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import load
import time

# -------------------------
# 1. Configuration
# -------------------------
CONFIDENCE_THRESHOLD = 0.75  # For SVM pseudo-confidence

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
# 6. Predict Intent Function
# -------------------------
def predict_intent(user_input):
    if svm_model is None or vectorizer is None:
        return "setup_error", "Model not loaded.", "N/A", 0.0

    start_time = time.time()
    cleaned = preprocess_text(user_input)
    vec = vectorizer.transform([cleaned])

    decision_scores = svm_model.decision_function(vec)
    predicted_index = decision_scores.argmax() if len(decision_scores.shape) > 1 else 0
    if len(decision_scores.shape) > 1:
        confidence_score = max(decision_scores[0])
    else:
        confidence_score = abs(decision_scores[0])

    intent_name = svm_model.classes_[predicted_index]
    response = responses.get(intent_name, "Sorry, I do not understand your request.")
    confidence_display = f"{confidence_score*100:.2f}%"

    end_time = time.time()
    response_time = end_time - start_time

    return intent_name, response, confidence_display, response_time

# -------------------------
# 7. Streamlit Chatbot
# -------------------------
def main():
    st.set_page_config(page_title="Hotel AI Assistant (SVM + spaCy)", layout="centered")
    st.title("🏨 Astra Imperium Hotel Chatbot (SVM + spaCy)")
    st.caption(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        greeting = responses.get("greeting", "Hello! How can I assist you?")
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "intent" in message:
                st.caption(f"Intent: **{message['intent']}** | Confidence: **{message['confidence']}** | Time: **{message['time']:.4f}s**")
            st.markdown(message["content"])

    # Handle user input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Analyzing query..."):
            intent_name, response, confidence_display, response_time = predict_intent(user_input)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "intent": intent_name,
                "confidence": confidence_display,
                "time": response_time
            })
            st.experimental_rerun()  # Immediately refresh to show new message

if __name__ == "__main__":
    main()
