# =====================================================
# Hotel Customer Support Chatbot
# SVM (Intent Classification) + spaCy (NER)
# Streamlit Final Version (Assignment Ready)
# =====================================================

import streamlit as st
import re
import spacy
import time
from joblib import load

# =====================================================
# 1. Configuration
# =====================================================
CONFIDENCE_MARGIN_THRESHOLD = 0.3

# =====================================================
# 2. Load spaCy Model
# =====================================================
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# =====================================================
# 3. Load ML Model & Vectorizer
# =====================================================
@st.cache_resource
def load_models():
    model = load("intent_model_spacy.joblib")
    vectorizer = load("tfidf_vectorizer_spacy.joblib")
    return model, vectorizer

svm_model, vectorizer = load_models()

# =====================================================
# 4. Text Preprocessing (for SVM)
# =====================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return " ".join(tokens)

# =====================================================
# 5. Entity Extraction (spaCy NER)
# =====================================================
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {"PERSON": [], "DATE": [], "GPE": []}

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)

    return entities

# =====================================================
# 6. Response Templates
# =====================================================
responses = {
    "greeting": "Hello! Welcome to our hotel service. How may I assist you today?",

    "book_hotel": (
        "Sure{PERSON}! I can help you book a room"
        "{LOCATION}{DATE}. Please let me know if you have any special requests."
    ),

    "cancel_booking": (
        "No problem{PERSON}. Your booking"
        "{LOCATION}{DATE} has been successfully canceled."
    ),

    "ask_room_price": "Our deluxe room costs RM180 per night.",

    "ask_wifi": "Yes, free Wi-Fi is available in all rooms and public areas.",

    "ask_checkin_time": "Check-in time starts from 2:00 PM.",

    "ask_checkout_time": "Check-out time is before 12:00 PM.",

    "unknown": "Sorry, I am not sure I understand. Could you please rephrase?"
}

# =====================================================
# 7. Fill Entity Placeholders
# =====================================================
def fill_entities(template, entities):
    person = ", ".join(entities["PERSON"]) if entities["PERSON"] else ""
    date = ", ".join(entities["DATE"]) if entities["DATE"] else ""
    location = ", ".join(entities["GPE"]) if entities["GPE"] else ""

    person = f" {person}" if person else ""
    date = f" for {date}" if date else ""
    location = f" in {location}" if location else ""

    return template.format(
        PERSON=person,
        DATE=date,
        LOCATION=location
    )

# =====================================================
# 8. Intent Prediction (Rule + SVM)
# =====================================================
def predict_intent(user_input):
    start_time = time.time()
    text = user_input.lower()

    # ---------- Rule-based (High precision FAQ) ----------
    if any(k in text for k in ["wifi", "internet"]):
        return "ask_wifi", "Rule", time.time() - start_time

    if any(k in text for k in ["price", "cost", "rate"]):
        return "ask_room_price", "Rule", time.time() - start_time

    if "check in" in text or "check-in" in text:
        return "ask_checkin_time", "Rule", time.time() - start_time

    if "check out" in text or "checkout" in text:
        return "ask_checkout_time", "Rule", time.time() - start_time

    # ---------- ML-based (SVM) ----------
    cleaned = preprocess_text(user_input)
    vec = vectorizer.transform([cleaned])
    scores = svm_model.decision_function(vec)

    best_index = scores.argmax()
    intent = svm_model.classes_[best_index]

    sorted_scores = sorted(scores[0], reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    elapsed = time.time() - start_time

    if margin < CONFIDENCE_MARGIN_THRESHOLD:
        return "unknown", f"Low confidence ({margin:.2f})", elapsed

    return intent, f"SVM ({margin:.2f})", elapsed

# =====================================================
# 9. Generate Final Response
# =====================================================
def generate_response(user_input):
    intent, confidence, response_time = predict_intent(user_input)
    entities = extract_entities(user_input)

    template = responses.get(intent, responses["unknown"])

    if any(tag in template for tag in ["{PERSON}", "{DATE}", "{LOCATION}"]):
        reply = fill_entities(template, entities)
    else:
        reply = template

    return intent, reply, confidence, response_time

# =====================================================
# 10. Streamlit UI
# =====================================================
st.set_page_config(page_title="Hotel AI Chatbot", layout="centered")
st.title("Hotel Customer Support Chatbot")
st.caption("SVM Intent Classification + spaCy NER")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": responses["greeting"]}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "intent" in msg:
            st.caption(
                f"Intent: {msg['intent']} | Confidence: {msg['confidence']} | Time: {msg['time']:.4f}s"
            )
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    intent, reply, confidence, response_time = generate_response(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "intent": intent,
        "confidence": confidence,
        "time": response_time
    })

    st.rerun()
