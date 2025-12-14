# =====================================================
# Hotel Customer Support Chatbot (Streamlit + SVM + spaCy)
# Multi-turn slot filling version with PERSON slot
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
# 4. Text Preprocessing
# =====================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# =====================================================
# 5. Entity Extraction (NER)
# =====================================================
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {"PERSON": [], "DATE": [], "GPE": [], "ROOM_TYPE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    # 规则识别房型
    for room in ["single", "double", "deluxe", "premier", "suite"]:
        if room in user_input.lower():
            entities["ROOM_TYPE"].append(room)
    return entities

# =====================================================
# 6. Response Templates
# =====================================================
responses = {
    "greeting": "Welcome to Astra Imperium Hotel. How may I assist you today?",
    "goodbye": "Thank you for choosing Astra Imperium Hotel. We look forward to welcoming you again soon!",
    "unknown": "I'm sorry, I don't understand your question. Could you please rephrase?",
    "book_hotel": "Sure{PERSON}! I can help you book a room{ROOM_TYPE}{DATE}.",
    "cancel_hotel_reservation": "I can help you cancel your booking{PERSON}{DATE}.",
    "change_hotel_reservation": "To modify your reservation{ROOM_TYPE}{DATE}, please contact our Reservations Team.",
    "add_night": "To extend your stay or add extra nights{ROOM_TYPE}{DATE}, please contact the Front Desk.",
    "book_parking_space": "Parking can be reserved{DATE}. Additional charges may apply.",
    "ask_room_price": "Our deluxe room costs RM180 per night. Breakfast and free Wi-Fi included.",
    "ask_wifi": "Yes, free Wi-Fi is available in all rooms and public areas.",
    "check_in": "Check-in starts at 3:00 PM. Early check-in subject to availability. Security deposit required.",
    "check_out": "Check-out is before 12:00 PM. Late check-out until 2:00 PM is RM50 if available."
}

# =====================================================
# 7. Define required slots for each intent
# =====================================================
intent_slots = {
    "book_hotel": ["PERSON", "ROOM_TYPE", "DATE"],
    "cancel_hotel_reservation": ["PERSON", "DATE"],
    "change_hotel_reservation": ["DATE", "ROOM_TYPE"],
    "add_night": ["DATE", "ROOM_TYPE", "ADDITIONAL_BED"],
    "book_parking_space": ["DATE", "VEHICLE_TYPE"]
}

# =====================================================
# 8. Fill entity placeholders
# =====================================================
def fill_entities(template, entities):
    person = ", ".join(entities.get("PERSON", []))
    date = ", ".join(entities.get("DATE", []))
    room_type = ", ".join(entities.get("ROOM_TYPE", []))
    
    person = f" {person}" if person else ""
    date = f" for {date}" if date else ""
    room_type = f" ({room_type})" if room_type else ""
    
    return template.format(PERSON=person, DATE=date, ROOM_TYPE=room_type)

# =====================================================
# 9. Intent Prediction
# =====================================================
def predict_intent(user_input):
    start_time = time.time()
    text = user_input.lower()

    # --- Rule-based ---
    if any(k in text for k in ["wifi", "internet"]):
        return "ask_wifi", "Rule", time.time() - start_time
    if any(k in text for k in ["price", "cost", "rate"]):
        return "ask_room_price", "Rule", time.time() - start_time
    if "check in" in text or "check-in" in text:
        return "check_in", "Rule", time.time() - start_time
    if "check out" in text or "checkout" in text:
        return "check_out", "Rule", time.time() - start_time

    # --- ML-based ---
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
# 10. Generate Response (Multi-turn)
# =====================================================
def generate_response(user_input):
    if "pending_intent" not in st.session_state:
        st.session_state.pending_inte
