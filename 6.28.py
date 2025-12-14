# =====================================================
# Hotel Customer Support Chatbot - Streamlit Version
# SVM + spaCy NER + Multi-turn Slot Filling
# =====================================================

import streamlit as st
import re
import spacy
import time
from joblib import load
from sklearn.metrics import accuracy_score

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
    # 房型识别
    for room in ["single", "double", "deluxe", "premier", "suite"]:
        if room in user_input.lower():
            entities["ROOM_TYPE"].append(room)
    return entities

# =====================================================
# 6. Response Templates
# =====================================================
responses = {
    "greeting": "Welcome to Astra Imperium Hotel. I'm your virtual assistant. How may I assist you today?",
    "check_functions": "I can help with room reservations, hotel information, facilities, services, and general inquiries.",
    "check_in": "Check-in begins at 3:00 PM. Early check-in subject to availability. Security deposit required.",
    "check_out": "Check-out is 12:00 PM. Late check-out until 2:00 PM is RM50 if available.",
    "book_hotel": "To make a reservation, visit www.astraimperium.com, call +60-3-5555-0199, or book at the Front Desk. A valid ID is required.",
    "cancel_hotel_reservation": "To cancel your booking, contact Reservations Team at +60-3-5555-0199 or bookings@astragroup.com.",
    "change_hotel_reservation": "To modify your booking, contact Reservations at +60-3-5555-0199 or email bookings@astragroup.com.",
    "add_night": "To extend your stay or add additional guests, contact the Front Desk.",
    "book_parking_space": "Parking can be reserved during booking or upon arrival. RM20/day for in-house guests.",
    "bring_pets": "Pets under 10kg are allowed with RM50 cleaning fee. Service animals free. Not allowed in dining/pool areas.",
    "goodbye": "Thank you for choosing Astra Imperium Hotel. We look forward to welcoming you again soon!",
    "unknown_intent": "I'm sorry, I don't understand your question."
}

# =====================================================
# 7. Define required slots for each operational intent
# =====================================================
intent_slots = {
    "book_hotel": ["ROOM_TYPE", "DATE"],
    "cancel_hotel_reservation": ["DATE"],
    "change_hotel_reservation": ["DATE", "ROOM_TYPE"],
    "add_night": ["DATE", "ROOM_TYPE"],
    "book_parking_space": ["DATE"]
}
OPERATIONAL_INTENTS = list(intent_slots.keys())

# =====================================================
# 8. Fill entity placeholders
# =====================================================
def fill_entities(template, entities):
    person = ", ".join(entities.get("PERSON", []))
    date = ", ".join(entities.get("DATE", []))
    location = ", ".join(entities.get("GPE", []))
    person = f" {person}" if person else ""
    date = f" for {date}" if date else ""
    location = f" in {location}" if location else ""
    return template.format(PERSON=person, DATE=date, LOCATION=location)

# =====================================================
# 9. Intent Prediction
# =====================================================
def predict_intent(user_input):
    cleaned = preprocess_text(user_input)
    vec = vectorizer.transform([cleaned])
    scores = svm_model.decision_function(vec)
    classes = svm_model.classes_
    best_index = scores.argmax()
    intent = classes[best_index]

    sorted_scores = sorted(scores[0], reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    if intent in OPERATIONAL_INTENTS and margin < CONFIDENCE_MARGIN_THRESHOLD:
        return "unknown", margin
    return intent, margin

# =====================================================
# 10. Generate Response (Multi-turn)
# =====================================================
def generate_response(user_input):
    if "pending_intent" not in st.session_state:
        st.session_state.pending_intent = None
    if "collected_info" not in st.session_state:
        st.session_state.collected_info = {}

    intent, confidence = predict_intent(user_input)
    entities = extract_entities(user_input)

    # Multi-turn for operational intents
    if st.session_state.pending_intent:
        current_intent = st.session_state.pending_intent
        for slot in intent_slots.get(current_intent, []):
            if entities.get(slot):
                st.session_state.collected_info[slot] = entities[slot][0]

        missing = [slot for slot in intent_slots[current_intent] 
                   if slot not in st.session_state.collected_info]
        if missing:
            reply = f"Please provide the following information: {', '.join(missing)}."
            return reply

        reply = f"I have recorded your information: {st.session_state['collected_info']}. Please proceed to the website or Front Desk to complete the operation."
        st.session_state.pending_intent = None
        st.session_state.collected_info = {}
        return reply

    # Trigger multi-turn for new operational intent
    if intent in OPERATIONAL_INTENTS:
        missing_entities = [slot for slot in intent_slots[intent] if not entities.get(slot)]
        if missing_entities:
            st.session_state.pending_intent = intent
            for slot in intent_slots[intent]:
                if entities.get(slot):
                    st.session_state.collected_info[slot] = entities[slot][0]
            reply = f"Sure! I can help you with that. Please provide: {', '.join(missing_entities)}."
            return reply

    # Single-turn reply for info or unknown intents
    template = responses.get(intent, responses.get("unknown_intent"))
    reply = fill_entities(template, entities)
    return reply

# =====================================================
# 11. Streamlit UI
# =====================================================
st.set_page_config(page_title="Hotel AI Chatbot", layout="centered")
st.title("Hotel Customer Support Chatbot")
st.caption("SVM Intent Classification + spaCy NER + Multi-turn Slot Filling")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": responses["greeting"]}]

# show history message
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# user input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    intent, reply, confidence, response_time = generate_response(user_input)

    intent_info = f"<sub>Predicted Intent: {intent} | Confidence: {confidence}</sub>"
    display_reply = f"{intent_info}\n\n{reply}"

    st.session_state.messages.append({"role": "assistant", "content": display_reply})
    st.rerun()



