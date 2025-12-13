# =====================================================
# Hotel Chatbot (SVM + spaCy + Rule-based + Templates)
# Streamlit Version
# =====================================================

import streamlit as st
import re
import spacy
from joblib import load

# -------------------------
# 1. Page Config
# -------------------------
st.set_page_config(
    page_title="Hotel Customer Support Chatbot",
    page_icon="🏨",
    layout="centered"
)

st.title("Hotel Customer Support Chatbot")
st.write("AI-powered hotel assistant using SVM + spaCy")

# -------------------------
# 2. Load spaCy Model
# -------------------------
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# -------------------------
# 3. Load ML Model
# -------------------------
@st.cache_resource
def load_models():
    model = load("intent_model_spacy.joblib")
    vectorizer = load("tfidf_vectorizer_spacy.joblib")
    return model, vectorizer

model, vectorizer = load_models()

# -------------------------
# 4. Text Preprocessing
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.ent_type_ == "PERSON":
            tokens.append("<PERSON>")
        elif token.ent_type_ == "DATE":
            tokens.append("<DATE>")
        elif token.ent_type_ == "GPE":
            tokens.append("<GPE>")
        elif not token.is_stop and token.is_alpha:
            tokens.append(token.lemma_)

    return " ".join(tokens)

# -------------------------
# 5. Intent Prediction
# -------------------------
def predict_intent(user_input):
    cleaned = preprocess_text(user_input)
    vec = vectorizer.transform([cleaned])
    intent = model.predict(vec)[0]

    # Rule-based fallback
    text_lower = user_input.lower()

    if any(k in text_lower for k in ["book", "reserve", "room"]):
        intent = "book_hotel"
    elif "cancel" in text_lower:
        intent = "cancel_booking"
    elif any(k in text_lower for k in ["pet", "dog", "cat"]):
        intent = "bring_pets"
    elif any(k in text_lower for k in ["add night", "extend"]):
        intent = "add_night"
    elif "parking" in text_lower:
        intent = "book_parking"

    return intent

# -------------------------
# 6. Entity Extraction
# -------------------------
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {"PERSON": [], "DATE": [], "GPE": []}

    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)

    return entities

# -------------------------
# 7. Response Templates
# -------------------------
responses = {
    "greeting": "Hello! Welcome to our hotel service. How may I assist you today?",

    "goodbye": "Thank you for choosing our hotel. Have a great day.",

    "book_hotel": (
        "Sure{PERSON}! I can help you with your hotel booking"
        "{LOCATION}{DATE}. Please let me know if you have any special requests."
    ),

    "cancel_booking": (
        "No problem{PERSON}. Your booking"
        "{LOCATION}{DATE} has been successfully canceled."
    ),

    "bring_pets": (
        "Yes{PERSON}, pets are allowed at our hotel"
        "{LOCATION}{DATE}. Additional charges may apply."
    ),

    "add_night": (
        "Certainly{PERSON}. Your stay has been extended by one night"
        "{LOCATION}{DATE}."
    ),

    "book_parking": (
        "Parking has been successfully reserved for you"
        "{LOCATION}{DATE}{PERSON}."
    ),

    "ask_room_price": (
        "Our deluxe room is priced at RM180 per night. "
        "Breakfast and free Wi-Fi are included."
    ),

    "ask_booking": (
        "I can assist you with a room booking. "
        "Please provide your check-in date and number of guests."
    ),

    "ask_checkin_time": "Check-in time starts from 2:00 PM.",

    "ask_checkout_time": "Check-out time is before 12:00 PM."
}

# -------------------------
# 8. Fill Entity Placeholders
# -------------------------
def fill_entities(template, entities):
    person = ", ".join(entities["PERSON"]) if entities["PERSON"] else ""
    date = ", ".join(entities["DATE"]) if entities["DATE"] else ""
    location = ", ".join(entities["GPE"]) if entities["GPE"] else ""

    person = f" {person}" if person else ""
    date = f" for {date}" if date else ""
    location = f" in {location}" if location else ""

    return template.format(PERSON=person, DATE=date, LOCATION=location)

# -------------------------
# 9. Chatbot Logic
# -------------------------
def chatbot_response(user_input):
    intent = predict_intent(user_input)
    entities = extract_entities(user_input)
    template = responses.get(intent, "Sorry, I do not understand your request.")

    if any(tag in template for tag in ["{PERSON}", "{DATE}", "{LOCATION}"]):
        return fill_entities(template, entities)
    else:
        return template

# -------------------------
# 10. Chat UI
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    reply = chatbot_response(user_input)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

    with st.chat_message("assistant"):
        st.markdown(reply)
