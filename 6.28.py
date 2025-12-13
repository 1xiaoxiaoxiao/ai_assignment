# =====================================================
# Streamlit Hotel Chatbot (SVM + spaCy NER + Template Responses)
# =====================================================

import streamlit as st
import spacy
from joblib import load
import re
import random
import time

# -------------------------
# 1. Load spaCy Model
# -------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------
# 2. Load Trained SVM Model & Vectorizer
# -------------------------
model = load("intent_model_spacy.joblib")
vectorizer = load("tfidf_vectorizer_spacy.joblib")

# -------------------------
# 3. Template Responses
# -------------------------
responses = {
    "greeting": "Welcome to Solaris Grand Hotel! I'm your virtual assistant. How may I assist you today?",
    "check_functions": "I can provide information on room bookings, hotel facilities, services, events, and general inquiries.",
    "book_hotel": "Reserve a room via www.solarisgrand.com, call +60-3-1234-5678, or visit the Front Desk. A valid ID is required{PERSON}{DATE}{LOCATION}.",
    "cancel_hotel_reservation": "To cancel, contact reservations@solarisgrand.com or call +60-3-1234-5678 with your booking reference{PERSON}{DATE}{LOCATION}.",
    "add_night": "To extend your stay, contact the Front Desk or call +60-3-1234-5678{PERSON}{DATE}{LOCATION}.",
    "bring_pets": "We welcome small pets under 10kg with a RM50 cleaning fee. Service animals are free{PERSON}{DATE}{LOCATION}.",
    "book_parking_space": "Parking can be reserved upon booking or arrival. RM25 per day for hotel guests{PERSON}{DATE}{LOCATION}.",
    "check_hotel_prices": "Room rates vary by date and room type. Visit www.solarisgrand.com or call reservations{PERSON}{DATE}{LOCATION}.",
    "check_room_availability": "Check availability via our booking page or call the Reservations Team{PERSON}{DATE}{LOCATION}.",
    "check_nearby_attractions": "Nearby: Petronas Towers, Pavilion KL, National Museum, Jalan Alor Street Food. City maps at the concierge.",
    "goodbye": "Thank you for choosing Solaris Grand Hotel. We hope to welcome you again soon!",
    "unknown_intent": "I'm sorry, I didn't understand that. Could you please rephrase?"
}

# Intent buttons
PROMPT_MAPPING = {
    "check_functions": "What services do you offer?",
    "book_hotel": "I want to book a room.",
    "check_hotel_prices": "What are your room rates?",
    "check_room_availability": "Do you have rooms available?",
    "add_night": "Can I extend my stay?",
    "bring_pets": "Are pets allowed?",
    "book_parking_space": "Do you have parking available?",
    "check_nearby_attractions": "Any nearby attractions?"
}

# -------------------------
# 4. Helper Functions
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
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

def predict_intent(user_input):
    cleaned = preprocess_text(user_input)
    vec = vectorizer.transform([cleaned])
    try:
        intent = model.predict(vec)[0]
    except:
        intent = "unknown_intent"

    # Rule-based fallback
    text_lower = user_input.lower()
    if "book" in text_lower or "reserve" in text_lower:
        intent = "book_hotel"
    elif "parking" in text_lower:
        intent = "book_parking_space"
    elif "pet" in text_lower or "dog" in text_lower or "cat" in text_lower:
        intent = "bring_pets"
    elif "cancel" in text_lower:
        intent = "cancel_hotel_reservation"
    elif "price" in text_lower or "cost" in text_lower:
        intent = "check_hotel_prices"
    elif "available" in text_lower or "availability" in text_lower:
        intent = "check_room_availability"
    elif "recommend" in text_lower or "nearby" in text_lower:
        intent = "check_nearby_attractions"

    return intent

def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {"PERSON": [], "DATE": [], "GPE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

def fill_entities(template, entities):
    person = ", ".join(entities["PERSON"]) if entities["PERSON"] else ""
    date = ", ".join(entities["DATE"]) if entities["DATE"] else ""
    location = ", ".join(entities["GPE"]) if entities["GPE"] else ""
    person = f" {person}" if person else ""
    date = f" for {date}" if date else ""
    location = f" in {location}" if location else ""
    return template.format(PERSON=person, DATE=date, LOCATION=location)

def chatbot_response(user_input):
    intent = predict_intent(user_input)
    entities = extract_entities(user_input)
    template = responses.get(intent, responses["unknown_intent"])
    response = fill_entities(template, entities)
    return intent, response

# -------------------------
# 5. Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Hotel Chatbot", layout="centered")
    st.title("🏨 Solaris Grand Hotel Chatbot")
    st.caption("Powered by SVM + spaCy NER + Template Responses")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        greeting = responses.get("greeting", "Hello! How may I help you?")
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested buttons
    if "suggested_intents" not in st.session_state:
        st.session_state.suggested_intents = random.sample(list(PROMPT_MAPPING.keys()), min(4, len(PROMPT_MAPPING)))

    cols = st.columns(len(st.session_state.suggested_intents))
    for i, key in enumerate(st.session_state.suggested_intents):
        prompt = PROMPT_MAPPING.get(key, key)
        with cols[i]:
            if st.button(prompt, key=f"btn_{key}", use_container_width=True):
                user_input = prompt
                st.session_state.messages.append({"role": "user", "content": user_input})
                intent, response = chatbot_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.suggested_intents = random.sample(list(PROMPT_MAPPING.keys()), min(4, len(PROMPT_MAPPING)))
                st.experimental_rerun()

    # User input
    user_input = st.chat_input("How can I help you?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        intent, response = chatbot_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()

if __name__ == "__main__":
    main()
