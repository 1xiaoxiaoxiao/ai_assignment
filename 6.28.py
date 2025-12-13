# =====================================================
# Solaris Grand Hotel Chatbot
# Streamlit Version with SVM + spaCy NER + Template Responses + Suggested Questions
# =====================================================

import streamlit as st
import spacy
from joblib import load
import random
import re

# -------------------------
# 1. Load spaCy model
# -------------------------
nlp = spacy.load("en_core_web_sm")

# -------------------------
# 2. Load trained SVM model & vectorizer
# -------------------------
model = load("intent_model_spacy.joblib")
vectorizer = load("tfidf_vectorizer_spacy.joblib")

# -------------------------
# 3. Template Responses
# -------------------------
responses = {
    "greeting": "Welcome to Solaris Grand Hotel! I'm your virtual assistant. How may I assist you today?",
    "book_hotel": "Reserve a room{PERSON}{LOCATION}{DATE}. Visit www.solarisgrand.com or call +60-3-1234-5678.",
    "cancel_hotel_reservation": "Your booking has been canceled{PERSON}{LOCATION}{DATE}. Contact reservations@solarisgrand.com.",
    "check_hotel_prices": "Room rates vary by date and type{DATE}. Check www.solarisgrand.com for details.",
    "check_room_availability": "You can check room availability{DATE} on our booking page.",
    "check_nearby_attractions": "Nearby attractions include Petronas Towers, Pavilion KL, National Museum{LOCATION}.",
    "bring_pets": "Pets are allowed under 10kg with a RM50 fee{PERSON}. Service animals are free.",
    "add_night": "Extend your stay by contacting the Front Desk{PERSON}{DATE}{LOCATION}.",
    "book_parking_space": "Parking can be reserved{DATE}{LOCATION} for RM25 per day.",
    "unknown_intent": "I'm sorry, I didn't understand that. Could you please rephrase?"
}

# --- Suggested Questions / Buttons ---
PROMPT_MAPPING = {
    "book_hotel": "I want to book a room.",
    "cancel_hotel_reservation": "I want to cancel my booking.",
    "check_hotel_prices": "What are your room rates?",
    "check_room_availability": "Do you have rooms available?",
    "check_nearby_attractions": "What attractions are nearby?",
    "bring_pets": "Are pets allowed?",
    "add_night": "Can I extend my stay?",
    "book_parking_space": "Do you have parking available?"
}

EXCLUDED_FROM_SUGGESTIONS = ["greeting", "unknown_intent"]
SUGGESTED_INTENTS = [key for key in PROMPT_MAPPING.keys() if key not in EXCLUDED_FROM_SUGGESTIONS]

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
    entities = {"PERSON": None, "DATE": None, "GPE": None}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_] = ent.text
    return entities

def fill_entities(template, context_entities):
    person = f" for {context_entities['PERSON']}" if context_entities['PERSON'] else ""
    date = f" on {context_entities['DATE']}" if context_entities['DATE'] else ""
    location = f" in {context_entities['GPE']}" if context_entities['GPE'] else ""
    return template.format(PERSON=person, DATE=date, LOCATION=location)

# -------------------------
# 5. Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Solaris Grand Hotel Chatbot", layout="centered")
    st.title("🏨 Solaris Grand Hotel Chatbot")
    st.caption("Powered by SVM + spaCy NER + Template Responses")
    
    # Initialize chat history & context
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": responses["greeting"]}]
        st.session_state.context_entities = {"PERSON": None, "DATE": None, "GPE": None}
        st.session_state.random_intents = random.sample(SUGGESTED_INTENTS, min(4, len(SUGGESTED_INTENTS)))
        st.session_state.pending_input = None

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested buttons
    if st.session_state.random_intents:
        st.markdown("**Suggested Questions:**")
        cols = st.columns(len(st.session_state.random_intents))
        for i, intent_key in enumerate(st.session_state.random_intents):
            with cols[i]:
                if st.button(PROMPT_MAPPING[intent_key], key=f"btn_{intent_key}", use_container_width=True):
                    st.session_state.pending_input = PROMPT_MAPPING[intent_key]
                    st.session_state.random_intents = random.sample(SUGGESTED_INTENTS, min(4, len(SUGGESTED_INTENTS)))

    # Handle user input
    if st.session_state.pending_input:
        user_input = st.session_state.pending_input
        st.session_state.pending_input = None
    else:
        user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Update context entities
        new_entities = extract_entities(user_input)
        for key in st.session_state.context_entities:
            if new_entities[key]:
                st.session_state.context_entities[key] = new_entities[key]
        # Predict intent and respond
        intent = predict_intent(user_input)
        reply = fill_entities(responses.get(intent, responses["unknown_intent"]), st.session_state.context_entities)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.experimental_rerun()  # refresh the chat interface

if __name__ == "__main__":
    main()
