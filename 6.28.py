import streamlit as st
import spacy
from joblib import load
import re
import random

# 1. Load spaCy model
nlp = spacy.load("en_core_web_sm")

# 2. Load trained SVM model & vectorizer
model = load("intent_model_spacy.joblib")
vectorizer = load("tfidf_vectorizer_spacy.joblib")

# 3. Responses
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

# 4. Helper functions
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
    return fill_entities(template, entities), intent

# 5. Streamlit App
def main():
    st.set_page_config(page_title="Hotel Chatbot", layout="centered")
    st.title("🏨 Solaris Grand Hotel Chatbot")
    st.markdown("Ask about bookings, facilities, policies, or nearby attractions!")

    # Initialize chat state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": responses["greeting"]}]

    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input form to handle submission immediately
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message")
        submit = st.form_submit_button("Send")
        if submit and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            reply, intent = chatbot_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.experimental_rerun()  # refresh to show the new message immediately

if __name__ == "__main__":
    main()
