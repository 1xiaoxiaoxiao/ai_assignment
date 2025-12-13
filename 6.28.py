# =====================================================
# Hotel Chatbot (SVM + spaCy NER + Rule-based + Entity Templates)
# Streamlit Version
# =====================================================

import streamlit as st
import spacy
from joblib import load
import re

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
    "invoices": "To request an invoice, visit the Front Desk or email accounting@solarisgrand.com. Include your booking reference for faster processing.",
    "cancellation_fees": "Cancellations made more than 24 hours before check-in are free. Cancellations within 24 hours or no-shows will incur a fee equal to one night's stay.",
    "check_in": "Check-in starts at 3:00 PM. Early check-in depends on room availability. Luggage storage is complimentary. A refundable deposit is required upon arrival.",
    "check_out": "Check-out is at 12:00 PM. Late check-out until 2:00 PM is RM50, subject to availability. Return your room key at departure.",
    "customer_service": "For urgent matters, call the Front Desk at +60-3-1234-5678 or email support@solarisgrand.com.",
    "human_agent": "To speak with a live agent, contact the Front Desk at +60-3-1234-5678 or leave your name and phone number for a callback.",
    "host_event": "To book an event space, email events@solarisgrand.com or call +60-3-1234-5679. Our team will assist with venue setup and catering.",
    "file_complaint": "File a complaint at the Front Desk or email feedback@solarisgrand.com. We respond within 24 hours.",
    "leave_review": "You can leave a review on Google, TripAdvisor, or our website under 'Guest Feedback'. Your feedback is appreciated.",
    "book_hotel": "Reserve a room via www.solarisgrand.com, call +60-3-1234-5678, or visit the Front Desk. A valid ID is required.",
    "cancel_hotel_reservation": "To cancel, contact reservations@solarisgrand.com or call +60-3-1234-5678 with your booking reference.",
    "change_hotel_reservation": "Modify your reservation by emailing reservations@solarisgrand.com or calling +60-3-1234-5678.",
    "check_hotel_facilities": "Facilities include rooftop pool, fitness center, spa, business center, meeting rooms, 24-hour concierge, and dining options.",
    "check_hotel_offers": "See current offers and packages on our website under 'Special Deals' or call reservations for in-house promotions.",
    "check_hotel_prices": "Room rates vary by date and room type. Visit www.solarisgrand.com or call reservations for exact pricing.",
    "check_hotel_reservation": "Check your reservation status by providing your booking reference to the Front Desk or via email reservations@solarisgrand.com.",
    "search_hotel": "Solaris Grand Hotel is at 99 Sunset Boulevard, Kuala Lumpur, 5 minutes from Bukit Bintang MRT Station.",
    "store_luggage": "Complimentary luggage storage is available. Leave your bags before check-in or after check-out while exploring the city.",
    "check_menu": "Our restaurant offers a variety of dishes. View the digital menu in your room via QR code or request a copy at the concierge.",
    "add_night": "To extend your stay, contact the Front Desk or call +60-3-1234-5678. Extensions are subject to availability and rate adjustments.",
    "book_parking_space": "Parking can be reserved upon booking or arrival. RM25 per day for hotel guests, subject to availability.",
    "bring_pets": "We welcome small pets under 10kg with a RM50 cleaning fee. Service animals are free. Pets are not allowed in the dining area or pool.",
    "redeem_points": "Redeem loyalty points for discounts or complimentary nights. Log in online or visit the Front Desk for assistance.",
    "get_refund": "Refunds are processed in 7-14 business days. Contact billing@solarisgrand.com with your booking reference.",
    "shuttle_service": "Airport transfers are available: Sedan RM85, Van RM130. Book 24h in advance. E-hailing options also available (~RM60-70).",
    "check_room_type": "We offer Standard, Deluxe, Executive, and Presidential Suites, each with unique amenities and views.",
    "check_room_availability": "Check availability via our booking page or call the Reservations Team with your dates.",
    "check_nearby_attractions": "Nearby: Petronas Towers, Pavilion KL, National Museum, Jalan Alor Street Food. City maps at the concierge.",
    "check_child_policy": "Children under 12 stay free with existing beds. Baby cots and high chairs available. Babysitting not offered.",
    "check_smoking_policy": "All rooms are non-smoking. RM500 fine for violations. Smoking allowed only in designated outdoor areas.",
    "check_payment_methods": "We accept cash (MYR), Visa, Mastercard, Amex, GrabPay, and Touch 'n Go. Refundable RM100 deposit required.",
    "check_lost_item": "Report lost items to the Front Desk. Our Security Team logs and contacts you when found.",
    "goodbye": "Thank you for choosing Solaris Grand Hotel. We hope to welcome you again soon!",
    "unknown_intent": "I'm sorry, I didn't understand that. Could you please rephrase?"
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
    return response

# -------------------------
# 5. Streamlit App
# -------------------------
st.set_page_config(page_title="Solaris Grand Hotel Chatbot", page_icon="🏨")
st.title("🏨 Solaris Grand Hotel Virtual Assistant")
st.write("Ask me anything about bookings, facilities, services, and more!")

user_input = st.text_input("You:", "")

if user_input:
    reply = chatbot_response(user_input)
    st.markdown(f"**Bot:** {reply}")
