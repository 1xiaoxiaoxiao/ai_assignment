import streamlit as st
import re
import spacy
import time
from joblib import load

# -----------------------------
# Load spaCy
# -----------------------------
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# -----------------------------
# Load SVM model & vectorizer
# -----------------------------
@st.cache_resource
def load_models():
    model = load("intent_model_spacy.joblib")
    vectorizer = load("tfidf_vectorizer_spacy.joblib")
    return model, vectorizer

svm_model, vectorizer = load_models()

# -----------------------------
# Text preprocessing
# -----------------------------
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

# -----------------------------
# Intent prediction
# -----------------------------
def predict_intent(user_input):
    cleaned = preprocess_text(user_input)
    vec = vectorizer.transform([cleaned])
    scores = svm_model.decision_function(vec)
    classes = svm_model.classes_

    best_index = scores.argmax()
    intent = classes[best_index]
    confidence = scores[0][best_index]

    return intent, confidence

# -----------------------------
# FAQ Responses
# -----------------------------
responses = {
    "greeting": "Welcome to Astra Imperium Hotel. I'm your virtual assistant. How may I assist you today?",
    "check_functions": "I can help with room reservations, hotel information, facilities, services, and general inquiries.",

    "invoices": "To request an invoice, please visit the Front Desk or email billing@astraimperium.com.",
    "cancellation_fees": "Cancellations are free up to 24 hours before check-in. Late cancellations incur one-night charges.",

    "check_in": "Check-in begins at 3:00 PM. Early check-in is subject to availability. A refundable security deposit is required.",
    "check_out": "Check-out time is 12:00 PM. Late check-out until 2:00 PM is available for RM50.",

    "customer_service": "For urgent assistance, call the Front Desk at +60-3-5555-0199.",
    "human_agent": "To speak with a hotel representative, please contact the Front Desk.",

    "host_event": "To host an event, please email events@astragroup.com or call +60-3-5555-0200.",
    "file_complaint": "To file a complaint, please email quality@astragroup.com.",

    "leave_review": "You may leave a review on Google Maps, TripAdvisor, or our website.",

    "book_hotel": "To make a reservation, visit www.astraimperium.com or contact Reservations.",
    "cancel_hotel_reservation": "To cancel your reservation, contact bookings@astragroup.com.",
    "change_hotel_reservation": "To modify your reservation, contact the Reservations Team.",

    "check_hotel_facilities": "Facilities include an infinity pool, gym, spa, rooftop lounge, and business centre.",
    "check_hotel_offers": "Current promotions are available on our website under the Offers section.",
    "check_hotel_prices": "Room prices vary by date and availability. Please check our website.",

    "search_hotel": "Astra Imperium Hotel is located at Jalan Alor, Kuala Lumpur City Centre.",
    "store_luggage": "Complimentary luggage storage is available 24/7.",

    "check_menu": "Our restaurant menu is available at SkyDine Restaurant (Level 8).",

    "add_night": "To extend your stay, please contact the Front Desk.",
    "book_parking_space": "Parking is available at RM20 per day for in-house guests.",

    "bring_pets": "Pets under 10kg are allowed with a RM50 cleaning fee.",

    "redeem_points": "Astra Rewards members may redeem points at the Front Desk or online.",
    "get_refund": "Refunds are processed within 7–14 business days.",

    "shuttle_service": "Airport transfer is available from RM80. Advance booking is required.",

    "check_room_type": "Room types include Superior, Deluxe, Premier, and Suites.",
    "check_room_availability": "Please check availability via our website or Reservations Team.",

    "check_nearby_attractions": "Nearby attractions include Petronas Twin Towers and Pavilion Bukit Bintang.",

    "check_child_policy": "Children under 12 stay free using existing bedding.",
    "check_smoking_policy": "All rooms are non-smoking. A RM500 penalty applies.",
    "check_payment_methods": "We accept cash, credit cards, and major e-wallets.",

    "check_lost_item": "Please report lost items to the Front Desk.",

    "goodbye": "Thank you for choosing Astra Imperium Hotel. We hope to see you again!",
    "unknown_intent": "I'm sorry, I don't understand your question."
}

# -----------------------------
# Generate response
# -----------------------------
def generate_response(user_input):
    intent, confidence = predict_intent(user_input)
    reply = responses.get(intent, responses["unknown_intent"])
    return intent, reply, confidence

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hotel FAQ Chatbot", layout="centered")
st.title("Astra Imperium Hotel FAQ Chatbot")
st.caption("SVM-based Intent Classification (Single-turn FAQ)")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": responses["greeting"]}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a hotel-related question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    intent, reply, confidence = generate_response(user_input)

    info = f"<sub>Predicted Intent: {intent} | Confidence: {confidence:.2f}</sub>"
    final_reply = f"{info}\n\n{reply}"

    st.session_state.messages.append(
        {"role": "assistant", "content": final_reply}
    )
    st.rerun()
