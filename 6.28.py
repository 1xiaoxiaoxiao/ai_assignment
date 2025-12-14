import streamlit as st
import re
import spacy
import time
from joblib import load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# =====================================================
# 1. Configuration
# =====================================================
CONFIDENCE_MARGIN_THRESHOLD = 0.3
SMOOTH_FN = SmoothingFunction().method1  # BLEU smoothing

# =====================================================
# 2. Load spaCy Model
# =====================================================
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# =====================================================
# 3. Load SVM Model & TF-IDF Vectorizer
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
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
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
    "greeting": "Welcome to Astra Imperium Hotel. I'm your virtual assistant. How may I assist you today?",
    "goodbye": "Thank you for choosing Astra Imperium Hotel. We look forward to welcoming you again soon!",
    "unknown_intent": "I'm sorry, I don't understand your question. Could you please rephrase?",

    # Booking-related
    "book_hotel": (
        "I can assist you with booking a room{LOCATION}{DATE}. "
        "Please visit our website or contact the Reservations Team to finalize your booking."
    ),
    "cancel_hotel_reservation": (
        "I understand you want to cancel your booking{LOCATION}{DATE}. "
        "Please contact our Reservations Team at +60-3-5555-0199 or email bookings@astragroup.com to complete the cancellation."
    ),
    "change_hotel_reservation": (
        "If you want to change your booking{LOCATION}{DATE}, "
        "please reach out to our Reservations Team at +60-3-5555-0199 or via email bookings@astragroup.com."
    ),
    "add_night": (
        "To extend your stay{LOCATION}{DATE}, please contact the Front Desk or Reservations Team. "
        "Extensions depend on availability and rate adjustments."
    ),
    "book_parking_space": (
        "Parking can be reserved during booking or upon arrival. "
        "Please contact the Reservations Team for availability and rates."
    ),

    # Facilities & Services
    "bring_pets": (
        "Astra Imperium is pet-friendly. Dogs and cats under 10kg are allowed with a cleaning fee. "
        "Service animals are welcome. Pets are not permitted in dining or pool areas."
    ),
    "ask_wifi": "Yes, free Wi-Fi is available in all rooms and public areas.",
    "check_hotel_facilities": (
        "Our facilities include an infinity pool, rooftop lounge, fitness centre, spa, business centre, "
        "event halls, all-day dining restaurant, and 24-hour concierge service."
    ),
    "check_hotel_offers": (
        "Current promotions and packages are listed on our website under the 'Offers' section. "
        "You may also call the Reservations Team for exclusive in-house deals."
    ),
    "check_hotel_prices": (
        "Room rates vary by date, room type, and availability. "
        "For accurate pricing, please check our website or contact the Reservations Team."
    ),
    "check_room_type": "Our room categories include Superior, Deluxe, Premier, Executive Suite, and the Astra Imperial Suite.",
    "check_room_availability": (
        "To check room availability, please visit our website's booking page or contact the Reservations Team."
    ),
    "check_nearby_attractions": (
        "Nearby attractions include the Petronas Twin Towers, Pavilion Bukit Bintang, "
        "the National Museum, and Jalan Alor Street Food Market."
    ),
    "check_child_policy": (
        "Children under 12 stay for free using existing bedding. Baby cots and high chairs are available on request."
    ),
    "check_smoking_policy": (
        "All guest rooms are non-smoking. Smoking is allowed only in designated outdoor areas."
    ),
    "check_payment_methods": (
        "We accept cash (MYR), major credit cards, and e-wallets. A refundable security deposit is required during check-in."
    ),
    "check_lost_item": (
        "For lost items, please report immediately to the Front Desk. "
        "Our Security Team will review the Lost & Found log and contact you once the item is located."
    ),
    "check_menu": (
        "Our restaurant menu is available at the SkyDine Restaurant (Level 8) or via the QR code in your room."
    ),
    "shuttle_service": (
        "We provide private airport transfers and can also guide you to e-hailing options. "
        "Please contact the Front Desk to arrange transport."
    ),
    "store_luggage": (
        "Complimentary luggage storage is available 24/7. You may store your bags before check-in or after check-out."
    ),

    # Customer support & feedback
    "customer_service": (
        "For urgent assistance, please call the Front Desk at +60-3-5555-0199 or email support@astragroup.com."
    ),
    "human_agent": (
        "To speak with a hotel representative, please contact the Front Desk at +60-3-5555-0199 or request a callback."
    ),
    "file_complaint": (
        "To file a complaint, speak to the Duty Manager at the Front Desk or email quality@astragroup.com."
    ),
    "leave_review": (
        "You may leave a review on Google Maps, TripAdvisor, or our website under the 'Guest Reviews' section."
    ),
    "redeem_points": (
        "If you are an Astra Rewards member, you may redeem points for discounts or complimentary nights. "
        "Visit the Front Desk or your member account online for assistance."
    ),
    "get_refund": (
        "Refunds are processed within 7-14 business days depending on payment method. "
        "Contact billing@astragroup.com with your booking reference number."
    ),
    "invoices": (
        "To request an invoice, please visit the Front Desk or email billing@astraimperium.com. "
        "Provide your booking reference number for quicker processing."
    ),
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
    return template.format(PERSON=person, DATE=date, LOCATION=location)

# =====================================================
# 8. Intent Prediction (Rule + SVM)
# =====================================================
def predict_intent(user_input):
    start_time = time.time()
    text = user_input.lower()

    # --- Rule-based FAQ ---
    if any(k in text for k in ["wifi", "internet"]):
        return "ask_wifi", "Rule", time.time() - start_time
    if any(k in text for k in ["price", "cost", "rate"]):
        return "ask_room_price", "Rule", time.time() - start_time
    if "check in" in text or "check-in" in text:
        return "ask_checkin_time", "Rule", time.time() - start_time
    if "check out" in text or "checkout" in text:
        return "ask_checkout_time", "Rule", time.time() - start_time

    # --- ML-based (SVM) ---
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
# 9. Generate Response + BLEU
# =====================================================
def generate_response(user_input, reference=None):
    intent, confidence, response_time = predict_intent(user_input)
    entities = extract_entities(user_input)
    template = responses.get(intent, responses["unknown"])
    reply = fill_entities(template, entities) if any(tag in template for tag in ["{PERSON}", "{DATE}", "{LOCATION}"]) else template

    # --- BLEU Score ---
    bleu_score = 0.0
    if reference:
        bleu_score = sentence_bleu([reference.split()], reply.split(), smoothing_function=SMOOTH_FN)

    return intent, reply, confidence, response_time, bleu_score

# =====================================================
# 10. Streamlit UI
# =====================================================
st.set_page_config(page_title="Hotel AI Chatbot", layout="centered")
st.title("Hotel Customer Support Chatbot")
st.caption("SVM Intent Classification + spaCy NER")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": responses["greeting"]}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "intent" in msg:
            st.caption(
                f"Intent: {msg['intent']} | Confidence: {msg['confidence']} | Time: {msg['time']:.4f}s"
            )
            if "bleu" in msg:
                st.caption(f"BLEU: {msg['bleu']:.4f}")
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 可选：提供参考回复用于BLEU计算
    reference_response = None  # e.g., "I can assist you with booking a room."
    intent, reply, confidence, response_time, bleu_score = generate_response(user_input, reference=reference_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "intent": intent,
        "confidence": confidence,
        "time": response_time,
        "bleu": bleu_score
    })

    st.rerun()



