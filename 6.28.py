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
    "greeting":                     "Welcome to Astra Imperium Hotel. I'm your virtual assistant. How may I assist you today?",
    "goodbye":                       "Thank you for choosing Astra Imperium Hotel. We look forward to welcoming you again soon!",
    "unknown":                       "I'm sorry, I don't understand your question. Could you please rephrase?",
    
    # Booking & Reservation
    "book_hotel":                     "Sure{PERSON}! I can help you book a room{LOCATION}{DATE}. Please let me know if you have any special requests.",
    "cancel_hotel_reservation":       "No problem{PERSON}. Your booking{LOCATION}{DATE} has been successfully canceled.",
    "change_hotel_reservation":       "To modify your reservation{LOCATION}{DATE}, please contact our Reservations Team.",
    "add_night":                      "To extend your stay or add extra nights{LOCATION}{DATE}, please contact the Front Desk.",
    "book_parking_space":             "Parking can be reserved{LOCATION}{DATE}. Additional charges may apply.",

    # Hotel Info
    "check_hotel_facilities":         "Our facilities include gym, spa, infinity pool, rooftop lounge, business center, and all-day dining.",
    "check_hotel_offers":             "Current promotions and packages are listed on our website under 'Offers'.",
    "check_hotel_prices":             "Room rates vary by date and room type. Please check our website or contact Reservations for exact pricing.",
    "check_room_type":                "Our room categories include Superior, Deluxe, Premier, Executive Suite, and Astra Imperial Suite.",
    "check_room_availability":        "To check room availability, please provide your preferred dates or check our booking page online.",
    "check_menu":                     "Our restaurant menu is available at SkyDine Restaurant or via QR code in your room.",
    "check_nearby_attractions":       "Nearby attractions include Petronas Twin Towers, Pavilion Bukit Bintang, National Museum, and Jalan Alor Street Food Market.",
    "check_child_policy":             "Children under 12 stay free with existing bedding. Baby cots and high chairs available on request.",
    "check_smoking_policy":           "All rooms are non-smoking. Designated smoking areas are available outside.",
    "check_payment_methods":          "We accept cash, credit cards (Visa, Mastercard, Amex), and e-wallets (GrabPay, Touch 'n Go).",
    "check_lost_item":                "Report lost items immediately to the Front Desk. Security will contact you once located.",
    "check_hotel_reservation":        "To check your reservation status, provide your booking reference to the Front Desk or email bookings@astragroup.com.",

    # Services
    "customer_service":               "For assistance, call the Front Desk or email support@astragroup.com.",
    "human_agent":                    "Connecting you to a hotel representative. Please wait.",
    "host_event":                     "To host an event, email events@astragroup.com or call our Events Team.",
    "file_complaint":                 "To file a complaint, speak to the Duty Manager or email quality@astragroup.com.",
    "leave_review":                   "Leave a review on Google Maps, TripAdvisor, or our website under 'Guest Reviews'.",
    "invoices":                       "Request invoices at the Front Desk or email billing@astraimperium.com with your booking reference.",
    "get_refund":                     "Refunds are processed within 7-14 business days depending on payment method.",
    "redeem_points":                  "Redeem membership points at checkout or via your online account.",
    "shuttle_service":                "We provide airport transfers (sedan RM80, van RM120). Book at least 24 hours in advance.",

    # Check-in / Check-out
    "check_in":                       "Check-in starts at 3:00 PM. Early check-in subject to availability. Security deposit required.",
    "check_out":                      "Check-out is before 12:00 PM. Late check-out until 2:00 PM is RM50 if available.",
    
    # Amenities / Wi-Fi / Pets
    "ask_room_price":                  "Our deluxe room costs RM180 per night. Breakfast and free Wi-Fi included.",
    "ask_wifi":                        "Yes, free Wi-Fi is available in all rooms and public areas.",
    "bring_pets":                      "Pets under 10kg are allowed with a RM50 cleaning fee. Service animals are welcome. Pets not allowed in dining/pool areas.",

    # Miscellaneous
    "search_hotel":                    "Astra Imperium Hotel is located at 18 Jalan Alor, Kuala Lumpur City Centre.",
    "store_luggage":                   "Complimentary luggage storage is available before check-in or after check-out.",
    "check_functions":                 "I can help with room reservations, hotel information, facilities, services, and general inquiries."
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




