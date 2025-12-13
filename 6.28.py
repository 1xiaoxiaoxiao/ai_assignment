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
    "greeting": "Hello! Welcome to our hotel service. How may I assist you today?",
    "goodbye": "Thank you for visiting. Have a nice day!",
    "book_hotel": "Sure{PERSON}! I can help you book a room{LOCATION}{DATE}.",
    "cancel_hotel_reservation": "Your booking{LOCATION}{DATE} has been successfully canceled.",
    "add_night": "Your stay has been extended by one night{LOCATION}{DATE}.",
    "bring_pets": "Yes{PERSON}, pets are allowed at our hotel{LOCATION}{DATE}. Additional charges may apply.",
    "book_parking_space": "Parking has been successfully reserved for you{LOCATION}{DATE}{PERSON}.",
    "cancellation_fees": "Early cancellation fee is RM50 if canceled within 24 hours of check-in.",
    "change_hotel_reservation": "You can change your reservation by contacting the front desk or online portal.",
    "check_child_policy": "Children under 12 stay free. Accessibility services are available on request.",
    "check_functions": "I can assist with bookings, cancellations, room info, and hotel services.",
    "check_hotel_facilities": "The hotel offers gym, swimming pool, spa, and conference rooms.",
    "check_hotel_offers": "Current offers include 10% off for early bookings and weekend discounts.",
    "check_hotel_prices": "Room prices vary: Deluxe RM180, Suite RM250, Executive RM300 per night.",
    "check_hotel_reservation": "You can check your reservation status online or via email confirmation.",
    "check_in": "Check-in starts at 2:00 PM.",
    "check_out": "Check-out is before 12:00 PM.",
    "check_lost_item": "Report lost items at the reception desk immediately.",
    "check_menu": "Our restaurant menu includes local and international cuisine.",
    "check_nearby_attractions": "Nearby attractions include the city museum, beach, and shopping mall.",
    "check_payment_methods": "We accept cash, credit cards, and online payments.",
    "check_room_availability": "Please provide dates to check room availability.",
    "check_room_type": "We offer Single, Double, Deluxe, and Suite rooms.",
    "check_smoking_policy": "All rooms are non-smoking except designated areas.",
    "customer_service": "You can contact the front desk at extension 0.",
    "file_complaint": "You can file a complaint at reception or via our website form.",
    "get_refund": "Refunds are processed within 5 business days after cancellation.",
    "host_event": "You can rent event spaces by contacting our events coordinator.",
    "human_agent": "Connecting you to a human agent, please wait.",
    "invoices": "Invoices are emailed automatically after your stay.",
    "leave_review": "You can leave a review on our website or TripAdvisor.",
    "redeem_points": "Redeem your membership points at checkout or online portal.",
    "search_hotel": "Our hotel is located at 123 Main Street, City Center.",
    "shuttle_service": "We offer shuttle service to the airport and nearby attractions.",
    "store_luggage": "Luggage storage is available at reception before check-in and after check-out.",
    "unknown": "Sorry, I am not sure I understand. Could you please rephrase?"
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

