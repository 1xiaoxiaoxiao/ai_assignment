# =====================================================
# Hotel Customer Support Chatbot (Multi-turn, Info Recording)
# SVM + spaCy NER + Streamlit
# =====================================================
import streamlit as st
import re
import spacy
import time
from joblib import load

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
# 3. Load SVM Model & Vectorizer
# =====================================================
@st.cache_resource
def load_models():
    model = load("intent_model_spacy.joblib")
    vectorizer = load("tfidf_vectorizer_spacy.joblib")
    return model, vectorizer

svm_model, vectorizer = load_models()

# =====================================================
# 4. Preprocess text
# =====================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# =====================================================
# 5. Entity Extraction
# =====================================================
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {"PERSON": [], "DATE": [], "GPE": [], "ROOM_TYPE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    # 额外简单正则提取房型
    room_types = ["superior", "deluxe", "premier", "executive suite", "astra imperial suite"]
    for rt in room_types:
        if rt in user_input.lower() and rt not in entities["ROOM_TYPE"]:
            entities["ROOM_TYPE"].append(rt)
    return entities

# =====================================================
# 6. Response Templates
# =====================================================
responses = {
    "greeting": "Welcome to Astra Imperium Hotel. I'm your virtual assistant. How may I assist you today?",
    "goodbye": "Thank you for choosing Astra Imperium Hotel. We look forward to welcoming you again soon!",
    "unknown": "I'm sorry, I don't understand your question. Could you please rephrase?",

    # Booking & Reservation
    "book_hotel": "Sure{PERSON}! I can help you book a room{LOCATION}{DATE}. Please provide the room type and check-in/check-out dates.",
    "cancel_hotel_reservation": "Sure{PERSON}. I can help you cancel a booking. Please provide your booking date and name used for reservation.",
    "change_hotel_reservation": "I can help you modify your booking. Please provide the current booking details and desired changes.",
    "add_night": "To extend your stay, please provide your current booking date and room type.",
    "book_parking_space": "Please provide your booking date and vehicle details to reserve parking.",
    
    # After collecting info
    "recorded_info": "Thank you! I have recorded your information: {COLLECTED_INFO}. Please proceed to our website or Front Desk to complete the operation.",

    # FAQ
    "check_hotel_facilities": "Our facilities include gym, spa, infinity pool, rooftop lounge, business center, and all-day dining.",
    "check_hotel_offers": "Current promotions and packages are listed on our website under 'Offers'.",
    "check_hotel_prices": "Room rates vary by date and room type. Please check our website or contact Reservations for exact pricing.",
    "check_room_type": "Our room categories include Superior, Deluxe, Premier, Executive Suite, and Astra Imperial Suite.",
    "check_room_availability": "To check room availability, please provide your preferred dates or check our booking page online.",
    "check_menu": "Our restaurant menu is available at SkyDine Restaurant or via QR code in your room.",
    "check_nearby_attractions": "Nearby attractions include Petronas Twin Towers, Pavilion Bukit Bintang, National Museum, and Jalan Alor Street Food Market.",
    "ask_room_price": "Our deluxe room costs RM180 per night. Breakfast and free Wi-Fi included.",
    "ask_wifi": "Yes, free Wi-Fi is available in all rooms and public areas.",
    "bring_pets": "Pets under 10kg are allowed with a RM50 cleaning fee. Service animals are welcome. Pets not allowed in dining/pool areas.",
}

# =====================================================
# 7. Fill Entities
# =====================================================
def fill_entities(template, entities, collected_info=None):
    person = ", ".join(entities.get("PERSON", [])) if entities.get("PERSON") else ""
    date = ", ".join(entities.get("DATE", [])) if entities.get("DATE") else ""
    location = ", ".join(entities.get("GPE", [])) if entities.get("GPE") else ""
    room_type = ", ".join(entities.get("ROOM_TYPE", [])) if entities.get("ROOM_TYPE") else ""
    
    person = f" {person}" if person else ""
    date = f" for {date}" if date else ""
    location = f" in {location}" if location else ""
    
    reply = template.format(PERSON=person, DATE=date, LOCATION=location)
    
    if collected_info:
        reply = responses["recorded_info"].format(COLLECTED_INFO=collected_info)
    
    return reply

# =====================================================
# 8. Predict Intent (Rule + SVM)
# =====================================================
def predict_intent(user_input):
    start_time = time.time()
    text = user_input.lower()
    
    # Rule-based
    if any(k in text for k in ["wifi", "internet"]):
        return "ask_wifi", "Rule", time.time() - start_time
    if any(k in text for k in ["price", "cost", "rate"]):
        return "ask_room_price", "Rule", time.time() - start_time
    
    # ML-based
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
# 9. Multi-turn State Storage
# =====================================================
if "collected_info" not in st.session_state:
    st.session_state.collected_info = {}

if "pending_intent" not in st.session_state:
    st.session_state.pending_intent = None

# =====================================================
# 10. Generate Response
# =====================================================
def generate_response(user_input):
    intent, confidence, response_time = predict_intent(user_input)
    entities = extract_entities(user_input)
    
    # 如果之前有待完成的意图（multi-turn）
    if st.session_state.pending_intent:
        # 更新收集信息
        for k, v in entities.items():
            if v:
                st.session_state.collected_info[k] = v
        reply = fill_entities("", {}, collected_info=st.session_state.collected_info)
        st.session_state.pending_intent = None
        st.session_state.collected_info = {}
        return st.session_state.pending_intent or intent, reply, confidence, response_time
    
    # 否则新意图
    missing_entities = any(v==[] for v in entities.values())
    if intent in ["book_hotel", "cancel_hotel_reservation", "change_hotel_reservation", "add_night", "book_parking_space"] and missing_entities:
        # 设置为待收集状态
        st.session_state.pending_intent = intent
        for k, v in entities.items():
            if v:
                st.session_state.collected_info[k] = v
        template = responses.get(intent, responses["unknown"])
        reply = fill_entities(template, entities)
        return intent, reply, confidence, response_time
    
    template = responses.get(intent, responses["unknown"])
    reply = fill_entities(template, entities)
    return intent, reply, confidence, response_time

# =====================================================
# 11. Streamlit UI
# =====================================================
st.set_page_config(page_title="Hotel AI Chatbot", layout="centered")
st.title("Hotel Customer Support Chatbot")
st.caption("SVM Intent Classification + spaCy NER + Multi-turn Info Collection")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": responses["greeting"]}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    intent, reply, confidence, response_time = generate_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()
