# =====================================================
# Hotel Customer Support Chatbot (Streamlit + SVM + spaCy)
# Multi-turn Slot Filling + Evaluation
# =====================================================

import streamlit as st
import re
import spacy
import time
from joblib import load
from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

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
# 3. Load ML Model & Vectorizer
# =====================================================
@st.cache_resource
def load_models():
    model = load("intent_model_spacy.joblib")
    vectorizer = load("tfidf_vectorizer_spacy.joblib")
    return model, vectorizer

svm_model, vectorizer = load_models()

# =====================================================
# 4. Text Preprocessing
# =====================================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# =====================================================
# 5. Entity Extraction (NER)
# =====================================================
def extract_entities(user_input):
    doc = nlp(user_input)
    entities = {"PERSON": [], "DATE": [], "GPE": [], "ROOM_TYPE": []}  # ROOM_TYPE手动识别
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    # 简单规则识别房型
    for room in ["single", "double", "deluxe", "premier", "suite"]:
        if room in user_input.lower():
            entities["ROOM_TYPE"].append(room)
    return entities

# =====================================================
# 6. Response Templates
# =====================================================
responses = {
    "greeting": "Welcome to Astra Imperium Hotel. How may I assist you today?",
    "goodbye": "Thank you for choosing Astra Imperium Hotel. We look forward to welcoming you again soon!",
    "unknown": "I'm sorry, I don't understand your question. Could you please rephrase?",
    "book_hotel": "Sure{PERSON}! I can help you book a room{LOCATION}{DATE}.",
    "cancel_hotel_reservation": "I can help you cancel your booking{LOCATION}{DATE}.",
    "change_hotel_reservation": "To modify your reservation{LOCATION}{DATE}, please contact our Reservations Team.",
    "add_night": "To extend your stay or add extra nights{LOCATION}{DATE}, please contact the Front Desk.",
    "book_parking_space": "Parking can be reserved{LOCATION}{DATE}. Additional charges may apply.",
    "ask_room_price": "Our deluxe room costs RM180 per night. Breakfast and free Wi-Fi included.",
    "ask_wifi": "Yes, free Wi-Fi is available in all rooms and public areas.",
    "check_in": "Check-in starts at 3:00 PM. Early check-in subject to availability. Security deposit required.",
    "check_out": "Check-out is before 12:00 PM. Late check-out until 2:00 PM is RM50 if available."
}

# =====================================================
# 7. Define required slots for each intent
# =====================================================
intent_slots = {
    "book_hotel": ["ROOM_TYPE", "DATE"],
    "cancel_hotel_reservation": ["DATE"],
    "change_hotel_reservation": ["DATE", "ROOM_TYPE"],
    "add_night": ["DATE", "ROOM_TYPE"],
    "book_parking_space": ["DATE"]
}

# =====================================================
# 8. Fill entity placeholders
# =====================================================
def fill_entities(template, entities):
    person = ", ".join(entities.get("PERSON", []))
    date = ", ".join(entities.get("DATE", []))
    location = ", ".join(entities.get("GPE", []))
    person = f" {person}" if person else ""
    date = f" for {date}" if date else ""
    location = f" in {location}" if location else ""
    return template.format(PERSON=person, DATE=date, LOCATION=location)

# =====================================================
# 9. Intent Prediction
# =====================================================
def predict_intent(user_input):
    start_time = time.time()
    text = user_input.lower()

    # --- Rule-based ---
    if any(k in text for k in ["wifi", "internet"]):
        return "ask_wifi", "Rule", time.time() - start_time
    if any(k in text for k in ["price", "cost", "rate"]):
        return "ask_room_price", "Rule", time.time() - start_time
    if "check in" in text or "check-in" in text:
        return "check_in", "Rule", time.time() - start_time
    if "check out" in text or "checkout" in text:
        return "check_out", "Rule", time.time() - start_time

    # --- ML-based ---
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
# 10. Generate Response (Multi-turn)
# =====================================================
def generate_response(user_input):
    if "pending_intent" not in st.session_state:
        st.session_state.pending_intent = None
    if "collected_info" not in st.session_state:
        st.session_state.collected_info = {}

    intent, confidence, response_time = predict_intent(user_input)
    entities = extract_entities(user_input)

    # --- Check if multi-turn in progress ---
    if st.session_state.pending_intent:
        current_intent = st.session_state.pending_intent
        # 更新收集到的槽位
        for slot in intent_slots.get(current_intent, []):
            if entities.get(slot):
                st.session_state.collected_info[slot] = entities[slot][0]
        # 检查是否还缺槽位
        missing = [slot for slot in intent_slots.get(current_intent, []) if slot not in st.session_state.collected_info]
        if missing:
            reply = f"Please provide the following information: {', '.join(missing)}."
            return current_intent, reply, confidence, response_time
        else:
            reply = f"I have recorded your information: {st.session_state.collected_info}. Please proceed to the website or Front Desk to complete the operation."
            st.session_state.pending_intent = None
            st.session_state.collected_info = {}
            return current_intent, reply, confidence, response_time

    # --- New intent with required slots ---
    if intent in intent_slots:
        missing_entities = [slot for slot in intent_slots[intent] if not entities.get(slot)]
        if missing_entities:
            st.session_state.pending_intent = intent
            # 保存已提供槽位
            for slot in intent_slots[intent]:
                if entities.get(slot):
                    st.session_state.collected_info[slot] = entities[slot][0]
            reply = f"Sure! I can help you with that. Please provide: {', '.join(missing_entities)}."
            return intent, reply, confidence, response_time

    # --- Single-turn reply ---
    template = responses.get(intent, responses["unknown"])
    reply = fill_entities(template, entities)
    return intent, reply, confidence, response_time

# =====================================================
# 11. Evaluation Utilities
# =====================================================
if "evaluation_log" not in st.session_state:
    st.session_state.evaluation_log = []

def evaluate_intent(test_dataset):
    y_true = []
    y_pred = []
    for item in test_dataset:
        intent, _, _ = predict_intent(item["input"])
        y_true.append(item["true_intent"])
        y_pred.append(intent)
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred

def evaluate_response(test_dataset):
    bleu_scores = []
    smoothie = SmoothingFunction().method4
    for item in test_dataset:
        _, reply, _, _ = generate_response(item["input"])
        reference = [item["true_response"].split()]
        candidate = reply.split()
        score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
        bleu_scores.append(score)
    avg_bleu = sum(bleu_scores)/len(bleu_scores) if bleu_scores else 0
    return avg_bleu, bleu_scores

def collect_feedback(user_input, bot_reply):
    rating = st.slider(f"Rate the response for: '{bot_reply}'", 1, 5, 3, key=f"fb_{len(st.session_state.evaluation_log)}")
    st.session_state.evaluation_log.append({
        "input": user_input,
        "response": bot_reply,
        "rating": rating
    })

# =====================================================
# 12. Streamlit UI
# =====================================================
st.set_page_config(page_title="Hotel AI Chatbot", layout="centered")
st.title("Hotel Customer Support Chatbot")
st.caption("SVM Intent Classification + spaCy NER + Multi-turn Slot Filling")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": responses["greeting"]}]

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# 用户输入
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    intent, reply, confidence, response_time = generate_response(user_input)

    # 显示小字意图和置信度在回答上方
    intent_info = f"<sub>Predicted Intent: {intent} | Confidence: {confidence}</sub>"
    display_reply = f"{intent_info}\n\n{reply}"

    st.session_state.messages.append({"role": "assistant", "content": display_reply})
    st.rerun()



