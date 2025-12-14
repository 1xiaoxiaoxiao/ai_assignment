# =====================================================
# Streamlit Hotel FAQ Chatbot (SVM + TF-IDF + spaCy NER)
# Chat UI + Intent + High/Medium/Low Confidence + Response Time
# =====================================================

import streamlit as st
import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from collections import defaultdict
import joblib
import time
import numpy as np

# -----------------------------
# Load models & spaCy
# -----------------------------
nlp = spacy.load("en_core_web_sm")
clf = joblib.load("svm_faq_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# -----------------------------
# Responses dictionary
# -----------------------------
responses = {
    "greeting": "Welcome to Astra Imperium Hotel. I'm your virtual assistant. How may I assist you today?",
    "check_functions": "I can help with room reservations, hotel information, facilities, services, and general inquiries.",
    "invoices": "To request an invoice, please visit the Front Desk or email us at billing@astraimperium.com. Provide your booking reference number for quicker processing.",
    "cancellation_fees": "Cancellations are free up to 24 hours before your check-in date. Cancellations within 24 hours, or no-shows, will incur a charge equivalent to the first night's stay.",
    "check_in": "Check-in begins at 3:00 PM. Early check-in is subject to room availability. Luggage may be stored at the concierge at no additional cost. A refundable security deposit is required during check-in.",
    "check_out": "Check-out time is 12:00 PM. Late check-out until 2:00 PM is available for RM50, depending on availability. Please return your key card to the Front Desk upon departure.",
    "book_hotel": "To make a reservation, visit www.astraimperium.com, call our Reservations Department at +60-3-5555-0199, or book in person at the Front Desk. A valid ID is required for all bookings.",
    "cancel_hotel_reservation": "To cancel your reservation, please contact our Reservations Team at +60-3-5555-0199 or email bookings@astragroup.com with your booking reference number.",
    "unknown_intent": "I'm sorry, I don't understand your question."
}

# -----------------------------
# Text preprocessing
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# -----------------------------
# Extract entities using spaCy NER
# -----------------------------
def extract_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities

# -----------------------------
# Predict intent and High/Medium/Low confidence
# -----------------------------
def predict_intent(text):
    vec = vectorizer.transform([preprocess_text(text)])
    scores = clf.decision_function(vec)
    best_index = np.argmax(scores)
    intent = clf.classes_[best_index]
    
    # Calculate margin
    if len(scores[0]) > 1:
        sorted_scores = np.sort(scores[0])[::-1]
        margin = sorted_scores[0] - sorted_scores[1]
    else:
        margin = scores[0][0]
    
    # Convert margin to High/Medium/Low
    if margin > 1.0:
        confidence = "High"
    elif margin > 0.5:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    return intent, confidence

# -----------------------------
# Streamlit Chat UI
# -----------------------------
st.set_page_config(page_title="Hotel FAQ Chatbot", layout="centered")
st.title("Astra Imperium Hotel FAQ Chatbot")
st.caption("SVM + TF-IDF + spaCy NER (Fixed Responses)")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("You:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    start_time = time.time()
    intent, confidence = predict_intent(user_input)
    response = responses.get(intent, responses["unknown_intent"])
    response_time = time.time() - start_time
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "intent": intent,
        "confidence": confidence,
        "time": response_time
    })

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            st.caption(f"Intent: **{msg['intent']}** | Confidence: **{msg['confidence']}** | Time: **{msg['time']:.3f}s**")
