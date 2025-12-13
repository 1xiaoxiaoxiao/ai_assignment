# =========================
# hotel_chatbot_streamlit.py
# =========================
import streamlit as st
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from joblib import load
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# -------------------------
# 1️⃣ NLTK Setup
# -------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

# -------------------------
# 2️⃣ Load Model & Vectorizer
# -------------------------
svm_model = load("intent_model_spacy.joblib")
vectorizer = load("tfidf_vectorizer_spacy.joblib")

# -------------------------
# 3️⃣ Load SpaCy NER
# -------------------------
nlp = spacy.load("en_core_web_sm")

def get_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities

# -------------------------
# 4️⃣ Response Policy / Templates
# -------------------------
policy = {
    ("ask_room_price", "ROOM_TYPE"): "The {ROOM_TYPE} costs RM180 per night.",
    ("ask_booking", "NUM_GUESTS"): "I can book a {ROOM_TYPE} for {NUM_GUESTS} starting {DATE}.",
    ("ask_checkin_time", "TIME"): "Check-in starts at {TIME}.",
    ("ask_checkout_time", "TIME"): "Check-out is at {TIME}.",
    ("default", "none"): "Sorry, I don't understand. Could you please rephrase?",
    ("greeting", "none"): "Hello! How can I help you today?"
}

# -------------------------
# 5️⃣ Intent Prediction
# -------------------------
def get_intent(text, model=svm_model, vectorizer=vectorizer, threshold=0.5):
    text_cleaned = preprocess_text(text)
    vector = vectorizer.transform([text_cleaned])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]
        max_prob = max(probs)
        if max_prob < threshold:
            return "default"
    return model.predict(vector)[0]

def respond(user_text):
    intent = get_intent(user_text)
    entities = get_entities(user_text)
    
    # Check policy templates
    if intent != "default":
        for key in ["ROOM_TYPE", "NUM_GUESTS", "TIME", "DATE"]:
            if (intent, key) in policy:
                template = policy[(intent, key)]
                response = template.format(
                    ROOM_TYPE=entities.get("ROOM_TYPE", ["room"])[0],
                    NUM_GUESTS=entities.get("CARDINAL", ["1"])[0],
                    DATE=entities.get("DATE", ["today"])[0],
                    TIME=entities.get("TIME", ["2 PM"])[0]
                )
                return response, intent
        # fallback if no entity matches
        response = f"I'm not sure how to answer that regarding {intent}."
        return response, intent
    else:
        return policy[("default", "none")], intent

# -------------------------
# 6️⃣ Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Hotel Chatbot", layout="centered")
    st.title("🏨 Hotel Chatbot (SVM + spaCy)")
    st.markdown("Ask about bookings, room rates, check-in/out times, etc.")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        greeting = policy.get(("greeting", "none"))
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response_text, intent = respond(user_input)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "intent": intent
        })

if __name__ == "__main__":
    main()



