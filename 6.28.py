# =====================================================
# Streamlit Hotel Chatbot (ML-based with spaCy)
# Auto-download spaCy model if missing
# =====================================================

import streamlit as st
import pandas as pd
import re
from joblib import load
import spacy
from spacy.cli import download

# -------------------------
# 1. Load spaCy Model safely
# -------------------------
import spacy
from spacy.cli import download

# 直接下载模型
download("en_core_web_sm")

# 下载完成后加载模型
nlp = spacy.load("en_core_web_sm")

# 测试用
doc = nlp("Hello, my name is Zhiqiang.")
for ent in doc.ents:
    print(ent.text, ent.label_)


# -------------------------
# 2. Load trained model and vectorizer
# -------------------------
model = load("intent_model_spacy.joblib")
vectorizer = load("tfidf_vectorizer_spacy.joblib")

# -------------------------
# 3. Text preprocessing
# -------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

# -------------------------
# 4. Rule-based responses
# -------------------------
responses = {
    "ask_room_price": "Our deluxe room costs RM180 per night.",
    "ask_booking": "I can help you book a room. Please provide your date and number of guests.",
    "ask_checkin_time": "Check-in time starts from 2:00 PM.",
    "ask_checkout_time": "Check-out time is before 12:00 PM.",
    "greeting": "Hello! How can I help you today?",
    "goodbye": "Thank you for visiting. Have a nice day!"
}

# -------------------------
# 5. Chatbot logic
# -------------------------
def predict_intent(user_input):
    cleaned = preprocess_text(user_input)
    vec = vectorizer.transform([cleaned])
    return model.predict(vec)[0]

def chatbot_response(user_input):
    try:
        intent = predict_intent(user_input)
        return responses.get(intent, "Sorry, I do not understand your request.")
    except:
        return "Sorry, something went wrong."

# -------------------------
# 6. Streamlit UI
# -------------------------
st.title("Hotel Customer Support Chatbot (ML-based)")

st.write("Type your message below and press Enter or click Send. Type 'exit' to quit.")

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

user_input = st.text_input("You:", "")

if st.button("Send") and user_input:
    if user_input.lower() == "exit":
        st.session_state.conversation.append(("User", user_input))
        st.session_state.conversation.append(("Bot", "Goodbye!"))
    else:
        st.session_state.conversation.append(("User", user_input))
        reply = chatbot_response(user_input)
        st.session_state.conversation.append(("Bot", reply))

# 显示聊天记录
for speaker, message in st.session_state.conversation:
    if speaker == "User":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")

