# -------------------------
# 5. Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Hotel Chatbot", layout="centered")
    st.title("🏨 Solaris Grand Hotel Chatbot")
    st.caption("Powered by SVM + spaCy NER + Template Responses")

    if "messages" not in st.session_state:
        st.session_state.messages = []
        greeting = responses.get("greeting", "Hello! How may I help you?")
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Suggested buttons
    if "suggested_intents" not in st.session_state:
        st.session_state.suggested_intents = random.sample(list(PROMPT_MAPPING.keys()), min(4, len(PROMPT_MAPPING)))

    cols = st.columns(len(st.session_state.suggested_intents))
    for i, key in enumerate(st.session_state.suggested_intents):
        prompt = PROMPT_MAPPING.get(key, key)
        with cols[i]:
            if st.button(prompt, key=f"btn_{key}", use_container_width=True):
                user_input = prompt
                st.session_state.messages.append({"role": "user", "content": user_input})
                intent, response = chatbot_response(user_input)
                st.session_state.messages.append({"role": "assistant", "content": response})
                # Refresh suggestions
                st.session_state.suggested_intents = random.sample(list(PROMPT_MAPPING.keys()), min(4, len(PROMPT_MAPPING)))
                
                # FIX 1: Use st.rerun() instead of experimental_rerun()
                st.rerun()

    # User input
    user_input = st.chat_input("How can I help you?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        intent, response = chatbot_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # FIX 2: Use st.rerun() instead of experimental_rerun()
        st.rerun()
