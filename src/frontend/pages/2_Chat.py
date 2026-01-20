import streamlit as st

st.set_page_config(page_title="Chat - NextGen RecSys", layout="wide")

st.title("💬 AI Assistant")

st.info("This feature is coming soon! You will be able to chat with a Large Language Model to get personalized recommendations via natural language.")

# Placeholder UI
st.chat_message("assistant").write("Hello! I am your recommendation assistant. How can I help you today? (Demo only)")
st.chat_message("user").write("I'm looking for some good running shoes.")
st.chat_message("assistant").write("I can help with that! (This is a static placeholder)")

st.text_input("Type your message here...", disabled=True, placeholder="Chat functionality disabled")
