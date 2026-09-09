import streamlit as st
import requests
import uuid


API_URL = "http://localhost:8000/chat"


st.set_page_config(page_title="DocuMind Agent",  layout="centered")

st.title(" DocuMind AI")
st.markdown("Ask me anything about your corporate documents!")


if "session_id" not in st.session_state:
    
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
    
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I help you with your documents today?"})


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
if user_input := st.chat_input("Type your question here..."):
   
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    
    with st.spinner("Searching documents..."):
        try:
            
            payload = {
                "session_id": st.session_state.session_id,
                "message": user_input
            }
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            
           
            data = response.json()
            agent_response = data["response"]
            
        except requests.exceptions.ConnectionError:
            agent_response = "Error: Could not connect to the backend API. Is `server.py` running on port 8000?"
        except Exception as e:
            agent_response = f"Error: {e}"

   
    st.chat_message("assistant").markdown(agent_response)
    st.session_state.messages.append({"role": "assistant", "content": agent_response})
