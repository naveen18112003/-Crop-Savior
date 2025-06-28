# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf
import requests
import json

# Loading the Model
model = load_model('plant_disease_model.h5')

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Mistral AI API configuration
MISTRAL_API_KEY = "yQdfM8MLbX9uhInQ7id4iUTwN4h4pDLX"  # Replace with your actual API key
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Setting Title of App
st.title("🌾 Crop Savior")
st.markdown("Upload an image of the plant leaf for disease detection and get AI-powered treatment recommendations")

# Sidebar for API key and chat
with st.sidebar:
    st.header("Mistral AI Settings")
    api_key = st.text_input("Enter your Mistral API key", type="password")
    if api_key:
        MISTRAL_API_KEY = api_key

# Tab interface
tab1, tab2 = st.tabs(["Disease Detection", "Plant Health Chat"])

# Disease Detection Tab
with tab1:
    # Uploading the plant image
    plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="detection_uploader")
    submit = st.button('Predict Disease')

    # On predict button click
    if submit:
        if plant_image is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)

            # Displaying the image
            st.image(opencv_image, channels="BGR", caption="Uploaded Leaf Image")

            # Resizing the image
            opencv_image = cv2.resize(opencv_image, (256, 256))

            # Convert image to 4 Dimension
            opencv_image.shape = (1, 256, 256, 3)

            # Make Prediction
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]
            disease_name = result.split('-')[1]
            plant_name = result.split('-')[0]

            st.success(f"**Detection Result:** This is {plant_name} leaf with {disease_name}")

            # Get treatment recommendation from Mistral AI
            if MISTRAL_API_KEY:
                with st.spinner("Getting treatment recommendations..."):
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {MISTRAL_API_KEY}"
                    }

                    prompt = f"""You are an expert agricultural botanist. Provide detailed treatment recommendations for {plant_name} affected by {disease_name}. 
                    Include organic and chemical treatment options, preventive measures, and expected recovery timeline. 
                    Format your response with clear headings and bullet points."""

                    data = {
                        "model": "mistral-tiny",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7
                    }

                    try:
                        response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
                        if response.status_code == 200:
                            treatment = response.json()['choices'][0]['message']['content']
                            st.subheader("💊 Treatment Recommendations")
                            st.markdown(treatment)
                        else:
                            st.warning("Couldn't get treatment recommendations. API error.")
                    except Exception as e:
                        st.error(f"Error contacting Mistral AI: {str(e)}")
            else:
                st.warning("Please enter Mistral API key in the sidebar to get treatment recommendations")

# Plant Health Chat Tab
with tab2:
    st.subheader("Ask Mistral AI about plant health")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about plant care, diseases, or treatments..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        if not MISTRAL_API_KEY:
            with st.chat_message("assistant"):
                st.error("Please enter your Mistral API key in the sidebar to enable chat")
            st.session_state.messages.append({"role": "assistant", "content": "API key missing"})
        else:
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {MISTRAL_API_KEY}"
                    }

                    # Prepare conversation history
                    messages_for_api = [{"role": m["role"], "content": m["content"]}
                                        for m in st.session_state.messages]

                    data = {
                        "model": "mistral-tiny",
                        "messages": messages_for_api,
                        "temperature": 0.7
                    }

                    try:
                        response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
                        if response.status_code == 200:
                            full_response = response.json()['choices'][0]['message']['content']
                            st.markdown(full_response)
                        else:
                            st.error(f"API Error: {response.status_code}")
                            full_response = "Sorry, I couldn't process your request."
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        full_response = "Sorry, I encountered an error."

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add some styling
st.markdown("""
<style>
    .stChatInput {
        position: fixed;
        bottom: 20px;
    }
    .stTab {
        margin-bottom: 50px;
    }
</style>
""", unsafe_allow_html=True)