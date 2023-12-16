import streamlit as st
import cv2
import base64
import time
import requests
import os
from elevenlabs import generate, play, set_api_key
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Initialize ElevenLabs for text-to-speech
set_api_key(ELEVENLABS_API_KEY)

# Function to capture an image from the webcam and encode it in base64
def capture_and_encode_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    else:
        return None

# Function to send request to OpenAI API
def get_image_description(base64_image):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            }
        ],
        "max_tokens": 500
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json().get('choices', [{}])[0].get('message', {}).get('content', '')

# Main Streamlit app function
def main():
    st.title("Webcam Image Analysis and Narration")

    while True:
        base64_image = capture_and_encode_image()

        if base64_image:
            st.image(base64_image, channels="BGR", use_column_width=True)
            description = get_image_description(base64_image)
            st.write(description)

            # Generate and play audio for the description
            audio = generate(description, voice=ELEVENLABS_VOICE_ID)
            play(audio)

            time.sleep(5)  # Capture image every 5 seconds

if __name__ == "__main__":
    main()
