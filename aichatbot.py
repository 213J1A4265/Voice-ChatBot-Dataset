import streamlit as st
import speech_recognition as sr
import pyttsx3
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from icondf_chatbot_dataset import dataset  # Import the dataset
import whisper
import io  # For handling audio data in memory

# 1. Prepare the training data (moved outside for efficiency)
user_queries = [item["user_query"] for item in dataset]
intents = [item["intent"] for item in dataset]

# 2. Feature extraction using TF-IDF (moved outside for efficiency)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(user_queries)

# 3. Train the intent classification model (moved outside for efficiency)
intent_model = LogisticRegression(random_state=0)
intent_model.fit(X, intents)

# 4. Create a dictionary to map intents to responses (moved outside for efficiency)
response_dict = {}
for item in dataset:
    if item["intent"] not in response_dict:
        response_dict[item["intent"]] = []
    response_dict[item["intent"]].append(item["bot_response"])


# Initialize text-to-speech engine (pyttsx3) - move to session state
if 'engine' not in st.session_state:
    st.session_state['engine'] = pyttsx3.init()

# Load the Whisper model (you might need to install whisper: pip install openai-whisper) - move to session state
if 'whisper_model' not in st.session_state:
    st.session_state['whisper_model'] = whisper.load_model("base")  # Load model once
    st.write("Whisper model loaded successfully!") #Feedback to the user



def speak(text):
    """Converts text to speech using pyttsx3."""
    engine = st.session_state['engine']  # Access from session state
    engine.say(text)
    engine.runAndWait()


def listen_whisper():
    """Listens for audio and transcribes it using the Whisper ASR model.

    Returns:
        str: The transcribed text, or an empty string if transcription fails.
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!") # Use Streamlit's info message
        audio = r.listen(source)

    try:
        # Transcribe the audio using Whisper
        whisper_model = st.session_state['whisper_model'] #Access from session state
        # Convert audio data to bytes and create a file-like object
        audio_bytes = audio.get_wav_data()
        audio_io = io.BytesIO(audio_bytes)
        result = whisper_model.transcribe(audio_io)  # Pass audio data in memory

        text = result["text"]
        st.write(f"You said (Whisper): {text}") # Use Streamlit's write
        return text.lower()

    except Exception as e:  # Catch more general exceptions
        st.error(f"Error during Whisper transcription: {e}") # Use Streamlit's error message
        return ""


def get_response(user_input):
    """Predicts the intent and returns a suitable bot response."""
    # 1. Predict the intent of the user input
    input_vector = vectorizer.transform([user_input])
    predicted_intent = intent_model.predict(input_vector)[0]

    # 2. Select a response based on the predicted intent
    if predicted_intent in response_dict:
        responses = response_dict[predicted_intent]
        return random.choice(responses)  # Choose a random response from the list
    else:
        return "I'm sorry, I don't have information on that topic. Please visit our website for more." #Or a fallback intent response


# Streamlit UI
st.title("Iconic Dream Focus Chatbot")

# Initialize conversation history in session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
    st.session_state.conversation.append(("Bot", "Hello! Welcome to Iconic Dream Focus. How can I help you today?"))

# Display conversation history
for role, text in st.session_state.conversation:
    if role == "User":
        st.write(f'<div style="background-color:#ADD8E6;padding:10px;border-radius:5px;"><strong>User:</strong> {text}</div>', unsafe_allow_html=True)
    else:
        st.write(f'<div style="background-color:#90EE90;padding:10px;border-radius:5px;"><strong>Bot:</strong> {text}</div>', unsafe_allow_html=True)



# Button to start listening
if st.button("Speak"):
    user_input = listen_whisper()
    if user_input:
        st.session_state.conversation.append(("User", user_input)) #Add user input to conversation history
        response = get_response(user_input)
        st.session_state.conversation.append(("Bot", response))  #Add bot response to conversation history
        speak(response)
        # Rerun to display updated conversation
        st.rerun()



# Manual input for debugging/testing purposes.  You can comment this out if you don't need it
user_input_manual = st.text_input("Or type your message here:", "")
if st.button("Send (Manual)", disabled=not user_input_manual): # disable button if input empty
    user_input = user_input_manual
    st.session_state.conversation.append(("User", user_input))
    response = get_response(user_input)
    st.session_state.conversation.append(("Bot", response))
    speak(response)
    st.rerun()  # Rerun to update the display

st.markdown("---")
st.caption("Created using Streamlit, Whisper, and scikit-learn")