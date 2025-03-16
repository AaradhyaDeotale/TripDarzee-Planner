import os
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Load environment variables from .env file
load_dotenv()

# Set API Keys
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key is not set. Please set the GEMINI_API_KEY environment variable.")
mapquest_api_key = os.getenv("MAPS_API_KEY")
weather_api_key = os.getenv("WEATHER_API_KEY")

# Configure Gemini API
genai.configure(api_key=api_key)

# Create the Gemini model
generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Chatbot initialization
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction=(
        "You are a chatbot for a travel planner website named TripDarzee, and your name is 'TravelMitra.' "
        "Your primary role is to assist users in planning their trips by gathering their preferences, such as "
        "destination, travel dates, and budget. You will generate personalized itineraries based on user inputs and "
        "provide recommendations for accommodations, transportation, and local attractions. You should also offer "
        "real-time updates on weather, local events, and travel advisories. Ensure that your responses are helpful, "
        "accurate, and tailored to individual user needs. Be proactive in suggesting follow-up questions to keep the "
        "conversation flowing. After each user input, provide 2-3 relevant follow-up questions to help the user explore "
        "more options or provide more details about their travel plans. Make sure the questions are concise, context-aware, "
        "and useful. Only suggest follow-up questions related to travel, itineraries, or planning."
    ),
)

history = []  # Conversation history
itinerary = []  # User itinerary

# Initial welcome message
welcome_message = (
    "Hello there! 👋 Welcome to TripDarzee! I'm TravelMitra, your personal travel assistant. "
    "What kind of trip are you dreaming of? Tell me all about it, and I'll help you weave a perfect travel tapestry! 🧵✨"
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Helper functions
def is_travel_related(user_input):
    travel_keywords = [
        "trip", "hotel", "flights", "destination", "itinerary", "transport", 
        "restaurant", "book", "continue", "yes", "ok", "confirm", "cancel", 
        "change", "next", "plan", "schedule", "tourism", "accommodation", 
        "travel", "weather", "route", "flight", "place", "sightseeing"
    ]
    return any(keyword in user_input.lower() for keyword in travel_keywords)

def summarize_history(limit=1000):
    global history
    total_length = sum(len(entry["parts"][0]) for entry in history)
    if total_length > limit:
        history = history[-5:]  # Keep the last 5 interactions for context

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    logging.debug(f"User Input: {user_input}")

    if not user_input:
        logging.warning("Received empty user input.")
        return jsonify({"response": "Please provide a message."}), 400

    # Enforce travel-related topics
    if not is_travel_related(user_input):
        logging.warning(f"Non-travel-related query: {user_input}")
        return jsonify({"response": "I can only help with travel-related queries! 😊 Please ask about your trip, itinerary, hotels, etc."}), 400

    # Use the generative model for the main query
    chat_session = model.start_chat(history=history)
    bot_response = chat_session.send_message(user_input).text

    # Extract follow-up questions
    follow_up_questions = []
    if "?" in bot_response:
        sentences = bot_response.split(". ")
        follow_up_questions = [sentence for sentence in sentences if "?" in sentence][:3]

    # Update conversation history
    history.append({"role": "user", "parts": [user_input]})
    history.append({"role": "model", "parts": [bot_response]})
    summarize_history()

    return jsonify({
        "response": bot_response,
        "followUpQuestions": follow_up_questions
    })

@app.route("/")
def index():
    return welcome_message

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
