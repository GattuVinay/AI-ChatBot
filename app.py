import json
import random
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import nltk

# Load the intents file
with open("PreDefinedintents.json", "r") as file:
    intents = json.load(file)

# Prepare training data
patterns = []
tags = []
responses_dict = {}
    
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    responses_dict[intent["tag"]] = intent["responses"]

# Train the model
vectorizer = TfidfVectorizer()
model = MultinomialNB()
pipeline = make_pipeline(vectorizer, model)
pipeline.fit(patterns, tags)

# Create Flask app
app = Flask(__name__)

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")

# Define the chatbot response route
@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    if user_input:
        predicted_tag = pipeline.predict([user_input])[0]
        response = random.choice(responses_dict[predicted_tag])
        return jsonify({"response": response})
    return jsonify({"response": "I'm not sure how to respond to that."})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
