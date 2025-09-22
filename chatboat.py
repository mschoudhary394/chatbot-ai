import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Set your Gemini API key in environment variables

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get('prompt', '')

    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        json=payload
    )
    if response.status_code == 200:
        gemini_output = response.json()
        text = gemini_output['candidates'][0]['content']['parts'][0]['text']
        return jsonify({"response": text})
    else:
        return jsonify({"error": "Failed to get response from Gemini API"}), 500

if __name__ == '__main__':
    app.run(debug=True)