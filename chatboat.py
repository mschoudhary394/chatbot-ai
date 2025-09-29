import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify

app = Flask(__name__)

genai.configure(api_key="AIzaSyCJ_ejqTU0jJsUqRh7ImAV6ck_-ww9uLcg")
@app.route('/chat', methods=['POST'])
def chat():
    data=request.get_json()
    prompt=data.get("prompt")
    model=genai.GenerativeModel('gemini-2.5-flash')
    response=model.generate_content(prompt)
    return jsonify(response.text)

if __name__ == '__main__':
    app.run(debug=True)