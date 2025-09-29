
import os
import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
from rag import retrieve_context

app = Flask(__name__)

genai.configure(api_key="AIzaSyCJ_ejqTU0jJsUqRh7ImAV6ck_-ww9uLcg")
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    prompt = data.get("prompt")
    # RAG: Retrieve semantic context from rag.py
    context_list = retrieve_context(prompt, top_k=2)
    context = "\n".join(context_list)
    augmented_prompt = f"Context:\n{context}\n\nUser: {prompt}"
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(augmented_prompt)
    return jsonify({
        "result": response.text,
        "context": context_list,
        "augmented_prompt": augmented_prompt,
        "model": "gemini-2.5-flash"
    })

if __name__ == '__main__':
    app.run(debug=True)