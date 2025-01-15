from flask import Flask, request, jsonify
import json
from transformers import pipeline
from difflib import get_close_matches

app = Flask(__name__)

# Load the dataset
with open("qa_dataset.json", "r") as file:
    qa_data = json.load(file)

# Load the pre-trained question-answering model
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Function to find the best answer
def get_answer(user_question, qa_data, qa_model):
    questions = [item["question"] for item in qa_data]
    closest_match = get_close_matches(user_question, questions, n=1, cutoff=0.5)
    if closest_match:
        for item in qa_data:
            if item["question"] == closest_match[0]:
                result = qa_model(question=user_question, context=item["answer"])
                return result["answer"]
    else:
        return "Sorry, I couldn't find an answer to your question."

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    answer = get_answer(question, qa_data, qa_model)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
