import streamlit as st
import json
from transformers import pipeline
from difflib import get_close_matches

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

# Streamlit UI
st.title("AI-Powered Question Answering System")
st.subheader("Ask me any question!")

# Input field for user question
user_question = st.text_input("Enter your question here:")

if st.button("Get Answer"):
    if user_question.strip():
        with st.spinner("Thinking..."):
            answer = get_answer(user_question, qa_data, qa_model)
        st.success("Answer:")
        st.write(answer)
    else:
        st.error("Please enter a valid question!")