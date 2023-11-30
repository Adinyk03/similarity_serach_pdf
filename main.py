from flask import Flask, request, jsonify
import openai
from config import apikey
from sklearn.metrics.pairwise import cosine_similarity
from func import extract_text_from_pdf, generate_tfidf_vectors

app = Flask(__name__)

openai.api_key = apikey

# Flask endpoint to handle PDF and question
@app.route('/pdf', methods=['POST'])
def get_answer():
    data = request.get_json()
    pdf_path = data['pdf_path']
    question = data['question']
    pdf_text = extract_text_from_pdf(pdf_path)
    tfidf_matrix, vectorizer = generate_tfidf_vectors(pdf_text)

    try:
        question_vector = vectorizer.transform([question])
        similarity_scores = cosine_similarity(question_vector, tfidf_matrix)
        similar_text_indices = similarity_scores.argsort()[0][-5:][::-1]

        answers = []
        for idx in similar_text_indices:
            part = pdf_text[idx * 4096: (idx + 1) * 4096]
            history = [
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": part},
                {"role": "assistant", "content": question}
            ]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=history,
            )
            answer = response['choices'][0]['message']['content']
            answers.append(answer)

        return jsonify({"answers": answers})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
