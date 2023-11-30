from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for pages in pdf_reader.pages:
        text += pages.extract_text()
    return text


def generate_tfidf_vectors(text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    return tfidf_matrix, vectorizer

