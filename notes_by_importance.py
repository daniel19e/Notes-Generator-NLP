from transformers import pipeline
import nltk.data
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# Load fine-tuned T5 model
summary_pipe = pipeline("summarization", model="final_model")
gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return sentence_transformer.encode(text)

def calculate_paragraph_importance(doc, window):
    doc_embedding = get_embedding(doc)
    window_embedding = get_embedding(window)
    return cosine_similarity([doc_embedding], [window_embedding])[0][0]

def capitalize(page):
    sentences = sentence_tokenizer.tokenize(page)
    return " ".join([sent.capitalize() for sent in sentences])

def sliding_window_summarize(text, window_size=300):
    sentences = text.split('.') 
    grouped_sentences = []
    current_group = []
    current_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_count + word_count <= window_size:
            current_group.append(sentence)
            current_count += word_count
        else:
            grouped_sentences.append('. '.join(current_group) + '.')
            current_group = [sentence]
            current_count = word_count

    if current_group:
        grouped_sentences.append('. '.join(current_group) + '.')

    summaries = []
    for group in grouped_sentences:
        if group.strip():
            summary = summary_pipe(capitalize(group), **gen_kwargs)[0]['summary_text']
            summaries.append(capitalize(summary))

    return summaries


def summarize_article(pdf):
    bullet_points = []
    full_text = ""
    similarity_threshold = 0.2
    for page in pdf.pages:
        extracted = page.extract_text(x_tolerance=1) or ""
        full_text += " " + extracted

    window_summaries = sliding_window_summarize(full_text)
    bullet_points.extend(window_summaries)

    return "• " + "\n• ".join([b for b in bullet_points if calculate_paragraph_importance(full_text, b) >= similarity_threshold])


