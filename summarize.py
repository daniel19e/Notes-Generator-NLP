from transformers import pipeline
import nltk.data
import math

sentence_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# load fine-tuned t5-small model
summary_pipe = pipeline("summarization", model="final_model")
gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}


def capitalize(page):
    sentences = sentence_tokenizer.tokenize(page)
    return " ".join([sent.capitalize() for sent in sentences])


def summarize_page(page):
    """
    This function summarizes a page of a scientific article using a fine-tuned t5 language model.
    Since the model can only summarize 512 tokens, we are processing the text in batches by computing the
    optimal number of batches to process and adding the result of each batch to the final summary.
    Args:
        paragraph (string): text to be summarized

    Returns:
        string: summary of the page
    """
    sentences = sentence_tokenizer.tokenize(page)
    total_tokens = sum(len(s.split()) for s in sentences)

    optimum_tokens = 512
    num_batches = math.ceil(total_tokens / optimum_tokens)

    if num_batches == 1:
        return capitalize(summary_pipe(page, **gen_kwargs)[0]["summary_text"])

    # determine the number of tokens per batch based on num_batches
    tokens_per_batch = math.ceil(total_tokens / num_batches)

    current_batch = ""
    current_tokens = 0
    full_summary = ""

    for sentence in sentences:
        if current_tokens + len(sentence.split()) > tokens_per_batch:
            batch_summary = summary_pipe(current_batch, **gen_kwargs)[0]["summary_text"]
            full_summary += batch_summary + " "

            # reset current batch
            current_batch = sentence
            current_tokens = len(sentence.split())
        else:
            current_batch += sentence + " "
            current_tokens += len(sentence.split())

    return capitalize(full_summary)

def summarize_article(pdf):
    text = ""
    for page in pdf.pages:
        extracted = page.extract_text(x_tolerance=1)
        summary = summarize_page(extracted)
        text += "• "
        text += summary or ""
        text += "\n"
    return text