from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Load summarization model
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

def generate_response(prompt, history=[]):
    """
    Summarize input text safely with truncation.
    """
    try:
        output = summarizer(
            prompt,
            max_length=150,
            min_length=40,
            do_sample=False,
            truncation=True  # <--- prevents 'index out of range'
        )
        return output[0]["summary_text"]
    except Exception as e:
        return f"⚠️ Error while summarizing: {str(e)}"
