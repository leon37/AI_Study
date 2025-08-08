from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
results = classifier("I love learning NLP with Transformers!")
print(results)