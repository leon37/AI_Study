from transformers import pipeline

# classifier = pipeline('sentiment-analysis')
# print(classifier("I've been waiting for a HuggingFace course my whole life."))

generator = pipeline('text-generation')
print(generator('In this course, we will teach you how to', max_length=15, num_return_sequences=2))
