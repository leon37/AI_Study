import nltk
from nltk import PorterStemmer
from nltk.corpus import gutenberg
import re

raw_data = gutenberg.raw('austen-emma.txt')[:1000]
print("====================raw_data====================")
print(raw_data[:100])

# Step 1：文本清洗（Cleaning）
cleaned_raw_data = " ".join(re.findall(r"\w+", raw_data))
print("====================cleaned_raw_data====================")
print(cleaned_raw_data[:100])

# Step 2：英文分词（Tokenization）
tokens = nltk.word_tokenize(cleaned_raw_data)
print("====================tokens====================")
print(tokens[:20])

# Step 3：转为小写 + 去除停用词
stop_words = nltk.corpus.stopwords.words('english')
tokens = [token.lower() for token in tokens if token.lower() not in stop_words and len(token) >= 2]
print("====================tokens_without_stopwords=====================")
print(len(tokens))
print(tokens)

# Step 4：词干提取（Stemming）
porter = PorterStemmer()
print("====================stemming=====================")
for token in tokens[:20]:
    stemmed = porter.stem(token)
    print(f"{token} --> {stemmed}")