from nltk.corpus import brown
import nltk

# news_text = brown.words(categories='news')
# fdist = nltk.FreqDist(w.lower() for w in news_text)
#
# modals = ['can', 'could', 'may', 'might', 'must', 'will']
# for m in modals:
#     print(m+': ', fdist[m], end=' ')

# genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
# modals = ['can', 'could', 'may', 'might', 'must', 'will']
# cfd = nltk.ConditionalFreqDist(
#     (genre, word)
#     for genre in brown.categories()
#     for word in brown.words(categories=genre)
# )
# cfd.tabulate(conditions=genres, samples=modals)

# genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
cfd = nltk.ConditionalFreqDist(
    (genre, word.lower())
    for genre in ['news', 'romance']
    for word in brown.words(categories=genre)
)
cfd.tabulate(conditions=['news', 'romance'], samples=days)