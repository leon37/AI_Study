import nltk, re, pprint
from nltk import word_tokenize, PorterStemmer, LancasterStemmer
from nltk.corpus import gutenberg, nps_chat

# moby = nltk.Text(gutenberg.words('melville-moby_dick.txt'))
# print(moby.findall(r"<a><.*><man>"))


raw = """DENNIS: Listen, strange women lying in ponds distributing swords
... is no basis for a system of government.  Supreme executive power derives from
... a mandate from the masses, not from some farcical aquatic ceremony."""
tokens = word_tokenize(raw)

porter = PorterStemmer()
lancaster = LancasterStemmer()

print([porter.stem(t) for t in tokens])
print([lancaster.stem(t) for t in tokens])