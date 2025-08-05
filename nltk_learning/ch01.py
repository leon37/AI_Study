from nltk.book import *
import matplotlib.pyplot as plt
# text1.concordance('monstrous')
# text1.similar('monstrous')
# text2.similar('monstrous')
# text2.common_contexts(['monstrous', 'very'])
# text4.dispersion_plot(['citizens', 'democracy', 'freedom', 'duties', 'America', 'liberty', 'constitution'])
# plt.show()

fdist1 = FreqDist(text1)
fdist1.plot(50, cumulative=True)
plt.show()