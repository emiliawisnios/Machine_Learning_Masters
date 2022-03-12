"""Practical 1

Greatly inspired by Stanford CS224 2019 class.
"""

import sys

import pprint

import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import random
import nltk

nltk.download('reuters')
nltk.download('pl196x')
import random

import numpy as np
import scipy as sp
from nltk.corpus import reuters
from nltk.corpus.reader import pl196x
from sklearn.decomposition import PCA, TruncatedSVD

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)


#################################
# TODO: a)
def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the 
            corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the 
            corpus
    """
    corpus_words = []
    num_corpus_words = -1
    
    # ------------------
    # Write your implementation here.
    for text in corpus:
        for word in text:
            corpus_words.append(word)
    corpus_words = sorted(list(set(corpus_words)))
    num_corpus_words = len(corpus_words)
    # ------------------

    return corpus_words, num_corpus_words


# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# ---------------------

# Define toy corpus
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]     
test_corpus_words, num_corpus_words = distinct_words(test_corpus)

# Correct answers
ans_test_corpus_words = sorted(list(set([
    'Ala', 'END', 'START', 'i', 'kot', 'lubic', 'miec', 'pies'])))
ans_num_corpus_words = len(ans_test_corpus_words)

# Test correct number of words
assert(num_corpus_words == ans_num_corpus_words), "Incorrect number of distinct words. Correct: {}. Yours: {}".format(ans_num_corpus_words, num_corpus_words)

# Test correct words
assert (test_corpus_words == ans_test_corpus_words), "Incorrect corpus_words.\nCorrect: {}\nYours:   {}".format(str(ans_test_corpus_words), str(test_corpus_words))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)


#################################
# TODO: b)
def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).
    
        Note: Each word in a document should be at the center of a window.
            Words near edges will have a smaller number of co-occurring words.
              
              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of corpus words, number of corpus words)): 
                Co-occurence matrix of word counts. 
                The ordering of the words in the rows/columns should be the 
                same as the ordering of the words given by the distinct_words 
                function.
            word2Ind (dict): dictionary that maps word to index 
                (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}

    # ------------------
    # Write your implementation here.
    for i, word in enumerate(words):
        word2Ind[word] = i

    d = dict()
    for text in corpus:
        for count, word in enumerate(text):

            tokens = text[count + 1:count + 1 + window_size]

            for token in tokens:
                key = tuple(sorted([token, word]))
                if key in d:
                    d[key] += 1
                else:
                    d[key] = 1

    M = np.zeros((num_words, num_words))

    for key, value in d.items():
        x = word2Ind[key[0]]
        y = word2Ind[key[1]]
        M[x, y] = value
        M[y, x] = value
    # ------------------

    return M, word2Ind

# ---------------------
# Run this sanity check
# Note that this is not an exhaustive check for correctness.
# ---------------------

# Define toy corpus and get student's co-occurrence matrix
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]     
M_test, word2Ind_test = compute_co_occurrence_matrix(
    test_corpus, window_size=1)

# Correct M and word2Ind
M_test_ans = np.array([
    [0., 0., 2., 0., 0., 1., 1., 0.],
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [2., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 1., 0., 0., 1.],
    [0., 1., 0., 1., 0., 1., 1., 0.],
    [1., 0., 0., 0., 1., 0., 0., 0.],
    [1., 0., 0., 0., 1., 0., 0., 0.],
    [0., 1., 0., 1., 0., 0., 0., 0.]
])

word2Ind_ans = {
    'Ala': 0, 'END': 1, 'START': 2, 'i': 3, 'kot': 4, 'lubic': 5, 'miec': 6,
    'pies': 7}

# Test correct word2Ind
assert (word2Ind_ans == word2Ind_test), "Your word2Ind is incorrect:\nCorrect: {}\nYours: {}".format(word2Ind_ans, word2Ind_test)

# Test correct M shape
assert (M_test.shape == M_test_ans.shape), "M matrix has incorrect shape.\nCorrect: {}\nYours: {}".format(M_test.shape, M_test_ans.shape)

# Test correct M values
for w1 in word2Ind_ans.keys():
    idx1 = word2Ind_ans[w1]
    for w2 in word2Ind_ans.keys():
        idx2 = word2Ind_ans[w2]
        student = M_test[idx1, idx2]
        correct = M_test_ans[idx1, idx2]
        if student != correct:
            print("Correct M:")
            print(M_test_ans)
            print("Your M: ")
            print(M_test)
            raise AssertionError("Incorrect count at index ({}, {})=({}, {}) in matrix M. Yours has {} but should have {}.".format(idx1, idx2, w1, w2, student, correct))

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)


#################################
# TODO: c)
def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurence count matrix of dimensionality
        (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following
         SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of corpus words, number 
                of corpus words)): co-occurence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)):
            matrix of k-dimensioal word embeddings.
            In terms of the SVD from math class, this actually returns U * S
    """
    n_iters = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ------------------
    # Write your implementation here.
    svd = TruncatedSVD(n_components=k, n_iter=n_iters, random_state=42)
    M_reduced = svd.fit_transform(M)
    # ------------------

    print("Done.")
    return M_reduced

# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness 
# In fact we only check that your M_reduced has the right dimensions.
# ---------------------

# Define toy corpus and run student code
test_corpus = ["START Ala miec kot i pies END".split(" "),
               "START Ala lubic kot END".split(" ")]  
M_test, word2Ind_test = compute_co_occurrence_matrix(test_corpus, window_size=1)
M_test_reduced = reduce_to_k_dim(M_test, k=2)

# Test proper dimensions
assert (M_test_reduced.shape[0] == 8), "M_reduced has {} rows; should have {}".format(M_test_reduced.shape[0], 8)
assert (M_test_reduced.shape[1] == 2), "M_reduced has {} columns; should have {}".format(M_test_reduced.shape[1], 2)

# Print Success
print ("-" * 80)
print("Passed All Tests!")
print ("-" * 80)

#################################
# TODO: d)
def plot_embeddings(M_reduced, word2Ind, words, save=False, f_name=None):
    """ Plot in a scatterplot the embeddings of the words specified 
        in the list "words".
        NOTE: do not plot all the words listed in M_reduced / word2Ind.
        Include a label next to each point.
        
        Params:
            M_reduced (numpy matrix of shape (number of unique words in the
            corpus , k)): matrix of k-dimensioal word embeddings
            word2Ind (dict): dictionary that maps word to indices for matrix M
            words (list of strings): words whose embeddings we want to
            visualize
    """

    # ------------------
    # Write your implementation here.
    x_val = []
    y_val = []

    for word in words:
        [x, y] = M_reduced[word2Ind[word]]
        x_val.append(x)
        y_val.append(y)

    fig, ax = plt.subplots()
    ax.scatter(x_val, y_val)

    for word in words:
        [x, y] = M_reduced[word2Ind[word]]
        ax.annotate(word, (x, y))

    if save:
        filename = f_name + '.png'
        plt.savefig(filename)
    # ------------------#

# ---------------------
# Run this sanity check
# Note that this not an exhaustive check for correctness.
# The plot produced should look like the "test solution plot" depicted below. 
# ---------------------

print ("-" * 80)
print ("Outputted Plot:")

M_reduced_plot_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]])
word2Ind_plot_test = {
    'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4}
words = ['test1', 'test2', 'test3', 'test4', 'test5']
plot_embeddings(M_reduced_plot_test, word2Ind_plot_test, words, True, 'plot_test')

print ("-" * 80)


#################################
# TODO: e)
# -----------------------------
# Run This Cell to Produce Your Plot
# ------------------------------

def read_corpus_pl():
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    pl196x_dir = nltk.data.find('corpora/pl196x')
    pl = pl196x.Pl196xCorpusReader(
        pl196x_dir, r'.*\.xml', textids='textids.txt',cat_file="cats.txt")
    tsents = pl.tagged_sents(fileids=pl.fileids(),categories='cats.txt')[:5000]

    return [[START_TOKEN] + [
        w[0].lower() for w in list(sent)] + [END_TOKEN] for sent in tsents]


def plot_unnormalized(corpus, words, save=False, f_name=None):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    plot_embeddings(M_reduced_co_occurrence, word2Ind_co_occurrence, words, save=save, f_name=f_name)


def plot_normalized(corpus, words, save=False, f_name=None):
    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(
        corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, save=save, f_name=f_name)

pl_corpus = read_corpus_pl()
words = [
    "sztuka", "śpiewaczka", "literatura", "poeta", "obywatel"]

plot_normalized(pl_corpus, words, True, 'plot_normalized')
plot_unnormalized(pl_corpus, words, True, 'plot_unnormalized')

# What clusters together in 2-dimensional embedding space?
# Answer:
# In normalized plot we can see two clusters: one with word 'sztuka'
# and one with words 'śpiewaczka', 'poeta', 'obywatel', literatura'
# In unnormalized plot we have three clusters: one with word 'sztuka',
# one with word 'literatura' and one with words 'śpiewaczka', 'poeta', 'obywatel'


# What doesn’t cluster together that you might think should have?
# Answer: 'sztuka' and 'literatura' are far away from each other in both cases, it's rather unintuitive


# TruncatedSVD returns U × S, so we normalize the returned vectors in the second plot, so
# that all the vectors will appear around the unit circle. Is normalization necessary?
# Answer: I think that normalized plot is worse than unnormalized (obtained clusters are not very good).


#################################
# Section 2:
#################################
# Then run the following to load the word2vec vectors into memory. 
# Note: This might take several minutes.
wv_from_bin_pl = KeyedVectors.load("word2vec_100_3_polish.bin")

# -----------------------------------
# Run Cell to Load Word Vectors
# Note: This may take several minutes
# -----------------------------------


#################################
# TODO: a)
def get_matrix_of_vectors(wv_from_bin, required_words):
    """ Put the word2vec vectors into a matrix M.
        Param:
            wv_from_bin: KeyedVectors object; the 3 million word2vec vectors
                         loaded from file
        Return:
            M: numpy matrix shape (num words, 300) containing the vectors
            word2Ind: dictionary mapping each word to its row number in M
    """
    words = list(wv_from_bin.key_to_index.keys())
    print("Shuffling words ...")
    random.shuffle(words)
    words = words[:10000]
    print("Putting %i words into word2Ind and matrix M..." % len(words))
    word2Ind = {}
    M = []
    curInd = 0
    for w in words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    for w in required_words:
        try:
            M.append(wv_from_bin.word_vec(w))
            word2Ind[w] = curInd
            curInd += 1
        except KeyError:
            continue
    M = np.stack(M)
    print("Done.")
    return M, word2Ind

# -----------------------------------------------------------------
# Run Cell to Reduce 300-Dimensinal Word Embeddings to k Dimensions
# Note: This may take several minutes
# -----------------------------------------------------------------

#################################
# TODO: a)
M, word2Ind = get_matrix_of_vectors(wv_from_bin_pl, words)
M_reduced = reduce_to_k_dim(M, k=2)

words = [
    "sztuka", "śpiewaczka", "literatura", "poeta", "obywatel"]
plot_embeddings(M_reduced, word2Ind, words, True, 'plot_reduced')

# What clusters together in 2-dimensional embedding space?
# Answer: We can cluster 'śpiewaczka', 'poeta' and 'obywatel' in one cluster (as professions) and
# 'literatura' and 'sztuka' in the other. It's worth observing that female profession ('śpiewaczka') is further
# than male ones.

# What doesn’t cluster together that you might think should have?
# Answer: I think that received clusters are correct.

# How is the plot different from the one generated earlier from the co-occurrence matrix?
# Answer: We can see that the closest word to 'sztuka' is 'literatura' and the closest word to 'śpiewaczka' is 'poeta'.
# This is very reasonable in case of the word meaning. It wasn't visible in the previous plots


#################################
# TODO: b)
# Polysemous Words
# ------------------
# Write your polysemous word exploration code here.

polysemous = wv_from_bin_pl.most_similar("stówa")
for i in range(10):
    key, similarity = polysemous[i]
    print(i, key, similarity)

# ------------------
# Please state the polysemous word you discover and the multiple meanings that occur in the top 10.
# Why do you think many of the polysemous words you tried didn’t work?

# Answer: Words I've tried (top 10 words in brackets):

# zamek: (pałac, zamczysko, forteca, grodźm warownia, gród, donżon, dworzyszcze, cytadela)
# babka: (babcia, ciotka, wnuczka, siostra, teściowa, matka, kuzynka, ciocia, babki, synowa)
# klucz: (kluczyk, wytrych, sejf, skrytka, schowek, kasetka, krypteks, zawiaska, zatrzask, szufladka)
# kolejka: (wagonik, peron, rampa, budka, wagon, przystanek, taksówka, zajezdnia, autobus, rozjazd)
# pióro: (ołówek, kałamarz, długopis, stalówka, pióra, pędzelek, obsadka, pergamin, inkaust, piórko)
# igła: (szpilka, rurka, szpilki, rurki, nitka, nożyczki, pręt, kulka, patyczek, drucik)
# szpilka: (szpilki, igła, spinka, agrafka, łańcuszek, gumka, kulka, patyczek, sznurek)
# pokój: (sypialnia, pokoik, bawialnia, jadalnia, przedpokój, alkowa, komnata, salon, mieszkanie, koju)
# piła: (pilić, pić, żłopać, pijać, wypijać, popijać, jedla, chlać, jadło, popijała)
# sztuka : (sztuca, sztuk, technika, kunszt, poezja, estetyka, dramaturgia, proza, arcydzieło, retoryka)
# prawy: (lewy, lewa, dolny, środkowy, przedni, podbródkowy, górny, przyśrodkowy, obrąbka, dystalny)
# góra: (dół, zbicze, góry, stok, urwisko, wzgórze, szczyt, pagórek, wydma, dola)
# para: (par, czwórka, dwójka, trójka, gromadka, grupka, tancerka, szóstka, blondynka, kitka)

# I think they didn't work out due to the word corpus the embeddings were trained on. We can see a lot of strange words
# that we don't use nowadays or words that are very specific in some areas (like in the case of word 'komórka').
# With that intuition I tried to find the right word. I suspected that embeddings were trained on wikipedia so I checked
# the 'pager' article. I found 'telefon komórkowy' in the text -- I turned out that 'komórkowy' is close
# to biological terminology and mobile phone ('przewodowy', 'bezprzewodowy').


# komórka :(neuron, tkanka, synapsa, mitochondrium, mórki, organellum, mórek, molekuła, jądro, błona)
# telefon: (pager, wideofon, faks, telefonia, domofon, lefonu, wideotelefon, telefonistka, radiotelefon, interkom)
# komórkowy: (receptorowy, błonowy, neurotransmiter, acetylocholina, mitochondrium, przewodowy, organellum, neuronowy,
# bezprzewodowy, neurohormon)


polysemous = wv_from_bin_pl.most_similar("komórkowy")
for i in range(10):
    key, similarity = polysemous[i]
    print(i+1, key, similarity)
# ------------------

#################################
# TODO: c)
# Synonyms & Antonyms
# ------------------
# Write your synonym & antonym exploration code here.

w1 = "radosny"
w2 = "pogodny"
w3 = "smutny"
w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
w1_w3_dist = wv_from_bin_pl.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

# Synonyms mądry, rozumny have cosine distance: 0.3054819107055664
# Antonyms mądry, głupi have cosine distance: 0.3935270309448242
# Synonyms stary, wiekowy have cosine distance: 0.8029355704784393
# Antonyms stary, młody have cosine distance: 0.45429128408432007
# Synonyms prosty, łatwy have cosine distance: 0.8007050156593323
# Antonyms prosty, krzywy have cosine distance: 0.9099887385964394
# Synonyms błyszczący, lśniący have cosine distance: 0.04743552207946777
# Antonyms błyszczący, matowy have cosine distance: 0.21790605783462524

w1 = "silny"
w2 = "mocny"
w3 = "słaby"
w1_w2_dist = wv_from_bin_pl.distance(w1, w2)
w1_w3_dist = wv_from_bin_pl.distance(w1, w3)

# Synonyms silny, mocny have cosine distance: 0.23193776607513428
# Antonyms silny, słaby have cosine distance: 0.18360120058059692

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

# Once you have found your example, please give a possible explanation for why this counterintuitive result
# may have happened.
# Answer: Antonymous words often occur in similar context -- if two words are often used in a similar context,
# they should have a small cosine similarity between the embeddings.

#################################
# TODO: d)
# Solving Analogies with Word Vectors
# ------------------

# ------------------
# Write your analogy exploration code here.
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["syn", "kobieta"], negative=["mezczyzna"]))

pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["król", "kobieta"], negative=["mezczyzna"]))

# Answer:  mezczyzna: król :: kobieta : królowa
# 'Królowa' is in the top 3

# [('książę', 0.6735260486602783),
#  ('księżniczka', 0.6372401714324951),
#  ('królowa', 0.6154096722602844),
#  ('monarcha', 0.6138051748275757),
#  ('królewicz', 0.5941848158836365),
#  ('możnowładca', 0.5807306170463562),
#  ('szahdżahana', 0.5804842114448547),
#  ('księżna', 0.5784489512443542),
#  ('cesarz', 0.5768533945083618),
#  ('królewna', 0.5737603306770325)]


#################################
# TODO: e)
# Incorrect Analogy
# ------------------
# Write your incorrect analogy exploration code here.
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=["szef", "kobieta"], negative=["mezczyzna"]))

# Answer: mezczyzna: szef :: kobieta : szefowa
# Incorrect  values below:

# [('własika', 0.5678122639656067),
#  ('agent', 0.5483713150024414),
#  ('oficer', 0.5411549210548401),
#  ('esperów', 0.5383270978927612),
#  ('interpol', 0.5367037653923035),
#  ('antyterrorystyczny', 0.5327680110931396),
#  ('komisarz', 0.5326411128044128),
#  ('europolu', 0.5274547338485718),
#  ('bnd', 0.5271410346031189),
#  ('pracownik', 0.5215375423431396)]
# ------------------


#################################
# TODO: f)
# Guided Analysis of Bias in Word Vectors
# Here `positive` indicates the list of words to be similar to and 
# `negative` indicates the list of words to be most dissimilar from.
# ------------------
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['kobieta', 'szef'], negative=['mezczyzna']))
print()
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['mezczyzna', 'prezes'], negative=['kobieta']))


# Which terms are most similar to ”kobieta” and ”szef” and most dissimilar to ”mezczyzna”,
# Answer: 'szefowa', but in top 10 we don't find words connected directly to females:
# 'własika', 'agent', 'oficer', 'esperów', 'interpol', 'antyterrorystyczny', 'komisarz', 'europolu', 'bnd', 'pracownik'

# Which terms are most similar to ”mezczyzna” and ”prezes” and most dissimilar to ”kobieta”.
# Answer: 'prezes' and we find similar words in the top 10:
# 'wiceprezes', 'czlonkiem', 'przewodniczący', 'czlonek', 'przewodniczącym', 'wiceprzewodniczący', 'obowiazków',
# 'obowiazani', 'dyrektor', 'obowiązany'


#################################
# TODO: g)
# Independent Analysis of Bias in Word Vectors 
# ------------------
pprint.pprint(wv_from_bin_pl.most_similar(
    positive=['kobieta', 'żołnierz'], negative=['mezczyzna']))

# Use the most similar function to find another case where some bias is exhibited by the
# vectors. Briefly explain the example of bias that you discover.
# Answer: I checked the analogies to mezczyzna : żołnierz :: kobieta : żołnierka/wojowniczka
# I obtained following words which are not close to the female version of 'żołnierz':

# [('jeniec', 0.6585068106651306),
#  ('partyzant', 0.6348068714141846),
#  ('nierzy', 0.6302312016487122),
#  ('najemnik', 0.6168861389160156),
#  ('ludzie', 0.6151184439659119),
#  ('cywil', 0.6111915707588196),
#  ('marynarz', 0.603709876537323),
#  ('legionista', 0.6008935570716858),
#  ('dziewczę', 0.5912114977836609),
#  ('czerwonoarmista', 0.5888139605522156)]

# TODO: h)
# What might be the cause of these biases in the word vectors?
# Answer: The main source of the bias in word vectors is the data that we use to train the embeddings.
# Some words are traditionally associated with sex (like some professions or attributes).

################################
# Section 3:
# English part
#################################
def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.key_to_index.keys())
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin

wv_from_bin = load_word2vec()

#################################
# TODO: 
# Find English equivalent examples for points b) to g).
#################################
# TODO: b)


polysemous = wv_from_bin.most_similar("cellular")
for i in range(10):
    key, similarity = polysemous[i]
    print(i, key, similarity)

# 0 Dr._Andrei_Gudkov 0.6398824453353882
# 1 phone_reverse_lookup 0.62388014793396
# 2 celluar 0.6222706437110901
# 3 Femto_cells 0.6154134273529053
# 4 wireless 0.611581027507782
# 5 ceramic_particulate_filter 0.6001943945884705
# 6 telecommunication 0.5893813371658325
# 7 cell_phone 0.5882241725921631
# 8 GSM_cellular 0.582930862903595
# 9 cellphone 0.5823144316673279


# TODO: c)


w1 = "strong"
w2 = "hard"
w3 = "weak"
w1_w2_dist = wv_from_bin.distance(w1, w2)
w1_w3_dist = wv_from_bin.distance(w1, w3)

print("Synonyms {}, {} have cosine distance: {}".format(w1, w2, w1_w2_dist))
print("Antonyms {}, {} have cosine distance: {}".format(w1, w3, w1_w3_dist))

# Synonyms strong, hard have cosine distance: 0.6976182162761688
# Antonyms strong, weak have cosine distance: 0.3842710852622986


# TODO: d)

pprint.pprint(wv_from_bin.most_similar(
    positive=["son", "woman"], negative=["men"]))

# [('daughter', 0.7800601124763489),
#  ('mother', 0.7436666488647461),
#  ('niece', 0.6925152540206909),
#  ('granddaughter', 0.6870002746582031),
#  ('grandson', 0.6627058386802673),
#  ('grandmother', 0.646586537361145),
#  ('stepdaughter', 0.6465805768966675),
#  ('father', 0.640291154384613),
#  ('aunt', 0.6323437094688416),
#  ('husband', 0.6284565925598145)]


pprint.pprint(wv_from_bin.most_similar(
    positive=["king", "woman"], negative=["men"]))

# [('queen', 0.5957391858100891),
#  ('monarch', 0.5637064576148987),
#  ('princess', 0.5488066673278809),
#  ('ruler', 0.5087037086486816),
#  ('prince', 0.502140998840332),
#  ('crown_prince', 0.4895453155040741),
#  ('sultan', 0.46843448281288147),
#  ('King_Ahasuerus', 0.45483770966529846),
#  ('maharaja', 0.4490197002887726),
#  ('Queen_Consort', 0.43428054451942444)]


# TODO: e)

pprint.pprint(wv_from_bin.most_similar(
    positive=["boss", "woman"], negative=["men"]))

# [('receptionist', 0.4684194028377533),
#  ('coworker', 0.4647681713104248),
#  ('Fiz_Jennie_McAlpine', 0.45498374104499817),
#  ('businesswoman', 0.4429871439933777),
#  ('lady', 0.4387196898460388),
#  ('exec', 0.4349324703216553),
#  ('girlfriend', 0.43153348565101624),
#  ('worker', 0.43021029233932495),
#  ('staffer', 0.4214950501918793),
#  ('MUM_TO_BE', 0.420085608959198)]


# TODO: f)

pprint.pprint(wv_from_bin.most_similar(
    positive=['woman', 'boss'], negative=['men']))

# [('receptionist', 0.4684194028377533),
#  ('coworker', 0.4647681713104248),
#  ('Fiz_Jennie_McAlpine', 0.45498374104499817),
#  ('businesswoman', 0.4429871439933777),
#  ('lady', 0.4387196898460388),
#  ('exec', 0.4349324703216553),
#  ('girlfriend', 0.43153348565101624),
#  ('worker', 0.43021029233932495),
#  ('staffer', 0.4214950501918793),
#  ('MUM_TO_BE', 0.420085608959198)]


print()
pprint.pprint(wv_from_bin.most_similar(
    positive=['men', 'chairman'], negative=['woman']))

# [('Chairman', 0.6223933100700378),
#  ('chariman', 0.5649006366729736),
#  ('chief_executive', 0.5222398638725281),
#  ('chairmen', 0.5125986337661743),
#  ('Vice_Chairmen', 0.48710301518440247),
#  ('chairmain', 0.485930472612381),
#  ('chaiman', 0.4807630479335785),
#  ('managing_director', 0.47778111696243286),
#  ('cochairman', 0.4742610454559326),
#  ('Chief_Executive', 0.47092965245246887)]

# TODO: g)

pprint.pprint(wv_from_bin.most_similar(
    positive=['woman', 'solider'], negative=['men']))

# [('soldier', 0.7221205830574036),
#  ('serviceman', 0.5974410176277161),
#  ('airman', 0.5473331809043884),
#  ('Soldier', 0.5266920328140259),
#  ('girl', 0.5174804925918579),
#  ('paratrooper', 0.5063685178756714),
#  ('Army_Reservist', 0.5004420876502991),
#  ('National_Guardsman', 0.4948287010192871),
#  ('guardsman', 0.48919495940208435),
#  ('policewoman', 0.48758482933044434)]

# Load vectors for English and run similar analysis for points from b) to g). Have you observed
# any qualitative differences? Answer with up to 7 sentences.
# Answer: We can observe few differences. In b) we see that 'cellular' was interpreted mostly as mobile phone and not
# something connected to biology -- this is probably caused by very different word corpus on which the embeddings were
# trained on (google news corpus consists more current words and may be less formal than wikipedia). In c) we obtained
# the same effect (with even better results). In d) we obtained assumed results, similar to those in Polish
# (maybe slightly better than in Polish). In e) we can see bias, but we can also observe a word 'businesswoman' -- it may
# also be caused by different corpus. In f) we can see that word 'chairman' is biased. In g) we also can see bias, but
# there is also word 'policewoman'.
# To conclude: The results are similar but the English ones are quite better - especially with biased words. I think it
# is mostly caused by the data we used to train the word embeddings. The language is changing, people are now more
# concered about geneder equallity in language. It is also different when we train embeddings on words in specific domain
# (we could observe that with biological words in Polish).