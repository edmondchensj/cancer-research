# Text Preprocessing
import nltk
import spacy
from nltk.corpus import stopwords

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# Misc
import re
import numpy as np
import pandas as pd
from pprint import pprint
import pickle

# This code is based on the tutorial from: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

def get_stopwords():
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    return stop_words

def sentence_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
    # yield is like return, but for "for" loops so that you get iterated results.

def remove_stopwords(texts):
    stop_words = stopwords.words('english')
    more_stop_words = 'breast cancer woman women female' # women->woman after lemmatization.
    stop_words += more_stop_words.split()
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    #https://spacy.io/api/annotation
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def make_ngrams(texts):
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def refine_data(file_name):
    pd.options.display.max_colwidth = 15
    df = pd.read_csv(file_name)
    print("\nPreviewing input data:")
    pprint(df.head())
    print("No. of entries in input data file: %d" %df.shape[0])
    df_new = df[(df['Year'] >= 1997) & (df['Year'] <= 2017)]
    print("\nDrop entries outside of year range (1997 - 2017). New no. of entries: %d, entries dropped: %d" 
        %(df_new.shape[0],df.shape[0]-df_new.shape[0]))
    print("Previewing new input data:")
    pprint(df_new.head())
    print("Reset index:")
    df = df_new.reset_index(drop=True)
    pprint(df.head())
    return df

def preprocess(file_name):
    df = refine_data(file_name)

    print("\nNow executing first phase of text processing (abstract-to-words, remove stopwords, make ngrams, lemmatize)...")
    data = df.Abstract.values.tolist()
    print("Sample abstract:")
    print((data[1])[:150])
    # Make texts into a list of words
    data_words = list(sentence_to_words(data))
    print("Preview after sentence_to_words:")
    pprint((data_words[1])[:10])
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    print("Preview after removing stopwords:")
    pprint((data_words_nostops[1])[:10])
    # Form Bi- and tri-grams (words that appear together might be a term e.g. false_positive)
    data_words_ngrams = make_ngrams(data_words_nostops)
    print("Preview after adding bigrams and trigrams:")
    pprint((data_words_ngrams[1])[:10])
    # Initialize spacy 'en' model. 
    # nlp = spacy.load('en', disable=['parser', 'ner'])
    # Do lemmatization keeping only noun, adj, vb, adv. Plural -> singular etc. 
    data_lemmatized = lemmatization(data_words_ngrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print("Preview after lemmatization:")
    pprint((data_lemmatized[1])[:10])

    print("\nNow executing second phase (build corpus for LDA modeling)...")
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    print("Second phase complete. Previewing term document frequency corpus [(word_id, word_frequency)]:")
    #print(corpus[:1])
    print((corpus[1])[:10])
    print("\nPreviewing term document freq corpus in readable form [(word, word_frequency)]:")
    print((([[(id2word[id], freq) for id, freq in cp] for cp in corpus[1:2]])[0])[:10])
    input("Pause:")

    # Store for later use
    id2word.save('saved_files/preprocess/test/id2word.dict')
    corpora.MmCorpus.serialize('saved_files/preprocess/test/corpus.mm', corpus)
    with open('saved_files/preprocess/test/texts.pkl', 'wb') as f:
        pickle.dump(texts, f)
    df.to_csv('saved_files/preprocess/test/breastcancer_reviews_refined.csv')
    return corpus, id2word, texts, df

if __name__ == "__main__":
    file_name = "saved_files/data/breastcancer_reviews_hasabstract_1997to2017.csv"
    preprocess(file_name)