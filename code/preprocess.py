# Text Preprocessing
import nltk
import spacy
from nltk.corpus import stopwords

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# Misc
import os
import re
import numpy as np
import pandas as pd
from pprint import pprint
import pickle
import time

# This code is built on the tutorial from: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

def save_for_later(corpus,id2word,texts,df,directory):
    id2word.save('%s/id2word.dict' %directory)
    corpora.MmCorpus.serialize('%s/corpus.mm' %directory, corpus)
    with open('%s/texts.pkl' %directory, 'wb') as f:
        pickle.dump(texts, f)
    df.to_csv('%s/breastcancer_reviews_refined.csv' %directory)

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
    breastcancer_stopwords = ['breast','cancer','woman','women','female','risk','risks','patient','patients',
    'screening','screen','screens','screenings','treatment','therapy','therapies','study','studies','research','diagnosis',
    'management','disease','survive','surviving','survival','stage','trial','trials']
    stop_words += breastcancer_stopwords
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

def refine_data(file_name,start_year,end_year):
    pd.options.display.max_colwidth = 12
    df = pd.read_csv(file_name)
    print("\nPreviewing input data:")
    pprint(df.head())
    print("No. of entries in input data file: %d" %df.shape[0])
    df_new = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    print("\nDrop entries outside of year range (%s - %s). New no. of entries: %d, entries dropped: %d" 
        %(start_year,end_year,df_new.shape[0],df.shape[0]-df_new.shape[0]))
    print("Previewing new input data:")
    pprint(df_new.head())
    print("Reset index:")
    df = df_new.reset_index(drop=True)
    pprint(df.head())
    return df

def get_preprocess_summary(data,data_words,data_words_nostops,data_words_ngrams,data_lemmatized,corpus,id2word,directory):
    f = open('%s/phase1_summary.txt' %directory,'w')
    f.write('Phase 1: Text preprocessing (abstract-to-words, remove stopwords, make ngrams, lemmatize)\n')
    f.write("\nSample abstract:\n")
    f.write(str((data[0])[:150]))
    f.write('\nAbstract to words:\n')
    f.write(str((data_words[0])[:10]))
    f.write('\nRemove stopwords:\n')
    f.write(str((data_words_nostops[0])[:10]))
    f.write('\nAdd bigrams and trigrams:\n')
    f.write(str((data_words_ngrams[0])[:10]))
    f.write('\nLemmatize (e.g. recommended, recommends -> recommend):\n')
    f.write(str((data_lemmatized[0])[:10]))
    f.write('\n\nPhase 2: Build corpus for LDA modeling.\n')
    f.write('\nCorpus:\n')
    f.write(str((corpus[0])[:10]))
    f.write('\nPreview Term Document Frequency in readable form: \n')
    f.write(str((([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])[0])[:10]))
    f.close()

def make_directory(start_year,end_year):
    directory = 'saved_files/%s_to_%s/preprocess'%(start_year,end_year)
    os.makedirs(directory, exist_ok=True)
    return directory

def main():
    '''
    Phase 1: Text preprocessing (abstract-to-words, remove stopwords, make ngrams, lemmatize)
    Phase 2: Build corpus for LDA modeling. 
    '''
    tic = time.time()
    file_name = "data/breastcancer_reviews_hasabstract_1997to2017.csv"
    start_year = 2007
    end_year = 2017
    
    df = refine_data(file_name,start_year,end_year)

    # Get abstracts as list
    data = df.Abstract.values.tolist()
    # Make texts into a list of words
    data_words = list(sentence_to_words(data))
    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)
    # Form Bi- and tri-grams (words that appear together might be a term e.g. false_positive)
    data_words_ngrams = make_ngrams(data_words_nostops)
    # Do lemmatization keeping only noun, adj, vb, adv. Plural -> singular etc. 
    data_lemmatized = lemmatization(data_words_ngrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    directory = make_directory(start_year,end_year)
    get_preprocess_summary(data,data_words,data_words_nostops,data_words_ngrams,data_lemmatized,corpus,id2word,directory)
    save_for_later(corpus,id2word,texts,df,directory)

    toc = time.time()
    print('Preprocessing complete. Time taken: %.2fs' %(toc-tic))

    return corpus, id2word, texts, df

if __name__ == "__main__":
    main()