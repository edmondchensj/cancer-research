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
import numpy as np
import pandas as pd
import pickle
import time

''' Table of Contents
    [To improve] avoid lower casing. try making ngram with spacy. 
    1. Preprocessing functions
    2. Generate summary
    3. Main function '''

# 1. Preprocessing functions
def refine_data(file_name,start_year,end_year):
    pd.options.display.max_colwidth = 12
    df = pd.read_csv(file_name)
    print("* -> [INFO] No. of entries in input data file: %d" %df.shape[0])
    print("* -> Previewing input data:")
    print(df.head())

    df_new = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
    print("* -> [INFO] Drop entries outside of year range (%s - %s). New no. of entries: %d, entries dropped: %d" 
        %(start_year,end_year,df_new.shape[0],df.shape[0]-df_new.shape[0]))
    print(df_new.head())

    print("* -> Shuffle rows randomly: ")
    df_new = df_new.reindex(np.random.permutation(df_new.index))
    df_new.reset_index(drop=True,inplace=True)
    print(df_new.head())
    return df_new

def abstracts_to_list(df):
    print('* -> Converting Abstracts column to list ...')
    return df.Abstract.values.tolist()

def tokenize(texts):
    print('* -> Converting list of abstract to list of list of words ...')
    return [simple_preprocess(str(doc),max_len=22) for doc in texts]

def remove_stopwords_prelim(texts):
    print('* -> Perform preliminary removal of stopwords ...')
    # This helps to prevent trivial n-grams from forming later. e.g. woman_who_have
    stops = stopwords.words('english')
    stops += breastcancer_stopwords()['l1']
    return [[word for word in doc if word not in stops] for doc in texts]

def breastcancer_stopwords():
    # l1 terms are common across all breast cancer topics and not part of an important n-gram. 
    # l2 terms describe broad categories and/or may be part of n-gram. examples: fracture_risk and elevated_risk; not sure about patient/woman. 
    return {'l1':['breast','cancer','cancers','female','research','model','effect','association',
                'outcome','outcomes','question','finding','findings','control','study','studies',
                'percent','confidence','interval','hazard','ratio','article','data','risk','program',
                'review','reviews','disease','stage','trial','clinical','factor','role','tumor','tumour',
                'evidence','definition','practice','paper','current','woman','women'],
            'l2':['treatment','therapy','therapeutic','cell','use','survivor',
                'patient','surgery','case','benefit',
                'survival','technique','detection','method','management','diagnosis',
                'care','gene','efficacy','mechanism','development','pathway',
                'receptor','activity','function','estrogen','protein','expression','target','growth','estrogen',
                'outcome','symptom','biomarker','marker','approach','year','drug','result',
                'progression','response','age','incidence','information','literature','prevention','strategy',
                'testing','identification','intervention','regimen','test','et_al','group',
                'bc','prognosis','subtype','inhibitor','biology','analysis','rate','procedure','tissue','level','change',
                'datum','bca','specimen','mortality','evaluation','signaling_pathway','activation','signal','regulation',
                'status','assessment','addition','health',
                'signature','stat','screening','focus','property','action','molecule','application',
                'process','discovery','characteristic','assay','panel','guideline','report','syndrome','examination','exam',
                'user','outlook','key','recommendation','option','challenge','tool','guide','issue','measure']}
                # comments: some are difficult to decide - "intervention' -> could refer to population-wide intervention policies. 
                # "bc" appears after lemmatizing - perhaps from LTQOL-BC or breast-conserving surgery (BCS) - we lose some information here, though BCS has other terms (mastectomy). 
                # HR_CI stands for hazard ratio and confidence intervals. 
                # BCa is short for breast cancer, but could also refer to other things. 
                # Mammogram/MRI is better than "screening". "Examination" also can be specified (self- or physical-)
def make_ngrams(texts):
    print('* -> Forming Bi- and tri-grams (words that appear together might be a term e.g. false_positive')
    bigram = gensim.models.Phrases(texts,threshold=10) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts],threshold=10)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatize(texts, allowed_postags=['PROPN','NOUN']): 
    # Part-of-speech tags: 
    # - PROPN (proper noun) keeps unique medical terms that do not have a root. 
    # - remove ADV, VERB, ADJ. 
    # - ADJ may be useful, but papers discussing a certain topic may see them differently, thus having oppposite adjectives. 
    # concern -> each doc is already tokenized before hand. would this affect POS accuracy? 
    # https://spacy.io/api/annotation
    timer = time.time()
    print('* -> Lemmatization i.e. find root form of words.')
    nlp = spacy.load('en',disable=['parser', 'ner']) # NLP pipeline includes POS-tagging, parser and NER. We only need POS. 
    texts_out = []
    for doc in texts: 
        doc = nlp(" ".join(doc)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags]) 
    print(f'Time taken for lemmatization: {time.time()-timer:.2f}s')
    return texts_out

def remove_stopwords_final(texts):
    print('* -> Final removal of stopwords ...')
    stops = stopwords.words('english')
    stops += ['-PRON-','per_cent'] # "-PRON-" and 'per_cent' emerges from spaCy lemmatizing.
    bc_stops = breastcancer_stopwords()['l1'] + breastcancer_stopwords()['l2']
    stops += bc_stops
    return [[word for word in doc if word not in stops] for doc in texts], bc_stops

# 2. Generate summary
def get_preprocess_summary(data,data_toks,data_nostopI,data_ngrams,data_lemmatize,data_nostopII,
                        bc_stops,id2word,corpus,directory):
    f = open('%s/preprocess_summary.txt' %directory,'w')
    f.write('Data Info:\n')
    f.write('Total Number of Documents: %s\n' %len(data))
    f.write('\nPhase 1: Text preprocessing (abstract-to-words, remove stopwords, make ngrams, lemmatize)\n')
    f.write("\nSample abstract:\n")
    f.write(str((data[0])[:250]))
    f.write('\nAbstract to list-of-words:\n')
    f.write(str((data_toks[0])[:30]))
    f.write('\nRemove stopwords (preliminary):\n')
    f.write(str((data_nostopI[0])[:30]))
    f.write('\nAdd bigrams and trigrams:\n')
    f.write(str((data_ngrams[0])[:30]))
    f.write('\nLemmatize (e.g. recommended, recommends -> recommend):\n')
    f.write(str((data_lemmatize[0])[:30]))
    f.write(f'\n\nBreast Cancer stopwords (total count:{len(bc_stops)}):\n')
    f.write(str(bc_stops))
    f.write('\nRemove stopwords (final):\n')
    f.write(str((data_nostopII[0])[:30]))
    f.write('\n\n\nPhase 2: Build corpus for LDA modeling.\n')
    f.write('\nCorpus:\n')
    f.write(str((corpus[0])[:50]))
    f.write('\nPreview Term Document Frequency in readable form: \n')
    f.write(str((([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])[0])[:50]))
    f.close()

# 3. Main Function and helpers
def save_for_later(corpus,id2word,texts,df,directory):
    id2word.save('%s/id2word.dict' %directory)
    corpora.MmCorpus.serialize('%s/corpus.mm' %directory, corpus)
    with open('%s/texts.pkl' %directory, 'wb') as f:
        pickle.dump(texts, f)
    df.to_csv('%s/breastcancer_reviews_refined.csv' %directory)

def make_directory(start_year,end_year):
    directory = 'saved_files/%s_to_%s/preprocess'%(start_year,end_year)
    os.makedirs(directory, exist_ok=True)
    return directory

def main():
    tic = time.time()
    file_name = "data/breastcancer_reviews_hasabstract_1997to2017.csv"
    start_year = 1997  # set start year
    end_year = 2017    # set end year

    print('\n* Phase 1: Text preprocessing ...')
    df = refine_data(file_name,start_year,end_year)
    data = abstracts_to_list(df)
    data_toks = tokenize(data)
    data_nostopI = remove_stopwords_prelim(data_toks)
    data_ngrams = make_ngrams(data_nostopI)
    data_lemmatize = lemmatize(data_ngrams)
    data_nostopII, bc_stops = remove_stopwords_final(data_lemmatize)

    print('\n* Phase 2: Build corpus for LDA modeling ...')
    texts = data_nostopII
    id2word = corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]

    print('\n* Preprocessing complete. Now saving files ...')
    directory = make_directory(start_year,end_year)
    get_preprocess_summary(data,data_toks,data_nostopI,data_ngrams,data_lemmatize,data_nostopII,
                        bc_stops,id2word,corpus,directory)
    save_for_later(corpus,id2word,texts,df,directory)

    toc = time.time()
    print('* Files saved. Time taken: %.2fs' %(toc-tic))
    return corpus, id2word, texts, df

if __name__ == "__main__":
    main()