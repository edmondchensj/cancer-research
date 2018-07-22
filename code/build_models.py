import re
import numpy as np
import pandas as pd
from pprint import pprint
import pickle
import time
import os
import heapq

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

def get_user_cutoff():
    for i in range(3):
        try:
            cutoff = int(input("\n* Based on coherence graph, enter cutoff number of topics: "))
            break
        except:
            print("* Not a valid number. %d tries left." %(2-i))
    return cutoff

def get_best_model_manually(models,coherence_values,num_topics_range,model_dir):
    input('* Press enter to get best models manually. Ctl-c to exit. ')
    get = 'y'
    while get=='y':

        accept = 'n'
        while accept=='n':
            cutoff = get_user_cutoff()
            print("* Now we will pick the best number of topics below %d." %cutoff)
            best_score,index = max((score,(trial_idx,topic_idx)) 
                                                    for trial_idx,scores in enumerate(coherence_values)
                                                    for topic_idx,score in enumerate(scores[:(cutoff-num_topics_range[0]+1)]))
            optimal_topics = num_topics_range[index[1]]
            print("* -> Best number of topics is %d, with coherence score of %.3f, in Trial %s" %(optimal_topics,best_score,index[0]+1))
            accept = input("Accept? [y/n]: ")

        best_model = models[index[0]][index[1]]
        with open('%s/model_%stopics_score%s.pkl'%(model_dir,optimal_topics,str(best_score)[2:5]),'wb') as f:
            pickle.dump(best_model, f)
        print("* -> Model saved.")
        getter = input("* Get more models? [y/n]: ")
        if getter!='y':
            get = input("* Confirm again: Get more models? [y/n]: ")

    print('* -> Saving complete. ')

def get_top_scores_and_index(coherence_values,num_topics_range,cutoff):
    '''
    Returns a list of tuples (score,(trial_index,topic_index))
    '''
    seq = ((score,(trial_idx,topic_idx)) for trial_idx,scores in enumerate(coherence_values) 
        for topic_idx,score in enumerate(scores[:(cutoff-num_topics_range[0]+1)]))
    return seq

def get_topn_models(models,coherence_values,num_topics_range,model_dir,n=7):
    '''
    Saves top n models below cutoff+1 number of topics.  
    '''
    cutoff = 18
    print('* Now getting top %d models at max %d topics' %(n,cutoff))
    seq = get_top_scores_and_index(coherence_values,num_topics_range,cutoff)
    topn_scores_idx = heapq.nlargest(n, seq)
    for i,(score,index) in enumerate(topn_scores_idx):
        optimal_topics = num_topics_range[index[1]]
        model = models[index[0]][index[1]]
        with open('%s/model_%stopics_score%s.pkl'%(model_dir,optimal_topics,str(score)[2:5]),'wb') as f:
            pickle.dump(model, f)
        print("* -> Model %d: Number of topics = %d, coherence score = %.3f, in Trial %s" %(i+1,optimal_topics,score,index[0]+1))
    print('* -> Top %s models saved as .pkl files in %s' %(n,model_dir))

def show_coherence_graph(coherence_values,num_topics_range,num_trials,model_dir):
    print("\n* Showing coherence graph ...")
    fig, ax = plt.subplots()
    plt.style.use('seaborn-notebook')
    x = num_topics_range

    for i in range(num_trials):
        ax.plot(x, coherence_values[i], label="Trial %s"%(i+1))
    avg_coherence = np.mean(np.array(coherence_values),axis=0)
    ax.plot(x,avg_coherence,color='gray',label='Average',linestyle='--')

    ax.xaxis.set_ticks(num_topics_range)
    ax.set_xlabel("Number of Topics")
    ax.set_ylabel("Coherence Score")
    ax.set_title("Coherence Scores of LDA models vs. Number of Topics")
    ax.grid(linestyle='--')
    ax.legend(loc="best")
    fig.savefig('%s/coherence_graph.png' %model_dir)
    plt.show(block=True)

def get_coherence_values(corpus, id2word, texts, num_topics_range, num_trials):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    id2word (dictionary) : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    models : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    print("\n* Running %s trials up to %s topics each to determine optimal number of topics." %(num_trials,num_topics_range[-1]))
    tic = time.time()
    coherence_values = [[] for i in range(num_trials)]
    models = [[] for i in range(num_trials)]
    for i in range(num_trials):
        print("* Running Trial %s ..." %(i+1))
        for num_topics in num_topics_range:
            tfidf_model = gensim.models.TfidfModel(corpus=corpus,id2word=id2word)
            model = gensim.models.ldamodel.LdaModel(corpus=tfidf_model[corpus], 
                                                    id2word=id2word, 
                                                    num_topics=num_topics)
            models[i].append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=id2word, coherence='c_v')
            coherence_values[i].append(coherencemodel.get_coherence())

            toc = time.time()
            print("* -> Computed for %s topics. Time taken to compute: %.2fs" %(num_topics,toc - tic))

    return models, coherence_values

def make_model_dir(filepath):
    model_dir = filepath + '/models'
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def load_preprocess_data(filepath):
    corpus = corpora.MmCorpus('%s/preprocess/corpus.mm' %filepath)
    id2word = np.load('%s/preprocess/id2word.dict' %filepath)
    with open('%s/preprocess/texts.pkl' %filepath, 'rb') as f:
        texts = pickle.load(f)
    model_dir = make_model_dir(filepath)
    return corpus,id2word,texts,model_dir

def main():
    filepath = 'saved_files/1997_to_2017'
    print('* Building models with data from %s' %filepath.split('/')[-1])
    num_topics_range = range(2,25,1) 
    num_trials = 5
    num_models_to_save = 7

    corpus,id2word,texts,model_dir = load_preprocess_data(filepath)

    models, coherence_values = get_coherence_values(corpus,
                                                    id2word, 
                                                    texts,
                                                    num_topics_range,
                                                    num_trials)
    
    show_coherence_graph(coherence_values,num_topics_range,num_trials,model_dir)

    get_topn_models(models,coherence_values,num_topics_range,model_dir,n=num_models_to_save)

    get_best_model_manually(models,coherence_values,num_topics_range,model_dir)

if __name__ == "__main__":
    main()