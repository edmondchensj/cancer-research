import numpy as np
import pandas as pd
import pickle
import time
import gensim.corpora as corpora
import os
import math

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this

# Customized Graphing Tool
import graphing_tools.graphing as gr
import graphing_tools.wordcloudgrid as wcg
        
def get_pyLDAvis(model,corpus,id2word,current_dir):
    print("\n* Now we will visualize the topics using pyLDAvis.")
    vis = pyLDAvis.gensim.prepare(model, corpus, id2word, sort_topics=False)
    pyLDAvis.save_html(vis,'%s/topic_model.html'%current_dir)
    print("PyLDAvis saved to html.")

def make_dir(parent_dir,model):
    current_dir = '%s/visualize_models/%stopics'%(parent_dir,model.num_topics)
    os.makedirs(current_dir, exist_ok=True)
    return current_dir

def load_preprocess_data(parent_dir):
    print('* Getting data from %s ...' %parent_dir)
    corpus = corpora.MmCorpus('%s/preprocess/corpus.mm' %parent_dir)
    id2word = np.load('%s/preprocess/id2word.dict' %parent_dir)
    return corpus,id2word

def load_models(model_dir):
    lst = os.listdir(model_dir)
    model_lst = filter(lambda m: 'model' in m, lst)
    models = []
    for model_file in model_lst:
        with open(str(model_dir+'/'+model_file),'rb') as f:
            models.append(pickle.load(f))
    return models

def visualize_models(parent_dir,selected_models=None):
    corpus,id2word = load_preprocess_data(parent_dir)
    models = load_models(parent_dir+'/models')

    for model in models:
        # Select models to run. 
        num_topics = model.num_topics
        print(f'\n* Now visualizing for {num_topics} topics model ...')

        ''' To select models '''
        if selected_models is not None:
            if num_topics not in selected_models:
                print('* --Skip-- ')
                continue

        current_dir = make_dir(parent_dir,model)
        get_pyLDAvis(model,corpus,id2word,current_dir)
        wcg.get_wordclouds(model,current_dir)
        wcg.basicGrid(num_topics,wordcloud_dir=current_dir,target_dir=current_dir)

def main():
    ''' 
    Purpose:
        1. Visualizes coherence scores from models built in build_models.py
        2. Visualizes selected models as a wordcloud grid. 

    Usage:
        To visualize selected models, enter topic number in "selected_models" below.
        To visualize all saved models, set selected_models=None.
    '''
    parent_dir = 'saved_files/1997_to_2017'

    # Part I: Show Coherence Graph
    gr.show_coherence_graph(parent_dir+'/models')

    # Part II: Visualize Selected Models
    visualize_models(parent_dir,selected_models=None)
   
if __name__ == "__main__":
    main()