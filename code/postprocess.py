import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import pickle
import gensim.corpora as corpora
import os

# My helper scripts
import graphing_tools.graphing as gr

''' Preparation '''
def load_corpus(parent_dir):
    return corpora.MmCorpus('%s/preprocess/corpus.mm'%parent_dir)

def load_documents(parent_dir):
    pd.options.display.max_colwidth = 10
    df = pd.read_csv('%s/preprocess/breastcancer_reviews_refined.csv'%parent_dir)
    select_cols = ['PMID','Year','Title','Abstract','Total_times_cited']
    df = df[select_cols]
    return df

def load_models(parent_dir):
    lst = os.listdir(parent_dir+'/models')
    model_files = [f for f in lst if 'model' in f]
    model_files.sort(key = lambda f: f.split('score')[1], reverse=True)
    models = []
    for model_file in model_files:
        with open(str(parent_dir+'/models/'+model_file),'rb') as f:
            models.append(pickle.load(f))
    print('* [INFO] Models loaded: %s' %len(models))
    print('* [INFO] List of models in order of score: %s'%list(model_files))
    return models

def make_dir(parent_dir,model):
    print("\n* -> Creating directory for this model ...")
    parent_dir = '/'.join(parent_dir.split('/')[0:2])
    current_dir = '%s/postprocess/%stopics'%(parent_dir,model.num_topics)
    os.makedirs(current_dir, exist_ok=True)
    wordcloud_dir = '%s/visualize_models/%stopics'%(parent_dir,model.num_topics)
    return current_dir,wordcloud_dir

def get_doc_topic_scores(model,corpus,df,directory):
    print("* -> Getting topic scores for documents ...")
    topic_cols = []
    for topic_num in range(model.num_topics):
        topic_cols.append('Topic_%s'%(topic_num+1))
    doc_topics_df = pd.DataFrame(columns=topic_cols)
    
    for i, row in enumerate(model[corpus]):
        doc = pd.Series(index=topic_cols)
        for j, (topic_num, prop_topic) in enumerate(row): # prop_topic is the percentage contribution of a topic to a comment.
            doc['Topic_%s'%(topic_num+1)] = prop_topic
        doc_topics_df = doc_topics_df.append(doc,ignore_index=True)

    doc_topics_df['Dominant Topic'] = doc_topics_df.idxmax(axis=1)
    df = pd.concat([df, doc_topics_df], axis=1)
    df.to_csv('%s/df_topic_scores.csv'%directory,index=False)
    return df

def load_topic_scores(model,current_dir):
    print('* -> Loading topic scores ...')
    df = pd.read_csv('%s/df_topic_scores.csv'%current_dir)
    try:
        df.drop(columns=['Unnamed: 0'])
    except:
        print('Document does not have "Unnamed" column.')
    return df

''' Output '''
def most_representative_titles(model,df,directory):
    num_topics = model.num_topics

    print("* -> Getting the most representative titles of each topic ...")
    df_most_rep_titles = pd.DataFrame()
    for topic in range(num_topics):
        top_documents = df.nlargest(1,'Topic_%s'%(topic+1))
        df_most_rep_titles = df_most_rep_titles.append(top_documents,ignore_index=True)
    df_most_rep_titles.drop(columns=['Abstract'],inplace=True)
    df_most_rep_titles.to_csv('%s/most_representative_titles.csv'%directory)
    return df_most_rep_titles

def get_topic_distribution(df,model,current_dir,threshold,wordcloud_dir):

    topic_mentions = _topic_mentions(df,model,current_dir,threshold,wordcloud_dir)   

    #topic_dist = _topics_distrib(df,model,threshold,current_dir,wordcloud_dir,dominant=True)

    return topic_mentions

def _topic_mentions(df,model,current_dir,threshold,wordcloud_dir):
    print('* -> Getting Topic Distribution (Mentions) ...')
    topic_mentions = df[df>=threshold].count()
    topic_mentions = topic_mentions[[col for col in df.columns if 'Topic_' in col]]
    topic_mentions.to_csv('%s/topic_dist_mentions.csv'%current_dir)
    gr.show_distribution(topic_mentions,model,current_dir,wordcloud_dir,dominant=False)
    return topic_mentions

def _topic_distrib(df,model,threshold,current_dir,wordcloud_dir):
    print('* -> Getting Topic Distribution (Dominant) ...')
    topic_dist = df['Dominant Topic'].value_counts()
    topic_dist = topic_dist.reindex(index=topic_cols).fillna(0)
    topic_dist.to_csv('%s/topic_dist_dominant.csv'%current_dir)
    gr.show_distribution(topic_dist,model,threshold,current_dir,wordcloud_dir,dominant=True)
    return topic_dist

def get_year_trend(df,num_topics,current_dir,threshold,wordcloud_dir):
    print('* -> Getting Year Trends  ...')

    print('* - - > Get trend in terms of absolute papers  ...')
    year_trend,total_growth = _trend(df,num_topics,threshold,relative=False)
    gr.show_trend(year_trend,total_growth,current_dir,wordcloud_dir,relative=False)

    print('* - - > Get trend in terms of proportion of total papers ...')
    year_trend,total_growth = _trend(df,num_topics,threshold,relative=True)
    gr.show_trend(year_trend,total_growth,current_dir,wordcloud_dir,relative=True)

    return year_trend,total_growth

def _total_growth(year_trend):
    # Gets total growth as % of 1997 number. Does not work if there were zero papers published in 1997. 
    return list(map(lambda x: (x.fillna(0).values[-1] - x.values[0])/x.values[0]*100,year_trend))

def _trend(df,num_topics,threshold,relative=False):
    year_trend = []
    papers_per_year = df.groupby('Year').size()
    for i in range(num_topics):
        topic_mentions = df[df['Topic_%s'%(i+1)]>=threshold].groupby('Year').size()
        if relative:
            topic_mentions = topic_mentions/papers_per_year
        year_trend.append(topic_mentions)
    total_growth = _total_growth(year_trend)
    return year_trend, total_growth

def get_venn(df,year_trend,total_growth,threshold,current_dir,wordcloud_dir):
    print('* -> Getting Venn Diagrams to show intersectionality ... ')
    #__ = gr.venn_top3(df,year_trend,threshold,current_dir,wordcloud_dir)
    venn1 = gr.venn_growth(df,year_trend,total_growth,threshold,current_dir,wordcloud_dir)
    #__ = gr.venn_top3(df,year_trend,0.25,current_dir,wordcloud_dir)
    venn2 = gr.venn_growth(df,year_trend,total_growth,0.25,current_dir,wordcloud_dir)
    gr.merge_two_venns(venn1,venn2,current_dir)

def main():
    parent_dir = 'saved_files/1997_to_2017'
    corpus = load_corpus(parent_dir)
    models = load_models(parent_dir)

    for model in models:
        num_topics = model.num_topics
        print(f'\n* Now postprocessing for {num_topics} topics model ...')

        ''' To select models '''
        if num_topics not in [11]:
            print('* --Skip-- ')
            continue
        ''' Declare if model has been run before (default: False) '''
        prev_run = True

        print('\n* Preparation step: ')
        current_dir,wordcloud_dir = make_dir(parent_dir,model)
        df = load_documents(parent_dir)
        if not prev_run:
            df = get_doc_topic_scores(model,corpus,df,current_dir)
        else:
            df = load_topic_scores(model,current_dir) # if topic_scores already retrieved. 

        print('\n* Output step: ')
        threshold = 0.10
        topic_mentions = get_topic_distribution(df,model,current_dir,threshold,wordcloud_dir)
        year_trend,total_growth = get_year_trend(df,num_topics,current_dir,threshold,wordcloud_dir)
        #get_venn(df,year_trend,total_growth,threshold,current_dir,wordcloud_dir)
        #most_rep_titles = most_representative_titles(model,df,current_dir)

if __name__ == "__main__":
    main()