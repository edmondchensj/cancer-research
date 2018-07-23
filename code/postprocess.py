import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import pickle
import gensim.corpora as corpora
import os
import seaborn as sns

# My own helper scripts
import wordcloudgrid as wcg
import graphing as gr

'''
Load data
'''
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
    print("\n* Creating directory for this model ...")
    parent_dir = '/'.join(parent_dir.split('/')[0:2])
    current_dir = '%s/postprocess/%stopics'%(parent_dir,model.num_topics)
    os.makedirs(current_dir, exist_ok=True)
    wordcloud_dir = '%s/visualize_models/%stopics'%(parent_dir,model.num_topics)
    return current_dir,wordcloud_dir

'''
Preparation step:
'''
def get_doc_topic_scores(model,corpus,df,directory):
    print("* -> Getting topic scores for documents ...")
    topic_cols = []
    for topic_num in range(model.num_topics):
        topic_cols.append('Topic_%s'%(topic_num+1))
    doc_topics_df = pd.DataFrame(columns=topic_cols)
    
    # Access each document by row
    for i, row in enumerate(model[corpus]):
        doc = pd.Series(index=topic_cols)
        for j, (topic_num, prop_topic) in enumerate(row): # prop_topic is the percentage contribution of a topic to a comment.
            doc['Topic_%s'%(topic_num+1)] = prop_topic
        doc_topics_df = doc_topics_df.append(doc,ignore_index=True)

    # Get 'dominant' topic of each document. 
    doc_topics_df['Dominant Topic'] = doc_topics_df.idxmax(axis=1)

    # Combine with original dataframe
    df = pd.concat([df, doc_topics_df], axis=1)
    df.to_csv('%s/df_topic_scores.csv'%directory)
    return df

def load_topic_scores(model,current_dir):
    print('* -> Loading topic scores ...')
    df = pd.read_csv('%s/df_topic_scores.csv'%current_dir)
    df.drop(columns=['Unnamed: 0'])
    return df

'''
Output step:
'''
def get_topic_distribution(df,model,corpus,current_dir,topic_sensitivity,wordcloud_dir,std_footer):
    topic_cols = [col for col in df.columns if 'Topic_' in col]

    print('* -> Getting Topic Distribution (Mentions) ...')
    topic_mentions = df[df>=topic_sensitivity].count()
    topic_mentions = topic_mentions[topic_cols]
    topic_mentions.to_csv('%s/topic_dist_mentions.csv'%current_dir)

    gr.show_distribution(topic_mentions,model,corpus,topic_sensitivity,current_dir,wordcloud_dir,std_footer,dominant=False)

    print('* -> Getting Topic Distribution (Dominant) ...')
    topic_dist = df['Dominant Topic'].value_counts()
    topic_dist = topic_dist.reindex(index=topic_cols).fillna(0)
    topic_dist.to_csv('%s/topic_dist_dominant.csv'%current_dir)

    gr.show_distribution(topic_dist,model,corpus,topic_sensitivity,current_dir,wordcloud_dir,std_footer,dominant=True)

def get_year_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,std_footer):
    print('* -> Getting Year Trends  ...')
    topic_range = range(model.num_topics)

    print('* - - > Get trend in terms of absolute papers  ...')
    relative = False
    year_trend,total_growth = _trend(df,topic_range,topic_sensitivity,relative)
    gr.show_trend(year_trend,total_growth,model,topic_sensitivity,current_dir,wordcloud_dir,std_footer,relative)

    print('* - - > Get trend in terms of proportion of total papers ...')
    relative = True
    year_trend,total_growth = _trend(df,topic_range,topic_sensitivity,relative)
    gr.show_trend(year_trend,total_growth,model,topic_sensitivity,current_dir,wordcloud_dir,std_footer,relative)

def _total_growth(year_trend):
    func = lambda x: (x.values[-1] - x.values[0])/x.values[0]*100
    total_growth = list(map(func,year_trend))
    return total_growth

def _trend(df,topic_range,topic_sensitivity,relative=False):
    year_trend = []
    papers_per_year = df.groupby('Year').size()
    for i in topic_range:
        topic_mentions = df[df['Topic_%s'%(i+1)]>=topic_sensitivity].groupby('Year').size()
        if relative:
            topic_mentions = topic_mentions/papers_per_year
        year_trend.append(topic_mentions)
    total_growth = _total_growth(year_trend)
    return year_trend, total_growth

def most_cited_per_topic():
    void()

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

def main():
    '''
    Preparation:
    1. Get topic scores for each document

    Output:
    1. Distribution of research papers with mentions of topic (also share link to dominant topic distribution.)
    2. Year-on-year trend* -> interested to know about. 
    3. Most cited docs for each topic as a recommended list. 
        a. Histogram of intra-PubMed citations (x) (20)
        b. Topic mentions among top N (e.g. 100) most cited papers. (x) (15)
        c. Most cited in each topic as recommended list.
    4. (Possible - not sure.) Go in depth with the most popular 3-5 topics. Identify keywords - can show on heatmap.
        Possibly just discuss along the way. 

    Summary:
    [To do] set background color? (peach) (7)
    [To do] Make bar chart for total growth alongside trend. (15) Change orientation of grids to vertical.** Continue this.  (10)
    [To do] Make Venn. (30)
    [To do] Most cited in each topic, with heatmap. (1.5h)

    1. What is ONE insight? 
        -> 1) Highest mentions: Chemo (treatment) + Tamoxifen (drug/hormonal) + Molecular biology. 
                -> They account for .... % of total. 
                -> Are they independent or interconnected? [To do] Make Venn.  (30)
        -> 2) Highest growth: Stem (cancer stem cells - cscs), chemotherapy-resistance proteins (bcrp), ctc, immunotherapy... at ... ~200% total growth. whole brain radiotherapy -> note that doesn't mean positive. One of the top searches is paper that critiques WBRT. Ro is an antigen. 
        -> 3) Negative growth: erbeta (estrogen receptor), gemcitabine, older_adult. 
    2. What is ONE secondary benefit? 
        -> 1) Knowledge of medical terms in cancer research. 
        -> 2) Recommended list?*
    3. Data Source: Abstracts from 12000+ breast cancer review papers, published between 1997 and 2017. \nRetrieved from PubMed in June 2018. Modeled into topics using Gensim, an open-source library. 
    '''
    parent_dir = 'saved_files/1997_to_2017'
    std_footer = 'Data Source: Abstracts from 12,000+ breast cancer review papers on PubMed, published between 1997 and 2017.'

    corpus = load_corpus(parent_dir)
    models = load_models(parent_dir)

    for model in models:
        print('\n* Now postprocessing for %s topics model ...' %model.num_topics)
        if model.num_topics != 17:
            print('* --Skip-- ')
            continue

        current_dir,wordcloud_dir = make_dir(parent_dir,model)
        df = load_documents(parent_dir)

        print('\n* Preparation step: ')
        #df = get_doc_topic_scores(model,corpus,df,current_dir)
        df = load_topic_scores(model,current_dir) # for if topic_scores already retrieved. 

        print('\n* Output step: ')
        topic_sensitivity = 0.05
        #get_topic_distribution(df,model,corpus,current_dir,topic_sensitivity,wordcloud_dir,std_footer)
        get_year_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,std_footer)
        #most_rep_titles = most_representative_titles(model,df,current_dir)
        input('* Press Enter to continue.')

if __name__ == "__main__":
    main()