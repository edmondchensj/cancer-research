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
Preparation step:
'''
def load_topic_scores(model,current_dir):
    print('* -> Loading topic scores ...')
    df = pd.read_csv('%s/df_topic_scores.csv'%current_dir)
    df.drop(columns=['Unnamed: 0'])
    return df

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
Output step:
'''

def get_topic_mentions(df,model,corpus,current_dir,topic_sensitivity,wordcloud_dir):
    print('* -> Getting Topic Mentions ...')

    topic_cols = [col for col in df.columns if 'Topic_' in col]
    topic_mentions = df[df>=topic_sensitivity].count()
    topic_mentions = topic_mentions[topic_cols]
    topic_mentions.to_csv('%s/topic_mentions.csv'%current_dir)

    topic_keywords = _topic_keywords(model,corpus)
    footnote = "$\mathregular{^{1}}$ Based on topic contribution of at least 0.05, according to Gensim's topic modeling algorithm.\nEach paper can have several topic mentions." %topic_sensitivity
    graph = gr.plot_graph(topic_mentions,topic_keywords,model,current_dir,tag='mention',title='Topic Mentions$\mathregular{^{1}}$',footnote=footnote)
    wcgrid = wcg.gradientGrid(model,topic_mentions,wordcloud_dir,current_dir,tag='mention',cbar_label='No. of Papers')
    gr.merge_graph_wcgrid(graph,wcgrid,model,current_dir,tag='mention')

def _topic_keywords(model,corpus,n=1):
    topic_keywords = []
    if n==0:
        topic_keywords = range(1,model.num_topics+1,1)
    else:
        for i in range(model.num_topics):
            words = list(dict(model.show_topic(i, n)).keys())
            words_as_string = '%s (%s,...)' %(i+1,','.join(words))
            topic_keywords.append(words_as_string)
    return topic_keywords

def get_topic_distribution(df,model,corpus,current_dir,wordcloud_dir):
    print('* -> Getting Topic Distribution ...')

    topic_cols = [col for col in df.columns if 'Topic_' in col]
    topic_dist = df['Dominant Topic'].value_counts()
    topic_dist = topic_dist.reindex(index=topic_cols).fillna(0)
    topic_dist.to_csv('%s/topic_dist.csv'%current_dir)

    topic_keywords = _topic_keywords(model,corpus)
    footnote = '$\mathregular{^{1}}$ Shows the number of papers in which the topic has the highest contribution.\n Only one main topic per paper.'
    graph = gr.plot_graph(topic_dist,topic_keywords,model,current_dir,tag='distribution',title='Topic Distribution$\mathregular{^{1}}$',footnote=footnote)
    wcgrid = wcg.gradientGrid(model,topic_dist,wordcloud_dir,current_dir,tag='distribution',cbar_label='No. of Papers')
    gr.merge_graph_wcgrid(graph,wcgrid,model,current_dir,tag='distribution')

def get_year_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir):
    print('* -> Getting Year Trends  ...')
    start_year = df['Year'].min()
    end_year = df['Year'].max()
    year_range = range(start_year,end_year+1,1)
    topic_range = range(model.num_topics)

    print('* - - > Get trend in terms of absolute papers  ...')
    _absolute_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,year_range,topic_range)

    print('* - - > Get trend in terms of proportion of total papers ...')
    _relative_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,year_range,topic_range)

def _absolute_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,year_range,topic_range):
    year_trend = gr.trend(df,topic_range,topic_sensitivity)
    graph = gr.plot_trend(year_trend,year_range,model,current_dir)
    total_growth = gr.total_growth(year_trend)
    wcgrid = wcg.gradientGrid(model,total_growth,wordcloud_dir,current_dir,tag='trend_abs',cbar_label='Total growth (%)')
    gr.merge_graph_wcgrid(graph,wcgrid,model,current_dir,tag='trend_abs')

def _relative_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,year_range,topic_range):
    year_trend_rel = gr.trend(df,topic_range,topic_sensitivity,relative=True)
    graph = gr.plot_trend(year_trend_rel,year_range,model,current_dir,relative=True)
    total_growth = gr.total_growth(year_trend_rel)
    wcgrid = wcg.gradientGrid(model,total_growth,wordcloud_dir,current_dir,tag='trend_rel',cmap_relative=True,cbar_label='Total growth (%)')
    gr.merge_graph_wcgrid(graph,wcgrid,model,current_dir,tag='trend_rel')

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
    [To do] adopt footer for data source (15), set background color? (peach) (7)
    [To do] Remove 'tumor', 'factor', 'clinical', 'trial' and preprocess and model up to 25. (10)

    [To do] Make bar chart for total growth alongside trend. (15)
    [To do] Make Venn.  (30)
    [To do] Most cited in each topic, with heatmap. (1.5h)

    1. What is ONE insight? 
        -> 1) Highest mentions: Chemo (treatment) + Tamoxifen (drug/hormonal) + Molecular biology. 
                -> They account for .... % of total. 
                -> Are they independent or interconnected? [To do] Make Venn.  (30)
        -> 2) Highest growth: Stem (cancer stem cells - cscs), chemotherapy-resistance proteins (bcrp), ctc, immunotherapy... at ... ~200% total growth. [To do] Make bar chart for total growth alongside trend. (15) Change orientation of grids to vertical. (10)
            * mentions based on 0.05~ sensitivity. 
        -> 3) Negative growth: erbeta (estrogen receptor), gemcitabine, older_adult. 
    2. What is ONE secondary benefit? 
        -> 1) Knowledge of medical terms in cancer research. 
        -> 2) Recommended list?* [To do] Most cited in each topic, with heatmap. (1.5h)
    '''
    parent_dir = 'saved_files/1997_to_2017'

    corpus = load_corpus(parent_dir)
    models = load_models(parent_dir)

    for model in models:
        print('\n* Now postprocessing for %s topics model ...' %model.num_topics)
        if model.num_topics != 13:
            print('* --Skip-- ')
            continue

        current_dir,wordcloud_dir = make_dir(parent_dir,model)
        df = load_documents(parent_dir)

        print('\n* Preparation step: ')
        #df = get_doc_topic_scores(model,corpus,df,current_dir)
        df = load_topic_scores(model,current_dir)

        print('\n* Output step: ')
        topic_sensitivity = 0.05
        get_topic_mentions(df,model,corpus,current_dir,topic_sensitivity,wordcloud_dir)
        get_topic_distribution(df,model,corpus,current_dir,wordcloud_dir)
        #get_year_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir)
        #most_rep_titles = most_representative_titles(model,df,current_dir)
        #input('* Press Enter to continue.')

if __name__ == "__main__":
    main()