import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import pandas as pd
import pickle
import gensim.corpora as corpora
import os
import seaborn as sns
import wordcloudgrid

'''
Preparation step:
'''
def load_topic_scores(model,current_dir):
    print('* -> Loading topic scores ...')
    df = pd.read_csv('%s/df_topic_scores.csv'%current_dir)
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

def load_model(parent_dir):
    with open(parent_dir, 'rb') as f:
        model = pickle.load(f)
    return model

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
    #model_lst = filter(lambda m: 'model' in m, lst)
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
    topic_mentions = np.zeros(model.num_topics)
    for i in range(model.num_topics):
        topic_mentions[i] = df[df['Topic_%s'%(i+1)]>=topic_sensitivity].shape[0]
    topic_keywords = _topic_keywords(model,corpus)
    footnote = '$\mathregular{^{1}}$ I consider a topic as "mentioned" in a document if its contribution to the document \nis greater than or equal to %s according to the Gensim topic modelling algorithm.' %topic_sensitivity
    graph = _plot_graph(topic_mentions,topic_keywords,model,current_dir,tag='mention',title='Topic Mentions$\mathregular{^{1}}$',footnote=footnote)
    wcgrid = wordcloudgrid.gradientGrid(model,topic_mentions,wordcloud_dir,current_dir,tag='mention')
    _merge_graph_wcgrid(wcgrid,graph,model,current_dir,tag='mention')

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
    topic_dist = np.zeros(model.num_topics)
    for i in range(model.num_topics):
        topic_dist[i] = df[df['Dominant Topic']=='Topic_%s'%(i+1)].shape[0]
    topic_keywords = _topic_keywords(model,corpus)
    footnote = '$\mathregular{^{1}}$ Each paper is assigned one "dominant" topic, the one with highest percentage contribution. '
    graph = _plot_graph(topic_dist,topic_keywords,model,current_dir,tag='distribution',title='Topic Distribution$\mathregular{^{1}}$',footnote=footnote)
    wcgrid = wordcloudgrid.gradientGrid(model,topic_dist,wordcloud_dir,current_dir,tag='distribution')
    _merge_graph_wcgrid(wcgrid,graph,model,current_dir,tag='distribution')

def _plot_graph(topic_dist,topic_keywords,model,current_dir,tag,title,footnote):
    fig = plt.figure()
    ax = fig.add_axes((0.2, 0.18, 0.75, 0.7)) # (left,bottom,width,height)
    y = np.arange(model.num_topics)
    ax.barh(y,topic_dist,align='center')
    ax.set_yticks(y)
    ax.set_yticklabels(topic_keywords) # might have to set fontsize
    ax.invert_yaxis() #reorder topics to start from top to bottom. 
    ax.set_xlabel('Number of papers')
    ax.set_title(title)
    ax.grid(axis='x',linestyle='--')
    plt.style.use('seaborn-notebook')
    plt.tight_layout()
    fig.text(0.6,0.025, footnote, size=8, ha="center")
    fig_path = '%s/topic_%s.png' %(current_dir,tag)
    fig.savefig(fig_path,bbox_inches='tight')
    return fig_path

def _merge_graph_wcgrid(wcgrid,graph,model,current_dir,tag):
    print('* - - -> Merging graph and wordcloud grid')
    # set axes:
    fig = plt.figure()
    graph_ax = plt.axes([0,0,0.56,1])
    wc_ax = plt.axes([0.56,0,0.44,1])
    graph_ax.imshow(mpimg.imread(graph))
    wc_ax.imshow(mpimg.imread(wcgrid))
    graph_ax.axis('off')
    graph_ax.autoscale_view('tight')
    wc_ax.axis('off')
    wc_ax.autoscale_view('tight')
    plt.tight_layout()
    fig_path = '%s/topic_%s_wc.png' %(current_dir,tag)
    fig.savefig(fig_path,bbox_inches='tight',dpi = 1000)

def get_year_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir):
    print('* -> Getting Year Trends  ...')
    start_year = df['Year'].min()
    end_year = df['Year'].max()
    year_range = range(start_year,end_year+1,1)
    topic_range = range(model.num_topics)

    # year trend in terms of absolute number of papers. 
    _absolute_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,year_range,topic_range)

    input('Pause')
    # year_trend as proportion of total papers per year. 
    _relative_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,year_range,topic_range)

def _absolute_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,year_range,topic_range):
    year_trend = _year_trend(df,topic_range,topic_sensitivity)
    graph = _plot_year_trend(year_trend,year_range,model,current_dir)
    pct_change = _percent_change(year_trend)
    wcgrid = wordcloudgrid.gradientGrid(model,pct_change,wordcloud_dir,current_dir,tag='trend_abs',cbar_label='Total growth (%)')
    _merge_graph_wcgrid(wcgrid,graph,model,current_dir,tag='trend_abs')

def _relative_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir,year_range,topic_range):
    year_trend_rel = _year_trend(df,topic_range,topic_sensitivity,relative=True)
    graph = _plot_year_trend(year_trend_rel,year_range,model,current_dir,relative=True)
    pct_change = _percent_change(year_trend_rel)
    wcgrid = wordcloudgrid.gradientGrid(model,pct_change,wordcloud_dir,current_dir,tag='trend_rel',cmap_relative=True,cbar_label='Total growth (%)')
    _merge_graph_wcgrid(wcgrid,graph,model,current_dir,tag='trend_rel')

def _percent_change(year_trend):
    print(year_trend[0].values.max())
    func = lambda x: (x.values.max() - x.values.min())/x.values.min()*100
    pct_change = list(map(func,year_trend))
    return pct_change

def _year_trend(df,topic_range,topic_sensitivity,relative=False):
    year_trend = []
    papers_per_year = df.groupby('Year').size()
    for i in topic_range:
        topic_mentions = df[df['Topic_%s'%(i+1)]>=topic_sensitivity].groupby('Year').size().to_frame('Topic_%s'%(i+1))  
        if relative==True:
            topic_mentions = topic_mentions/papers_per_year
        year_trend.append(topic_mentions)
    return year_trend

def _plot_year_trend(year_trend,year_range,model,current_dir,relative=False):
    fig, ax = plt.subplots()
    plt.style.use('seaborn-notebook')

    linestyle = ['-','--',':','-.'] # alternate linestyles to differentiate topics better.
    for i in range(model.num_topics):
        ax.plot(year_range,year_trend[i],label="Topic %s"%(i+1),linestyle=linestyle[i%4])

    xticks = list(year_range)
    del xticks[1::2]
    ax.xaxis.set_ticks(xticks)
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion of total yearly papers (%)" if relative else "No. of Papers")
    ax.set_title("Topic Mentions across years")
    ax.grid(linestyle='--')
    legend_elements = []
    ax.legend(loc="upper left",fontsize='small')
    #plt.tight_layout()
    fig_path = '%s/topic_trend%s.png' %(current_dir,'_rel' if relative else '')
    fig.savefig(fig_path,bbox_inches='tight')
    return fig_path

def most_cited_per_topic():
    void()

def main():
    '''
    Preparation:
    1. Get topic scores for each document

    Output:
    1. Distribution of research papers with mentions of topic (also share link to dominant topic distribution.)
    2. Year-on-year trend* -> interested to know about. 
    3. Most cited docs in each topic and most representative titles for each topic. 
    4. (Discuss: Go in depth with the most popular 3-5 topics. Identify keywords - can show on heatmap.)
    '''
    parent_dir = 'saved_files/1997_to_2017'

    corpus = load_corpus(parent_dir)
    models = load_models(parent_dir)

    for model in models:

        current_dir,wordcloud_dir = make_dir(parent_dir,model)
        df = load_documents(parent_dir)

        print('\n* Preparation step: ')
        #df = get_doc_topic_scores(model,corpus,df,current_dir)
        df = load_topic_scores(model,current_dir)

        print('\n* Output step: ')
        topic_sensitivity = 0.005
        #get_topic_mentions(df,model,corpus,current_dir,topic_sensitivity,wordcloud_dir)
        #get_topic_distribution(df,model,corpus,current_dir,wordcloud_dir)
        get_year_trend(df,model,current_dir,topic_sensitivity,wordcloud_dir)
        '''
        Make year-trend as percentage of total papers each year.
        '''
        most_rep_titles = most_representative_titles(model,df,current_dir)

if __name__ == "__main__":
    main()