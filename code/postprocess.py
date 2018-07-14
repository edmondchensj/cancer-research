import numpy as np
import pandas as pd
import pickle
import gensim.corpora as corpora
import os

def get_topic_scores(model,corpus,df,directory):
    print("\nNow let's get the topic scores for each document.")
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
    print("\nPreview topic_scores:")
    print(doc_topics_df.head(10))

    # Combine with original dataframe
    df = pd.concat([df, doc_topics_df], axis=1)
    df.to_csv('%s/df_topic_scores.csv'%directory)
    return df

def most_representative_titles(model,df,directory):
    pd.options.display.max_colwidth = 10
    print("Let's preview the dataframe with topic scores.")
    print(df.head(10))
    num_topics = model.num_topics

    print("\nNow we will get the most representative titles of each topic.")
    print("Total number of topics (just checking): %s" %num_topics)
    df_most_rep_titles = pd.DataFrame()
    for topic in range(num_topics):
        top_documents = df.nlargest(1,'Topic_%s'%(topic+1))
        df_most_rep_titles = df_most_rep_titles.append(top_documents,ignore_index=True)
    df_most_rep_titles.drop(columns=['Abstract'],inplace=True)
    print(df_most_rep_titles)
    df_most_rep_titles.to_csv('%s/most_representative_titles.csv'%directory)
    return df_most_rep_titles

def postprocess(model,corpus,df):
    pd.options.display.max_colwidth = 10
    print("\nCreating directory for this model ...")
    directory = 'saved_files/postprocess/%stopics'%model.num_topics
    os.makedirs(directory, exist_ok=True)

    print("\nLet's preview the original dataframe with our retrieved papers.")
    print(df.head(10))

    df = get_topic_scores(model,corpus,df,directory)

    df_most_rep_titles = most_representative_titles(model,df,directory)
    return df

def choose_model_from_files(filepath):
    print("\nSee list of saved files from build_model.py:")
    models = os.listdir(filepath)
    for i,model in enumerate(models):
        print(i,model)
    for i in range(3):
        try: 
            index = int(input("Select a model by its file index: "))
            break
        except: 
            error("Not a number. %d tries left."%(2-i))
    with open('saved_files/models/%s'%models[index], 'rb') as f:
        model = pickle.load(f)
    print("Model chosen: %s"%models[index])
    return model

if __name__ == "__main__":
    corpus = corpora.MmCorpus('saved_files/preprocess/corpus.mm')
    model = choose_model_from_files('saved_files/models')
    df = pd.read_csv('saved_files/preprocess/breastcancer_reviews_refined.csv')
    cols = ['PMID','Year','Title','Abstract','Total_times_cited']
    df = df[cols]
    postprocess(model,corpus,df)