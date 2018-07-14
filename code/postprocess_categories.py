import numpy as np
import pandas as pd
import pickle
import gensim.corpora as corpora
import os

def classify_keywords(df):
    '''
    This may have to be customized for each model.
    '''
    categories = ['Treatment','Diagnosis','Lymphatic',
    'Prevention','Hormone Therapy','Maternity','Genetics',
    'Chemotherapy',' Molecular Biology',
    'Radiotherapy','Epidemiology']
    df = pd.DataFrame(columns=categories).rename_index('Topics')

wp = model.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])