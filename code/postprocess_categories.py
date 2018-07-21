import numpy as np
import pandas as pd
import pickle
import gensim.corpora as corpora
import os

def classify_keywords(model,categories,dir):
    '''
    This may have to be customized for each model.
    '''
    
    df = pd.DataFrame(columns=categories).rename_index('Topics')

wp = model.show_topic(topic_num)
topic_keywords = ", ".join([word for word, prop in wp])


def main():
    categories = ['Treatment - Chemotherapy','Diagnosis','Biology - Molecular',
    'Lifestyle-Environment Factors-Causes','Treatment - Hormonal','Maternity','Genetics',
    'Chemotherapy',' Molecular Biology',
    'Radiotherapy','Epidemiology']





if __name__ == "__main__":
    main()