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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from PIL import Image
from wordcloud import WordCloud, STOPWORDS
from scipy.misc import imsave
from itertools import product

from glob import iglob
from PIL import Image

# My helper script
import graphing_tools.wordcloudgrid as wcg

class colormap_size_func(object):
    """Color func created from matplotlib colormap. The larger the word, the darker the color. 
    Parameters
    ----------
    colormap : string or matplotlib colormap
        Colormap to sample from
    Example
    -------
    >>> WordCloud(color_func=colormap_color_func("magma"))
    """
    def __init__(self, colormap, max_font_size):
        import matplotlib.pyplot as plt
        self.colormap = plt.cm.get_cmap(colormap)
        self.max_font_size = max_font_size

    def __call__(self, word, font_size, position, orientation,
                 random_state=None, **kwargs):
        if random_state is None:
            random_state = Random()
        r, g, b, _ = 255 * np.array(self.colormap(font_size / self.max_font_size))
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(r, g, b)

def get_wordcloud(model,current_dir):
    print("\n* Now creating wordclouds...")
    from PIL import Image
    icon_path = "helper_files/circle.png"
    icon = Image.open(icon_path)
    mask = Image.new("RGB", icon.size, (255,255,255))
    mask.paste(icon,icon)
    mask = np.array(mask)

    for t in range(model.num_topics):
        words = dict(model.show_topic(t, 15))
        wc = WordCloud(font_path='/Library/Fonts/HelveticaNeue.dfont', 
                       mask=mask,
                       prefer_horizontal=0.95,
                       relative_scaling=0.4,
                       background_color="white",
                       max_font_size=500,
                       color_func=colormap_size_func("Blues",500))
        plt.style.use('seaborn-notebook')
        plt.imshow(wc.generate_from_frequencies(words))
        plt.axis("off")
        imsave('%s/wordcloud_topic%s.png'%(current_dir,t+1), wc.generate_from_frequencies(words))
        #plt.title("Topic #" + str(t+1))
    print("Wordclouds saved.")
        
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

def load_models(parent_dir):
    lst = os.listdir(parent_dir+'/models')
    model_lst = filter(lambda m: 'model' in m, lst)
    models = []
    for model_file in model_lst:
        with open(str(parent_dir+'/models/'+model_file),'rb') as f:
            models.append(pickle.load(f))
    return models

def main():
    parent_dir = 'saved_files/1997_to_2017'

    corpus,id2word = load_preprocess_data(parent_dir)
    models = load_models(parent_dir)

    for model in models:
        # Select models to run. 
        num_topics = model.num_topics
        print(f'\n* Now visualizing for {num_topics} topics model ...')

        ''' To select models '''
        if not num_topics in [13,15,17]:
            print('* --Skip-- ')
            continue

        current_dir = make_dir(parent_dir,model)
        get_pyLDAvis(model,corpus,id2word,current_dir)
        get_wordcloud(model,current_dir)
        wcg.basicGrid(num_topics,wordcloud_dir=current_dir,target_dir=current_dir)

if __name__ == "__main__":
    main()