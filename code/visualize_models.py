import re
import numpy as np
import pandas as pd
from pprint import pprint
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

def sorted_ls(path):
    mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
    # The body of a lambda is a single expression, the result of which is used as the lambda's return value.
    return list(sorted(os.listdir(path), key=mtime))

def wordclouds_to_grid(model,vis_dir):
    print("\nNow combining wordclouds to grid...")
    width = math.ceil(math.sqrt(model.num_topics))
    print("Gridsize: %s x %s"%(width,width))
    fig = plt.figure(figsize=(width,width))
    gs = gridspec.GridSpec(width, width, wspace=0, hspace=0.1)
    #fig, ax = plt.subplots(width,width)

    images = []
    for i in sorted_ls(vis_dir):
        if 'wordcloud_topic' in i:
            images.append(i)
    print("List of wordclouds:")
    print(images)

    idx = 0
    for i,j in product(range(width),range(width)):
        if idx==model.num_topics:
            break
        else:
            img = mpimg.imread(vis_dir + '/' + images[idx])
            ax = plt.subplot(gs[i,j])
            ax.imshow(img)
            for pos in ['top', 'bottom', 'right', 'left']:
                ax.spines[pos].set_color('#E5E7E9')
            ax.set_xticks([])
            ax.set_yticks([])
            idx += 1
            '''
            ax[i,j].imshow(img)
            for pos in ['top', 'bottom', 'right', 'left']:
                ax[i,j].spines[pos].set_color('#E5E7E9')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            idx += 1
            '''

    fig.savefig('%s/wordcloud_grid.png'%vis_dir,dpi = 1000)
    print('Wordcloud grid saved.')

class colormap_size_func(object):
    """Color func created from matplotlib colormap.
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

def get_wordcloud(model,vis_dir):
    #colors = ["Blues", "Oranges", "Greens", "Purples"]
    print("\nNow creating wordclouds...")
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
        imsave('%s/wordcloud_topic%s.png'%(vis_dir,t+1), wc.generate_from_frequencies(words))
        #plt.title("Topic #" + str(t+1))
    print("Wordclouds saved.")
        
def get_pyLDAvis(model,corpus,id2word,vis_dir):
    print("\nNow we will visualize the topics using pyLDAvis.")
    vis = pyLDAvis.gensim.prepare(model, corpus, id2word, sort_topics=False)
    pyLDAvis.save_html(vis,'%s/topic_model.html'%vis_dir)
    print("PyLDAvis saved to html.")

def choose_model_from_files(filepath):
    print("See list of saved files from build_model.py:")
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

def make_dir(filepath,model):
    vis_dir = '%s/visualize_models/%stopics'%(filepath,model.num_topics)
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def load_preprocess_data(filepath):
    print('Getting data from %s ...' %filepath)
    corpus = corpora.MmCorpus('%s/preprocess/corpus.mm' %filepath)
    id2word = np.load('%s/preprocess/id2word.dict' %filepath)
    return corpus,id2word

def load_models(filepath):
    lst = os.listdir(filepath+'/models')
    model_lst = filter(lambda m: 'model' in m, lst)
    models = []
    for model_file in model_lst:
        with open(str(filepath+'/models/'+model_file),'rb') as f:
            models.append(pickle.load(f))
    return models

def main():
    filepath = 'saved_files/2007_to_2017'

    corpus,id2word = load_preprocess_data(filepath)
    models = load_models(filepath)

    for model in models:
        vis_dir = make_dir(filepath,model)
        get_pyLDAvis(model,corpus,id2word,vis_dir)
        get_wordcloud(model,vis_dir)
        wordclouds_to_grid(model,vis_dir)

if __name__ == "__main__":
    main()