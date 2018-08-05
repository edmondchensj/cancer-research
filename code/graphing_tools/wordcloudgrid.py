import math
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import colorcet as cc

''' Table of Contents:
    1   Plot Style
    3   Make Wordclouds
    4   Make Wordcloud Grids
'''

# 1     Plot Style 
def set_plot_style(): 
    plt.style.use('ggplot')

    # Background color (off-white; could consider pure white)
    plt.rcParams['axes.facecolor'] = '#FDFDFD' 
    plt.rcParams['axes.edgecolor'] = '#FDFDFD'
    plt.rcParams['figure.facecolor'] = '#FDFDFD' 
    plt.rcParams['savefig.facecolor'] = '#FDFDFD'

    # Text
    plt.rcParams['text.color'] = '#8F8F8F'  # color of topic number
    plt.rcParams['axes.labelsize'] = 'x-small'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['legend.fontsize'] = 'x-small'

    # Layout
    plt.rcParams['figure.autolayout'] = True # equiv to plt.tight_layout()
    plt.rcParams['savefig.bbox'] = 'tight' # might be bbox_inches instead
    plt.rcParams['savefig.dpi'] = 1000


# 2     Make wordclouds
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

def get_wordclouds(model,current_dir):
    ''' Makes a wordcloud for each topic and saves to .png file in current_dir. '''
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
                       color_func=colormap_size_func(cc.m_linear_blue_95_50_c20,500))
        plt.style.use('seaborn-notebook')
        plt.imshow(wc.generate_from_frequencies(words))
        plt.axis("off")
        imsave('%s/wordcloud_topic%s.png'%(current_dir,t+1), wc.generate_from_frequencies(words))
    print("Wordclouds saved.")


# 3     Make Wordcloud Grids
def get_wordcloud_grid(fig,gs,num_topics,wordcloud_dir,target_dir,dynamic_color=None,sm=None,save=True,figpath=None):
    images = retrieve_wordclouds(wordcloud_dir)

    for i in range(num_topics):
        img = mpimg.imread(images[i])
        ax = plt.subplot(gs[i])
        ax.imshow(img)
        for pos in ['top', 'bottom', 'right', 'left']:
            if dynamic_color is not None: ax.spines[pos].set_color(dynamic_color[i])
            else: ax.spines[pos].set_color('#E5E7E9')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate('%s'%(i+1), xy=(0,0), xytext=(0.0175,0.875), textcoords='axes fraction',fontsize='x-small')

    figpath = _save_and_close(fig,f'{target_dir}/wordcloud_grid.png') if save else None
    return figpath

def get_dynamic_colors(data,cmap_relative=False):
    if cmap_relative: 
        max_val = abs(max(data,key=abs))
        bound = max_val + max_val/20
        norm = Normalize(vmin=-bound, vmax=bound)
        color = cc.m_diverging_gwr_55_95_c38_r
    else:
        norm = Normalize(vmin=min(data), vmax=max(data))
        color = cc.m_linear_blue_95_50_c20
    cmap = cm.get_cmap(color)
    colors = list(map(lambda x: cmap(norm(x)), data))
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    return colors,sm

def basicGrid(num_topics,wordcloud_dir,target_dir):
    print("* [wordcloudgrid.py] Now visualizing model as wordcloud grid...")
    set_plot_style()
    fig, gs = initialize_grid(num_topics)
    figpath = get_wordcloud_grid(fig,gs,num_topics,wordcloud_dir,target_dir)
    return figpath

def _sort_by_date(path):
    return list(sorted(os.listdir(path), key=lambda f: os.stat(os.path.join(path, f)).st_mtime))

def retrieve_wordclouds(wordcloud_dir):
    return [str(wordcloud_dir + '/' + img) for img in _sort_by_date(wordcloud_dir) if 'wordcloud_topic' in img]

def initialize_grid(num_topics,gradient=False):
    row = math.ceil(math.sqrt(num_topics))
    col = row
    fig = plt.figure(figsize=(col-0.35,row))
    gs = gridspec.GridSpec(row, col, wspace=0, hspace=0.10)
    return fig, gs

def _save_and_close(fig,figpath):
    fig.savefig(figpath)
    plt.close()
    print('* [wordcloudgrid.py] File saved: %s.' %figpath)
    return figpath