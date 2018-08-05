import math
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import colorcet as cc

def set_plot_style(): 
    plt.style.use('ggplot')

    # Background color (off-white; could consider pure white)
    plt.rcParams['axes.facecolor'] = '#FDFDFD' 
    plt.rcParams['axes.edgecolor'] = '#FDFDFD'
    plt.rcParams['figure.facecolor'] = '#FDFDFD' 
    plt.rcParams['savefig.facecolor'] = '#FDFDFD'

    # Text
    plt.rcParams['text.color'] = '#8F8F8F'  # color of topic number
    plt.rcParams['axes.labelsize'] = 'small'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['legend.fontsize'] = 'x-small'

    # Layout
    plt.rcParams['figure.autolayout'] = True # equiv to plt.tight_layout()
    plt.rcParams['savefig.bbox'] = 'tight' # might be bbox_inches instead
    plt.rcParams['savefig.dpi'] = 1000

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

def get_dynamic_colors(data,cmap_relative=False):
    if cmap_relative: 
        max_val = abs(max(data,key=abs))
        bound = max_val + max_val/20
        norm = Normalize(vmin=-bound, vmax=bound)
        color = cc.m_diverging_gwr_55_95_c38_r
    else:
        norm = Normalize(vmin=min(data), vmax=max(data)+(max(data)/7))
        color = cc.m_fire_r
    cmap = cm.get_cmap(color)
    colors = list(map(lambda x: cmap(norm(x)), data))
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    return colors,sm

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

def gradientGrid(fig,gs,data,wordcloud_dir,target_dir,cmap_relative=False,cbar_label=False):
    print("* [wordcloudgrid.py] Now making wordcloud grid with color gradients ...")
    set_plot_style()
    dynamic_color,sm = get_dynamic_colors(data,cmap_relative)
    get_wordcloud_grid(fig,gs,len(data),wordcloud_dir,target_dir,dynamic_color=dynamic_color,sm=sm,cbar_label=cbar_label,save=False)

def basicGrid(num_topics,wordcloud_dir,target_dir):
    print("* [wordcloudgrid.py] Now visualizing model as wordcloud grid...")
    set_plot_style()
    fig, gs = initialize_grid(num_topics)
    figpath = get_wordcloud_grid(fig,gs,num_topics,wordcloud_dir,target_dir)
    return figpath