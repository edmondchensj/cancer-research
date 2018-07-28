import math
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from itertools import product

styling()

def styling(): 
    plt.style.use('ggplot')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.facecolor']='#FDF6D8'
    plt.rcParams['axes.labelsize'] = 'small'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['legend.fontsize'] = 'x-small'
    plt.rcParams['figure.autolayout'] = True # equiv to plt.tight_layout()
    plt.rcParams['savefig.facecolor']='#FDF6D8'
    plt.rcParams['savefig.bbox'] = 'tight' # might be bbox_inches instead
    plt.rcParams['savefig.dpi'] = 1000

def _sort_by_date(path):
    return list(sorted(os.listdir(path), key=lambda f: os.stat(os.path.join(path, f)).st_mtime))

def retrieve_wordclouds(wordcloud_dir):
    return [str(wordcloud_dir + '/' + img) for img in _sort_by_date(wordcloud_dir) if 'wordcloud_topic' in img]

def initialize_grid(num_topics,gradient):
    row = math.ceil(math.sqrt(num_topics))
    col = row
    fig = plt.figure(figsize=(row,col))
    if gradient and (math.ceil(num_topics/row) < row): # auto adjust to rectangle 
        col = col - 1
        fig.set_size_inches(col,row)
    gs = gridspec.GridSpec(row, col, wspace=0, hspace=0.10)
    return fig, gs, row, col

def _save_and_close(fig,figpath):
    fig.savefig(figpath)
    plt.close()
    print('* [wordcloudgrid.py] File saved: %s.' %figpath)
    return figpath

def get_dynamic_colors(data,cmap_relative):
    if cmap_relative: 
        max_val = abs(max(data,key=abs))
        bound = max_val + max_val/10
        norm = Normalize(vmin=-bound, vmax=bound)
        color = 'RdYlGn'
    else:
        norm = Normalize(vmin=min(data), vmax=max(data)+(max(data)/8))
        color = 'Oranges'
    cmap = cm.get_cmap(color)
    colors = list(map(lambda x: cmap(norm(x)), data))
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    return colors,sm

def generate_wc_grid(num_topics,wordcloud_dir,target_dir,gradient=False,dynamic_color=None,sm=None,cbar_label=None,tag=''):
    fig, gs, row, col = initialize_grid(num_topics,gradient)
    images = retrieve_wordclouds(wordcloud_dir)

    for i in range(num_topics):
        img = mpimg.imread(images[i])
        ax = plt.subplot(gs[i])
        ax.imshow(img)
        for pos in ['top', 'bottom', 'right', 'left']:
            if gradient:
                ax.spines[pos].set_color(dynamic_color[i])
            else:
                ax.spines[pos].set_color('#E5E7E9')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.annotate('%s'%(i+1), xy=(0,0), xytext=(0.0175,0.875), textcoords='axes fraction',fontsize='x-small', color='gray')

    if gradient: # Emphasize topics if gradient==True
        if row == col:
            gs.update(left=0,right=0.92,wspace=-0.3)
        else:
            gs.update(left=0,right=0.99,wspace=0.1)
        cbar_ax = fig.add_axes([0.9,0.12,0.02,0.76])
        sc = plt.gca().get_children()[0]
        cbar = plt.colorbar(sc,cax=cbar_ax)
        cbar.set_label('%s' %cbar_label,size='xx-small',labelpad=-0.05)
        cbar.ax.tick_params(labelsize='xx-small',pad=-0.15)

    figpath = _save_and_close(fig,f'{target_dir}/wordcloud_grid{'_'+tag}.png')
    return figpath

def gradientGrid(data,wordcloud_dir,target_dir,tag='',cmap_relative=False,cbar_label=False):
    print("* [wordcloudgrid.py] Now making wordcloud grid with color gradients ...")
    dynamic_color,sm = get_dynamic_colors(data,cmap_relative)
    figpath = generate_wc_grid(len(data),wordcloud_dir,target_dir,gradient=True,dynamic_color=dynamic_color,sm=sm,cbar_label=cbar_label,tag=tag)
    return figpath

def basicGrid(num_topics,wordcloud_dir,target_dir):
    print("* [wordcloudgrid.py] Now visualizing model as wordcloud grid...")
    figpath = generate_wc_grid(num_topics,wordcloud_dir,target_dir)
    return figpath